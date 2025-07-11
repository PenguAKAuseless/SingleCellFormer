import os
import glob
import sys
import gc
import logging
from datetime import datetime

# Add the project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import json
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import cudf
import cugraph
import rmm
from sklearn.metrics import calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
import umap
import matplotlib.pyplot as plt
import anndata
from models.scEncoder import scEncoder
from data.SingleCellDataset import SingleCellDataset
from utils.utils import load_vocabulary
import shutil
from torch.cuda.amp import autocast
import scanpy as sc
import time
import psutil
import GPUtil
from memory_profiler import profile

# Configuration
DATA_DIR = "/mnt/nasdev2/pengu-space/eval"  # Directory containing .h5ad files
GENE_VOCAB_PATH = "vocab/gene_vocab.json"
CELL_TYPE_VOCAB_PATH = "vocab/celltype_vocab.json"
CHECKPOINT_PATH = "final/encoder_final_model.pth"
SEQ_LEN = 512
NUM_BINS = 51
D_MODEL = 128
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT = 0.1
BATCH_SIZE = 64  # Reduced from 256 to lower memory usage
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
N_CLUSTERS = 10  # Target number of clusters for Leiden
MAX_EMBEDDINGS_PER_BATCH = 50000  # Reduced from 100000 to limit memory
UMAP_SAMPLE_SIZE = 25000  # Reduced from 50000 for UMAP visualization
METRIC_SAMPLE_SIZE = 25000  # Reduced from 50000 for metrics computation
K_NEIGHBORS = 15  # Number of neighbors for KNN graph in Leiden

def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/clustering_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.info(f"Logging started. Log file: {log_file}")
    return logger

class ScalableClustering:
    def __init__(self, n_clusters: int, batch_size: int = 5000, temp_dir: str = "temp_clustering"):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.temp_dir = temp_dir
        self.embedding_files = []
        self.total_samples = 0
        os.makedirs(self.temp_dir, exist_ok=True)

    def save_embedding_batch(self, embeddings: np.ndarray, batch_idx: int) -> str:
        filename = os.path.join(self.temp_dir, f"embeddings_batch_{batch_idx}.npy")
        np.save(filename, embeddings)
        return filename

    def load_embedding_batch(self, filename: str) -> np.ndarray:
        return np.load(filename)

    def cleanup_temp_files(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def get_embedding_iterator(self):
        for filename in self.embedding_files:
            embeddings = self.load_embedding_batch(filename)
            for i in range(0, len(embeddings), self.batch_size):
                yield embeddings[i:i + self.batch_size]

    def get_sample_for_metrics(self, sample_size: int) -> np.ndarray:
        samples = []
        samples_per_file = sample_size // len(self.embedding_files) if self.embedding_files else sample_size
        for filename in self.embedding_files:
            embeddings = self.load_embedding_batch(filename)
            if len(embeddings) <= samples_per_file:
                cuda_array = cudf.DataFrame(embeddings)
                samples.append(cuda_array.to_pandas())
            else:
                indices = np.random.choice(len(embeddings), samples_per_file, replace=False)
                cuda_array = cudf.DataFrame(embeddings[indices])
                samples.append(cuda_array.to_pandas())
        return np.concatenate(samples, axis=0) if samples else np.array([])

def clear_gpu_memory():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        logger.debug("GPU memory cleared successfully")
    except Exception as e:
        logger.warning(f"Failed to clear GPU memory: {e}")

def extract_embeddings(model, dataloader, device, max_embeddings_per_batch=50000):
    logger = logging.getLogger(__name__)
    clustering_manager = ScalableClustering(N_CLUSTERS, batch_size=5000)
    model.eval()
    current_batch_embeddings = []
    batch_idx = 0
    
    logger.info("Starting embedding extraction...")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i % 50 == 0:
                clear_gpu_memory()
                if i > 0:
                    logger.info(f"Processed {i} batches, current memory usage logged")
            
            # Extract inputs from batch
            gene_ids = batch['gene_ids'].to(device, non_blocking=True)
            gene_expr = batch['gene_expr'].to(device, non_blocking=True)
            cell_type = batch['cell_type'].to(device, non_blocking=True)
            
            # Use updated autocast syntax
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available() and device.startswith('cuda')):
                mlm_gene_id_out, mlm_gene_expr_out, contrastive_out, attn_mask = model(
                    gene_ids, gene_expr, cell_type
                )
            
            # Convert to float32 to avoid cuDF compatibility issues
            embeddings = contrastive_out.detach().cpu().numpy().astype(np.float32)
            current_batch_embeddings.append(embeddings)
            
            if len(current_batch_embeddings) * BATCH_SIZE >= max_embeddings_per_batch:
                batch_embeddings = np.concatenate(current_batch_embeddings, axis=0)
                filename = clustering_manager.save_embedding_batch(batch_embeddings, batch_idx)
                clustering_manager.embedding_files.append(filename)
                clustering_manager.total_samples += len(batch_embeddings)
                current_batch_embeddings = []
                batch_idx += 1
                clear_gpu_memory()
                logger.info(f"Saved embedding batch {batch_idx}, total samples: {clustering_manager.total_samples}")
    
    if current_batch_embeddings:
        batch_embeddings = np.concatenate(current_batch_embeddings, axis=0)
        filename = clustering_manager.save_embedding_batch(batch_embeddings, batch_idx)
        clustering_manager.embedding_files.append(filename)
        clustering_manager.total_samples += len(batch_embeddings)
        clear_gpu_memory()
        logger.info(f"Final batch saved. Total embeddings extracted: {clustering_manager.total_samples}")
    
    return clustering_manager

def perform_leiden_clustering(clustering_manager: ScalableClustering, n_neighbors: int = K_NEIGHBORS, resolution: float = 1.0):
    """
    Perform Leiden clustering using RAPIDS cuGraph for GPU acceleration
    """
    try:
        # Initialize RAPIDS memory manager
        rmm.reinitialize(
            managed_memory=True,
            pool_allocator=True,
            initial_pool_size=2**30  # 1GB initial pool
        )

        print("Loading all embeddings for clustering...")
        # Collect all embeddings
        all_embeddings = []
        for filename in clustering_manager.embedding_files:
            embeddings = clustering_manager.load_embedding_batch(filename)
            all_embeddings.append(embeddings)
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        print(f"Loaded {len(all_embeddings)} embeddings for clustering")

        # Convert to float32 if needed (cuDF doesn't support float16)
        if all_embeddings.dtype == np.float16:
            print("Converting embeddings from float16 to float32 for cuDF compatibility")
            all_embeddings = all_embeddings.astype(np.float32)

        # Convert embeddings to cuDF DataFrame for GPU processing
        embeddings_df = cudf.DataFrame(all_embeddings)

        # Build KNN graph using cuML
        print("Building KNN graph on GPU...")
        from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
        nn_model = cuNearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean')
        nn_model.fit(embeddings_df)
        distances, indices = nn_model.kneighbors(embeddings_df)

        # Create edge list for cuGraph
        print("Creating cuGraph from KNN graph...")
        sources = []
        targets = []
        weights = []
        for i in range(len(all_embeddings)):
            for j in range(1, n_neighbors + 1):  # Skip self-loops
                neighbor_idx = indices.iloc[i, j]
                weight = 1.0 / (1.0 + distances.iloc[i, j])  # Convert distance to similarity
                sources.append(i)
                targets.append(neighbor_idx)
                weights.append(weight)

        # Create cuDF DataFrame for edges
        edge_list = cudf.DataFrame({
            'src': cudf.Series(sources, dtype='int32'),
            'dst': cudf.Series(targets, dtype='int32'),
            'weight': cudf.Series(weights, dtype='float32')
        })

        # Create cuGraph graph
        G = cugraph.Graph()
        G.from_cudf_edgelist(edge_list, source='src', destination='dst', edge_attr='weight')

        # Perform GPU-accelerated Leiden clustering
        print("Performing GPU Leiden clustering...")
        start_time = time.time()
        leiden_result = cugraph.leiden(G, resolution=resolution)
        end_time = time.time()
        print(f"GPU Leiden clustering completed in {end_time - start_time:.2f} seconds")

        # Extract cluster labels
        cluster_labels = leiden_result['partition'].to_pandas().values
        print(f"Found {len(np.unique(cluster_labels))} clusters")

        return cluster_labels

    except ImportError as e:
        print(f"RAPIDS not available: {e}. Falling back to CPU-based clustering...")
        # Fallback to original CPU-based implementation
        import igraph as ig
        import leidenalg

        all_embeddings = []
        for filename in clustering_manager.embedding_files:
            embeddings = clustering_manager.load_embedding_batch(filename)
            all_embeddings.append(embeddings)
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        print(f"Loaded {len(all_embeddings)} embeddings for clustering")

        # Convert to float32 if needed for consistency
        if all_embeddings.dtype == np.float16:
            print("Converting embeddings from float16 to float32")
            all_embeddings = all_embeddings.astype(np.float32)

        print("Building KNN graph on CPU...")
        nn_model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean', n_jobs=-1)
        nn_model.fit(all_embeddings)
        distances, indices = nn_model.kneighbors(all_embeddings)

        edges = []
        weights = []
        for i in range(len(all_embeddings)):
            for j in range(1, n_neighbors + 1):
                neighbor_idx = indices[i, j]
                weight = 1.0 / (1.0 + distances[i, j])
                edges.append((i, neighbor_idx))
                weights.append(weight)

        print("Creating igraph from KNN graph...")
        ig_graph = ig.Graph(n=len(all_embeddings), edges=edges, directed=False)
        ig_graph.es['weight'] = weights
        ig_graph.simplify(combine_edges='sum')

        print("Performing CPU Leiden clustering...")
        try:
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.RBConfigurationVertexPartition,
                weights='weight',
                n_iterations=-1,
                resolution_parameter=resolution
            )
        except Exception as e:
            print(f"RBConfigurationVertexPartition failed: {e}")
            print("Falling back to ModularityVertexPartition...")
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.ModularityVertexPartition,
                weights='weight',
                n_iterations=-1
            )

        cluster_labels = np.array(partition.membership)
        print(f"Found {len(np.unique(cluster_labels))} clusters")
        return cluster_labels

def compute_metrics(clustering_manager: ScalableClustering, cluster_labels: np.ndarray):
    metrics = {}
    sample_size = min(METRIC_SAMPLE_SIZE, clustering_manager.total_samples)
    if sample_size < 2:
        return metrics  # Not enough samples to compute metrics
    
    sample_embeddings = clustering_manager.get_sample_for_metrics(sample_size)
    actual_sample_size = len(sample_embeddings)
    
    sample_indices = np.random.choice(len(cluster_labels), actual_sample_size, replace=False)
    sample_labels = cluster_labels[sample_indices]
    
    if len(sample_embeddings) > 1 and len(np.unique(sample_labels)) > 1:
        metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(sample_embeddings, sample_labels))
    
    return metrics

def create_umap_visualization(clustering_manager: ScalableClustering, cluster_labels: np.ndarray):
    sample_embeddings = clustering_manager.get_sample_for_metrics(UMAP_SAMPLE_SIZE)
    actual_sample_size = len(sample_embeddings)
    
    sample_indices = np.random.choice(len(cluster_labels), actual_sample_size, replace=False)
    sample_labels = cluster_labels[sample_indices]
    
    sample_labels = sample_labels.astype(int)
    
    print("Computing UMAP embedding...")
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
    emb_2d = umap_reducer.fit_transform(sample_embeddings)
    
    # Ensure emb_2d is a numpy array
    if not isinstance(emb_2d, np.ndarray):
        emb_2d = np.array(emb_2d)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=sample_labels, cmap='tab10', s=20, alpha=0.7)
    plt.title(f"scEncoder Clusters (UMAP, n={len(sample_embeddings)})", fontsize=14)
    plt.xlabel("UMAP 1", fontsize=12)
    plt.ylabel("UMAP 2", fontsize=12)
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/scencoder_clusters_umap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("UMAP visualization saved to output/scencoder_clusters_umap.png")
    
    return emb_2d

def analyze_clustering_performance(adata, resolution=0.5):
    """Analyze current Leiden clustering performance"""
    
    # Monitor CPU and GPU usage
    print("=== Performance Analysis ===")
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"Available RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    # Time the clustering
    start_time = time.time()
    cpu_start = psutil.cpu_percent()
    
    # Current leiden implementation
    sc.tl.leiden(adata, resolution=resolution)
    
    end_time = time.time()
    cpu_end = psutil.cpu_percent()
    
    print(f"Clustering time: {end_time - start_time:.2f} seconds")
    print(f"CPU usage: {cpu_end - cpu_start:.1f}%")
    
    return adata

def optimized_leiden_gpu(adata, resolution=0.5, use_rapids=True):
    """GPU-optimized Leiden clustering"""
    
    if use_rapids:
        try:
            import cudf
            import cugraph
            import rmm
            
            # Initialize RAPIDS memory pool
            rmm.reinitialize(
                managed_memory=True,
                pool_allocator=True,
                initial_pool_size=2**30  # 1GB
            )
            
            # Convert to GPU-friendly format
            print("Converting data to GPU format...")
            
            # Get connectivity matrix
            if 'connectivities' not in adata.obsp:
                sc.pp.neighbors(adata, use_rep='X_pca')
            
            # Convert sparse matrix to COO format for GPU
            coo = adata.obsp['connectivities'].tocoo()
            
            # Create GPU graph
            sources = cudf.Series(coo.row, dtype='int32')
            targets = cudf.Series(coo.col, dtype='int32')
            weights = cudf.Series(coo.data, dtype='float32')
            
            edge_list = cudf.DataFrame({
                'src': sources,
                'dst': targets,
                'weight': weights
            })
            
            # Create graph
            G = cugraph.Graph()
            G.from_cudf_edgelist(edge_list, 
                                source='src', 
                                destination='dst', 
                                edge_attr='weight')
            
            # Run GPU Leiden
            print("Running GPU Leiden clustering...")
            start_time = time.time()
            
            leiden_result = cugraph.leiden(G, resolution=resolution)
            
            end_time = time.time()
            print(f"GPU Leiden time: {end_time - start_time:.2f} seconds")
            
            # Convert back to CPU and assign to adata
            clusters = leiden_result['partition'].to_pandas().values
            adata.obs['leiden_gpu'] = clusters.astype('category')
            
        except ImportError:
            print("RAPIDS not available, falling back to optimized CPU version")
            return optimized_leiden_cpu(adata, resolution)
    
    else:
        return optimized_leiden_cpu(adata, resolution)
    
    return adata

def optimized_leiden_cpu(adata, resolution=0.5):
    """CPU-optimized Leiden clustering with better parallelization"""
    
    # Use igraph backend with optimizations
    import igraph as ig
    import leidenalg
    
    print("Running optimized CPU Leiden...")
    
    # Ensure we have connectivity matrix
    if 'connectivities' not in adata.obsp:
        sc.pp.neighbors(adata, use_rep='X_pca', n_jobs=-1)  # Use all cores
    
    # Convert to igraph format more efficiently
    adj_matrix = adata.obsp['connectivities']
    
    # Create igraph object
    sources, targets = adj_matrix.nonzero()
    weights = adj_matrix.data
    
    # Build graph more efficiently
    g = ig.Graph(n=adj_matrix.shape[0], directed=False)
    g.add_edges(list(zip(sources, targets)))
    g.es['weight'] = weights
    
    # Optimize Leiden parameters
    start_time = time.time()
    
    # Use optimized Leiden with better parameters
    partition = leidenalg.find_partition(
        g, 
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        n_iterations=-1,  # Run until convergence
        max_comm_size=0,  # No limit on community size
        seed=42  # Fixed seed for reproducibility
    )
    
    end_time = time.time()
    print(f"Optimized Leiden time: {end_time - start_time:.2f} seconds")
    
    # Get cluster labels
    clusters = np.array(partition.membership)
    adata.obs['leiden_optimized'] = clusters.astype('category')
    
    return adata

if __name__ == "__main__":
    logger = setup_logging()
    logger.info(f"Using device: {DEVICE}")
    
    # Load vocabularies
    logger.info("Loading vocabularies...")
    gene_vocab = load_vocabulary(GENE_VOCAB_PATH)
    cell_type_vocab = load_vocabulary(CELL_TYPE_VOCAB_PATH)
    
    if gene_vocab is None:
        logger.error("Gene vocabulary is required")
        raise ValueError("Gene vocabulary is required")
    
    logger.info(f"Loaded gene vocabulary with {len(gene_vocab)} genes")
    if cell_type_vocab:
        logger.info(f"Loaded cell type vocabulary with {len(cell_type_vocab)} types")
    
    # Load datasets
    logger.info(f"Looking for .h5ad files in {DATA_DIR}")
    data_files = glob.glob(os.path.join(DATA_DIR, "*.h5ad"))
    if not data_files:
        logger.error(f"No .h5ad files found in {DATA_DIR}")
        raise FileNotFoundError(f"No .h5ad files found in {DATA_DIR}")
    
    logger.info(f"Found {len(data_files)} data files")
    
    datasets = []
    for data_file in data_files:
        logger.info(f"Loading {data_file}...")
        adata = anndata.read_h5ad(data_file)
        logger.info(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes from {os.path.basename(data_file)}")
        dataset = SingleCellDataset(
            adata, gene_vocab, cell_type_vocab,
            num_bins=NUM_BINS, seq_len=SEQ_LEN
        )
        datasets.append(dataset)
    
    combined_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(
        combined_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True if torch.cuda.is_available() else False
    )
    
    logger.info(f"Total samples in dataset: {len(combined_dataset)}")
    
    # Load model
    logger.info("Loading model...")
    try:
        model = scEncoder(
            gene_vocab_size=len(gene_vocab),
            cell_vocab_size=len(cell_type_vocab) if cell_type_vocab else 1,
            num_bins=NUM_BINS,
            seq_len=SEQ_LEN,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            device=DEVICE
        ).to(DEVICE)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # Extract embeddings
    logger.info("Starting embedding extraction...")
    start_time = datetime.now()
    clustering_manager = extract_embeddings(model, dataloader, DEVICE)
    extraction_time = datetime.now() - start_time
    logger.info(f"Embedding extraction completed in {extraction_time}")
    
    # Perform Leiden clustering
    logger.info("Starting Leiden clustering...")
    start_time = datetime.now()
    cluster_labels = perform_leiden_clustering(clustering_manager, n_neighbors=K_NEIGHBORS)
    clustering_time = datetime.now() - start_time
    logger.info(f"Leiden clustering completed in {clustering_time}")
    
    # Ensure cluster labels are integers
    cluster_labels = cluster_labels.astype(int)
    n_clusters = len(np.unique(cluster_labels))
    logger.info(f"Generated {n_clusters} clusters")
    
    # Clear model from memory
    del model
    clear_gpu_memory()
    logger.info("Model cleared from memory")
    
    # Compute metrics
    logger.info("Computing clustering metrics...")
    metrics = compute_metrics(clustering_manager, cluster_labels)
    with open("output/clustering_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    logger.info("Clustering Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Create UMAP visualization
    logger.info("Creating UMAP visualization...")
    emb_2d = create_umap_visualization(clustering_manager, cluster_labels)
    
    # Save results
    logger.info("Saving results...")
    np.save("output/scencoder_cluster_labels.npy", cluster_labels)
    if isinstance(emb_2d, np.ndarray):
        np.save("output/scencoder_umap_embeddings.npy", emb_2d)
    
    metadata = {
        'total_samples': clustering_manager.total_samples,
        'n_clusters': len(np.unique(cluster_labels)),
        'batch_size': BATCH_SIZE,
        'num_embedding_files': len(clustering_manager.embedding_files),
        'k_neighbors': K_NEIGHBORS,
        'extraction_time': str(extraction_time),
        'clustering_time': str(clustering_time)
    }
    with open("output/clustering_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    clustering_manager.cleanup_temp_files()
    logger.info("Temporary files cleaned up")
    
    logger.info(f"Clustering analysis complete!")
    logger.info(f"Processed {clustering_manager.total_samples} samples")
    logger.info(f"Generated {len(np.unique(cluster_labels))} clusters")
    logger.info(f"Results saved in 'output/' directory")
    logger.info("Analysis completed successfully")