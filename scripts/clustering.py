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
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
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
        self.cell_type_files = []  # Add this to store cell type files
        self.total_samples = 0
        os.makedirs(self.temp_dir, exist_ok=True)

    def save_embedding_batch(self, embeddings: np.ndarray, batch_idx: int) -> str:
        # Ensure embeddings are in float32 format before saving
        if embeddings.dtype == np.float16:
            embeddings = embeddings.astype(np.float32)
        filename = os.path.join(self.temp_dir, f"embeddings_batch_{batch_idx}.npy")
        np.save(filename, embeddings)
        return filename

    def save_cell_type_batch(self, cell_types: np.ndarray, batch_idx: int) -> str:
        """Save cell types for a batch"""
        filename = os.path.join(self.temp_dir, f"cell_types_batch_{batch_idx}.npy")
        np.save(filename, cell_types)
        return filename

    def load_embedding_batch(self, filename: str) -> np.ndarray:
        embeddings = np.load(filename)
        # Ensure loaded embeddings are in float32 format
        if embeddings.dtype == np.float16:
            embeddings = embeddings.astype(np.float32)
        return embeddings

    def load_cell_type_batch(self, filename: str) -> np.ndarray:
        """Load cell types for a batch"""
        return np.load(filename)

    def get_all_cell_types(self) -> np.ndarray:
        """Load all cell types from saved batches"""
        all_cell_types = []
        for filename in self.cell_type_files:
            cell_types = self.load_cell_type_batch(filename)
            all_cell_types.append(cell_types)
        return np.concatenate(all_cell_types, axis=0) if all_cell_types else np.array([])

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
            # Ensure embeddings are float32 before any operations
            if embeddings.dtype == np.float16:
                embeddings = embeddings.astype(np.float32)
            
            if len(embeddings) <= samples_per_file:
                samples.append(embeddings)
            else:
                indices = np.random.choice(len(embeddings), samples_per_file, replace=False)
                sampled_embeddings = embeddings[indices]
                # Ensure sampled embeddings are float32
                if sampled_embeddings.dtype == np.float16:
                    sampled_embeddings = sampled_embeddings.astype(np.float32)
                samples.append(sampled_embeddings)
        
        result = np.concatenate(samples, axis=0) if samples else np.array([])
        # Final check to ensure result is float32
        if result.dtype == np.float16:
            result = result.astype(np.float32)
        return result

    def get_sample_cell_types(self, sample_size: int, sample_indices: np.ndarray) -> np.ndarray:
        """Get cell types corresponding to sampled embeddings"""
        all_cell_types = self.get_all_cell_types()
        if len(all_cell_types) == 0:
            return np.array([])
        return all_cell_types[sample_indices]
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
    current_batch_cell_types = []  # Add this to store cell types
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
            
            # Store cell types for visualization
            cell_types_batch = cell_type.detach().cpu().numpy()
            current_batch_cell_types.append(cell_types_batch)
            
            if len(current_batch_embeddings) * BATCH_SIZE >= max_embeddings_per_batch:
                batch_embeddings = np.concatenate(current_batch_embeddings, axis=0)
                batch_cell_types = np.concatenate(current_batch_cell_types, axis=0)
                
                # Save embeddings
                filename = clustering_manager.save_embedding_batch(batch_embeddings, batch_idx)
                clustering_manager.embedding_files.append(filename)
                
                # Save cell types
                cell_type_filename = clustering_manager.save_cell_type_batch(batch_cell_types, batch_idx)
                clustering_manager.cell_type_files.append(cell_type_filename)
                
                clustering_manager.total_samples += len(batch_embeddings)
                current_batch_embeddings = []
                current_batch_cell_types = []
                batch_idx += 1
                clear_gpu_memory()
                logger.info(f"Saved embedding batch {batch_idx}, total samples: {clustering_manager.total_samples}")
    
    if current_batch_embeddings:
        batch_embeddings = np.concatenate(current_batch_embeddings, axis=0)
        batch_cell_types = np.concatenate(current_batch_cell_types, axis=0)
        
        # Save final embeddings
        filename = clustering_manager.save_embedding_batch(batch_embeddings, batch_idx)
        clustering_manager.embedding_files.append(filename)
        
        # Save final cell types
        cell_type_filename = clustering_manager.save_cell_type_batch(batch_cell_types, batch_idx)
        clustering_manager.cell_type_files.append(cell_type_filename)
        
        clustering_manager.total_samples += len(batch_embeddings)
        clear_gpu_memory()
        logger.info(f"Final batch saved. Total embeddings extracted: {clustering_manager.total_samples}")
    
    return clustering_manager

def perform_cpu_clustering_fallback(clustering_manager, n_neighbors, resolution):
    """CPU-based clustering fallback"""
    print("Using CPU clustering fallback...")
    
    all_embeddings = []
    for filename in clustering_manager.embedding_files:
        embeddings = clustering_manager.load_embedding_batch(filename)
        if embeddings.dtype == np.float16:
            embeddings = embeddings.astype(np.float32)
        all_embeddings.append(embeddings)
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Loaded {len(all_embeddings)} embeddings for CPU clustering")
    
    return perform_chunked_clustering(all_embeddings, n_neighbors, resolution)

def perform_leiden_clustering(clustering_manager: ScalableClustering, n_neighbors: int = K_NEIGHBORS, resolution: float = 1.0):
    """
    Perform Leiden clustering with better error handling and memory management
    """
    print("=== Starting Leiden Clustering ===")
    monitor_memory()
    
    # Get the correct GPU device number from DEVICE string
    gpu_device = 0
    if DEVICE.startswith('cuda:'):
        gpu_device = int(DEVICE.split(':')[1])
    
    try:
        # More conservative RAPIDS memory initialization
        try:
            if torch.cuda.is_available():
                available_memory = torch.cuda.get_device_properties(gpu_device).total_memory
                pool_size = min(2**28, available_memory // 8)  # Use 1/8 of GPU memory
                print(f"Initializing RMM with {pool_size / (1024**3):.1f}GB pool on GPU {gpu_device}")
                
                rmm.reinitialize(
                    managed_memory=False,
                    pool_allocator=True,
                    initial_pool_size=pool_size
                )
            else:
                print("CUDA not available, using CPU clustering")
                raise ImportError("CUDA not available")
        except Exception as e:
            print(f"Failed to initialize RMM: {e}, falling back to CPU clustering")
            raise ImportError("RMM initialization failed")

        print("Loading all embeddings for clustering...")
        all_embeddings = []
        total_memory_needed = 0
        
        for i, filename in enumerate(clustering_manager.embedding_files):
            print(f"Loading batch {i+1}/{len(clustering_manager.embedding_files)}")
            embeddings = clustering_manager.load_embedding_batch(filename)
            if embeddings.dtype == np.float16:
                embeddings = embeddings.astype(np.float32)
            all_embeddings.append(embeddings)
            total_memory_needed += embeddings.nbytes
            
            # Monitor memory after each batch
            if i % 3 == 0:  # Every 3 batches
                monitor_memory()
        
        print(f"Total memory needed: {total_memory_needed / (1024**3):.2f} GB")
        
        # Check if we have enough GPU memory (use correct GPU device)
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(gpu_device).total_memory
            memory_multiplier = 4  # Need 4x for processing
            if total_memory_needed * memory_multiplier > gpu_memory:
                print(f"Not enough GPU memory: need {total_memory_needed * memory_multiplier / (1024**3):.1f}GB, have {gpu_memory / (1024**3):.1f}GB")
                print("Falling back to CPU clustering")
                raise ImportError("Insufficient GPU memory")
        
        # Force CPU clustering for large datasets to avoid crashes
        if len(clustering_manager.embedding_files) > 10 or clustering_manager.total_samples > 500000:
            print(f"Large dataset detected ({clustering_manager.total_samples} samples), using CPU clustering for stability")
            raise ImportError("Large dataset - using CPU for stability")
        
        print("Concatenating embeddings...")
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        print(f"Loaded {len(all_embeddings)} embeddings for clustering (dtype: {all_embeddings.dtype})")
        monitor_memory()

        # Process in smaller chunks to avoid memory issues
        chunk_size = min(100000, len(all_embeddings))  # Increased chunk size
        if len(all_embeddings) > chunk_size:
            print(f"Processing in chunks of {chunk_size} samples")
            return perform_chunked_clustering(all_embeddings, n_neighbors, resolution, chunk_size)

        # Convert embeddings to cuDF DataFrame with error handling
        try:
            print("Creating cuDF DataFrame...")
            # Set the correct GPU device for cuDF
            with torch.cuda.device(gpu_device):
                embeddings_df = cudf.DataFrame(all_embeddings)
            print(f"cuDF DataFrame created successfully with shape: {embeddings_df.shape}")
            monitor_memory()
        except Exception as e:
            print(f"Failed to create cuDF DataFrame: {e}")
            raise ImportError("cuDF DataFrame creation failed")

        # Build KNN graph using cuML with memory monitoring
        print("Building KNN graph on GPU...")
        try:
            from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
            with torch.cuda.device(gpu_device):
                nn_model = cuNearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean')
                nn_model.fit(embeddings_df)
                distances, indices = nn_model.kneighbors(embeddings_df)
            print("KNN graph built successfully")
            monitor_memory()
        except Exception as e:
            print(f"Failed to build KNN graph: {e}")
            raise ImportError("KNN graph construction failed")

        # Create edge list with progress monitoring
        print("Creating edge list for cuGraph...")
        sources = []
        targets = []
        weights = []
        
        batch_size = 1000  # Process in smaller batches
        for start_idx in range(0, len(all_embeddings), batch_size):
            end_idx = min(start_idx + batch_size, len(all_embeddings))
            
            if start_idx % 10000 == 0:
                print(f"Processing nodes {start_idx}/{len(all_embeddings)}")
                monitor_memory()
                # Clear cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            for i in range(start_idx, end_idx):
                for j in range(1, n_neighbors + 1):
                    try:
                        neighbor_idx = int(indices.iloc[i, j])
                        weight = float(1.0 / (1.0 + distances.iloc[i, j]))
                        sources.append(i)
                        targets.append(neighbor_idx)
                        weights.append(weight)
                    except Exception as e:
                        print(f"Error processing edge {i},{j}: {e}")
                        continue

        print(f"Created {len(sources)} edges")

        # Create cuDF DataFrame for edges with error handling
        try:
            print("Creating edge list DataFrame...")
            with torch.cuda.device(gpu_device):
                edge_list = cudf.DataFrame({
                    'src': cudf.Series(sources, dtype='int32'),
                    'dst': cudf.Series(targets, dtype='int32'),
                    'weight': cudf.Series(weights, dtype='float32')
                })
            print("Edge list DataFrame created successfully")
            monitor_memory()
        except Exception as e:
            print(f"Failed to create edge list DataFrame: {e}")
            raise ImportError("Edge list creation failed")

        # Create cuGraph graph
        try:
            print("Creating cuGraph...")
            with torch.cuda.device(gpu_device):
                G = cugraph.Graph()
                G.from_cudf_edgelist(edge_list, source='src', destination='dst', edge_attr='weight')
            print("cuGraph created successfully")
            monitor_memory()
        except Exception as e:
            print(f"Failed to create cuGraph: {e}")
            raise ImportError("cuGraph creation failed")

        # Perform GPU-accelerated Leiden clustering
        print("Performing GPU Leiden clustering...")
        try:
            start_time = time.time()
            with torch.cuda.device(gpu_device):
                leiden_result = cugraph.leiden(G, resolution=resolution)
            end_time = time.time()
            print(f"GPU Leiden clustering completed in {end_time - start_time:.2f} seconds")
            monitor_memory()
        except Exception as e:
            print(f"GPU Leiden clustering failed: {e}")
            raise ImportError("Leiden clustering failed")

        # Extract cluster labels
        try:
            cluster_labels = leiden_result['partition'].to_pandas().values
            print(f"Found {len(np.unique(cluster_labels))} clusters")
            return cluster_labels
        except Exception as e:
            print(f"Failed to extract cluster labels: {e}")
            raise ImportError("Label extraction failed")

    except Exception as e:
        print(f"GPU clustering failed: {e}. Falling back to CPU-based clustering...")
        return perform_cpu_clustering_fallback(clustering_manager, n_neighbors, resolution)

def perform_chunked_clustering(embeddings, n_neighbors, resolution, chunk_size=50000):
    """Process clustering in chunks to avoid memory issues"""
    print(f"Processing {len(embeddings)} embeddings in chunks of {chunk_size}")
    
    # Use CPU-based clustering for large datasets
    from sklearn.neighbors import NearestNeighbors
    import igraph as ig
    import leidenalg
    
    print("Building KNN graph on CPU (chunked processing)...")
    nn_model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean', n_jobs=-1)
    nn_model.fit(embeddings)
    
    print("Finding neighbors...")
    distances, indices = nn_model.kneighbors(embeddings)
    
    print("Creating graph edges...")
    edges = []
    weights = []
    for i in range(len(embeddings)):
        if i % 10000 == 0:
            print(f"Processing node {i}/{len(embeddings)}")
        for j in range(1, n_neighbors + 1):
            neighbor_idx = indices[i, j]
            weight = 1.0 / (1.0 + distances[i, j])
            edges.append((i, neighbor_idx))
            weights.append(weight)

    print("Creating igraph...")
    ig_graph = ig.Graph(n=len(embeddings), edges=edges, directed=False)
    ig_graph.es['weight'] = weights
    ig_graph.simplify(combine_edges='sum')

    print("Performing Leiden clustering...")
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights='weight',
        n_iterations=-1,
        resolution_parameter=resolution
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

def create_umap_visualization(clustering_manager: ScalableClustering, cluster_labels: np.ndarray, cell_type_vocab: dict = None):
    """Create UMAP visualization with both cluster and cell type coloring"""
    sample_embeddings = clustering_manager.get_sample_for_metrics(UMAP_SAMPLE_SIZE)
    actual_sample_size = len(sample_embeddings)
    
    sample_indices = np.random.choice(len(cluster_labels), actual_sample_size, replace=False)
    sample_labels = cluster_labels[sample_indices]
    
    # Get cell types for the sampled indices
    sample_cell_types = clustering_manager.get_sample_cell_types(UMAP_SAMPLE_SIZE, sample_indices)
    
    print("Computing UMAP embedding...")
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
    emb_2d = umap_reducer.fit_transform(sample_embeddings)
    
    # Ensure emb_2d is a numpy array
    if not isinstance(emb_2d, np.ndarray):
        emb_2d = np.array(emb_2d)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(24, 8))
    
    # Plot 1: Color by cluster labels
    ax1 = plt.subplot(1, 3, 1)
    sample_labels = sample_labels.astype(int)
    scatter1 = ax1.scatter(emb_2d[:, 0], emb_2d[:, 1], c=sample_labels, cmap='tab10', s=15, alpha=0.7)
    ax1.set_title(f"Leiden Clusters (n={len(sample_embeddings)})", fontsize=14, fontweight='bold')
    ax1.set_xlabel("UMAP 1", fontsize=12)
    ax1.set_ylabel("UMAP 2", fontsize=12)
    cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8)
    cbar1.set_label('Cluster ID', fontsize=10)
    
    # Plot 2: Color by cell type
    ax2 = plt.subplot(1, 3, 2)
    if len(sample_cell_types) > 0:
        # Convert cell type indices to names if vocabulary is available
        if cell_type_vocab:
            # Create reverse mapping from index to cell type name
            idx_to_celltype = {v: k for k, v in cell_type_vocab.items()}
            cell_type_names = [idx_to_celltype.get(idx, f"Unknown_{idx}") for idx in sample_cell_types]
            unique_cell_types = list(set(cell_type_names))
            
            # Create numeric mapping for colors
            celltype_to_color = {ct: i for i, ct in enumerate(unique_cell_types)}
            color_values = [celltype_to_color[ct] for ct in cell_type_names]
            
            # Use distinct colors for cell types
            from matplotlib.colors import ListedColormap
            import matplotlib.cm as cm
            
            # Get distinct colors
            n_types = len(unique_cell_types)
            if n_types <= 10:
                colors = plt.cm.tab10(np.linspace(0, 1, n_types))
            elif n_types <= 20:
                colors = plt.cm.tab20(np.linspace(0, 1, n_types))
            else:
                colors = plt.cm.hsv(np.linspace(0, 1, n_types))
            
            cmap = ListedColormap(colors)
        else:
            # Use cell type indices directly
            color_values = sample_cell_types
            unique_cell_types = np.unique(sample_cell_types)
            cell_type_names = [f"CellType_{ct}" for ct in sample_cell_types]
            cmap = 'tab20'
        
        scatter2 = ax2.scatter(emb_2d[:, 0], emb_2d[:, 1], c=color_values, cmap=cmap, s=15, alpha=0.7)
        ax2.set_title(f"Cell Types (n={len(sample_embeddings)})", fontsize=14, fontweight='bold')
        ax2.set_xlabel("UMAP 1", fontsize=12)
        ax2.set_ylabel("UMAP 2", fontsize=12)
        
        # Add legend for cell types (if not too many)
        if len(unique_cell_types) <= 20:
            import matplotlib.patches as mpatches
            if cell_type_vocab:
                legend_elements = [mpatches.Patch(color=colors[celltype_to_color[ct]], 
                                                label=ct[:15] + '...' if len(ct) > 15 else ct) 
                                 for ct in sorted(unique_cell_types)]
            else:
                legend_elements = [mpatches.Patch(color=plt.cm.tab20(i/len(unique_cell_types)), 
                                                label=f"Type_{ct}") for i, ct in enumerate(unique_cell_types)]
            ax2.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        else:
            cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
            cbar2.set_label('Cell Type', fontsize=10)
    else:
        # Fallback: color by cluster if no cell type data
        scatter2 = ax2.scatter(emb_2d[:, 0], emb_2d[:, 1], c=sample_labels, cmap='tab10', s=15, alpha=0.7)
        ax2.set_title(f"No Cell Type Data - Showing Clusters", fontsize=14)
        ax2.set_xlabel("UMAP 1", fontsize=12)
        ax2.set_ylabel("UMAP 2", fontsize=12)
        plt.colorbar(scatter2, ax=ax2, label='Cluster')
    
    # Plot 3: Overlay - Cell types with cluster boundaries
    ax3 = plt.subplot(1, 3, 3)
    if len(sample_cell_types) > 0 and cell_type_vocab:
        # Background scatter with cell types
        scatter3 = ax3.scatter(emb_2d[:, 0], emb_2d[:, 1], c=color_values, cmap=cmap, s=15, alpha=0.6)
        
        # Add cluster boundaries using convex hulls
        from scipy.spatial import ConvexHull
        import matplotlib.patches as patches
        
        cluster_colors = plt.cm.Set3(np.linspace(0, 1, len(np.unique(sample_labels))))
        for cluster_id in np.unique(sample_labels):
            cluster_mask = sample_labels == cluster_id
            cluster_points = emb_2d[cluster_mask]
            
            if len(cluster_points) >= 3:  # Need at least 3 points for convex hull
                try:
                    hull = ConvexHull(cluster_points)
                    hull_points = cluster_points[hull.vertices]
                    hull_patch = patches.Polygon(hull_points, linewidth=2, 
                                               edgecolor=cluster_colors[cluster_id], 
                                               facecolor='none', alpha=0.8)
                    ax3.add_patch(hull_patch)
                except:
                    pass  # Skip if convex hull fails
        
        ax3.set_title(f"Cell Types with Cluster Boundaries", fontsize=14, fontweight='bold')
        ax3.set_xlabel("UMAP 1", fontsize=12)
        ax3.set_ylabel("UMAP 2", fontsize=12)
    else:
        # Fallback: just show clusters
        scatter3 = ax3.scatter(emb_2d[:, 0], emb_2d[:, 1], c=sample_labels, cmap='tab10', s=15, alpha=0.7)
        ax3.set_title(f"Clusters Only", fontsize=14)
        ax3.set_xlabel("UMAP 1", fontsize=12)
        ax3.set_ylabel("UMAP 2", fontsize=12)
    
    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/scencoder_umap_comprehensive.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Comprehensive UMAP visualization saved to output/scencoder_umap_comprehensive.png")
    
    # Create a separate detailed cell type visualization
    if len(sample_cell_types) > 0 and cell_type_vocab:
        plt.figure(figsize=(16, 12))
        
        # Create a more detailed cell type plot
        scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=color_values, cmap=cmap, s=25, alpha=0.8)
        plt.title(f"Cell Type Distribution (n={len(sample_embeddings)})", fontsize=18, fontweight='bold', pad=20)
        plt.xlabel("UMAP 1", fontsize=14)
        plt.ylabel("UMAP 2", fontsize=14)
        
        # Add detailed legend
        if len(unique_cell_types) <= 25:
            legend_elements = [mpatches.Patch(color=colors[celltype_to_color[ct]], 
                                            label=ct) for ct in sorted(unique_cell_types)]
            plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', 
                      fontsize=10, title="Cell Types", title_fontsize=12)
        else:
            cbar = plt.colorbar(scatter, shrink=0.8)
            cbar.set_label('Cell Type', fontsize=12)
        
        # Add cell type statistics
        cell_type_counts = {}
        for ct_name in cell_type_names:
            cell_type_counts[ct_name] = cell_type_counts.get(ct_name, 0) + 1
        
        # Add text box with statistics
        stats_text = f"Total cells: {len(sample_embeddings)}\n"
        stats_text += f"Cell types: {len(unique_cell_types)}\n"
        stats_text += f"Clusters: {len(np.unique(sample_labels))}"
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        plt.savefig("output/scencoder_celltypes_detailed.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Detailed cell type UMAP visualization saved to output/scencoder_celltypes_detailed.png")
        
        # Save cell type mapping for reference
        if cell_type_vocab:
            cell_type_mapping = {
                'vocabulary': cell_type_vocab,
                'reverse_mapping': idx_to_celltype,
                'sample_statistics': dict(sorted(cell_type_counts.items(), key=lambda x: x[1], reverse=True))
            }
            with open("output/cell_type_mapping.json", "w") as f:
                json.dump(cell_type_mapping, f, indent=4)
            print("Cell type mapping saved to output/cell_type_mapping.json")
    
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

def monitor_memory():
    """Monitor system and GPU memory usage"""
    import psutil
    
    # System memory
    mem = psutil.virtual_memory()
    print(f"System RAM: {mem.used / (1024**3):.1f}GB / {mem.total / (1024**3):.1f}GB ({mem.percent:.1f}% used)")
    
    # GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
            mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {mem_allocated:.1f}GB allocated, {mem_reserved:.1f}GB reserved, {mem_total:.1f}GB total")

if __name__ == "__main__":
    logger = setup_logging()
    logger.info(f"Using device: {DEVICE}")
    
    try:
        # Monitor initial memory state
        monitor_memory()
        
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
        
        # Load datasets with memory monitoring
        logger.info(f"Looking for .h5ad files in {DATA_DIR}")
        data_files = glob.glob(os.path.join(DATA_DIR, "*.h5ad"))
        if not data_files:
            logger.error(f"No .h5ad files found in {DATA_DIR}")
            raise FileNotFoundError(f"No .h5ad files found in {DATA_DIR}")
        
        logger.info(f"Found {len(data_files)} data files")
        
        datasets = []
        for data_file in data_files:
            logger.info(f"Loading {data_file}...")
            try:
                adata = anndata.read_h5ad(data_file)
                logger.info(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes from {os.path.basename(data_file)}")
                dataset = SingleCellDataset(
                    adata, gene_vocab, cell_type_vocab,
                    num_bins=NUM_BINS, seq_len=SEQ_LEN
                )
                datasets.append(dataset)
                monitor_memory()
            except Exception as e:
                logger.error(f"Failed to load {data_file}: {e}")
                continue
        
        if not datasets:
            raise ValueError("No datasets could be loaded")
        
        combined_dataset = ConcatDataset(datasets)
        dataloader = DataLoader(
            combined_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=True if torch.cuda.is_available() else False
        )
        
        logger.info(f"Total samples in dataset: {len(combined_dataset)}")
        
        # Load model with error handling
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
            monitor_memory()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Extract embeddings with monitoring
        logger.info("Starting embedding extraction...")
        start_time = datetime.now()
        try:
            clustering_manager = extract_embeddings(model, dataloader, DEVICE)
            extraction_time = datetime.now() - start_time
            logger.info(f"Embedding extraction completed in {extraction_time}")
            monitor_memory()
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            raise
        
        # Clear model from memory before clustering
        del model
        clear_gpu_memory()
        monitor_memory()
        logger.info("Model cleared from memory")
        
        # Perform clustering with monitoring and timeout
        logger.info("Starting Leiden clustering...")
        start_time = datetime.now()
        
        # Add a simple timeout mechanism
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Clustering operation timed out")
        
        try:
            # Set timeout to 30 minutes
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(1800)  # 30 minutes
            
            cluster_labels = perform_leiden_clustering(clustering_manager, n_neighbors=K_NEIGHBORS)
            
            signal.alarm(0)  # Cancel timeout
            clustering_time = datetime.now() - start_time
            logger.info(f"Leiden clustering completed in {clustering_time}")
            monitor_memory()
        except TimeoutError:
            logger.error("Clustering timed out after 30 minutes")
            raise
        except Exception as e:
            signal.alarm(0)  # Cancel timeout
            logger.error(f"Clustering failed: {e}")
            raise
        
        # Ensure cluster labels are integers
        cluster_labels = cluster_labels.astype(int)
        n_clusters = len(np.unique(cluster_labels))
        logger.info(f"Generated {n_clusters} clusters")
        
        # Clear model from memory
        clear_gpu_memory()
        logger.info("Model cleared from memory")
        
        # Compute metrics
        logger.info("Computing clustering metrics...")
        metrics = compute_metrics(clustering_manager, cluster_labels)
        os.makedirs("output", exist_ok=True)
        with open("output/clustering_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        logger.info("Clustering Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Create UMAP visualization with cell types
        logger.info("Creating UMAP visualization...")
        emb_2d = create_umap_visualization(clustering_manager, cluster_labels, cell_type_vocab)
        
        # Save results
        logger.info("Saving results...")
        np.save("output/scencoder_cluster_labels.npy", cluster_labels)
        if isinstance(emb_2d, np.ndarray):
            np.save("output/scencoder_umap_embeddings.npy", emb_2d)
        
        # Save cell types as well
        all_cell_types = clustering_manager.get_all_cell_types()
        if len(all_cell_types) > 0:
            np.save("output/scencoder_cell_types.npy", all_cell_types)
            logger.info("Cell types saved")
        
        metadata = {
            'total_samples': clustering_manager.total_samples,
            'n_clusters': len(np.unique(cluster_labels)),
            'batch_size': BATCH_SIZE,
            'num_embedding_files': len(clustering_manager.embedding_files),
            'num_cell_type_files': len(clustering_manager.cell_type_files),
            'k_neighbors': K_NEIGHBORS,
            'extraction_time': str(extraction_time),
            'clustering_time': str(clustering_time),
            'has_cell_type_data': len(all_cell_types) > 0
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
        
        # Analyze cell type vs cluster correspondence
        logger.info("Analyzing cell type vs cluster correspondence...")
        try:
            correspondence_matrix = analyze_cell_type_cluster_correspondence(
                clustering_manager, cluster_labels, cell_type_vocab
            )
            logger.info("Cell type analysis completed")
        except Exception as e:
            logger.warning(f"Cell type analysis failed: {e}")
        
    except Exception as e:
        logger.error(f"Critical error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        # Cleanup
        try:
            if 'clustering_manager' in locals():
                clustering_manager.cleanup_temp_files()
        except:
            pass
        clear_gpu_memory()

def analyze_cell_type_cluster_correspondence(clustering_manager, cluster_labels, cell_type_vocab=None):
    """Analyze how well clusters correspond to cell types"""
    
    all_cell_types = clustering_manager.get_all_cell_types()
    if len(all_cell_types) == 0:
        print("No cell type data available for analysis")
        return
    
    print("\n=== Cell Type vs Cluster Analysis ===")
    
    # Create confusion matrix-like analysis
    if cell_type_vocab:
        idx_to_celltype = {v: k for k, v in cell_type_vocab.items()}
        cell_type_names = [idx_to_celltype.get(idx, f"Unknown_{idx}") for idx in all_cell_types]
    else:
        cell_type_names = [f"CellType_{ct}" for ct in all_cell_types]
    
    # Create cross-tabulation
    unique_clusters = np.unique(cluster_labels)
    unique_cell_types = list(set(cell_type_names))
    
    # Build correspondence matrix
    correspondence_matrix = np.zeros((len(unique_cell_types), len(unique_clusters)))
    
    for i, ct in enumerate(unique_cell_types):
        ct_mask = np.array(cell_type_names) == ct
        ct_clusters = cluster_labels[ct_mask]
        for j, cluster_id in enumerate(unique_clusters):
            correspondence_matrix[i, j] = np.sum(ct_clusters == cluster_id)
    
    # Save correspondence matrix
    correspondence_df = {
        'cell_types': unique_cell_types,
        'clusters': unique_clusters.tolist(),
        'matrix': correspondence_matrix.tolist()
    }
    
    with open("output/cell_type_cluster_correspondence.json", "w") as f:
        json.dump(correspondence_df, f, indent=4)
    
    # Create heatmap
    plt.figure(figsize=(max(12, len(unique_clusters) * 0.8), max(8, len(unique_cell_types) * 0.4)))
    plt.imshow(correspondence_matrix, cmap='Blues', aspect='auto')
    plt.colorbar(label='Number of cells')
    plt.xlabel('Cluster ID')
    plt.ylabel('Cell Type')
    plt.title('Cell Type vs Cluster Correspondence')
    
    # Add cluster labels
    plt.xticks(range(len(unique_clusters)), unique_clusters)
    plt.yticks(range(len(unique_cell_types)), 
               [ct[:20] + '...' if len(ct) > 20 else ct for ct in unique_cell_types])
    
    # Add text annotations
    for i in range(len(unique_cell_types)):
        for j in range(len(unique_clusters)):
            if correspondence_matrix[i, j] > 0:
                plt.text(j, i, f'{int(correspondence_matrix[i, j])}', 
                        ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig("output/cell_type_cluster_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Cell type vs cluster heatmap saved to output/cell_type_cluster_heatmap.png")
    
    return correspondence_matrix