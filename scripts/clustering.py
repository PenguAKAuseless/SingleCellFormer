import os
import glob
import sys

# Add the project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import json
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import cudf
import igraph as ig
import leidenalg
from sklearn.metrics import calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
import umap
import matplotlib.pyplot as plt
import anndata
from models.scEncoder import scEncoder
from data.SingleCellDataset import SingleCellDataset
from utils.utils import load_vocabulary
import shutil
from torch.amp.autocast_mode import autocast

# Configuration
DATA_DIR = "data"  # Directory containing .h5ad files
GENE_VOCAB_PATH = "vocab/gene_vocab.json"
CELL_TYPE_VOCAB_PATH = "vocab/celltype_vocab.json"
DISEASE_VOCAB_PATH = "vocab/disease_vocab.json"
TISSUE_VOCAB_PATH = "vocab/tissue_vocab.json"
CHECKPOINT_PATH = "final/encoder_final_model.pth"
SEQ_LEN = 512
NUM_BINS = 51
BATCH_SIZE = 128  # Reduced from 256 to lower memory usage
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
N_CLUSTERS = 10  # Target number of clusters for Leiden
MAX_EMBEDDINGS_PER_BATCH = 50000  # Reduced from 100000 to limit memory
UMAP_SAMPLE_SIZE = 25000  # Reduced from 50000 for UMAP visualization
METRIC_SAMPLE_SIZE = 25000  # Reduced from 50000 for metrics computation
K_NEIGHBORS = 15  # Number of neighbors for KNN graph in Leiden

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
                samples.append(embeddings)
            else:
                indices = np.random.choice(len(embeddings), samples_per_file, replace=False)
                samples.append(embeddings[indices])
        return np.concatenate(samples, axis=0) if samples else np.array([])

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure all operations are complete
        import gc
        gc.collect()

def extract_embeddings(model, dataloader, device, max_embeddings_per_batch=50000):
    clustering_manager = ScalableClustering(N_CLUSTERS, batch_size=5000)
    model.eval()
    current_batch_embeddings = []
    batch_idx = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i % 50 == 0:  # Clear memory more frequently
                clear_gpu_memory()
            
            inputs = {
                'gene_ids': batch['gene_ids'].to(device, non_blocking=True),
                'gene_expr': batch['gene_expr'].to(device, non_blocking=True),
            }
            if 'cell_type' in batch and batch['cell_type'] is not None:
                inputs['cell_type'] = batch['cell_type'].to(device, non_blocking=True)
            if 'disease' in batch and batch['disease'] is not None:
                inputs['disease'] = batch['disease'].to(device, non_blocking=True)
            if 'tissue' in batch and batch['tissue'] is not None:
                inputs['tissue'] = batch['tissue'].to(device, non_blocking=True)
            
            with autocast(device_type=device, dtype=torch.float16):
                outputs = model(inputs, create_mask=False)
            
            hidden_states = outputs['hidden_states']
            valid_positions = (inputs['gene_expr'] > 0).float()
            pooled_repr = (hidden_states * valid_positions.unsqueeze(-1)).sum(dim=1) / (valid_positions.sum(dim=1, keepdim=True) + 1e-8)
            pooled_emb = pooled_repr.detach().cpu().numpy()
            current_batch_embeddings.append(pooled_emb)
            
            if len(current_batch_embeddings) * BATCH_SIZE >= max_embeddings_per_batch:
                batch_embeddings = np.concatenate(current_batch_embeddings, axis=0)
                filename = clustering_manager.save_embedding_batch(batch_embeddings, batch_idx)
                clustering_manager.embedding_files.append(filename)
                clustering_manager.total_samples += len(batch_embeddings)
                current_batch_embeddings = []
                batch_idx += 1
                clear_gpu_memory()
                print(f"Saved embedding batch {batch_idx}, total samples: {clustering_manager.total_samples}")
    
    if current_batch_embeddings:
        batch_embeddings = np.concatenate(current_batch_embeddings, axis=0)
        filename = clustering_manager.save_embedding_batch(batch_embeddings, batch_idx)
        clustering_manager.embedding_files.append(filename)
        clustering_manager.total_samples += len(batch_embeddings)
        clear_gpu_memory()
    
    print(f"Total embeddings extracted: {clustering_manager.total_samples}")
    return clustering_manager

def perform_leiden_clustering(clustering_manager: ScalableClustering, n_neighbors: int = K_NEIGHBORS, resolution: float = 1.0):
    """
    Perform Leiden clustering using sklearn for KNN graph construction
    """
    print("Loading all embeddings for clustering...")
    # Collect all embeddings
    all_embeddings = []
    for filename in clustering_manager.embedding_files:
        embeddings = clustering_manager.load_embedding_batch(filename)
        all_embeddings.append(embeddings)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Loaded {len(all_embeddings)} embeddings for clustering")
    
    # Build KNN graph using sklearn (more reliable than cuGraph for this use case)
    print("Building KNN graph...")
    nn_model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean', n_jobs=-1)
    nn_model.fit(all_embeddings)
    distances, indices = nn_model.kneighbors(all_embeddings)
    
    # Create edge list for igraph (excluding self-loops)
    edges = []
    weights = []
    for i in range(len(all_embeddings)):
        for j in range(1, n_neighbors + 1):  # Skip first neighbor (self)
            neighbor_idx = indices[i, j]
            weight = 1.0 / (1.0 + distances[i, j])  # Convert distance to similarity
            edges.append((i, neighbor_idx))
            weights.append(weight)
    
    print("Creating igraph from KNN graph...")
    # Create igraph from edge list
    ig_graph = ig.Graph(n=len(all_embeddings), edges=edges, directed=False)
    ig_graph.es['weight'] = weights
    
    # Remove duplicate edges and sum weights
    ig_graph.simplify(combine_edges='sum')
    
    print("Performing Leiden clustering...")
    # Option 1: Use RBConfigurationVertexPartition with resolution parameter
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
        # Option 2: Fall back to basic ModularityVertexPartition (no resolution parameter)
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.ModularityVertexPartition,
            weights='weight',
            n_iterations=-1
        )
    
    # Get cluster labels
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

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    # Load vocabularies
    gene_vocab = load_vocabulary(GENE_VOCAB_PATH)
    cell_type_vocab = load_vocabulary(CELL_TYPE_VOCAB_PATH)
    disease_vocab = load_vocabulary(DISEASE_VOCAB_PATH)
    tissue_vocab = load_vocabulary(TISSUE_VOCAB_PATH)
    
    # Load datasets
    data_files = glob.glob(os.path.join(DATA_DIR, "*.h5ad"))
    if not data_files:
        raise FileNotFoundError(f"No .h5ad files found in {DATA_DIR}")
    
    datasets = []
    for data_file in data_files:
        print(f"Loading {data_file}...")
        adata = anndata.read_h5ad(data_file)
        dataset = SingleCellDataset(
            adata, gene_vocab, cell_type_vocab, disease_vocab, tissue_vocab,
            num_bins=NUM_BINS, seq_len=SEQ_LEN
        )
        datasets.append(dataset)
    
    combined_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(
        combined_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Total samples in dataset: {len(combined_dataset)}")
    
    # Load model
    print("Loading model...")
    model = scEncoder(
        gene_vocab_size=len(gene_vocab),
        cell_type_vocab_size=len(cell_type_vocab) if cell_type_vocab else None,
        disease_vocab_size=len(disease_vocab) if disease_vocab else None,
        tissue_vocab_size=len(tissue_vocab) if tissue_vocab else None,
        num_bins=NUM_BINS, seq_len=SEQ_LEN,
        num_layers=12, hidden_dim=2048, dropout=0.1
    ).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    
    # Extract embeddings
    print("Extracting embeddings...")
    clustering_manager = extract_embeddings(model, dataloader, DEVICE_TYPE)
    
    # Perform Leiden clustering
    print("Performing Leiden clustering...")
    cluster_labels = perform_leiden_clustering(clustering_manager, n_neighbors=K_NEIGHBORS)
    
    # Ensure cluster labels are integers
    cluster_labels = cluster_labels.astype(int)
    
    # Clear model from memory
    del model
    clear_gpu_memory()
    
    # Compute metrics
    print("Computing clustering metrics...")
    metrics = compute_metrics(clustering_manager, cluster_labels)
    with open("output/clustering_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print("Clustering Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Create UMAP visualization
    emb_2d = create_umap_visualization(clustering_manager, cluster_labels)
    
    # Save results
    np.save("output/scencoder_cluster_labels.npy", cluster_labels)
    np.save("output/scencoder_umap_embeddings.npy", emb_2d)
    # Leiden does not provide cluster centers, so skip saving centers
    
    metadata = {
        'total_samples': clustering_manager.total_samples,
        'n_clusters': len(np.unique(cluster_labels)),
        'batch_size': BATCH_SIZE,
        'num_embedding_files': len(clustering_manager.embedding_files),
        'k_neighbors': K_NEIGHBORS
    }
    with open("output/clustering_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    clustering_manager.cleanup_temp_files()
    
    print(f"\nClustering analysis complete!")
    print(f"Processed {clustering_manager.total_samples} samples")
    print(f"Generated {len(np.unique(cluster_labels))} clusters")
    print(f"Results saved in 'output/' directory")