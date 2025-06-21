import os
import glob
import sys

# Add the project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import gc
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score
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
N_CLUSTERS = 10  # Adjustable based on dataset
MAX_EMBEDDINGS_PER_BATCH = 50000  # Reduced from 100000 to limit memory
UMAP_SAMPLE_SIZE = 25000  # Reduced from 50000 for UMAP visualization
METRIC_SAMPLE_SIZE = 25000  # Reduced from 50000 for metrics computation

class ScalableClustering:
    def __init__(self, n_clusters: int, batch_size: int = 5000, temp_dir: str = "temp_clustering"):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.temp_dir = temp_dir
        self.embedding_files = []
        self.total_samples = 0
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize MiniBatchKMeans for incremental clustering
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=batch_size,
            max_iter=100,
            n_init=3
        )

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
                clustering_manager.kmeans.partial_fit(batch_embeddings)
                filename = clustering_manager.save_embedding_batch(batch_embeddings, batch_idx)
                clustering_manager.embedding_files.append(filename)
                clustering_manager.total_samples += len(batch_embeddings)
                current_batch_embeddings = []
                batch_idx += 1
                clear_gpu_memory()
                print(f"Saved embedding batch {batch_idx}, total samples: {clustering_manager.total_samples}")
    
    if current_batch_embeddings:
        batch_embeddings = np.concatenate(current_batch_embeddings, axis=0)
        clustering_manager.kmeans.partial_fit(batch_embeddings)
        filename = clustering_manager.save_embedding_batch(batch_embeddings, batch_idx)
        clustering_manager.embedding_files.append(filename)
        clustering_manager.total_samples += len(batch_embeddings)
        clear_gpu_memory()
    
    print(f"Total embeddings extracted: {clustering_manager.total_samples}")
    return clustering_manager

def compute_metrics(clustering_manager: ScalableClustering, cluster_labels: np.ndarray):
    metrics = {}
    # Determine the number of samples to use (min of METRIC_SAMPLE_SIZE and total available samples)
    sample_size = min(METRIC_SAMPLE_SIZE, clustering_manager.total_samples)
    if sample_size < 2:
        return metrics  # Not enough samples to compute metrics
    
    # Sample embeddings
    sample_embeddings = clustering_manager.get_sample_for_metrics(sample_size)
    # Ensure the number of embeddings matches the requested sample size
    actual_sample_size = len(sample_embeddings)
    
    # Sample the same number of labels using random indices
    sample_indices = np.random.choice(len(cluster_labels), actual_sample_size, replace=False)
    sample_labels = cluster_labels[sample_indices]
    
    # Compute metrics if there are enough samples and clusters
    if len(sample_embeddings) > 1 and len(np.unique(sample_labels)) > 1:
        metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(sample_embeddings, sample_labels))
    
    return metrics

def create_umap_visualization(clustering_manager: ScalableClustering, cluster_labels: np.ndarray):
    # Get sample embeddings
    sample_embeddings = clustering_manager.get_sample_for_metrics(UMAP_SAMPLE_SIZE)
    actual_sample_size = len(sample_embeddings)
    
    # Sample the same number of labels to match embeddings
    sample_indices = np.random.choice(len(cluster_labels), actual_sample_size, replace=False)
    sample_labels = cluster_labels[sample_indices]
    
    # Ensure labels are integers for matplotlib
    sample_labels = sample_labels.astype(int)
    
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
    
    # Extract embeddings and perform incremental clustering
    print("Extracting embeddings and clustering...")
    clustering_manager = extract_embeddings(model, dataloader, DEVICE_TYPE)
    
    # Predict labels for all embeddings
    cluster_labels = []
    for batch_embeddings in clustering_manager.get_embedding_iterator():
        batch_labels = clustering_manager.kmeans.predict(batch_embeddings)
        cluster_labels.append(batch_labels)
    cluster_labels = np.concatenate(cluster_labels, axis=0)
    
    # Ensure cluster labels are integers
    cluster_labels = cluster_labels.astype(int)
    
    # Clear model from memory
    del model
    clear_gpu_memory()
    
    # Compute metrics
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
    np.save("output/scencoder_cluster_centers.npy", clustering_manager.kmeans.cluster_centers_)
    
    metadata = {
        'total_samples': clustering_manager.total_samples,
        'n_clusters': N_CLUSTERS,
        'batch_size': BATCH_SIZE,
        'num_embedding_files': len(clustering_manager.embedding_files)
    }
    with open("output/clustering_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    clustering_manager.cleanup_temp_files()
    
    print(f"\nClustering analysis complete!")
    print(f"Processed {clustering_manager.total_samples} samples")
    print(f"Generated {N_CLUSTERS} clusters")
    print(f"Results saved in 'output/' directory")