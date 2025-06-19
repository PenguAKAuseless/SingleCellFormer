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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.spatial.distance import cdist
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import anndata

from models.scEncoder import scEncoder
from dataset.SingleCellDataset import SingleCellDataset
from utils.utils import load_vocabulary, build_gene_vocab, build_celltype_tissue_disease_vocab

# --- Configuration ---
DATA_DIR = "dataset"  # Directory containing .h5ad files
GENE_VOCAB_PATH = "vocab/gene_vocab.json"
CELL_TYPE_VOCAB_PATH = "vocab/celltype_vocab.json"  # Optional
DISEASE_VOCAB_PATH = "vocab/disease_vocab.json"     # Optional
TISSUE_VOCAB_PATH = "vocab/tissue_vocab.json"       # Optional
CHECKPOINT_PATH = "final/encoder_final_model.pth"
SEQ_LEN = 512
NUM_BINS = 51
BATCH_SIZE = 32
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
N_CLUSTERS = 10  # Set as needed

def update_vocabularies_from_datasets():
    """
    Update vocabularies using the same pipeline as prepare_train.py.
    Creates combined vocabularies from all datasets in the DATA_DIR.
    """
    print("Updating vocabularies from all datasets...")
    
    # Create vocab directory if it doesn't exist
    os.makedirs("vocab", exist_ok=True)
    
    # Get all .h5ad files
    data_files = glob.glob(os.path.join(DATA_DIR, "*.h5ad"))
    if not data_files:
        raise FileNotFoundError(f"No .h5ad files found in {DATA_DIR}")
    
    print(f"Found {len(data_files)} dataset files:")
    for i, file in enumerate(data_files):
        print(f"  {i+1}. {file}")
    
    all_genes = set()
    all_celltypes = set()
    all_tissues = set()
    all_diseases = set()
    
    temp_vocab_files = []  # Track temporary vocab files for cleanup
    
    # Process each dataset to build individual vocabularies
    for idx, data_file in enumerate(data_files):
        print(f"\nProcessing dataset {idx+1}/{len(data_files)}: {os.path.basename(data_file)}")
        
        try:
            # Build vocabularies for this dataset
            temp_gene_vocab = f"vocab/gene_vocab_{idx}.json"
            temp_celltype_vocab = f"vocab/celltype_vocab_{idx}.json"
            temp_tissue_vocab = f"vocab/tissue_vocab_{idx}.json"
            temp_disease_vocab = f"vocab/disease_vocab_{idx}.json"
            
            temp_vocab_files.extend([temp_gene_vocab, temp_celltype_vocab, temp_tissue_vocab, temp_disease_vocab])
            
            # Build gene vocabulary
            gene_vocab = build_gene_vocab(
                adata_file=data_file,
                vocab_file=temp_gene_vocab
            )
            
            # Build cell type, tissue, and disease vocabularies
            celltype_vocab, tissue_vocab, disease_vocab = build_celltype_tissue_disease_vocab(
                adata_file=data_file,
                celltype_vocab_file=temp_celltype_vocab,
                tissue_vocab_file=temp_tissue_vocab,
                disease_vocab_file=temp_disease_vocab
            )
            
            # Collect all unique values
            with open(temp_gene_vocab, 'r') as f:
                gene_vocab_data = json.load(f)
                all_genes.update(gene_vocab_data.keys())
            
            with open(temp_celltype_vocab, 'r') as f:
                celltype_vocab_data = json.load(f)
                all_celltypes.update(celltype_vocab_data.keys())
            
            with open(temp_tissue_vocab, 'r') as f:
                tissue_vocab_data = json.load(f)
                all_tissues.update(tissue_vocab_data.keys())
            
            with open(temp_disease_vocab, 'r') as f:
                disease_vocab_data = json.load(f)
                all_diseases.update(disease_vocab_data.keys())
                
        except Exception as e:
            print(f"Warning: Could not process dataset {idx}: {e}")
            continue
    
    # Create combined vocabularies
    print(f"\nCreating combined vocabularies...")
    combined_gene_vocab = {gene: idx for idx, gene in enumerate(sorted(all_genes))}
    combined_celltype_vocab = {celltype: idx for idx, celltype in enumerate(sorted(all_celltypes))}
    combined_tissue_vocab = {tissue: idx for idx, tissue in enumerate(sorted(all_tissues))}
    combined_disease_vocab = {disease: idx for idx, disease in enumerate(sorted(all_diseases))}
    
    # Save combined vocabularies
    with open(GENE_VOCAB_PATH, 'w') as f:
        json.dump(combined_gene_vocab, f, indent=2)
    
    with open(CELL_TYPE_VOCAB_PATH, 'w') as f:
        json.dump(combined_celltype_vocab, f, indent=2)
    
    with open(TISSUE_VOCAB_PATH, 'w') as f:
        json.dump(combined_tissue_vocab, f, indent=2)
    
    with open(DISEASE_VOCAB_PATH, 'w') as f:
        json.dump(combined_disease_vocab, f, indent=2)
    
    print(f"Combined vocabularies created:")
    print(f"  - Genes: {len(combined_gene_vocab)}")
    print(f"  - Cell types: {len(combined_celltype_vocab)}")
    print(f"  - Tissues: {len(combined_tissue_vocab)}")
    print(f"  - Diseases: {len(combined_disease_vocab)}")
    
    # Clean up temporary vocabulary files
    print(f"\nCleaning up temporary vocabulary files...")
    for temp_file in temp_vocab_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"Removed: {temp_file}")
    
    print("Vocabulary update complete!")
    
    return combined_gene_vocab, combined_celltype_vocab, combined_tissue_vocab, combined_disease_vocab

def load_or_update_vocabularies():
    """
    Load existing vocabularies or create them if they don't exist or are outdated.
    """
    # Check if vocabulary files exist
    vocab_files_exist = all(os.path.exists(path) for path in [
        GENE_VOCAB_PATH, CELL_TYPE_VOCAB_PATH, DISEASE_VOCAB_PATH, TISSUE_VOCAB_PATH
    ])
    
    if not vocab_files_exist:
        print("Vocabulary files not found. Creating new vocabularies...")
        return update_vocabularies_from_datasets()
    else:
        print("Loading existing vocabularies...")
        gene_vocab = load_vocabulary(GENE_VOCAB_PATH)
        cell_type_vocab = load_vocabulary(CELL_TYPE_VOCAB_PATH)
        disease_vocab = load_vocabulary(DISEASE_VOCAB_PATH)
        tissue_vocab = load_vocabulary(TISSUE_VOCAB_PATH)
        
        print(f"Loaded vocabularies:")
        print(f"  - Genes: {len(gene_vocab)}")
        print(f"  - Cell types: {len(cell_type_vocab)}")
        print(f"  - Tissues: {len(tissue_vocab)}")
        print(f"  - Diseases: {len(disease_vocab)}")
        
        return gene_vocab, cell_type_vocab, tissue_vocab, disease_vocab

# --- Update/Load vocabularies ---
gene_vocab, cell_type_vocab, tissue_vocab, disease_vocab = load_or_update_vocabularies()

# --- Load all AnnData files from directory ---
data_files = glob.glob(os.path.join(DATA_DIR, "*.h5ad"))
if not data_files:
    raise FileNotFoundError(f"No .h5ad files found in {DATA_DIR}")

print(f"\nLoading {len(data_files)} datasets for clustering...")
datasets = []
all_cell_types = []
all_tissues = []
for i, data_file in enumerate(data_files):
    print(f"Loading dataset {i+1}/{len(data_files)}: {os.path.basename(data_file)}")
    adata = anndata.read_h5ad(data_file)
    dataset = SingleCellDataset(
        adata,
        gene_vocab,
        cell_type_vocab,
        disease_vocab,
        tissue_vocab,
        num_bins=NUM_BINS,
        seq_len=SEQ_LEN
    )
    datasets.append(dataset)
    # Collect cell type and tissue labels
    if cell_type_vocab and 'cell_type' in adata.obs:
        cell_types = adata.obs['cell_type'].values
        all_cell_types.extend(cell_types)
    if tissue_vocab and 'tissue' in adata.obs:
        tissues = adata.obs['tissue'].values
        all_tissues.extend(tissues)

# Combine all datasets
combined_dataset = ConcatDataset(datasets)
dataloader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Total samples for clustering: {len(combined_dataset)}")

# --- Load model ---
print(f"Loading model from {CHECKPOINT_PATH}...")
model = scEncoder(
    gene_vocab_size=len(gene_vocab),
    cell_type_vocab_size=len(cell_type_vocab) if cell_type_vocab else None,
    disease_vocab_size=len(disease_vocab) if disease_vocab else None,
    tissue_vocab_size=len(tissue_vocab) if tissue_vocab else None,
    num_bins=NUM_BINS,
    seq_len=SEQ_LEN,
    d_model=512,
    nhead=8,
    num_layers=12,
    hidden_dim=2048,
    dropout=0.1,
    gradient_checkpointing=False
).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

print(f"Model loaded successfully. Using device: {DEVICE}")

# --- Extract pooled embeddings ---
print("Extracting pooled embeddings...")
all_pooled_embeddings = []
with torch.no_grad():
    for i, batch in enumerate(dataloader):
        if (i + 1) % 100 == 0:
            print(f"Processing batch {i+1}/{len(dataloader)}...")
        
        # Prepare input dictionary
        inputs = {
            'gene_ids': batch['gene_ids'].to(DEVICE),
            'gene_expr': batch['gene_expr'].to(DEVICE),
        }
        
        # Add optional metadata inputs if available
        if 'cell_type' in batch and batch['cell_type'] is not None:
            inputs['cell_type'] = batch['cell_type'].to(DEVICE)
        if 'disease' in batch and batch['disease'] is not None:
            inputs['disease'] = batch['disease'].to(DEVICE)
        if 'tissue' in batch and batch['tissue'] is not None:
            inputs['tissue'] = batch['tissue'].to(DEVICE)
        
        # Forward pass without creating mask (inference mode)
        outputs = model(inputs, create_mask=False)
        
        # Get hidden states and create pooled representation
        hidden_states = outputs['hidden_states']  # (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = hidden_states.size()
        
        # Create pooling mask to exclude padded positions
        # Assume non-zero gene expressions indicate valid positions
        valid_positions = (inputs['gene_expr'] > 0).float()  # (batch_size, seq_len)
        
        # Compute mean pooling over valid positions
        pooled_repr = (hidden_states * valid_positions.unsqueeze(-1)).sum(dim=1) / (valid_positions.sum(dim=1, keepdim=True) + 1e-8)
        
        # Convert to numpy and store
        pooled_emb = pooled_repr.detach().cpu().numpy()
        all_pooled_embeddings.append(pooled_emb)

all_pooled_embeddings = np.concatenate(all_pooled_embeddings, axis=0)
print(f"Extracted embeddings shape: {all_pooled_embeddings.shape}")

# --- Clustering ---
print(f"Performing K-means clustering with {N_CLUSTERS} clusters...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
cluster_labels = kmeans.fit_predict(all_pooled_embeddings)

# --- Compute clustering metrics ---
print("Computing clustering metrics...")
metrics = {}
# Silhouette Coefficient
metrics['silhouette_score'] = float(silhouette_score(all_pooled_embeddings, cluster_labels))
# Davies-Bouldin Index
metrics['davies_bouldin_score'] = float(davies_bouldin_score(all_pooled_embeddings, cluster_labels))
# Calinski-Harabasz Index
metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(all_pooled_embeddings, cluster_labels))

# Dunn Index
def dunn_index(X, labels):
    distances = cdist(X, X, metric='euclidean')
    n_clusters = len(np.unique(labels))
    intra_cluster_dists = []
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        if len(cluster_points) > 1:
            intra_cluster_dists.append(np.max(cdist(cluster_points, cluster_points, metric='euclidean')))
    max_intra = np.max(intra_cluster_dists) if intra_cluster_dists else np.inf
    min_inter = np.min([np.min(cdist(X[labels == i], X[labels != i], metric='euclidean')) 
                        for i in range(n_clusters) if np.sum(labels != i) > 0])
    return float(min_inter / max_intra if max_intra > 0 else np.inf)

metrics['dunn_index'] = dunn_index(all_pooled_embeddings, cluster_labels)

# ARI and NMI (only if cell type labels are available)
if all_cell_types:
    print("Computing supervised clustering metrics...")
    cell_type_labels = np.array([cell_type_vocab.get(ct, 0) for ct in all_cell_types])
    metrics['adjusted_rand_score'] = float(adjusted_rand_score(cell_type_labels, cluster_labels))
    metrics['normalized_mutual_info_score'] = float(normalized_mutual_info_score(cell_type_labels, cluster_labels))

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Save metrics to JSON
with open("output/clustering_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nClustering Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# --- Visualization with UMAP ---
print("Generating UMAP visualization...")
umap_reducer = umap.UMAP(n_components=2, random_state=42)
emb_2d = umap_reducer.fit_transform(all_pooled_embeddings)

# Plot with cluster labels
plt.figure(figsize=(10, 8))
scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=cluster_labels, cmap='tab10', s=20, alpha=0.7)
plt.title("scEncoder Pooled Embedding Clusters (UMAP)", fontsize=14)
plt.xlabel("UMAP 1", fontsize=12)
plt.ylabel("UMAP 2", fontsize=12)
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.savefig("output/scencoder_clusters_umap.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot with cell types (if available)
if all_cell_types:
    print("Generating cell type visualization...")
    unique_cell_types = np.unique(all_cell_types)
    cell_type_indices = np.array([list(unique_cell_types).index(ct) for ct in all_cell_types])
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=cell_type_indices, cmap='tab20', s=20, alpha=0.7)
    plt.title("scEncoder Pooled Embeddings by Cell Type (UMAP)", fontsize=14)
    plt.xlabel("UMAP 1", fontsize=12)
    plt.ylabel("UMAP 2", fontsize=12)
    
    # Create custom legend
    unique_indices = np.unique(cell_type_indices)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=plt.cm.tab20(i/len(unique_indices)), 
                                  markersize=8, label=unique_cell_types[i]) 
                      for i in range(len(unique_cell_types))]
    plt.legend(handles=legend_elements, title="Cell Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("output/scencoder_cell_types_umap.png", dpi=300, bbox_inches='tight')
    plt.close()

# Plot with tissue types (if available)
if all_tissues:
    print("Generating tissue type visualization...")
    unique_tissues = np.unique(all_tissues)
    tissue_indices = np.array([list(unique_tissues).index(t) for t in all_tissues])
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=tissue_indices, cmap='tab20', s=20, alpha=0.7)
    plt.title("scEncoder Pooled Embeddings by Tissue Type (UMAP)", fontsize=14)
    plt.xlabel("UMAP 1", fontsize=12)
    plt.ylabel("UMAP 2", fontsize=12)
    
    # Create custom legend
    unique_indices = np.unique(tissue_indices)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=plt.cm.tab20(i/len(unique_indices)), 
                                  markersize=8, label=unique_tissues[i]) 
                      for i in range(len(unique_tissues))]
    plt.legend(handles=legend_elements, title="Tissue Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("output/scencoder_tissue_types_umap.png", dpi=300, bbox_inches='tight')
    plt.close()

# --- Save cluster labels and embeddings ---
print("Saving results...")
np.save("output/scencoder_cluster_labels.npy", cluster_labels)
np.save("output/scencoder_pooled_embeddings.npy", all_pooled_embeddings)
np.save("output/scencoder_umap_embeddings.npy", emb_2d)

print(f"\nClustering analysis complete!")
print(f"Found {len(all_pooled_embeddings)} samples")
print(f"Generated {N_CLUSTERS} clusters")
print(f"Results saved in 'output/' directory")
print(f"Vocabulary files updated and saved in 'vocab/' directory")