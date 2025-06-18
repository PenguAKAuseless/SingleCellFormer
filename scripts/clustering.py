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
from utils.utils import load_vocabulary

# --- Configuration ---
DATA_DIR = "eval"  # Directory containing .h5ad files
GENE_VOCAB_PATH = "vocab/gene_vocab.json"
CELL_TYPE_VOCAB_PATH = "vocab/celltype_vocab.json"  # Optional
DISEASE_VOCAB_PATH = "vocab/disease_vocab.json"     # Optional
TISSUE_VOCAB_PATH = "vocab/tissue_vocab.json"       # Optional
CHECKPOINT_PATH = "final/final_model.pth"
SEQ_LEN = 512
NUM_BINS = 51
BATCH_SIZE = 32
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
N_CLUSTERS = 10  # Set as needed

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# --- Load vocabularies ---
gene_vocab = load_vocabulary(GENE_VOCAB_PATH)
cell_type_vocab = load_vocabulary(CELL_TYPE_VOCAB_PATH)
disease_vocab = load_vocabulary(DISEASE_VOCAB_PATH)
tissue_vocab = load_vocabulary(TISSUE_VOCAB_PATH)

# --- Load all AnnData files from directory ---
data_files = glob.glob(os.path.join(DATA_DIR, "*.h5ad"))
if not data_files:
    raise FileNotFoundError(f"No .h5ad files found in {DATA_DIR}")

datasets = []
all_cell_types = []
all_tissues = []
for data_file in data_files:
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

# --- Load model ---
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

# --- Extract CLS embeddings ---
all_cls_embeddings = []
with torch.no_grad():
    for batch in dataloader:
        inputs = {
            'gene_ids': batch['gene_ids'].to(DEVICE),
            'gene_expr': batch['gene_expr'].to(DEVICE),
        }
        if 'cell_type' in batch:
            inputs['cell_type'] = batch['cell_type'].to(DEVICE)
        if 'disease' in batch:
            inputs['disease'] = batch['disease'].to(DEVICE)
        if 'tissue' in batch:
            inputs['tissue'] = batch['tissue'].to(DEVICE)
        # Forward pass
        batch_size = inputs['gene_ids'].size(0)
        gene_emb = model.gene_embedding(inputs['gene_ids'])
        value_emb = model.value_embedding(inputs['gene_expr'])
        emb = gene_emb + value_emb
        cls_token = model.cls_token.expand(batch_size, -1, -1)
        if model.cell_embedding and 'cell_type' in inputs:
            cls_token = cls_token + model.cell_embedding(inputs['cell_type']).unsqueeze(1)
        if model.disease_embedding and 'disease' in inputs:
            cls_token = cls_token + model.disease_embedding(inputs['disease']).unsqueeze(1)
        if model.tissue_embedding and 'tissue' in inputs:
            cls_token = cls_token + model.tissue_embedding(inputs['tissue']).unsqueeze(1)
        x = torch.cat((cls_token, emb), dim=1)
        if x.size(1) < model.seq_len:
            pad_size = model.seq_len - x.size(1)
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_size), value=0)
        elif x.size(1) > model.seq_len:
            x = x[:, :model.seq_len, :]
        x = model.dropout_layer(x)
        attention_mask = torch.ones(batch_size, model.seq_len, device=DEVICE)
        if x.size(1) < model.seq_len:
            attention_mask[:, x.size(1):] = 0
        x = model.transformer_encoder(x, src_key_padding_mask=(attention_mask == 0))
        x = model.norm(x)
        cls_emb = x[:, 0, :].detach().cpu().numpy()
        all_cls_embeddings.append(cls_emb)

all_cls_embeddings = np.concatenate(all_cls_embeddings, axis=0)

# --- Clustering ---
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
cluster_labels = kmeans.fit_predict(all_cls_embeddings)

# --- Compute clustering metrics ---
metrics = {}
# Silhouette Coefficient
metrics['silhouette_score'] = float(silhouette_score(all_cls_embeddings, cluster_labels))
# Davies-Bouldin Index
metrics['davies_bouldin_score'] = float(davies_bouldin_score(all_cls_embeddings, cluster_labels))
# Calinski-Harabasz Index
metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(all_cls_embeddings, cluster_labels))
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
metrics['dunn_index'] = dunn_index(all_cls_embeddings, cluster_labels)

# ARI and NMI (only if cell type labels are available)
if all_cell_types:
    cell_type_labels = np.array([cell_type_vocab.get(ct, 0) for ct in all_cell_types])
    metrics['adjusted_rand_score'] = float(adjusted_rand_score(cell_type_labels, cluster_labels))
    metrics['normalized_mutual_info_score'] = float(normalized_mutual_info_score(cell_type_labels, cluster_labels))

# Save metrics to JSON
with open("output/clustering_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# --- Visualization with UMAP ---
umap_reducer = umap.UMAP(n_components=2, random_state=42)
emb_2d = umap_reducer.fit_transform(all_cls_embeddings)

# Plot with cluster labels
plt.figure(figsize=(8, 6))
plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=cluster_labels, cmap='tab10', s=5)
plt.title("scEncoder CLS Embedding Clusters (UMAP)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.colorbar(label='Cluster')
plt.tight_layout()
plt.savefig("output/scencoder_clusters_umap.png", dpi=150)
plt.close()

# Plot with cell types (if available)
if all_cell_types:
    unique_cell_types = np.unique(all_cell_types)
    cell_type_indices = np.array([list(unique_cell_types).index(ct) for ct in all_cell_types])
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=cell_type_indices, cmap='tab20', s=5)
    plt.title("scEncoder CLS Embeddings by Cell Type (UMAP)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    
    # Fix legend creation by converting numpy array to list
    handles, labels = scatter.legend_elements()
    plt.legend(handles=handles, labels=list(unique_cell_types), title="Cell Type", 
               bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("output/scencoder_cell_types_umap.png", dpi=150, bbox_inches='tight')
    plt.close()

# Plot with tissue types (if available)
if all_tissues:
    unique_tissues = np.unique(all_tissues)
    tissue_indices = np.array([list(unique_tissues).index(t) for t in all_tissues])
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=tissue_indices, cmap='tab20', s=5)
    plt.title("scEncoder CLS Embeddings by Tissue Type (UMAP)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    
    # Fix legend creation by converting numpy array to list and limiting number of labels
    handles, labels = scatter.legend_elements()
    unique_tissues_list = list(unique_tissues)
    
    # If there are too many tissue types, show only the first 20 to avoid overcrowding
    if len(unique_tissues_list) > 20:
        unique_tissues_list = unique_tissues_list[:20]
        handles = handles[:20]
        print(f"Warning: Only showing first 20 tissue types out of {len(unique_tissues)} total")
    
    plt.legend(handles=handles, labels=unique_tissues_list, title="Tissue Type", 
               bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("output/scencoder_tissue_types_umap.png", dpi=150, bbox_inches='tight')
    plt.close()

# --- Save cluster labels ---
np.save("output/scencoder_cluster_labels.npy", cluster_labels)

print("Clustering analysis completed successfully!")
print(f"Results saved in output/ directory:")
print("- clustering_metrics.json: Clustering evaluation metrics")
print("- scencoder_clusters_umap.png: UMAP visualization of clusters")
if all_cell_types:
    print("- scencoder_cell_types_umap.png: UMAP visualization by cell types")
if all_tissues:
    print("- scencoder_tissue_types_umap.png: UMAP visualization by tissue types")
print("- scencoder_cluster_labels.npy: Cluster assignments for each cell")