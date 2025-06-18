import os
import sys

# Add the project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from pathlib import Path
from utils.utils import download_dataset, build_gene_vocab, build_celltype_tissue_disease_vocab, shuffle_and_split_anndata
import anndata
import scanpy as sc

# List of dataset URLs to download
datasets = [
    "https://datasets.cellxgene.cziscience.com/ca3aaa90-3720-4ea1-98bd-a2a81e2d3c33.h5ad",
    "https://datasets.cellxgene.cziscience.com/fd27ac78-f3de-48fb-b671-e35626c7b675.h5ad"
]

def prepare_data():
    """
    Downloads datasets, builds vocabularies, and splits data into training and evaluation sets.
    """
    # Create directories if they don't exist
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("train", exist_ok=True)
    os.makedirs("eval", exist_ok=True)
    os.makedirs("vocab", exist_ok=True)

    # Download datasets
    for idx, url in enumerate(datasets):
        filename = f"dataset/dataset_{idx}.h5ad"
        print(f"Downloading dataset {idx+1}/{len(datasets)}: {url}")
        download_dataset(url, filename)

    # Process each downloaded dataset
    for idx, url in enumerate(datasets):
        adata_file = f"dataset/dataset_{idx}.h5ad"
        if not os.path.exists(adata_file):
            print(f"Dataset file {adata_file} not found, skipping.")
            continue

        # Build vocabularies
        print(f"\nBuilding vocabularies for dataset {idx+1}")
        gene_vocab = build_gene_vocab(
            adata_file=adata_file,
            vocab_file=f"vocab/gene_vocab.json"
        )
        celltype_vocab, tissue_vocab, disease_vocab = build_celltype_tissue_disease_vocab(
            adata_file=adata_file,
            celltype_vocab_file=f"vocab/celltype_vocab.json",
            tissue_vocab_file=f"vocab/tissue_vocab.json",
            disease_vocab_file=f"vocab/disease_vocab.json"
        )

        # Load AnnData object
        adata = sc.read_h5ad(adata_file)

        # Shuffle and split dataset
        print(f"\nShuffling and splitting dataset {idx+1}")
        train_adata, eval_adata = shuffle_and_split_anndata(
            adata=adata,
            train_size=0.8,
            random_state=42,
            train_file=f"train/train_adata_{idx}.h5ad",
            eval_file=f"eval/eval_adata_{idx}.h5ad"
        )
        print(f"Training set size: {train_adata.n_obs} cells, saved to train/train_adata_{idx}.h5ad")
        print(f"Evaluation set size: {eval_adata.n_obs} cells, saved to eval/eval_adata_{idx}.h5ad")

if __name__ == "__main__":
    prepare_data()
    print("\nData preparation complete!")