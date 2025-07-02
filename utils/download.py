import pandas as pd
import numpy as np
import requests
import os
from pathlib import Path
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_dataset(url, save_path):
    """Download a dataset from a URL and save it locally."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Downloaded dataset to {save_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

def bin_expression(expression, num_bins=10):
    """Bin gene expression values into discrete categories."""
    try:
        bins = np.linspace(expression.min(), expression.max(), num_bins + 1)
        binned = np.digitize(expression, bins, right=True)
        return binned
    except Exception as e:
        logger.error(f"Error binning expression data: {e}")
        return expression

def create_vocabularies(df, gene_id_col, cell_type_col):
    """Create vocabularies for gene_id and cell_type."""
    try:
        # Create gene_id vocabulary
        unique_gene_ids = sorted(df[gene_id_col].unique())
        gene_id_vocab = {gene_id: idx for idx, gene_id in enumerate(unique_gene_ids)}
        
        # Create cell_type vocabulary
        unique_cell_types = sorted(df[cell_type_col].unique())
        cell_type_vocab = {cell_type: idx for idx, cell_type in enumerate(unique_cell_types)}
        
        return gene_id_vocab, cell_type_vocab
    except Exception as e:
        logger.error(f"Error creating vocabularies: {e}")
        return None, None

def save_vocabulary(vocab, output_path):
    """Save vocabulary to a JSON file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(vocab, f, indent=2)
        logger.info(f"Saved vocabulary to {output_path}")
    except Exception as e:
        logger.error(f"Error saving vocabulary to {output_path}: {e}")

def preprocess_dataset(file_path, batch_size=1000, gene_id_col='gene_id', cell_type_col='cell_type'):
    """Preprocess a dataset: extract gene_id, bin gene_expr, and extract cell_type."""
    try:
        # Read dataset in chunks to handle large files
        chunks = pd.read_csv(file_path, chunksize=batch_size)
        preprocessed_chunks = []

        for chunk in chunks:
            # Ensure required columns exist
            if gene_id_col not in chunk.columns or cell_type_col not in chunk.columns:
                logger.error(f"Required columns {gene_id_col} or {cell_type_col} not found in {file_path}")
                return None

            # Extract gene_id and cell_type
            preprocessed = chunk[[gene_id_col, cell_type_col]].copy()

            # Identify expression columns (all columns except gene_id and cell_type)
            expr_cols = [col for col in chunk.columns if col not in [gene_id_col, cell_type_col]]

            # Bin expression values for each expression column
            for col in expr_cols:
                preprocessed[f"{col}_binned"] = bin_expression(chunk[col])

            preprocessed_chunks.append(preprocessed)

        # Concatenate all processed chunks
        preprocessed_df = pd.concat(preprocessed_chunks, ignore_index=True)
        logger.info(f"Preprocessed dataset {file_path}")
        return preprocessed_df
    except Exception as e:
        logger.error(f"Error preprocessing {file_path}: {e}")
        return None

def save_preprocessed_dataset(df, output_path):
    """Save the preprocessed dataset to a file."""
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Saved preprocessed dataset to {output_path}")
    except Exception as e:
        logger.error(f"Error saving dataset to {output_path}: {e}")

def main(dataset_paths, output_dir="preprocessed_datasets", gene_id_vocab_path="gene_id_vocab.json", cell_type_vocab_path="cell_type_vocab.json", batch_size=1000):
    """Main function to download, preprocess datasets, and create/save vocabularies."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize lists to collect all gene_ids and cell_types across datasets
    all_preprocessed = []

    for path in dataset_paths:
        # Handle both local and remote paths
        file_name = os.path.basename(path)
        local_path = os.path.join(output_dir, file_name)
        output_path = os.path.join(output_dir, f"preprocessed_{file_name}")

        # Download if path is a URL
        if path.startswith(('http://', 'https://')):
            if not download_dataset(path, local_path):
                continue
        else:
            local_path = path  # Assume it's a local file

        # Preprocess the dataset
        preprocessed_df = preprocess_dataset(local_path, batch_size=batch_size)
        if preprocessed_df is not None:
            # Save the preprocessed dataset
            save_preprocessed_dataset(preprocessed_df, output_path)
            all_preprocessed.append(preprocessed_df)

    # Combine all preprocessed data to create vocabularies
    if all_preprocessed:
        combined_df = pd.concat(all_preprocessed, ignore_index=True)
        gene_id_vocab, cell_type_vocab = create_vocabularies(combined_df, 'gene_id', 'cell_type')
        
        if gene_id_vocab and cell_type_vocab:
            # Save vocabularies
            save_vocabulary(gene_id_vocab, os.path.join(output_dir, gene_id_vocab_path))
            save_vocabulary(cell_type_vocab, os.path.join(output_dir, cell_type_vocab_path))

if __name__ == "__main__":
    # Example list of dataset paths (replace with actual paths or URLs)
    dataset_paths = [
        "https://example.com/dataset1.csv",
        "https://example.com/dataset2.csv",
        "/local/path/to/dataset3.csv"
    ]
    
    # Run the preprocessing pipeline
    main(
        dataset_paths,
        output_dir="preprocessed",
        gene_id_vocab_path="vocab/gene_id_vocab.json",
        cell_type_vocab_path="vocab/cell_type_vocab.json",
        batch_size=1000
    )