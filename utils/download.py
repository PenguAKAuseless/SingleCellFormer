from tqdm import tqdm
import requests
import numpy as np
import os
import scanpy as sc
from sklearn.model_selection import train_test_split

def download_dataset(url, filename):
    """
    Downloads a dataset from a given URL and saves it to the specified filename with a progress bar.

    Parameters:
    - url: str, URL of the dataset to download
    - filename: str, local path to save the downloaded file

    Returns:
    - None
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        total_size = int(response.headers.get("content-length", 0))

        with open(filename, "wb") as f, tqdm(
            desc=os.path.basename(filename), total=total_size, unit="B", unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        print(f"Download complete: {filename}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        raise

def shuffle_and_split_anndata(adata, train_size=0.8, random_state=None, train_file="train/train_adata.h5ad", eval_file="eval/eval_adata.h5ad"):
    """
    Shuffles an AnnData dataset, splits it into training and evaluation sets, and saves them to files.
    
    Parameters:
    - adata: AnnData object
    - train_size: float, proportion of dataset to include in the train split (default: 0.8)
    - random_state: int, random seed for reproducibility (default: None)
    - train_file: str, file path to save the training AnnData object (default: 'train/train_adata.h5ad')
    - eval_file: str, file path to save the evaluation AnnData object (default: 'eval/eval_adata.h5ad')
    
    Returns:
    - train_adata: AnnData object for training
    - eval_adata: AnnData object for evaluation
    """
    # Get the number of observations
    n_obs = adata.n_obs
    
    # Generate indices and shuffle them
    indices = np.arange(n_obs)
    train_idx, eval_idx = train_test_split(
        indices, 
        train_size=train_size, 
        random_state=random_state, 
        shuffle=True
    )
    
    # Create new AnnData objects for train and eval sets
    train_adata = adata[train_idx, :].copy()
    eval_adata = adata[eval_idx, :].copy()
    
    # Save the AnnData objects to files
    train_adata.write_h5ad(train_file, compression="gzip", compression_opts=4)
    eval_adata.write_h5ad(eval_file, compression="gzip", compression_opts=4)
    
    return train_adata, eval_adata

def create_vocabularies(adata, gene_id_vocab_path="vocab/gene_id_vocab.txt", cell_type_vocab_path="vocab/cell_type_vocab.txt"):
    """
    Creates vocabularies for gene IDs and cell types from an AnnData object and saves them to files.
    
    Parameters:
    - adata: AnnData object
    - gene_id_vocab_path: str, file path to save the gene ID vocabulary (default: 'vocab/gene_id_vocab.txt')
    - cell_type_vocab_path: str, file path to save the cell type vocabulary (default: 'vocab/cell_type_vocab.txt')

    Returns:
    - None
    """
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(gene_id_vocab_path), exist_ok=True)
    os.makedirs(os.path.dirname(cell_type_vocab_path), exist_ok=True)
    
    # Load old vocabularies if they exist
    gene_id_vocab = set()
    cell_type_vocab = set()
    if os.path.exists(gene_id_vocab_path):
        with open(gene_id_vocab_path, 'r') as f:
            gene_id_vocab = set(f.read().splitlines())
    if os.path.exists(cell_type_vocab_path):
        with open(cell_type_vocab_path, 'r') as f:
            cell_type_vocab = set(f.read().splitlines())
    
    # Extract gene IDs and cell types from the AnnData object
    gene_ids = set(adata.var_names)
    cell_types = set(adata.obs['cell_type'].unique())
    
    # Update vocabularies with new entries
    gene_id_vocab.update(gene_ids)
    cell_type_vocab.update(cell_types)
    
    # Save the updated vocabularies to files
    with open(gene_id_vocab_path, 'w') as f:
        for gene_id in sorted(gene_id_vocab):
            f.write(f"{gene_id}\n")
    with open(cell_type_vocab_path, 'w') as f:
        for cell_type in sorted(cell_type_vocab):
            f.write(f"{cell_type}\n")

def main(dataset_paths, output_dir, vocab_dir, num_processes=4, download=True, preprocess=True):
    """
    Main function to handle downloading and preprocessing of datasets.

    Parameters:
    - dataset_paths: list of str, URLs or paths to the datasets
    - output_dir: str, directory to save the downloaded datasets
    - vocab_dir: str, directory to save vocabulary files
    - num_processes: int, number of processes for parallel execution
    - download: bool, whether to download the datasets
    - preprocess: bool, whether to preprocess the datasets

    Returns:
    - None
    """
    # Create necessary directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vocab_dir, exist_ok=True)
    
    if download:
        for idx, url in enumerate(dataset_paths):
            filename = os.path.join(output_dir, "dataset_" + str(idx) + ".h5ad")
            download_dataset(url, filename)
    
    if preprocess:
        # Create train and eval directories
        train_dir = os.path.join(output_dir, "train")
        eval_dir = os.path.join(output_dir, "eval")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        
        for idx, filename in enumerate(os.listdir(output_dir)):
            if filename.endswith(".h5ad") and filename.startswith("dataset_"):
                adata = sc.read_h5ad(os.path.join(output_dir, filename))
                
                # Define proper file paths for train and eval splits
                train_file = os.path.join(train_dir, f"train_adata_{idx}.h5ad")
                eval_file = os.path.join(eval_dir, f"eval_adata_{idx}.h5ad")
                
                # Split the data and save to proper locations
                train_adata, eval_adata = shuffle_and_split_anndata(
                    adata, 
                    train_file=train_file, 
                    eval_file=eval_file
                )
                
                # Create vocabularies
                create_vocabularies(adata, 
                                    gene_id_vocab_path=os.path.join(vocab_dir, "gene_id_vocab.txt"),
                                    cell_type_vocab_path=os.path.join(vocab_dir, "cell_type_vocab.txt"))
        

if __name__ == "__main__":
    dataset_paths = [
        "https://datasets.cellxgene.cziscience.com/ffdaa1f0-b1d1-4135-8774-9fed7bf039ba.h5ad",
        "https://datasets.cellxgene.cziscience.com/d80e6ae7-2848-48ea-a898-70c80ae349c2.h5ad",
        "https://datasets.cellxgene.cziscience.com/0483d387-4ef6-4bcc-af68-6e3127979711.h5ad",
        "https://datasets.cellxgene.cziscience.com/3cbdbdaa-098a-42db-a601-d4c6454925e5.h5ad",
        "https://datasets.cellxgene.cziscience.com/bdae7c8d-5d2c-45a2-a149-8ba7d9260926.h5ad",
        "https://datasets.cellxgene.cziscience.com/b8eeb150-0420-4b56-b8b1-b8e488783949.h5ad",
        "https://datasets.cellxgene.cziscience.com/b77b62f8-16f9-42d0-9967-5802c0cd8ee2.h5ad",
        "https://datasets.cellxgene.cziscience.com/06c971c0-569d-494a-9fd0-60ef96c2da45.h5ad",
        "https://datasets.cellxgene.cziscience.com/4e6cf682-3aa0-4c79-a6e1-8abc21a85146.h5ad",
        "https://datasets.cellxgene.cziscience.com/348af6fd-d958-4d8c-9bd1-1062f54e2cc8.h5ad",
        "https://datasets.cellxgene.cziscience.com/3091260a-9c1e-461d-91c3-ab5e309b90c9.h5ad",
        "https://datasets.cellxgene.cziscience.com/c6d0a970-5043-4e38-8509-125a80edf930.h5ad"
    ]

    main(
        dataset_paths=dataset_paths,
        output_dir="/mnt/nasdev2/pengu-space",
        vocab_dir="vocab",
        num_processes=4,
        download=False,
        preprocess=True
    )