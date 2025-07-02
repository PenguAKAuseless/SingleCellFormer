import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
from typing import Tuple, List
import logging

# python dataset_splitter.py --input_dir /path/to/datasets --output_dir /path/to/output

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def collect_dataset_files(directory: str, file_extensions: List[str] = ['.csv', '.npy']) -> List[str]:
    """
    Collect all dataset files from the specified directory with given extensions.
    
    Args:
        directory (str): Path to the directory containing dataset files
        file_extensions (List[str]): List of file extensions to include
    
    Returns:
        List[str]: List of file paths
    """
    try:
        all_files = []
        for ext in file_extensions:
            all_files.extend(glob.glob(os.path.join(directory, f'*{ext}')))
        logging.info(f"Found {len(all_files)} files in {directory}")
        return all_files
    except Exception as e:
        logging.error(f"Error collecting files: {str(e)}")
        return []

def load_and_concatenate_data(file_paths: List[str], chunk_size: int = 10000) -> np.ndarray:
    """
    Load and concatenate data from multiple files in chunks to avoid memory overflow.
    
    Args:
        file_paths (List[str]): List of file paths to load
        chunk_size (int): Size of chunks for loading large files
    
    Returns:
        np.ndarray: Concatenated dataset
    """
    data_list = []
    
    for file_path in file_paths:
        try:
            if file_path.endswith('.csv'):
                # Read CSV in chunks
                for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                    data_list.append(chunk.values)
            elif file_path.endswith('.npy'):
                # Load numpy array directly
                data = np.load(file_path, mmap_mode='r')
                data_list.append(data)
            logging.info(f"Loaded {file_path}")
        except Exception as e:
            logging.error(f"Error loading {file_path}: {str(e)}")
    
    # Concatenate all chunks
    if data_list:
        return np.concatenate(data_list, axis=0)
    else:
        return np.array([])

def shuffle_and_split_data(
    data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shuffle and split the dataset into train, validation, and test sets.
    
    Args:
        data (np.ndarray): Input dataset
        train_ratio (float): Ratio for training set
        val_ratio (float): Ratio for validation set
        test_ratio (float): Ratio for test set
        random_seed (int): Random seed for reproducibility
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Train, validation, and test sets
    """
    try:
        # Input validation
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Shuffle indices instead of data to save memory
        np.random.seed(random_seed)
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        
        # Calculate split sizes
        n_samples = len(data)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        # Split indices
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # Create splits using indices
        train_data = data[train_indices]
        val_data = data[val_indices]
        test_data = data[test_indices]
        
        logging.info(f"Split dataset: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")
        return train_data, val_data, test_data
    
    except Exception as e:
        logging.error(f"Error in shuffle_and_split_data: {str(e)}")
        return np.array([]), np.array([]), np.array([])

def save_splits(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    output_dir: str
) -> None:
    """
    Save the split datasets to the output directory.
    
    Args:
        train_data (np.ndarray): Training data
        val_data (np.ndarray): Validation data
        test_data (np.ndarray): Test data
        output_dir (str): Directory to save the split datasets
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'train_data.npy'), train_data)
        np.save(os.path.join(output_dir, 'val_data.npy'), val_data)
        np.save(os.path.join(output_dir, 'test_data.npy'), test_data)
        
        logging.info(f"Saved splits to {output_dir}")
    except Exception as e:
        logging.error(f"Error saving splits: {str(e)}")

def main(
    input_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    chunk_size: int = 10000
) -> None:
    """
    Main function to process dataset: collect, shuffle, split, and save.
    
    Args:
        input_dir (str): Input directory containing dataset files
        output_dir (str): Output directory to save split datasets
        train_ratio (float): Ratio for training set
        val_ratio (float): Ratio for validation set
        test_ratio (float): Ratio for test set
        chunk_size (int): Size of chunks for loading large files
    """
    # Collect dataset files
    file_paths = collect_dataset_files(input_dir)
    if not file_paths:
        logging.error("No dataset files found. Exiting.")
        return
    
    # Load and concatenate data
    data = load_and_concatenate_data(file_paths, chunk_size)
    if data.size == 0:
        logging.error("No data loaded. Exiting.")
        return
    
    # Shuffle and split data
    train_data, val_data, test_data = shuffle_and_split_data(
        data, train_ratio, val_ratio, test_ratio
    )
    
    # Save splits
    save_splits(train_data, val_data, test_data, output_dir)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect, shuffle, and split dataset")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing dataset files")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save split datasets")
    parser.add_argument('--train_ratio', type=float, default=0.7, help="Ratio for training set")
    parser.add_argument('--val_ratio', type=float, default=0.15, help="Ratio for validation set")
    parser.add_argument('--test_ratio', type=float, default=0.15, help="Ratio for test set")
    parser.add_argument('--chunk_size', type=int, default=10000, help="Size of chunks for loading large files")
    args = parser.parse_args()
    
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        chunk_size=args.chunk_size
    ) 