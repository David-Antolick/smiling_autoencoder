#!/usr/bin/env python3

import gzip
import numpy as np
import argparse
import os
from tqdm import tqdm

# Language class for SMILES vocab
class Lang:
    def __init__(self):
        self.chartoindex = {'$': 0, '^': 1, 'C': 2, '(': 3,
            '=': 4, 'O': 5, ')': 6, '[': 7, '-': 8, ']': 9,
            'N': 10, '+': 11, '1': 12, 'P': 13, '2': 14, '3': 15,
            '4': 16, 'S': 17, '#': 18, '5': 19, '6': 20, '7': 21,
            'H': 22, 'I': 23, 'B': 24, 'F': 25, '8': 26, '9': 27
        } 
        self.indextochar = {v: k for k, v in self.chartoindex.items()}
        self.nchars = len(self.chartoindex)

    def indexesFromSMILES(self, smiles_str):
        index_list = [self.chartoindex[char] for char in smiles_str]
        index_list.append(self.chartoindex["$"])  # EOS token
        return np.array(index_list, dtype=np.uint8)

# Detect whether file is gzipped
def smart_open(filepath, mode='rt'):
    if filepath.endswith(".gz"):
        return gzip.open(filepath, mode)
    else:
        return open(filepath, mode)

# Preprocessing logic
def preprocess(input_file, output_npy, max_length=150):
    lang = Lang()

    print(f"Counting lines in {input_file}...")
    with smart_open(input_file, 'rt') as f:
        N = sum(1 for _ in f)

    print(f"Allocating array of shape ({N}, {max_length})...")
    smiles_array = np.zeros((N, max_length), dtype=np.uint8)

    print(f"Processing SMILES strings...")
    with smart_open(input_file, 'rt') as f:
        for i, line in enumerate(tqdm(f, total=N)):
            smiles = line.strip()
            try:
                indices = lang.indexesFromSMILES(smiles)
                indices = indices[:max_length]  # truncate if too long
                smiles_array[i, :len(indices)] = indices
            except KeyError as e:
                print(f"Skipping line {i} due to unknown character: {smiles}")

    print(f"Saving preprocessed data to {output_npy}...")
    np.save(output_npy, smiles_array)
    print("Done.")

# Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SMILES (.smi or .smi.gz) into .npy format")
    parser.add_argument("--input_file", required=True, help="Input SMILES file (.smi or .smi.gz)")
    parser.add_argument("--output_npy", required=True, help="Output filename (e.g., smiles.npy)")
    parser.add_argument("--max_length", type=int, default=150, help="Max SMILES length (default=150)")
    args = parser.parse_args()

    preprocess(args.input_file, args.output_npy, args.max_length)
