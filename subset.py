import numpy as np
import os

# Define the EOS token index (0 for '$')
EOS_TOKEN = 0

def create_subset(input_npy, output_npy, max_smiles_length=50):
    print(f"Loading data from {input_npy}...")
    data = np.load(input_npy)

    print("Finding sequences with actual length < 50...")
    # Count length up to first EOS (or max_length)
    eos_positions = np.argmax((data == EOS_TOKEN), axis=1)
    eos_positions[eos_positions == 0] = data.shape[1]  # handle rows with no EOS

    mask = eos_positions < max_smiles_length
    filtered_data = data[mask]

    print(f"Selected {filtered_data.shape[0]} sequences out of {data.shape[0]}")
    print(f"Saving filtered subset to {output_npy}...")
    np.save(output_npy, filtered_data)
    print("Done.")

if __name__ == "__main__":
    input_path = "data/randoms.npy"
    output_path = "data/randoms_under_50.npy"

    create_subset(input_path, output_path, max_smiles_length=50)
