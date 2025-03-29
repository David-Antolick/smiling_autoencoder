import numpy as np
from tqdm import tqdm

EOS_TOKEN = 0

def stratified_smiles_subset_streamed(input_npy, output_npy, sample_per_bin=500_000, bins=((0, 50), (50, 100), (100, 150))):
    print(f"Loading data from {input_npy}...")
    data = np.load(input_npy, mmap_mode='r')  # streamed read

    bin_limits = {bin_range: sample_per_bin for bin_range in bins}
    bin_buckets = {bin_range: [] for bin_range in bins}

    print("Streaming through dataset and sorting into bins...")
    for i in tqdm(range(data.shape[0]), desc="Scanning", unit="seq"):
        row = data[i]
        eos_pos = np.where(row == EOS_TOKEN)[0]
        length = eos_pos[0] if eos_pos.size > 0 else row.shape[0]

        for bin_range in bins:
            lower, upper = bin_range
            if lower <= length < upper:
                if len(bin_buckets[bin_range]) < bin_limits[bin_range]:
                    bin_buckets[bin_range].append(i)
                break  # found the bin, move on

        # Stop early if all bins are full
        if all(len(bin_buckets[br]) >= bin_limits[br] for br in bins):
            break

    total_selected = sum(len(bucket) for bucket in bin_buckets.values())
    print(f"\nTotal selected: {total_selected:,} entries.")

    selected_indices = np.concatenate([np.array(bucket) for bucket in bin_buckets.values()])
    final_subset = data[selected_indices]

    print(f"Saving to {output_npy}...")
    np.save(output_npy, final_subset)
    print("Done.")

if __name__ == "__main__":
    input_path = "data/_random.npy"
    output_path = "data/randoms_stratified_subset.npy"

    stratified_smiles_subset_streamed(
        input_npy=input_path,
        output_npy=output_path,
        sample_per_bin=500_000,
        bins=((0, 50), (50, 100), (100, 149))
    )