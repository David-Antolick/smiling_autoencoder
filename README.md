# SMILES Variational Autoencoder (VAE)

This repository contains a PyTorch-based implementation of a Variational Autoencoder (VAE) designed to learn a continuous representation of chemical molecules represented as SMILES (Simplified Molecular Input Line Entry System) strings. The model enables generation of valid, unique, and novel molecular structures by sampling from the learned latent space.

## 🧬 Overview

The VAE architecture encodes tokenized SMILES strings into a latent vector and decodes them to reconstruct or generate new SMILES. It is trained on molecular datasets such as PubChem, with optional augmentation through SMILES randomization. The project includes modules for preprocessing, training, evaluation, and dataset stratification.

## 📁 Project Structure

- `main.py` — Training script that builds, trains, and saves the VAE decoder.
- `eval.py` — Evaluates the generative capacity of the trained decoder.
- `preprocess_smiles.py` — Converts `.smi` files to tokenized `.npy` arrays.
- `randomize_smiles.py` — Generates randomized (non-canonical) SMILES for data augmentation.
- `subset.py` — Creates a stratified subset of SMILES based on sequence length.
- `test.py` — Simple script to verify CUDA availability.
- `requirements.txt` — Lists required packages.
- `notes.txt` — Contains example file paths and dataset references.

## 🚀 Quickstart

### 1. Preprocess SMILES Data

```bash
python preprocess_smiles.py --input_file path/to/input.smi --output_npy path/to/output.npy
````

### 2. Train the VAE

```bash
python main.py --train_data path/to/output.npy --epochs 10 --out vae_model.pth
```

### 3. Evaluate Generative Performance

```bash
python eval.py --model vae_model.pth --evals 1000 --train_pickle path/to/train_smiles.pkl
```

Evaluation metrics include:

* Number of valid SMILES
* Number of unique and novel molecules
* Average number of rings per structure

## 🧰 Dependencies

Install all required packages via:

```bash
pip install -r requirements.txt
```

### Required packages

* `torch`
* `numpy`
* `rdkit`
* `scikit-learn`
* `tqdm`

## 📝 Notes

* The model uses a fixed vocabulary of 28 SMILES tokens including SOS/EOS markers.
* Latent dimensionality defaults to 1024 and can be adjusted via command-line.
* Example data paths are included in `notes.txt` for reference.
* Trained decoders are exported using TorchScript for easy loading and inference.

---

