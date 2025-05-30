# SMILES‑VAE – Generating Novel Small Molecules with a Character‑Level Variational Autoencoder

*Course project for **Scalable Machine Learning** (David Koes, University of Pittsburgh, Spring 2025)*

---

## 1 · Project overview

* \*\*Goal \*\*Learn a smooth latent space over canonical SMILES strings and sample **novel, chemically valid molecules** (up to 150 non‑hydrogen atoms).
* \*\*Model \*\*Bidirectional GRU encoder → 2‑layer latent projection → GRU decoder; optimised with a KL‑annealed ELBO.
* \*\*Dataset \*\*62 million PubChemLite SMILES (≤ 150 heavy atoms) canonicalised and deduplicated. A **10 % slice (≈ 6.2 M) containing only molecules ≤ 50 heavy atoms** is included in this repo to keep the download small; swap in the full `.npy` to reproduce larger‑scale runs.
* \*\*Key metric \*\*`NovelMols` – count of valid ∧ novel SMILES in a 10 k sample draw.

---

## 2 · Directory layout

```text
smiling_autoencoder/
├── src/
│   ├── data.py             # SMILESDataset, char vocab, train/val split
│   ├── vae_model.py        # CharVAE (encoder, sampler, decoder)
│   ├── loss.py             # KL‑annealed ELBO, teacher‑forcing scheduler
│   ├── train.py            # train loop + checkpointing
│   ├── sample.py           # latent → SMILES decode + RDKit validity test
│   ├── evaluate.py         # NovelMols, validity/uniqueness metrics
│   └── hyperopt.py         # Optuna sweep wrapper
├── configs/
│   └── default.yaml        # hidden_dim, β schedule, dropout, etc.
├── datasets/
│   └── pubchem_under50.npy # 6.2 M SMILES ≤50 heavy atoms (ships with repo)
├── checkpoints/            # *.pt saved by train.py
├── logs/                   # wandb or CSV logs
├── plots/                  # loss curves, latent TSNE, property hists
├── requirements.txt        # PyTorch‑CUDA, RDKit, tqdm, wandb
└── .devcontainer/          # CUDA 12 VS Code container
```

---

## 3 · Quick start (GPU)

```bash
# 1 · Clone
$ git clone https://github.com/<user>/smiling_autoencoder.git
$ cd smiling_autoencoder

# 2 · Install (venv/conda or VS Code dev‑container)
$ pip install -r requirements.txt

# 3 · Add dataset
#  – Small slice for quick tests already lives in datasets/pubchem_under50.npy
#  – To reproduce full experiments, drop pubchem_under150.npy (≈ 24 GB) here.

# 4 · Train (example: hidden_dim=2048, β‑anneal, no dropout)
$ python src/train.py \
      --config configs/default.yaml \
      --hidden_dim 2048 \
      --kl_beta 0.001 \
      --epochs 1 \
      --run_name run7
  # ↳ checkpoints/run7_epoch01.pt   logs/run7.csv  plots/run7_loss.png

# 5 · Sample 10 k molecules & evaluate
$ python src/sample.py --ckpt checkpoints/run7_epoch01.pt --num 10000 | \
  python src/evaluate.py --reference datasets/pubchem_test_smiles.txt
```



## 4 · Key hyper‑parameters

| flag                 | description                         | default |
| -------------------- | ----------------------------------- | ------- |
| `--hidden_dim`       | GRU hidden size                     | 1024    |
| `--latent_dim`       | z‑space size                        | 128     |
| `--kl_beta`          | final KL weight                     | 0.1     |
| `--kl_anneal_epochs` | epochs over which β rises 0 → value | 10      |
| `--dropout`          | decoder dropout probability         | 0.0     |
| `--teacher_forcing`  | p(TF) at epoch 0 (linear decay)     | 1.0     |

Optuna sweeps (`src/hyperopt.py`) explored hidden\_dim ∈ \[256, 4096], β ∈ \[1e‑4, 0.2], dropout ∈ \[0, 0.3], epochs ∈ \[1, 50].


## 5 · Engineering notes

* **Tokenisation** – character alphabet of 62 symbols (branching, ring digits, aromatic, etc.).
* **Batching** – PyTorch `PackedSequence` + gradient accumulation for memory‑efficient 200‑token sequences.
* **KL‑annealing** – linear schedule; optional cyclical cosine implemented.
* **Logging** – wandb captures ELBO, KL/sec, validity curves; artifacts in `plots/`.
* **Reproducibility** – Dev‑container with pinned CUDA 12.6 / PyTorch 2.2 ensures deterministic rebuild.

---

## 6 · Next steps

* Bayesian optimisation in latent space (BO‑EI) for property‑targeted generation.
* Conditional tokens for logP, MW, SA; multi‑task decoder.
* SELFIES tokenizer for 100 % validity.

---

## 7 · License

MIT — PRs and issues welcome!
