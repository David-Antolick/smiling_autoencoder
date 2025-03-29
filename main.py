#!/usr/bin/env python3

import gzip
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
import argparse
from tqdm import tqdm

class Lang:
    '''Your original mapping with EOS=0 and no separate PAD token.'''
    
    def __init__(self):
        # Here 0 = '$' is end-of-sequence, 1 = '^' is start-of-sequence
        # This matches what your eval file expects.
        self.chartoindex = {'$': 0,'^': 1, 'C': 2, '(': 3,
                            '=': 4, 'O': 5, ')': 6, '[': 7, '-': 8, ']': 9,
                            'N': 10, '+': 11, '1': 12, 'P': 13, '2': 14, '3': 15,
                            '4': 16, 'S': 17, '#': 18, '5': 19, '6': 20, '7': 21,
                            'H': 22, 'I': 23, 'B': 24, 'F': 25, '8': 26, '9': 27
                            } 
        self.indextochar = {v:k for k,v in self.chartoindex.items()}
        self.nchars = 28  # same total count as before
        
    def indexesFromSMILES(self, smiles_str):
        '''Convert SMILES into array of ints. Append 0 ($) as EoS.'''
        idxs = [self.chartoindex[ch] for ch in smiles_str]
        idxs.append(self.chartoindex["$"])  # 0
        return np.array(idxs, dtype=np.uint8)
        
    def indexToSmiles(self, indices):
        '''Convert list of indices to SMILES, stopping at first 0.'''
        out = []
        for x in indices:
            if x == 0:  # that's $
                break
            out.append(self.indextochar[int(x)])
        return ''.join(out)


class SmilesDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, max_length=150, limit=None):
        self.max_length = max_length
        self.language = Lang()
        self.examples = np.load(data_path)  # shape [N, max_len], presumably

        if limit is not None:
            self.examples = self.examples[:limit]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Return (input_seq, target_seq) in a teacher-forcing format:
            input_seq = [ ^,  token2, token3, ..., last-but-one ]
            target_seq = [ token2, token3, ..., last, 0($?), ...]
        Because 0 is also used for “padding,” everything after the real SMILES is also 0.
        """
        full_seq = self.examples[idx]
        # We'll slice them so that input_seq is everything except the last token,
        # and target_seq is everything except the first token, standard language modeling approach.
        input_seq  = torch.tensor(full_seq[:-1], dtype=torch.long)
        target_seq = torch.tensor(full_seq[1:], dtype=torch.long)
        return input_seq, target_seq

    def getIndexToChar(self):
        return self.language.indextochar


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        # We do NOT do "padding_idx=0" because that means ignoring the “end” tokens in the embedding.
        # If all trailing tokens are 0, the model sees them as real inputs, which is not typical,
        # but we can't fix that if we cannot separate EOS vs. PAD.
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.to_mean = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)         # (B, T, emb_dim)
        _, h_n = self.gru(embedded)                  # h_n -> (1, B, hidden_dim)
        h_n = h_n.squeeze(0)                         # (B, hidden_dim)
        mean = self.to_mean(h_n)                     # (B, latent_dim)
        logvar = self.to_logvar(h_n)                 # (B, latent_dim)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, latent_dim, max_length=150):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.to_vocab = nn.Linear(hidden_dim, vocab_size)

        self.max_length = max_length
        self.vocab_size = vocab_size
        # '^' = 1 is the start token
        self.start_token = 1

    def forward(self, z, actual_input=None):
        batch_size = z.size(0)
        hidden_0 = self.latent_to_hidden(z).unsqueeze(0)

        if actual_input is not None:
            # Teacher-forcing for training
            emb = self.embedding(actual_input)
            output, _ = self.gru(emb, hidden_0)
            decoder_output = self.to_vocab(output)
            return decoder_output, actual_input
        else:
            # Inference: step-by-step greedy sampling
            dummy_output = torch.zeros(
                (batch_size, self.max_length, self.vocab_size),
                device=z.device, dtype=torch.float
            )
            outputs = []
            input_step = torch.full((batch_size, 1), self.start_token, dtype=torch.long, device=z.device)
            hidden = hidden_0

            for _ in range(self.max_length):
                emb = self.embedding(input_step)
                out, hidden = self.gru(emb, hidden)
                logits = self.to_vocab(out[:, -1])
                next_token = torch.argmax(logits, dim=-1, keepdim=True)  # shape (B,1)
                outputs.append(next_token)
                input_step = next_token

            generated_sequence = torch.cat(outputs, dim=1)
            return dummy_output, generated_sequence


class VAE(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, latent_dim=1024, max_length=150):
        super(VAE, self).__init__()
        self.encoder = Encoder(vocab_size, emb_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(vocab_size, emb_dim, hidden_dim, latent_dim, max_length)

    def forward(self, input_seq):
        mean, logv = self.encoder(input_seq)
        std = torch.exp(0.5 * logv)
        eps = torch.randn_like(std)
        z = mean + std * eps
        decoder_output, generated_seq = self.decoder(z, actual_input=input_seq)
        return decoder_output, generated_seq, (mean, logv, z)


def run_training(vae, dataloader, device, epochs, lr=1e-3):
    """
    A basic training loop for a VAE that must treat EOS=0 also as 'padding'.
    We do *not* do ignore_index=0 in cross-entropy, so the model is penalized
    for incorrectly generating zeros or failing to generate them in the right places.
    """
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    # Simple linear schedule for KL from 0 to 1 over a few epochs
    kl_anneal_epochs = max(1, epochs // 2)  # half the training for ramp-up

    for epoch in range(epochs):
        vae.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        log_lines = []

        # Current KL weight
        fraction = min(1.0, epoch / float(kl_anneal_epochs))
        kl_weight = .001  # from 0.0 to 1.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for batch_idx, (inp, tgt) in enumerate(progress_bar):
            inp, tgt = inp.to(device), tgt.to(device)

            optimizer.zero_grad()
            decoder_out, _, (mean, logv, _) = vae(inp)

            # CrossEntropy with no ignore_index => the model is penalized for everything.
            # dimension check: decoder_out (B, T, vocab), need (B, vocab, T) for CE
            recon_loss = F.cross_entropy(decoder_out.transpose(1,2), tgt)

            # KL
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logv - mean.pow(2) - logv.exp(), dim=1))
            loss = recon_loss + kl_weight * kl_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

            if batch_idx % 100 == 0:
                log_line = (f"[Batch {batch_idx}] "
                            f"Loss: {loss.item():.4f}, "
                            f"Recon: {recon_loss.item():.4f}, "
                            f"KL: {kl_loss.item():.4f}, "
                            f"KL_weight={kl_weight:.3f}")
                log_lines.append(log_line)

        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon_loss / len(dataloader)
        avg_kl = total_kl_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}, w={kl_weight:.2f})")

        # Save logs
        with open("batch_logs.txt", "a") as f:
            for line in log_lines:
                f.write(line + "\n")
            f.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a Variational Autoencoder - keep EOS=0')
    parser.add_argument('--train_data', required=True, help='Path to .npy file with tokenized SMILES (0=EOS).')
    parser.add_argument('--out', default='vae_model.pth', help='File to save traced decoder to')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=2048)
    parser.add_argument('--latent_dim', type=int, default=1024)
    parser.add_argument('--max_length', type=int, default=150)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    dataset = SmilesDataset(args.train_data, max_length=args.max_length, limit=args.limit)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True
    )

    vocab_size = dataset.language.nchars  # should be 28 if you have $=0, etc.

    # Build model
    vae = VAE(
        vocab_size=vocab_size,
        emb_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        max_length=args.max_length
    ).to(device)

    # Train
    run_training(vae, dataloader, device, epochs=args.epochs, lr=args.lr)

    # Export only the decoder for your `eval.py` usage:
    vae.eval()
    z_1 = torch.normal(0, 1, size=(1, args.latent_dim), device=device)
    decoder = vae.decoder  # single-GPU version
    decoder.eval()

    with torch.no_grad():
        traced_decoder = torch.jit.trace(decoder, (z_1,), check_trace=False)
        torch.jit.save(traced_decoder, args.out)

    print(f"Traced decoder saved to {args.out}")
