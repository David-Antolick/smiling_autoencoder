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
    '''Predefined mapping from characters to indices for our
    reduced alphabet of SMILES with methods for converting.
    You must use this mapping.'''
    
    def __init__(self):
        # $ is the end of sequence token
        # ^ is the start of sequence token, which should never be generated
        self.chartoindex = {'$': 0,'^': 1, 'C': 2, '(': 3,
                '=': 4, 'O': 5, ')': 6, '[': 7, '-': 8, ']': 9,
                'N': 10, '+': 11, '1': 12, 'P': 13, '2': 14,'3': 15,
                '4': 16, 'S': 17, '#': 18, '5': 19,'6': 20, '7': 21,
                'H': 22, 'I': 23, 'B': 24, 'F': 25, '8': 26, '9': 27
                } 
        self.indextochar = {0: '$', 1: '^', 2: 'C', 3: '(',
                4: '=', 5: 'O', 6: ')', 7: '[', 8: '-', 9: ']',
                10: 'N', 11: '+', 12: '1', 13: 'P', 14: '2', 15: '3',
                16: '4', 17: 'S', 18: '#', 19: '5', 20: '6', 21: '7',
                22: 'H', 23: 'I', 24: 'B', 25: 'F', 26: '8', 27: '9'
                }
        self.nchars = 28
        
    def indexesFromSMILES(self, smiles_str):
        '''convert smiles string into numpy array of integers'''
        index_list = [self.chartoindex[char] for char in smiles_str]
        index_list.append(self.chartoindex["$"])
        return np.array(index_list, dtype=np.uint8)
        
    def indexToSmiles(self,indices):
        '''convert list of indices into a smiles string'''
        smiles_str = ''.join(list(map(lambda x: self.indextochar[int(x)] if x != 0.0 else 'E',indices)))
        return smiles_str.split('E')[0] #Only want values before output $ end of sequence token


class SmilesDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, max_length=150, limit=None):
        self.max_length = max_length
        self.language = Lang()
        self.examples = np.load(data_path)

        if limit is not None:
            self.examples = self.examples[:limit]

                
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        full_seq = self.examples[idx]  # already padded numpy array
        input_seq = torch.tensor(full_seq[:-1], dtype=torch.long)   # ^ C C ( = O ...
        target_seq = torch.tensor(full_seq[1:], dtype=torch.long)   #   C C ( = O $ ...
        return input_seq, target_seq


    
    def getIndexToChar(self):
        return self.language.indextochar


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
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
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.to_vocab = nn.Linear(hidden_dim, vocab_size)

        self.max_length = max_length
        self.vocab_size = vocab_size
        self.start_token = 1  # '^'

    def forward(self, z, actual_input=None):
        batch_size = z.size(0)
        hidden_0 = self.latent_to_hidden(z).unsqueeze(0)

        if actual_input is not None:
            # Training
            embedded = self.embedding(actual_input)   # (B, T, emb_dim)
            output, _ = self.gru(embedded, hidden_0)  # (B, T, hidden_dim)
            decoder_output = self.to_vocab(output)     # (B, T, vocab_size)
            return decoder_output, actual_input

        else:
            # Inferances
            dummy_decoder_output = torch.zeros(
                (batch_size, self.max_length, self.vocab_size),
                device=z.device,
                dtype=torch.float
            )

            outputs = []
            input_step = torch.full((batch_size, 1), self.start_token,
                                    dtype=torch.long, device=z.device)
            hidden = hidden_0

            for _ in range(self.max_length):
                embedded = self.embedding(input_step)
                out, hidden = self.gru(embedded, hidden)
                logits = self.to_vocab(out[:, -1])
                temperature = 0.9
                k = 5
                probs = F.softmax(logits / temperature, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
                topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
                sampled = torch.multinomial(topk_probs, num_samples=1)
                next_token = topk_indices.gather(-1, sampled)  # shape (B, 1)

                outputs.append(next_token)
                input_step = next_token

            generated_sequence = torch.cat(outputs, dim=1)

            # Return dummy_decoder_output and generated_sequence
            return dummy_decoder_output, generated_sequence



class VAE(nn.Module):      
    def __init__(self, vocab_size, emb_dim, hidden_dim, latent_dim=1024, max_length=150):
        super(VAE, self).__init__()        
        self.encoder = Encoder(vocab_size, emb_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(vocab_size, emb_dim, hidden_dim, latent_dim, max_length)

    def forward(self, input_seq):
        mean, logv = self.encoder(input_seq)
        std = torch.exp(0.5 * logv)
        eps = torch.randn_like(std)
        z = mean + std * eps  # Reparameterization
        decoder_output, generated_sequence = self.decoder(z, actual_input=input_seq)
        return decoder_output, generated_sequence, (mean, logv, z)


def run_training(vae, dataloader, device, epochs, lr=1e-3):
    """
    Basic training loop for the VAE.
    :param vae: The VAE model
    :param dataloader: PyTorch DataLoader of SMILES
    :param device: 'cuda' or 'cpu'
    :param epochs: number of epochs to train
    :param lr: learning rate
    """
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    for epoch in range(epochs):
        vae.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        phubar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for input_seq, target_seq in phubar:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            optimizer.zero_grad()

            decoder_output, _, (mean, logv, z) = vae(input_seq)

            recon_loss = F.cross_entropy(
                decoder_output.transpose(1, 2),  # (B, vocab, T)
                target_seq,                      # (B, T)
                ignore_index=0
            )

            kl_loss = -0.5 * torch.mean(torch.sum(1 + logv - mean.pow(2) - logv.exp(), dim=1) / logv.size(1))


            kl_weight = 0.001  # Fixed small value
            loss = recon_loss + kl_weight * kl_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()


        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon_loss / len(dataloader)
        avg_kl = total_kl_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} "
              f"(Recon: {avg_recon:.4f}, KL: {avg_kl:.4f})")



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a Variational Autoencoder')
    parser.add_argument('--train_data', required=True, help='Path to .npy file with tokenized SMILES')
    parser.add_argument('--out', default='vae_model.pth', help='File to save traced decoder to')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--epochs', type=int, default=7, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--latent_dim', type=int, default=1024)
    parser.add_argument('--max_length', type=int, default=150)
    parser.add_argument('--limit', type=int, default=None, help='Limit number of training examples')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset / DataLoader
    dataset = SmilesDataset(args.train_data, max_length=args.max_length, limit=args.limit)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             drop_last=True)

    vocab_size = dataset.language.nchars

    # Initialize model
    vae = VAE(
        vocab_size=vocab_size,
        emb_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        max_length=args.max_length
    ).to(device)

    # Use DataParallel if multiple GPUs
    if torch.cuda.device_count() > 1:
        vae = nn.DataParallel(vae)

    # TRAIN
    run_training(vae, dataloader, device, epochs=args.epochs, lr=args.lr)

    # Once trained, save a TorchScript version of just the decoder
    # for the evaluation script that calls model(z).
    # In practice, you may want to do this after a certain # of epochs or
    # using your best checkpoint, etc.
    vae.eval()
    z_1 = torch.normal(0, 1, size=(1, args.latent_dim), device=device)
    if isinstance(vae, nn.DataParallel):
        decoder = vae.module.decoder
    else:
        decoder = vae.decoder
    decoder.eval()
    with torch.no_grad():
        traced_decoder = torch.jit.trace(decoder, (z_1,), check_trace=False)
        torch.jit.save(traced_decoder, args.out)
    print(f"Traced decoder saved to {args.out}")
