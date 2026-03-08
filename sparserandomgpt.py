import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *

DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class SparseRandomGPT:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        
        # Reshape to (Tokens, Features)
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t() # Now (Features, Tokens)

        # 1. Update Sample Count
        old_nsamples = self.nsamples
        self.nsamples += tmp
        
        # 2. Scaling Factor (EMA style to keep H stable)
        self.H *= (old_nsamples / self.nsamples)
        
        # 3. CHUNKED ACCUMULATION (The OOM Fix)
        count = inp.shape[1] 
        chunk_size = 512    
        
        # Pre-scale inp to avoid doing it inside the loop
        inp = inp.float() * math.sqrt(2 / self.nsamples)
        
        for i in range(0, count, chunk_size):
            end = min(i + chunk_size, count)
            chunk = inp[:, i:end]
            self.H += chunk.matmul(chunk.t())
            
        del inp, chunk
        
        if self.dev.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.dev.type == 'mps':
            torch.mps.empty_cache()

    def fasterprune(
        self, sparsity, blocksize=128, percdamp=.01
    ):
        """
        Modified to randomly prune weights per block based on the sparsity ratio.
        Removed N:M arguments (prunen, prunem) as they conflict with pure random unstructured pruning.
        """
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            # --- NEW RANDOM PRUNING LOGIC ---
            # Calculate exact number of weights to prune in this block
            num_elements = W1.numel()
            num_prune = int(num_elements * sparsity)
            
            # Create a 1D boolean mask, set random indices to True, then reshape
            flat_mask = torch.zeros(num_elements, device=self.dev, dtype=torch.bool)
            rand_indices = torch.randperm(num_elements, device=self.dev)[:num_prune]
            flat_mask[rand_indices] = True
            mask1 = flat_mask.view_as(W1)
            # --------------------------------

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                q = w.clone()
                q[mask1[:, i]] = 0  # Apply the random mask for this column

                if hasattr(self, 'quantizer'):
                    q = quantize(
                        q.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = W[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        if self.dev.type == 'cuda':
            torch.cuda.synchronize()
        elif self.dev.type == 'mps':
            torch.mps.synchronize()

        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None

        if self.dev.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.dev.type == 'mps':
            torch.mps.empty_cache()