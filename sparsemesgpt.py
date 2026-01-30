import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *

from election_model import Voter, Candidate, Election
from rules import bounded_overspending

DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class SparseMESGPT:
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
        self.H_diags = []  # List to store individual Hessians
        self.H_global = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
    
    def add_batch(self, inp, out, blocksize=1024, percdamp=0.01):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        
        batch_size = inp.shape[0]
        lambda_val = percdamp    

        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        
        # Process each sample individually
        if len(inp.shape) == 2:
            # inp is now (features, batch_size) after transpose
            for i in range(inp.shape[1]):
                x = inp[:, i].float() # Vector of size D
                
                # 1. Compute the squared norm (sum of squares)
                sum_sq = torch.sum(x**2)
                
                # 2. Apply Sherman-Morrison to get the diagonal of (xx^T + λI)^-1
                # diag_inv = (1/lambda) * (1 - (x^2 / (lambda + sum_sq)))
                diag_inv = (1.0 / lambda_val) * (1.0 - (x**2 / (lambda_val + sum_sq)))
                
                # Store only the diagonal (Size D)
                self.H_diags.append(diag_inv)
                
                # 3. We STILL need the global H for the weight updates later
                # But we only need ONE global H, not N of them.
                self.H_global += x.matmul(x.t()) 
                self.nsamples += 1

def make_election(utility, sparsity):
    num_voters = utility.shape[0]
    num_rows = utility.shape[1]
    num_cols = utility.shape[2]
    total_votes = num_rows * num_cols

    candidates = [Candidate(id=j, cost=1) for j in range(total_votes)]
    voters = [Voter(id=i) for i in range(num_voters)]

    voter_pref = utility.view(num_voters, -1)
    voter_sums = voter_pref.sum(dim=1, keepdim=True)
    voter_sums[voter_sums == 0] = 1.0 
    normalized_utilities = voter_pref / voter_sums
    
    profile = {candidate : {} for candidate in candidates}
    
    for v_idx, voter in enumerate(voters):
        for c_idx, candidate in enumerate(candidates):
            profile[candidate][voter] = normalized_utilities[v_idx, c_idx].item()

    budget = int(num_rows * num_cols * (1 - sparsity))

    election = Election(
        name="Pruning Election",
        voters=set(voters),
        profile=profile,
        budget=budget
    )

    return election

def results_to_mask(winners, num_rows, num_cols, device):
    # 1. Create a flattened mask of "True" (Prune everything by default)
    # In SparseGPT, True usually means "Prune" and False means "Keep"
    # (Check your mask1 logic: mask1 = tmp <= thresh means True is pruned)
    flattened_mask = torch.ones(num_rows * num_cols, device=device, dtype=torch.bool)

    # 2. Set winners to False (Do NOT prune)
    # If your winners are Candidate objects:
    winner_ids = [c.id for c in winners]
    
    # If your winners are just IDs, use them directly
    flattened_mask[winner_ids] = False

    # 3. Reshape back to the 2D block shape (R, C)
    return flattened_mask.view(num_rows, num_cols)

def fasterprune(self, sparsity, blocksize=128, percdamp=.01): #for prunen and prunem 0
    W = self.layer.weight.data.clone()
    if isinstance(self.layer, nn.Conv2d): W = W.flatten(1)
    if isinstance(self.layer, transformers.Conv1D): W = W.t()
    W = W.float()

    # --- 1. PREPARE GLOBAL HESSIAN ---
    H = self.H_global
    del self.H_global
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(self.columns, device=self.dev)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv_global = H # This is our "Update Engine"

    Losses = torch.zeros(self.rows, device=self.dev)
    
    # Convert H_diags list to a single tensor for vectorized scoring
    # Shape: (num_samples, columns)
    voter_diags = torch.stack(self.H_diags) 

    for i1 in range(0, self.columns, blocksize):
        i2 = min(i1 + blocksize, self.columns)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Hinv_global1 = Hinv_global[i1:i2, i1:i2]
        
        # --- 2. THE ELECTION (SCORING) ---
        # Get voter diagonals for this specific block: (num_samples, blocksize)
        voter_diags1 = voter_diags[:, i1:i2] 
        
        # Compute scores for ALL voters simultaneously
        # Using broadcasting: (rows, blocksize) / (num_samples, 1, blocksize)
        # Result utility_tensor shape: (num_samples, rows, blocksize)
        utility_tensor = (W1.unsqueeze(0) ** 2) / (voter_diags1.unsqueeze(1) ** 2)

        # YOUR ELECTION LOGIC HERE:
        election = make_election(utility_tensor, sparsity)
        winners = bounded_overspending(election)
        mask1 = results_to_mask(winners, W1.shape[0], W1.shape[1], self.dev)

        # --- 3. PRUNE & COMPENSATE ---
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)

        for i in range(count):
            w = W1[:, i]
            d = Hinv_global1[i, i]

            q = w.clone()
            q[mask1[:, i]] = 0 # Apply the election result

            if hasattr(self, 'quantizer'):
                q = quantize(q.unsqueeze(1), self.quantizer.scale, 
                             self.quantizer.zero, self.quantizer.maxq).flatten()

            Q1[:, i] = q
            
            # Error is (original_weight - pruned_weight) / inverse_hessian_diagonal
            # We use Hinv_global here to ensure the model stays stable
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv_global1[i, i:].unsqueeze(0))
            Err1[:, i] = err1

        # Global update for the rest of the matrix
        W[:, i1:i2] = Q1
        W[:, i2:] -= Err1.matmul(Hinv_global[i1:i2, i2:])

    # Finalize layer weights
    if isinstance(self.layer, transformers.Conv1D): W = W.t()
    self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)


    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
