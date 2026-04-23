import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *

from election_model import Voter, Candidate, Election
from rules import bounded_overspending

from tqdm import tqdm

DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# def make_election_lazy(W1, voter_diags, sparsity, epsilon=1e-3):
#     # W1 shape: (num_rows, num_cols) -> Flattened candidates
#     # voter_diags shape: (num_voters, 1) or (num_voters,)
    
#     num_voters = voter_diags.shape[0]
#     num_rows, num_cols = W1.shape
#     total_votes = num_rows * num_cols
    
#     candidates = [Candidate(id=j, cost=1) for j in range(total_votes)]
#     voters = [Voter(id=i) for i in range(num_voters)]
#     profile = {candidate: {} for candidate in candidates}
    
#     # Pre-flatten W1 squared once to save redundant math
#     W1_flat_sq = (W1.view(-1) ** 2) 
    
#     # Process one voter at a time
#     for i in range(num_voters):
#         # Calculate utility for ONLY this voter
#         # Equivalent to: (W1**2) / (voter_diag[i]**2)
#         d_val = voter_diags[i].reshape(-1)[0] ** 2 
#         denom = max(d_val.item(), 1e-12)
        
#         # Now: (98304,) / scalar -> works perfectly
#         voter_utility = W1_flat_sq / denom
        
#         v_sum = voter_utility.sum().item()
#         v_sum = max(v_sum, 1e-12)
        
#         normalized_row = voter_utility / v_sum
        
#         # Pruning
#         mask = normalized_row > epsilon
#         cand_indices = mask.nonzero(as_tuple=True)[0]
        
#         if cand_indices.numel() > 0:
#             # Transfer to CPU immediately to free MPS memory
#             vals = normalized_row[cand_indices].tolist()
#             cands = cand_indices.tolist()
            
#             for c_idx, val in zip(cands, vals):
#                 profile[candidates[c_idx]][voters[i]] = val

#     budget = int(total_votes * (1 - sparsity))
#     nonempty_dict_count = sum(1 for v in profile.values() if v)
#     print(f"Election built with {len(profile)} candidates and {num_voters} voters, budget {budget} - {nonempty_dict_count} candidates with nonzero utility")
#     return Election(
#         name="Pruning Election",
#         voters=set(voters),
#         profile=profile,
#         budget=budget
#     )
import torch

import torch
import numpy as np

import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def _compute_utility_core(w_sq_flat, d_vals_sq, epsilon, num_voters, num_cands):
    # Pre-allocate output (CANDIDATES x VOTERS)
    out = np.zeros((num_cands, num_voters), dtype=np.float32)
    
    # Parallelize over voters (the large dimension)
    for v in prange(num_voters):
        # 1. Calculate the normalization sum for this voter
        voter_sum = 0.0
        d_val = d_vals_sq[v]
        
        # Inner loop: Calculate normalization constant
        for c in range(num_cands):
            voter_sum += w_sq_flat[c] / d_val
            
        # 2. Safety check for division
        if voter_sum < 1e-12:
            continue
            
        # 3. Fill the matrix for this voter
        for c in range(num_cands):
            utility = (w_sq_flat[c] / d_val) / voter_sum
            # Apply epsilon threshold (pruning)
            if utility > epsilon:
                out[c, v] = utility
            else:
                out[c, v] = 0.0
                
    return out

def make_utility_matrix_numba(W1, voter_diags, epsilon=1e-6):
    """
    CPU-only Numba implementation.
    Input: NumPy arrays or Tensors (converted to numpy)
    """
    # Convert to numpy if they are Tensors
    if hasattr(W1, 'numpy'): W1 = W1.detach().cpu().numpy()
    if hasattr(voter_diags, 'numpy'): voter_diags = voter_diags.detach().cpu().numpy()
    
    # Pre-flatten and square
    w_sq_flat = (W1**2).flatten().astype(np.float32)
    d_vals_sq = (voter_diags**2).astype(np.float32)
    print(d_vals_sq.shape)
    
    num_voters = d_vals_sq.shape[0]
    num_cands = w_sq_flat.shape[0]
    
    # Call the JIT-compiled core
    return _compute_utility_core(w_sq_flat, d_vals_sq, epsilon, num_voters, num_cands)

def make_election_lazy_optimized(W1, voter_diags, sparsity, epsilon=1e-8, batch_size=512):
    num_voters = voter_diags.shape[0]
    num_rows, num_cols = W1.shape
    total_votes = num_rows * num_cols
    
    candidates = [Candidate(id=j, cost=1) for j in range(total_votes)]
    voters = [Voter(id=i) for i in range(num_voters)]
    profile = {candidate: {} for candidate in candidates}
    
    # 1. Precompute Squared Values
    W1_sq = W1 ** 2
    d_vals_all = voter_diags ** 2
    d_vals_all = torch.clamp(d_vals_all, min=1e-12)
    
    # aprint(f"Processing {num_voters} voters in batches of {batch_size}...")
    for start_idx in range(0, num_voters, batch_size):
        end_idx = min(start_idx + batch_size, num_voters)
        
        d_vals_batch = d_vals_all[start_idx:end_idx]
        
        # 2. Proper Broadcasting
        matrix_batch = W1_sq.unsqueeze(0) / d_vals_batch.unsqueeze(1)
        matrix_batch = matrix_batch.view(d_vals_batch.size(0), -1)
        
        # 4. Vectorize the Pruning for the batch
        # FIX: Move mask to CPU *before* nonzero to prevent MPS async garbage-read bugs
        mask_cpu = (matrix_batch > epsilon).cpu()
        v_idx_local, c_idx = mask_cpu.nonzero(as_tuple=True)
        
        if len(v_idx_local) == 0:
            continue
            
        v_idx_global = v_idx_local + start_idx
        
        # 5. Populate dict
        # Safely index the device tensor using the validated indices mapped back to device
        v_idx_local_dev = v_idx_local.to(matrix_batch.device)
        c_idx_dev = c_idx.to(matrix_batch.device)
        
        vals_list = matrix_batch[v_idx_local_dev, c_idx_dev].cpu().tolist()
        
        # Lists are already on CPU
        v_idx_list = v_idx_global.tolist() 
        c_idx_list = c_idx.tolist()
        
        for v_idx, c_idx, val in zip(v_idx_list, c_idx_list, vals_list):
            profile[candidates[c_idx]][voters[v_idx]] = val
            
        del matrix_batch, mask_cpu, v_idx_local, c_idx, v_idx_global
        del v_idx_local_dev, c_idx_dev
    
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    budget = int(total_votes * (1 - sparsity))
    nonempty_dict_count = sum(1 for v in profile.values() if v)
    
    print(f"Election built: {budget} - budget, {nonempty_dict_count} - candidates with supporters, {total_votes} - total candidates ({num_rows} rows x {num_cols} cols).")
    
    return Election(
        name="Pruning Election",
        voters=set(voters),
        profile=profile,
        budget=budget
    )

def results_to_mask(winners, num_rows, num_cols, device, object_winners=True):
    # 1. Create a flattened mask of "True" (Prune everything by default)
    # In SparseGPT, True usually means "Prune" and False means "Keep"
    # (Check your mask1 logic: mask1 = tmp <= thresh means True is pruned)
    flattened_mask = torch.ones(num_rows * num_cols, device=device, dtype=torch.bool)

    # 2. Set winners to False (Do NOT prune)
    # If your winners are Candidate objects:
    if object_winners:
        winner_ids = [c.id for c in winners]
    else:
        winner_ids = [c for c, is_winner in enumerate(winners) if is_winner==1]

    # If your winners are just IDs, use them directly
    flattened_mask[winner_ids] = False

    # 3. Reshape back to the 2D block shape (R, C)
    return flattened_mask.view(num_rows, num_cols)

def results_ndarray_to_mask(winners, num_rows, num_cols, device):
    # 1. Create a flattened mask of "True" (Prune everything by default)
    # In SparseGPT, True usually means "Prune" and False means "Keep"
    # (Check your mask1 logic: mask1 = tmp <= thresh means True is pruned)
    flattened_mask = torch.ones(num_rows * num_cols, device=device, dtype=torch.bool)
    
    # If your winners are just IDs, use them directly
    flattened_mask[winners] = False

    # 3. Reshape back to the 2D block shape (R, C)
    return flattened_mask.view(num_rows, num_cols)


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
    
    def add_batch(self, inp, out, percdamp=0.01):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        
        # Ensure 2D (batch_size, features)
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        elif len(inp.shape) == 2 and not (isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D)):
            inp = inp.unsqueeze(0)

        inp = inp.float()

        # 1. Global H update (Vectorized Matmul)
        # This replaces the iterative self.H_global update.
        # inp.t() @ inp is mathematically equivalent to sum(x @ x.t() for x in batch)
        self.H_global += inp.t().matmul(inp)
        
        # 2. Vectorized Sherman-Morrison for Diagonals
        # Instead of looping, we compute all diagonals in one go.
        # x_sq: (batch_size, features) -> squared elements
        # sum_sq: (batch_size, 1) -> sum of squares per sample
        x_sq = inp.float()**2
        sum_sq = torch.sum(x_sq, dim=1, keepdim=True) # Per-sample norm
        
        # diag_inv calculation using broadcasting
        # result shape: (batch_size, features)
        diag_inv_batch = (1.0 / percdamp) * (1.0 - (x_sq / (percdamp + sum_sq)))
        
        # 3. Store the result
        # We extend the list with the rows of the batch
        self.H_diags.extend(list(diag_inv_batch))
        self.nsamples += inp.shape[0]

        # Optional: Release memory for M4
        if self.dev.type == 'mps':
            torch.mps.empty_cache()


    def fasterprune(self, sparsity, blocksize=128, percdamp=.05):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d): W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D): W = W.t()
        W = W.float()

        H = self.H_global.float()
        del self.H_global
        
        # --- NEW: Sanitize the Hessian ---
        # If the Hessian has NaNs or Infs, Cholesky is impossible.
        if not torch.isfinite(H).all():
            print("Warning: Non-finite values detected in H. Nan-to-num triggered.")
            H = torch.nan_to_num(H, nan=0.0, posinf=1e10, neginf=-1e10)

        # 1. Handle Dead Neurons
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # 2. Robust Damping & Symmetrization
        H = (H + H.t()) / 2.0
        
        # Safeguard mean_diag against being 0 or inf
        mean_diag = torch.mean(torch.diag(H))
        if not torch.isfinite(mean_diag) or mean_diag <= 0:
            mean_diag = torch.tensor(1.0, device=self.dev)
            
        base_damp = percdamp * mean_diag + 1e-4 
        diag_idx = torch.arange(self.columns, device=self.dev)
        H[diag_idx, diag_idx] += base_damp

        # 3. CPU Shuffle + Triangular Solve
        H_cpu = H.cpu().double()
        L_cpu = None
        
        # Use a fixed, non-inf penalty step
        for attempt in range(10):
            try:
                L_cpu = torch.linalg.cholesky(H_cpu)
                break
            except torch._C._LinAlgError:
                # Use a static multiplier if mean_diag is suspicious
                step = mean_diag.item() if torch.isfinite(mean_diag) else 1.0
                penalty = (10 ** (attempt - 4)) * step
                print(f"Cholesky failed (attempt {attempt+1}). Adding {penalty:.6f} damping...")
                H_cpu[torch.arange(self.columns), torch.arange(self.columns)] += penalty
            
            if L_cpu is None:
                raise RuntimeError("Hessian factorization failed completely.")

        # Math: H^-1 = L^-T @ L^-1. 
        # The upper Cholesky of the inverse (which SparseGPT needs) is L^-1 transposed.
        # This is much faster than Hinv = cholesky_inverse(L) followed by another cholesky.
        L_inv_cpu = torch.linalg.solve_triangular(
            L_cpu, torch.eye(self.columns, dtype=torch.double), upper=False
        )
        
        # Move back to MPS as float32 for the pruning loop
        Hinv_global = L_inv_cpu.t().float().to(self.dev)

        # Cleanup CPU memory
        del H_cpu, L_cpu, L_inv_cpu

        Losses = torch.zeros(self.rows, device=self.dev)
        
        # Convert H_diags list to a single tensor for vectorized scoring
        # Shape: (num_samples, columns)
        print(f"Preparing voter diagonals, {self.H_diags.__len__()} samples collected...")
        print(f"H_diags elt shape {self.H_diags[0].shape}")
        voter_diags = torch.stack(self.H_diags) 
        print(f"voter diags shape {voter_diags.shape}")

        pbar = tqdm(total=math.floor(self.columns / blocksize))
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()

            Hinv_global1 = Hinv_global[i1:i2, i1:i2]
            
            # --- 2. THE ELECTION (SCORING) ---
            # Get voter diagonals for this specific block: (num_samples, blocksize)
            voter_diags1 = voter_diags[:, i1:i2] 
            # print(f"voter diags 1 shape {voter_diags1.shape}")
            
            # Compute scores for ALL voters simultaneously
            # Using broadcasting: (rows, blocksize) / (num_samples, 1, blocksize)
            # Result utility_tensor shape: (num_samples, rows, blocksize)
            rows = W1.shape[0]
            voters = voter_diags1.shape[0]
            # print(f"W1 shape: {W1.shape}, voter_diags1 shape: {voter_diags1.shape}")

            # YOUR ELECTION LOGIC HERE:

            # NUMBA ELECTION
            # start_time = time.time()
            # utility_3d = (W1.unsqueeze(0)**2) / (voter_diags1.unsqueeze(1)**2)
            # utility = utility_3d.view(voters, rows * count).t()
            # utility = np.array(utility.cpu())
          
            # utility = utility.T
            # num_candidates, num_voters = utility.shape
            
            # costs = np.ones(num_candidates)
            # budgets = int(sparsity * num_candidates)
            # make_election_time = time.time() - start_time

            # start_time = time.time()

            # winners = bounded_overspending_numba(utility, costs, budgets)
            # run_election_time = time.time() - start_time


            # mask1 = results_to_mask(winners, W1.shape[0], W1.shape[1], self.dev, object_winners=False)

            # ------------------

            # OOP ELECTION
            start_time = time.time()
            election = make_election_lazy_optimized(W1, voter_diags1, sparsity)
 
            make_election_time = time.time() - start_time

            start_time = time.time()
            winners = bounded_overspending(election)
            run_election_time = time.time() - start_time

            print(f"Election prepared in {make_election_time:.2f} seconds, run in {run_election_time:.2f} seconds, selected {len(winners)} winners.")

            mask1 = results_to_mask(winners, W1.shape[0], W1.shape[1], self.dev)
            # -------------------

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
            pbar.update(1)
        
        pbar.close()
        # Finalize layer weights
        if isinstance(self.layer, transformers.Conv1D): W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)


    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        
        if self.dev.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.dev.type == 'mps':
            torch.mps.empty_cache()
