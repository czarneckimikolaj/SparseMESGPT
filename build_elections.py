import torch
from .election_model import Voter, Candidate, Election

def tensor_to_voter(id: str, utility_tensor, candidates, candidate_dict):
    # print(f"Voter {id} processed")
    new_voter = Voter(id)

    all_utils = torch.sum(utility_tensor) #Normalize utility of voter
    for neuron_id, value in enumerate(utility_tensor):
        if not torch.isclose(value, torch.tensor(0.)):
            candidate_dict[candidates[neuron_id]][new_voter] = value.item() / all_utils

    return new_voter

def build_elections(preferences_tensor, units_to_select):
    assert preferences_tensor.ndim == 2, "preferences tensor must be 2D, first dimension for voters and second for preferences for all candidates"

    num_candidates = preferences_tensor.shape[-1]
    # print(f"BE: {num_candidates} candidates")
    candidates = [Candidate(str(i), 1) for i in range(num_candidates)]
    # print(f"BE: candidates list built")
    pref_sums = preferences_tensor.sum(dim=1, keepdim=True)
    normalized = preferences_tensor/pref_sums

    eps = 1e-9
    mask = normalized.abs() > eps

    voters = [Voter(str(i)) for i in range(preferences_tensor.shape[0])]

    preferences = {candidate: {} for candidate in candidates}

    for cand_idx, candidate in enumerate(candidates):
        # print(f"BE: Processing candidate {cand_idx}")
        values = normalized[:, cand_idx]
        nonzero_mask = values != 0
        nonzero_indices = nonzero_mask.nonzero(as_tuple=True)[0]
        for voter_idx in nonzero_indices.tolist():
            preferences[candidate][voters[voter_idx]] = values[voter_idx].item()

    # print("BE: Voters list build, pref populated")
    for candidate in candidates:
        if len(preferences[candidate])==0:
            del preferences[candidate]
    
    election = Election(voters=voters, profile=preferences, budget=units_to_select)
    # print("BE: Elections built")
    return election