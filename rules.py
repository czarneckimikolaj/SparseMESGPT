from .election_model import Election, Candidate, Voter
import math


###############################################################################
######################## UTILITARIAN GREEDY ###################################
###############################################################################


def _utilitarian_greedy_internal(e : Election, W : set[Candidate]) -> set[Candidate]:
    costW = sum(c.cost for c in W)
    remaining = set(c for c in e.profile if c not in W)
    ranked = sorted(remaining, key=lambda c : -sum(e.profile[c].values()) / c.cost)
    for c in ranked:
        if costW + c.cost <= e.budget:
            # print("SELECTED " + str(c.id) + ", utilit = " + str(sum(e.profile[c].values()))) 
            W.add(c)
            costW += c.cost
    return W

def utilitarian_greedy(e : Election) -> set[Candidate]:
    return _utilitarian_greedy_internal(e, set())


###############################################################################
################# PHRAGMEN'S SEQUENTIAL RULE ##################################
###############################################################################


def _phragmen_internal(e : Election, endow : dict[Voter, float], W : set[Candidate]) -> set[Candidate]:
    payment = {i : {} for i in e.voters}
    remaining = set(c for c in e.profile if c not in W)
    costW = sum(c.cost for c in W)
    while True:
        next_candidate = None
        lowest_time = float("inf")
        for c in remaining:
            if costW + c.cost <= e.budget:
                time = float(c.cost - sum(endow[i] for i in e.profile[c])) / len(e.profile[c])
                if time < lowest_time:
                    next_candidate = c
                    lowest_time = time
        if next_candidate is None:
            break
        W.add(next_candidate)
        costW += next_candidate.cost
        remaining.remove(next_candidate)
        for i in e.voters:
            if i in e.profile[next_candidate]:
                payment[i][next_candidate] = endow[i]
                endow[i] = 0
            else:
                endow[i] += lowest_time
    return W

def phragmen(e : Election) -> set[Candidate]:
    endow = {i : 0.0 for i in e.voters}
    return _phragmen_internal(e, endow, set())


###############################################################################
####################### METHOD OF EQUAL SHARES ################################
###############################################################################


def _mes_epsilons_internal(e : Election, eps_cost : bool, endow : dict[Voter, float], W : set[Candidate]) -> set[Candidate]:
    costW = sum(c.cost for c in W)
    remaining = e.profile.keys() - W
    rho = {c : c.cost - sum(endow[i] for i in e.profile[c]) for c in remaining}
    cnt = 0
    while True:
        cnt += 1
        next_candidate = None
        lowest_rho = math.inf
        voters_sorted = sorted(e.voters, key=lambda i: endow[i])
        for c in sorted(remaining, key=lambda c: rho[c]):
            if costW + c.cost > e.budget:
                continue
            if rho[c] >= lowest_rho:
                break
            sum_supporters = sum(endow[i] for i in e.profile[c])
            price = c.cost - sum_supporters
            for i in voters_sorted:
                if i not in e.profile[c]:
                    continue
                if endow[i] >= price:
                    if eps_cost:
                        rho[c] = price / c.cost
                    else:
                        rho[c] = price
                    break
                price -= endow[i]
            if rho[c] < lowest_rho:
                next_candidate = c
                lowest_rho = rho[c]
        if next_candidate is None:
            break
        else:
            W.add(next_candidate)
            costW += next_candidate.cost
            remaining.remove(next_candidate)
            for i in e.voters:
                if i in e.profile[next_candidate]:
                    endow[i] = 0
                else:
                    endow[i] -= min(endow[i], lowest_rho)
    return W

def _mes_internal(e : Election, real_budget : int = 0) -> (dict[Voter, float], set[Candidate]):
    W = set()

    costW = 0
    remaining = set(c for c in e.profile)
    endow = {i : 1.0 * e.budget / len(e.voters) for i in e.voters}
    rho = {c : c.cost / sum(e.profile[c].values()) for c in e.profile}
    while True:
        next_candidate = None
        lowest_rho = float("inf")
        remaining_sorted = sorted(remaining, key=lambda c: rho[c])
        #print(" Endowments: " + str(endow) + ", sum = " + str(sum(endow.values())))
        for c in remaining_sorted:
            if rho[c] >= lowest_rho:
                break
            if sum(endow[i] for i in e.profile[c]) + 0.001 >= c.cost:
                supporters_sorted = sorted(e.profile[c], key=lambda i: endow[i] / e.profile[c][i])
                price = c.cost
                util = sum(e.profile[c].values())
                for i in supporters_sorted:
                    if endow[i] * util >= price * e.profile[c][i]:
                        break
                    price -= endow[i]
                    util -= e.profile[c][i]
                if price > 0.001:
                    rho[c] = price / util
                else:
                    rho[c] = endow[supporters_sorted[-1]] / e.profile[c][supporters_sorted[-1]]
                if rho[c] < lowest_rho:
                    next_candidate = c
                    lowest_rho = rho[c]
        if next_candidate is None:
            break
        else:
            #print(" MES selected " + str(next_candidate))
            #print(str(len(e.profile[next_candidate])) + ", " + str(len(e.voters)/ 20) + ", " + str(e.budget) + ", " + str(e.budget/ len(e.voters)))

            W.add(next_candidate)
            costW += next_candidate.cost
            remaining.remove(next_candidate)
            #payments = []
            for i in e.profile[next_candidate]:
                endow[i] -= min(endow[i], lowest_rho * e.profile[next_candidate][i])
            #    payments.append(min(endow[i], lowest_rho * e.profile[next_candidate][i]))
            #print(str(sorted(payments)))
            if real_budget: #optimization for 'increase-budget' completions
                if costW > real_budget:
                    return None
    return endow, W

  


def equal_shares_simple(e : Election) -> set[Candidate]:
    W = set()
    costW = 0

    per_voter_score = {v: {} for v in e.voters}
    for c, l in e.profile.items():
        for v, u in l.items():
            per_voter_score[v][c] = u
    active_voters = {v: True for v in e.voters}

    endow = {i : 1.0 * e.budget / len(e.voters) for i in e.voters}
    scores = {c : sum(e.profile[c].values()) for c in e.profile}

    while len(scores) > 0:
        next_candidate = max(scores.keys(), key=lambda x: scores[x])
        supporters_sorted = sorted(e.profile[next_candidate], key=lambda i: endow[i] / e.profile[next_candidate][i])
        price = c.cost
        util = sum(e.profile[c].values())
        rho = 0

        for i in supporters_sorted:
            remove_i = False
            if endow[i] * util >= price * e.profile[next_candidate][i]:
                rho = price / util
            if rho > 0:
                endow[i] -= min(endow[i], rho * e.profile[next_candidate][i])
                if endow[i] < 0.001:
                    remove_i = True
            else:
                price -= endow[i]
                util -= e.profile[next_candidate][i]
                remove_i = True
            if remove_i and active_voters[i]:
                for c, u in per_voter_score[i].items():
                    if c in scores.keys():
                        scores[c] -= u
                active_voters[i] = False
        W.add(next_candidate)
        costW += next_candidate.cost
        del scores[next_candidate]
        too_exp = [c for c in scores.keys() if c.cost + costW > e.budget]
        for c in too_exp:
            del scores[c]
            
    return W


def _is_exhaustive(e : Election, W : set[Candidate]) -> bool:
    costW = sum(c.cost for c in W)
    minRemainingCost = min([c.cost for c in e.profile if c not in W], default=math.inf)
    return costW + minRemainingCost > e.budget

def equal_shares(e : Election, completion : str = None) -> set[Candidate]:
    endow, W = _mes_internal(e)
    # print("MES computed: " + str(W))
    if completion is None:
        return W
    if completion == 'binsearch':
        initial_budget = e.budget
        while not _is_exhaustive(e, W): #we keep multiplying budget by 2 to find lower and upper bounds for budget
            b_low = e.budget
            e.budget *= 2
            res_nxt = _mes_internal(e, real_budget=initial_budget)
            if res_nxt is None:
                break
            _, W = res_nxt
        b_high = e.budget
        while not _is_exhaustive(e, W) and b_high - b_low >= 1: #now we perform the classical binary search
            e.budget = (b_high + b_low) / 2.0
            res_med = _mes_internal(e, real_budget=initial_budget)
            if res_med is None:
                b_high = e.budget
            else:
                b_low = e.budget
                _, W = res_med
        e.budget = initial_budget
        return W
    if completion == 'utilitarian_greedy':
        return _utilitarian_greedy_internal(e, W)
    if completion == 'phragmen':
        return _phragmen_internal(e, endow, W)
    if completion == 'add1':
        initial_budget = e.budget
        while not _is_exhaustive(e, W):
            e.budget *= 1.01
            res_nxt = _mes_internal(e, real_budget=initial_budget)
            if res_nxt is None:
                break
            _, W = res_nxt
        e.budget = initial_budget
        return W
    if completion == 'add1_utilitarian':
        initial_budget = e.budget
        while not _is_exhaustive(e, W):
            e.budget *= 1.01
            res_nxt = _mes_internal(e, real_budget=initial_budget)
            if res_nxt is None:
                break
            _, W = res_nxt
        e.budget = initial_budget
        return _utilitarian_greedy_internal(e, W)
    if completion == 'eps':
        return _mes_epsilons_internal(e, False, endow, W)
    assert False, f"""Invalid value of parameter completion. Expected one of the following:
        * 'binsearch',
        * 'utilitarian_greedy',
        * 'phragmen',
        * 'add1',
        * 'add1_utilitarian',
        * 'eps',
        * None."""



def bounded_overspending(e : Election, real_budget : int = 0) -> (set[Candidate]):
    W = set()
    costW = 0
    remaining = set(c for c in e.profile)
    endow = {i : 1.0 * e.budget / len(e.voters) for i in e.voters}
    ratio = {c : -1.0 for c in e.profile}
    while True:
        next_candidate = None
        lowest_ratio = float("inf")
        remaining_sorted = sorted(remaining, key=lambda c: ratio[c])
        best_util = 0
        for c in remaining_sorted:
            if ratio[c] >= lowest_ratio:
                break
            if costW + c.cost <= e.budget:
                supporters_sorted = sorted([i for i in e.profile[c]], key=lambda i: endow[i] / e.profile[c][i])
                util = sum(e.profile[c].values())
                money_used = 0
                last_rho = 0
                new_ratio = float("inf")
                for i in supporters_sorted:
                    alpha = min(1.0, (money_used + util * (endow[i] / e.profile[c][i])) / c.cost)
                    if round(alpha, 5) > 0 and round(util, 5) > 0:
                        rho = ((alpha * c.cost) - money_used) / (alpha * util)
                        if rho < last_rho:
                            break
                        if rho / alpha < new_ratio :
                            new_ratio = rho / alpha
                            new_rho = rho
                    util -= e.profile[c][i]
                    money_used += endow[i]
                    last_rho = endow[i] / e.profile[c][i]
                ratio[c] = new_ratio
                if ratio[c] < lowest_ratio:
                    lowest_ratio = ratio[c]
                    lowest_rho = new_rho
                    next_candidate = c
                    best_util = sum([e.profile[c][i] for i in e.profile[c]])
                elif ratio[c] == lowest_ratio:
                    util = sum([e.profile[c][i] for i in e.profile[c]])
                    if util > best_util:
                        next_candidate = c
                        best_util = util
        if next_candidate is None:
            break
        else:
            W.add(next_candidate)
            costW += next_candidate.cost
            remaining.remove(next_candidate)
            for i in e.profile[next_candidate]:
                endow[i] -= min(endow[i], lowest_rho * e.profile[next_candidate][i])
            if real_budget: #optimization for 'increase-budget' completions
                if costW > real_budget:
                    return None
    return W


def bounded_overspending2(e : Election, real_budget : int = 0) -> (set[Candidate]):
    W = set()
    costW = 0
    remaining = set(c for c in e.profile)
    endow = {i : 1.0 * e.budget / len(e.voters) for i in e.voters}
    ratio = {c : -1.0 for c in e.profile}
    while True:
        next_candidate = None
        lowest_ratio = float("inf")
        remaining_sorted = sorted(remaining, key=lambda c: ratio[c])
        best_util = 0
        for c in remaining_sorted:
            if ratio[c] >= lowest_ratio:
                break
            if costW + c.cost <= e.budget:
                supporters_sorted = sorted([i for i in e.profile[c]], key=lambda i: endow[i] / e.profile[c][i])
                util = sum(e.profile[c].values())
                money_used = 0
                last_rho = 0
                new_ratio = float("inf")
                for i in supporters_sorted:
                    alpha = min(1.0, (money_used + util * (endow[i] / e.profile[c][i])) / c.cost)
                    if round(alpha, 5) > 0:
                        rho = ((alpha * c.cost) - money_used) / (alpha * util)
                        if rho < last_rho:
                            break
                        if rho / (alpha) < new_ratio :
                            new_ratio = rho / (alpha)
                            new_rho = rho
                    util -= e.profile[c][i]
                    money_used += endow[i]
                    last_rho = endow[i] / e.profile[c][i]
                ratio[c] = new_ratio
                if ratio[c] < lowest_ratio:
                    lowest_ratio = ratio[c]
                    lowest_rho = new_rho
                    next_candidate = c
                    best_util = sum([e.profile[c][i] for i in e.profile[c]])
                elif ratio[c] == lowest_ratio:
                    util = sum([e.profile[c][i] for i in e.profile[c]])
                    if util > best_util:
                        next_candidate = c
                        best_util = util
        if next_candidate is None:
            break
        else:
            W.add(next_candidate)
            costW += next_candidate.cost
            remaining.remove(next_candidate)
            if sum(endow[i] for i in e.profile[next_candidate]) + 0.001 >= next_candidate.cost:
                supporters_sorted = sorted([i for i in e.profile[next_candidate]], key=lambda i: endow[i] / e.profile[next_candidate][i])
                price = next_candidate.cost
                util = sum(e.profile[next_candidate].values())
                for i in supporters_sorted:
                    if endow[i] * util >= price * e.profile[next_candidate][i]:
                        break
                    price -= endow[i]
                    util -= e.profile[next_candidate][i]
                if price > 0.001:
                    rho = price / util
                else:
                    rho = endow[supporters_sorted[-1]] / e.profile[next_candidate][supporters_sorted[-1]]

                for i in e.profile[next_candidate]:
                    endow[i] -= min(endow[i], rho * e.profile[next_candidate][i])
            else:
                for i in e.profile[next_candidate]:
                    endow[i] = 0
            if real_budget: #optimization for 'increase-budget' completions
                if costW > real_budget:
                    return None
    return W


def test_budget_overspending1():
    e = Election()
    e.budget = 100
    for i in range(10):
        e.voters.add(i)

    c0 = Candidate('0', 30)
    c1 = Candidate('1', 100)

    e.profile[c0] = {6: 1, 7: 1, 8: 1, 9: 1}
    e.profile[c1] = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}

    e.binary_to_cost_utilities()

    assert(list(bounded_overspending(e))[0].id == '0')

def test_budget_overspending2():
    e = Election()
    e.budget = 100
    for i in range(10):
        e.voters.add(i)

    c0 = Candidate('0', 30)
    c1 = Candidate('1', 100)

    e.profile[c0] = {7: 1, 8: 1, 9: 1}
    e.profile[c1] = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
    e.binary_to_cost_utilities()

    assert(list(bounded_overspending(e))[0].id == '1')

def test_budget_overspending3():
    e = Election()
    e.budget = 2
    for i in range(6):
        e.voters.add(i)

    c0 = Candidate('0', 1)
    c1 = Candidate('1', 1)
    c2 = Candidate('2', 1)

    e.profile[c0] = {0: 1, 1: 1, 2: 1, 3: 1, 4: 100, 5: 100}
    e.profile[c1] = {0: 1, 1: 1, 2: 100, 3: 100, 4: 1, 5: 1}
    e.profile[c2] = {0: 10, 1: 10, 2: 2, 3: 2, 4: 2, 5: 2}

    res = sorted([c.id for c in bounded_overspending(e)])
    assert(res == ['0', '1'])

def test_budget_overspending_with_parties(parties, k, expected):
    e = Election()
    e.budget = k

    v_id = 0
    for p in parties:
        for i in range(p):
            e.voters.add(v_id)
            v_id += 1

    candidates = []
    cand_id = 0
    for p in parties:
        for i in range(k):
            candidates.append(Candidate(str(cand_id), 1))
            e.profile[candidates[cand_id]] = {}
            cand_id += 1

    v_id = 0
    for p_id, p in enumerate(parties):
        for i in range(p):
            cand_id = 0
            for p2_id, p2 in enumerate(parties):
                for j in range(k):
                    if p_id == p2_id:
                        e.profile[candidates[cand_id]][v_id] = 1
                    cand_id += 1
            v_id += 1


    res = [c.id for c in bounded_overspending(e)]
    distr = [0 for p in parties]

    cand_id = 0
    for p_id, _ in enumerate(parties):
        for i in range(k):
            if candidates[cand_id].id in res:
                distr[p_id] += 1
            cand_id += 1

    assert(distr == expected)



#test_budget_overspending1()
#test_budget_overspending2()
#test_budget_overspending3()
#test_budget_overspending_with_parties([20, 30, 50], 10, [2, 3, 5])

def compute_max_spending(already_spent, bound_on_spending, cost):
    timestamps = [] 
    for spent in already_spent:
        timestamps.append((spent, 'S'))
    for bound in bound_on_spending:
        timestamps.append((bound, 'B'))
    timestamps.sort(key = lambda x: x[0])
    last_timestamp = timestamps[0][0]
    num_active_voters = 0
    remaining_cost = cost
    for v in timestamps:
        stamp = v[0]
        symb = v[1]
        assert(stamp >= last_timestamp)
        if (stamp - last_timestamp) * num_active_voters >= remaining_cost:
            return last_timestamp + (remaining_cost / num_active_voters)
        remaining_cost -= (stamp - last_timestamp) * num_active_voters
        last_timestamp = stamp
        if symb == 'S':
            num_active_voters += 1
        elif symb == 'B':
            num_active_voters -= 1
    return last_timestamp

def compute_max_overspending_exact(voters, cost, expected_rho, simply_afford = False):
    max_overspending = 0
    while True:
        endow = [max(v[0] + max_overspending - v[1], 0) for v in voters]
        supporters_sorted = sorted([i for i in range(len(endow))], key=lambda i: endow[i] / voters[i][2])
        supporters_utils_sum = [0 for i in supporters_sorted]
        utils_sum = 0
        for j in range(len(supporters_utils_sum) - 1, -1, -1):
            i = supporters_sorted[j]
            utils_sum += voters[i][2]
            supporters_utils_sum[j] = utils_sum
        new_rho = float("inf")
        for j in range(len(supporters_sorted)):
            i = supporters_sorted[j]
            if (endow[i] / voters[i][2]) * supporters_utils_sum[j] >= cost:
                new_rho = cost / supporters_utils_sum[j]
                break
                
        if simply_afford and new_rho < float("inf"):
            return max_overspending
        elif new_rho <= expected_rho * 1.00001:
            return max_overspending
        else:
            max_overspending += 1


def bounded_overspending_single_simple(e : Election, compute_minimal_overspenfing = "rescale") -> (set[Candidate]):
    W = set()

    costW = 0
    remaining = set(c for c in e.profile)
    endow_ini = 1.0 * e.budget / len(e.voters)
    endow = {i : endow_ini for i in e.voters}
    overspendings = {i : 0 for i in e.voters}
    ratio = {c : -1.0 for c in e.profile}
    while True:
        next_candidate = None
        lowest_ratio = float("inf")

        exceed_cost = [c for c in remaining if costW + c.cost > e.budget]
        for c in exceed_cost:
            remaining.remove(c)
        remaining_sorted = sorted(remaining, key=lambda c: ratio[c])

        for c in remaining_sorted:
            if ratio[c] >= lowest_ratio:
                continue
            supporters_sorted = sorted([i for i in e.profile[c]], key=lambda i: endow[i] / e.profile[c][i])
            util = sum(e.profile[c].values())
            money_used = 0
            last_rho = 0
            new_ratio = float("inf")
            for i in supporters_sorted:
                alpha = min(1.0, (money_used + util * (endow[i] / e.profile[c][i])) / c.cost)
                if round(alpha, 5) > 0:
                    rho = ((alpha * c.cost) - money_used) / (alpha * util)
                    if rho < last_rho:
                        break
                    if rho / alpha < new_ratio : 
                        new_ratio = rho / alpha
                        new_rho = rho
                        new_alpha = alpha
                    if alpha == 1:
                        break
                util -= e.profile[c][i]
                money_used += endow[i]
                last_rho = endow[i] / e.profile[c][i]
            ratio[c] = new_ratio
            if ratio[c] < lowest_ratio:
                lowest_ratio = ratio[c]
                best_rho = new_rho
                best_alpha = new_alpha
                next_candidate = c
        if next_candidate is None:
            break
        else:
            if compute_minimal_overspenfing == "opt":
                bound_on_spending = []
                already_spent = []
                for i in e.profile[next_candidate].keys():
                    bound_on_spending.append((endow_ini - endow[i]) + overspendings[i] + (best_rho * e.profile[next_candidate][i])) 
                    already_spent.append((endow_ini - endow[i]) + overspendings[i])
                max_spending = compute_max_spending(already_spent = already_spent,
                                                    bound_on_spending = bound_on_spending,
                                                    cost = next_candidate.cost)
                max_overspending = max(max_spending - endow_ini, 0)
            elif compute_minimal_overspenfing == "rescale":
                overspendings2 = {}
                for i in e.profile[next_candidate].keys():
                    payment = min(endow[i], best_alpha * best_rho * e.profile[next_candidate][i]) / best_alpha
                    overspendings2[i] = max(0, payment - endow[i])
                max_overspending = max(overspendings2.values())
            elif compute_minimal_overspenfing == "opt_exact" or compute_minimal_overspenfing == "opt_full_exact":
                max_overspending = compute_max_overspending_exact(voters = [(endow[i], overspendings[i], e.profile[next_candidate][i]) for i in e.profile[next_candidate].keys()],
                                                                  cost = next_candidate.cost,
                                                                  expected_rho = best_rho)
            elif compute_minimal_overspenfing == "opt_exact_afford":
                max_overspending = compute_max_overspending_exact(voters = [(endow[i], overspendings[i], e.profile[next_candidate][i]) for i in e.profile[next_candidate].keys()],
                                                                  cost = next_candidate.cost,
                                                                  expected_rho = best_rho,
                                                                  simply_afford= True)
            else:
                raise RuntimeError("Bos plus strategy for compute_minimal_overspenfing not recognised.")


            max_overspending += 0.0001

            endow_ = {i: endow[i] + max(0, max_overspending - overspendings[i]) for i in e.voters}
            best_rho_ = float("inf")
            next_candidate_ = None
            rho_map = {c: float('inf') for c in e.profile.keys()}
            if compute_minimal_overspenfing == "opt_full_exact":
                for c_ in remaining:
                    supporters_sorted_ = sorted([i for i in e.profile[c_]], key=lambda i: endow_[i] / e.profile[c_][i])
                    supporters_utils_sum_ = [0 for i in supporters_sorted_]
                    utils_sum = 0
                    for j in range(len(supporters_utils_sum_) - 1, -1, -1):
                        i = supporters_sorted_[j]
                        utils_sum += e.profile[c_][i]
                        supporters_utils_sum_[j] = utils_sum
                    new_rho = -1
                    for j in range(len(supporters_sorted_)):
                        i = supporters_sorted_[j]
                        if (endow_[i] / e.profile[c_][i]) * supporters_utils_sum_[j] >= c_.cost:
                            new_rho = c_.cost / supporters_utils_sum_[j]
                            break
                    if new_rho > 0 and new_rho < best_rho_:
                        best_rho_ = new_rho
                        next_candidate_ = c_
            else:
                for c_ in remaining:
                    if round(sum(endow_[i] for i in e.profile[c_].keys() if e.profile[c_][i]>0),5) >= c_.cost:
                        supporters_sorted = sorted([i for i in e.profile[c_]], key=lambda i: endow_[i] / e.profile[c_][i])
                        price = c_.cost
                        util = sum(e.profile[c_].values())
                        for i in supporters_sorted:
                            if endow_[i] * util >= price * e.profile[c_][i]:
                                break
                            price -= endow_[i]
                            util -= e.profile[c_][i]
                        if price > 0.001:
                            rho_ = price / util
                        else:
                            rho_ = endow[supporters_sorted[-1]] / e.profile[c_][supporters_sorted[-1]]
                        if round(float(rho_), 8) < round(float(best_rho_), 8):
                            next_candidate_ = c_
                            best_rho_ = rho_
                        rho_map[c_] = rho_
            W.add(next_candidate_)

            costW += next_candidate_.cost
            remaining.remove(next_candidate_)
            all_payments = 0
            if compute_minimal_overspenfing == "opt_full_exact":
                for i in e.profile[next_candidate_]:
                    if endow_[i] >= best_rho_ * e.profile[next_candidate_][i]:
                        payment = best_rho_ * e.profile[next_candidate_][i]
                    else:
                        payment = 0
                    all_payments += payment
                    endow_[i] -= payment
                    if endow_[i] >= max_overspending:
                        endow[i] = endow_[i] - max_overspending
                    else:
                        endow[i] = 0
                        overspendings[i] = max(overspendings[i], max_overspending - endow_[i])
            else:
                for i in e.profile[next_candidate_]:
                    payment = min(endow_[i], best_rho_ * e.profile[next_candidate_][i])
                    all_payments += payment
                    endow_[i] -= payment
                    if endow_[i] >= max_overspending:
                        endow[i] = endow_[i] - max_overspending
                    else:
                        endow[i] = 0
                        overspendings[i] = max(overspendings[i], max_overspending - endow_[i])
    return W
    