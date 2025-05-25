from collections import Counter

import math
import math

import numpy as np

def is_dominated(y, programs, program_at_pareto_front_valset):
    y_fronts = [front for front in program_at_pareto_front_valset if y in front]
    for front in y_fronts:
        found_dominator_in_front = False
        for other_prog in front:
            if other_prog in programs:
                found_dominator_in_front = True
                break
        if not found_dominator_in_front:
            return False
    
    return True

def remove_dominated_programs(program_at_pareto_front_valset, scores=None):
    freq = {}
    for front in program_at_pareto_front_valset:
        for p in front:
            freq[p] = freq.get(p, 0) + 1

    dominated = set()
    programs = list(freq.keys())

    if scores is None:
        scores = {p:1 for p in programs}
    
    programs = sorted(programs, key=lambda x: scores[x], reverse=False)

    found_to_remove = True
    while found_to_remove:
        found_to_remove = False
        for y in programs:
            if y in dominated:
                continue
            if is_dominated(y, set(programs).difference({y}).difference(dominated), program_at_pareto_front_valset):
                dominated.add(y)
                found_to_remove = True
                break
    
    dominators = [p for p in programs if p not in dominated]
    for front in program_at_pareto_front_valset:
        assert any(p in front for p in dominators)
    
    new_program_at_pareto_front_valset = [{prog_idx for prog_idx in front if prog_idx in dominators} for front in program_at_pareto_front_valset]
    assert len(new_program_at_pareto_front_valset) == len(program_at_pareto_front_valset)
    for front_old, front_new in zip(program_at_pareto_front_valset, new_program_at_pareto_front_valset):
        assert front_new.issubset(front_old)

    return new_program_at_pareto_front_valset

def calculate_entropy(selection):
    frequency = Counter(selection)
    total = sum(frequency.values())
    probabilities = [count / total for count in frequency.values()]
    return -sum(p * math.log2(p) for p in probabilities)

def minimize_entropy(pareto, sample_size=400, elite_ratio=0.4, iterations=50, seed=0, reversed=False):
    pareto = [list(s) for s in pareto]
    n_dimensions = len(pareto)
    # Initialize probabilities uniformly for each dimension
    probabilities = [np.ones(len(choices))/len(choices) for choices in pareto]

    # Set up a random generator with the given seed
    rng = np.random.default_rng(seed)
    
    for _ in range(iterations):
        # Generate samples based on current probabilities
        samples = []
        for _ in range(sample_size):
            sample = [rng.choice(len(choices), p=probs) 
                     for choices, probs in zip(pareto, probabilities)]
            samples.append([pareto[dim][idx] for dim, idx in enumerate(sample)])
        
        # Calculate entropy for each sample
        entropies = [calculate_entropy(sample) for sample in samples]
        
        # Select elite samples
        elite_size = max(int(sample_size * elite_ratio), 1)
        elite_indices = np.argsort(entropies)
        if reversed:
            elite_indices = elite_indices[::-1]
        elite_indices = elite_indices[:elite_size]
        elite_samples = [samples[i] for i in elite_indices]
        
        # Update probabilities based on elite samples
        for dim in range(n_dimensions):
            for val_idx in range(len(pareto[dim])):
                count = sum(1 for sample in elite_samples 
                          if sample[dim] == pareto[dim][val_idx])
                probabilities[dim][val_idx] = (count + 1e-10) / (elite_size + 1e-10)
    
    # Return best solution found
    final_sample = [pareto[dim][np.argmax(probs)] 
                   for dim, probs in enumerate(probabilities)]
    
    for elem, pareto_set in zip(final_sample, pareto):
        assert elem in pareto_set

    return final_sample