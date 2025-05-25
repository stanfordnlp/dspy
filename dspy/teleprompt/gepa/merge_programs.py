def find_common_ancestor_pair(rng, parent_list, program_indexes, max_attempts=10):
    def get_ancestors(node):
        ancestors = []
        current = parent_list[node]
        while current is not None:
            ancestors.append(current)
            current = parent_list[current]
        return ancestors

    def is_ancestor(ancestor, node):
        current = parent_list[node]
        while current is not None:
            if current == ancestor:
                return True
            current = parent_list[current]
        return False

    for _ in range(max_attempts):
        if len(program_indexes) < 2:
            return None
        i, j = rng.sample(program_indexes, 2)
        ancestors_i = get_ancestors(i)
        ancestors_j = get_ancestors(j)
        
        common_ancestors = []
        for an1, an2 in zip(ancestors_i[::-1], ancestors_j[::-1]):
            if an1 == an2:
                common_ancestors.append(an1)
            else:
                break
        if common_ancestors:
            if not is_ancestor(i, j) and not is_ancestor(j, i):
                return (i, j, common_ancestors[-1])
    
    return None

def sample_and_attempt_merge_programs_by_common_predictors(gepa_state, agg_scores, rng, merge_candidates, max_attempts=10):
    if len(merge_candidates) < 2:
        return (False, None, None, None, None)
    if len(gepa_state.parent_program_for_candidate) < 3:
        return (False, None, None, None, None)

    for _ in range(max_attempts):
        ids_to_merge = find_common_ancestor_pair(rng, gepa_state.parent_program_for_candidate, list(merge_candidates), max_attempts=10)
        if ids_to_merge is None:
            continue
        id1, id2, ancestor = ids_to_merge

        if agg_scores[ancestor] > agg_scores[id1] or agg_scores[ancestor] > agg_scores[id2]:
            continue

        # Now we have a common ancestor, which is outperformed by both its descendants

        found_predictors = []
        for pred_idx, (pred_anc, pred_id1, pred_id2) in enumerate(zip(
            gepa_state.program_candidates[ancestor].named_predictors(),
            gepa_state.program_candidates[id1].named_predictors(),
            gepa_state.program_candidates[id2].named_predictors()
        )):
            if (
                (pred_anc[1].signature.instructions == pred_id1[1].signature.instructions) or 
                (pred_anc[1].signature.instructions == pred_id2[1].signature.instructions)
            ) and (
                pred_id1[1].signature.instructions != pred_id2[1].signature.instructions
            ):
                # We have a predictor that is the same as one of its ancestors, so we can update it with the other
                same_as_ancestor_id = (1 if pred_anc[1].signature.instructions == pred_id1[1].signature.instructions else 2)
                found_predictors.append((pred_idx, same_as_ancestor_id))
        
        if len(found_predictors) == 0:
            continue

        selected_predictor_to_update = rng.choice(found_predictors)

        predictor_idx, same_as_ancestor_id = selected_predictor_to_update

        prog_to_update = id1 if same_as_ancestor_id == 1 else id2
        prog_to_get_instruction_from = id2 if same_as_ancestor_id == 1 else id1

        curr_prog_lm = gepa_state.program_candidates[prog_to_update].get_lm()
        new_program = gepa_state.program_candidates[prog_to_update].deepcopy()
        new_program.set_lm(curr_prog_lm)
        new_program.named_predictors()[predictor_idx][1].signature = gepa_state.program_candidates[prog_to_get_instruction_from].named_predictors()[predictor_idx][1].signature
        return (True, new_program, id1, id2, ancestor)
    
    return (False, None, None, None, None)
