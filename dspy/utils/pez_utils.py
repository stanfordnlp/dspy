from statistics import mean
import copy

import torch

from sentence_transformers.util import (semantic_search,
                                        dot_score,
                                        normalize_embeddings)


def nn_project(curr_embeds, embedding_layer, print_hits=False):
    with torch.no_grad():
        bsz, seq_len, emb_dim = curr_embeds.shape

        # Using the sentence transformers semantic search which is
        # a dot product exact kNN search between a set of
        # query vectors and a corpus of vectors
        curr_embeds = curr_embeds.reshape((-1, emb_dim))
        curr_embeds = normalize_embeddings(curr_embeds)  # queries

        embedding_matrix = embedding_layer.weight
        embedding_matrix = normalize_embeddings(embedding_matrix)

        hits = semantic_search(curr_embeds, embedding_matrix,
                               query_chunk_size=curr_embeds.shape[0],
                               top_k=1,
                               score_function=dot_score)

        if print_hits:
            all_hits = []
            for hit in hits:
                all_hits.append(hit[0]["score"])
            print(f"mean hits:{mean(all_hits)}")

        nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=curr_embeds.device)
        nn_indices = nn_indices.reshape((bsz, seq_len))

        projected_embeds = embedding_layer(nn_indices)

    return projected_embeds, nn_indices


def decode_ids(input_ids, tokenizer, by_token=False):
    input_ids = input_ids.detach().cpu().numpy()

    texts = []

    if by_token:
        for input_ids_i in input_ids:
            curr_text = []
            for tmp in input_ids_i:
                curr_text.append(tokenizer.decode([tmp]))

            texts.append('|'.join(curr_text))
    else:
        for input_ids_i in input_ids:
            texts.append(tokenizer.decode(input_ids_i))

    return texts


def get_target_feature(model, preprocess, tokenizer_funct, device, target_images=None, target_prompts=None):
    if target_images is not None:
        with torch.no_grad():
            curr_images = [preprocess(i).unsqueeze(0) for i in target_images]
            curr_images = torch.concatenate(curr_images).to(device)
            all_target_features = model.encode_image(curr_images)
    else:
        texts = tokenizer_funct(target_prompts).to(device)
        all_target_features = model.encode_text(texts)

    return all_target_features


def initialize_prompt(tokenizer, token_embedding, args, device):
    prompt_len = args.prompt_len

    # randomly optimize prompt embeddings
    prompt_ids = torch.randint(len(tokenizer.encoder), (args.prompt_bs, prompt_len)).to(device)
    prompt_embeds = token_embedding(prompt_ids).detach()
    prompt_embeds.requires_grad = True

    # initialize the template
    template_text = "{}"
    padded_template_text = template_text.format(" ".join(["<start_of_text>"] * prompt_len))
    dummy_ids = tokenizer.encode(padded_template_text)

    # -1 for optimized tokens
    dummy_ids = [i if i != 49406 else -1 for i in dummy_ids]
    dummy_ids = [49406] + dummy_ids + [49407]
    dummy_ids += [0] * (77 - len(dummy_ids))
    dummy_ids = torch.tensor([dummy_ids] * args.prompt_bs).to(device)

    # for getting dummy embeds; -1 won't work for token_embedding
    tmp_dummy_ids = copy.deepcopy(dummy_ids)
    tmp_dummy_ids[tmp_dummy_ids == -1] = 0
    dummy_embeds = token_embedding(tmp_dummy_ids).detach()
    dummy_embeds.requires_grad = False

    return prompt_embeds, dummy_embeds, dummy_ids


def optimize_prompt_loop(model, tokenizer, token_embedding, all_target_features, args, device):
    opt_iters = args.iter
    lr = args.lr
    weight_decay = args.weight_decay
    print_step = args.print_step
    batch_size = args.batch_size
    print_new_best = getattr(args, 'print_new_best', False)

    # initialize prompt
    prompt_embeds, dummy_embeds, dummy_ids = initialize_prompt(tokenizer, token_embedding, args, device)
    p_bs, p_len, p_dim = prompt_embeds.shape

    # get optimizer
    input_optimizer = torch.optim.AdamW([prompt_embeds], lr=lr, weight_decay=weight_decay)

    best_sim = -1000 * args.loss_weight
    best_text = ""

    for step in range(opt_iters):
        # randomly sample sample images and get features
        if batch_size is None:
            target_features = all_target_features
        else:
            curr_indx = torch.randperm(len(all_target_features))
            target_features = all_target_features[curr_indx][0:batch_size]

        universal_target_features = all_target_features

        # forward projection
        projected_embeds, nn_indices = nn_project(prompt_embeds, token_embedding, print_hits=False)

        # get cosine similarity score with all target features
        with torch.no_grad():
            # padded_embeds = copy.deepcopy(dummy_embeds)
            padded_embeds = dummy_embeds.detach().clone()
            padded_embeds[dummy_ids == -1] = projected_embeds.reshape(-1, p_dim)
            logits_per_image, _ = model.forward_text_embedding(padded_embeds, dummy_ids, universal_target_features)
            scores_per_prompt = logits_per_image.mean(dim=0)
            universal_cosim_score = scores_per_prompt.max().item()
            best_indx = scores_per_prompt.argmax().item()

        # tmp_embeds = copy.deepcopy(prompt_embeds)
        tmp_embeds = prompt_embeds.detach().clone()
        tmp_embeds.data = projected_embeds.data
        tmp_embeds.requires_grad = True

        # padding
        # padded_embeds = copy.deepcopy(dummy_embeds)
        padded_embeds = dummy_embeds.detach().clone()
        padded_embeds[dummy_ids == -1] = tmp_embeds.reshape(-1, p_dim)

        logits_per_image, _ = model.forward_text_embedding(padded_embeds, dummy_ids, target_features)
        cosim_scores = logits_per_image
        loss = 1 - cosim_scores.mean()
        loss = loss * args.loss_weight

        prompt_embeds.grad, = torch.autograd.grad(loss, [tmp_embeds])

        input_optimizer.step()
        input_optimizer.zero_grad()

        curr_lr = input_optimizer.param_groups[0]["lr"]
        cosim_scores = cosim_scores.mean().item()

        decoded_text = decode_ids(nn_indices, tokenizer)[best_indx]
        if print_step is not None and (step % print_step == 0 or step == opt_iters - 1):
            per_step_message = f"step: {step}, lr: {curr_lr}"
            if not print_new_best:
                per_step_message = f"\n{per_step_message}, cosim: {universal_cosim_score:.3f}, text: {decoded_text}"
            print(per_step_message)

        if best_sim * args.loss_weight < universal_cosim_score * args.loss_weight:
            best_sim = universal_cosim_score
            best_text = decoded_text
            if print_new_best:
                print(f"new best cosine sim: {best_sim}")
                print(f"new best prompt: {best_text}")

    if print_step is not None:
        print()
        print(f"best cosine sim: {best_sim}")
        print(f"best prompt: {best_text}")

    return best_text


def optimize_prompt(model, preprocess, args, device, target_images=None, target_prompts=None):
    token_embedding = model.token_embedding
    tokenizer = open_clip.tokenizer._tokenizer
    tokenizer_funct = open_clip.get_tokenizer(args.clip_model)

    # get target features
    all_target_features = get_target_feature(model, preprocess, tokenizer_funct, device, target_images=target_images,
                                             target_prompts=target_prompts)

    # optimize prompt
    learned_prompt = optimize_prompt_loop(model, tokenizer, token_embedding, all_target_features, args, device)

    return learned_prompt
