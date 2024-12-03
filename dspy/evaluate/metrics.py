# TODO: This should move internally. Same for passage_match. dspy.metrics.answer_exact_match, dspy.metrics.answer_passage_match

import dsp


def answer_exact_match(example, pred, trace=None, frac=1.0):
    assert(type(example.answer) is str or type(example.answer) is list)

    if type(example.answer) is str:
        return dsp.answer_match(pred.answer, [example.answer], frac=frac)
    else: # type(example.answer) is list
        return dsp.answer_match(pred.answer, example.answer, frac=frac)

answer_exact_match_str = dsp.answer_match

def answer_passage_match(example, pred, trace=None):
    assert(type(example.answer) is str or type(example.answer) is list)

    if type(example.answer) is str:
        return dsp.passage_match(pred.context, [example.answer])
    else: # type(example.answer) is list
        return dsp.passage_match(pred.context, example.answer)


# Function to compute spBLEU score
def calculate_spBLEU(pred_answer, reference_answer, tokenize="flores101"):
    """
    Calculate the spBLEU score between the predicted and reference answers.

    :param pred_answer: The predicted answer.
    :param reference_answer: The reference answer.
    :param tokenize: The tokenizer to use for tokenization.
    :return: The spBLEU score.
    """
    # sacrebleu expects a list of references and a single hypothesis
    # Here reference_answer is converted into a list for compatibility
    # List of potential tokenizers:
    #  - "13a" for moses
    #  - "flores101"
    #  - "char" for character-based
    try:
        import sacrebleu
    except ImportError:
        print(
            "The 'sacrebleu' package is not installed. You can install it using the command: pip install sacrebleu"
        )

    reference_list = [reference_answer]

    bleu = sacrebleu.sentence_bleu(pred_answer, reference_list, tokenize=tokenize)

    return bleu.score


# Function to validate answers using spBLEU score
def validate_answer_with_spBLEU(example, pred, trace=None, threshold=50):
    """
    Validate the answer by calculating the spBLEU score between predicted and
    example answers.

    :param example: An object containing the reference answer.
    :param pred: An object containing the predicted answer.
    :param trace: Optional trace parameter (not used in this function).
    :param threshold: The minimum spBLEU score required to consider the
    prediction correct.
    :return: A dictionary with the spBLEU score and a correctness boolean.
    """
    reference_answer = example.answer
    predicted_answer = pred.answer

    # Calculate the spBLEU score
    spBLEU_score = calculate_spBLEU(predicted_answer, reference_answer)

    # Consider the answer correct if the spBLEU score is above the threshold
    is_correct = spBLEU_score >= threshold

    return is_correct


# Function to calculate cosine similarity
def cosine_similarity(a, b):
    try:
        import torch.nn.functional as F
    except ImportError:
        raise ImportError(
            "torch is required to calculate cosine similarity. "
            "Please install it using 'pip install torch'."
        )
    # Compute cosine similarity along the last dimension
    return F.cosine_similarity(a, b, dim=-1).mean().item()


# Function to compute BERT-like score using the transformer model
def calculate_bert_score_with_transformers(
    pred_answer, reference_answer, tokenizer, model
):
    try:
        import torch
    except ImportError:
        print(
            "The 'torch' package is not installed. You can install it using the command: pip install torch"
        )
    # Encode both the predicted and reference answers
    pred_input = tokenizer(pred_answer, return_tensors="pt")
    ref_input = tokenizer(reference_answer, return_tensors="pt")

    # Get the embeddings from the last hidden state of [CLS] token
    with torch.no_grad():
        pred_embedding = model(**pred_input).last_hidden_state[:, 0, :]
        ref_embedding = model(**ref_input).last_hidden_state[:, 0, :]

    # Calculate the cosine similarity between the two embeddings
    similarity = cosine_similarity(pred_embedding, ref_embedding)
    return similarity


# Function to validate answers using cosine similarity score
def validate_answer_with_cosine_similarity(
    example, pred, trace=None, threshold=0.85, tokenizer=None, model=None
):
    """
    Validate the answer by calculating the cosine similarity between predicted
    and example answers.

    :param example: An object containing the reference answer.
    :param pred: An object containing the predicted answer.
    :param threshold: The minimum cosine similarity score required to consider
    the prediction correct.
    :param tokenizer: The tokenizer to use for tokenization.
    :param model: The transformer model to use for embedding.
    :return: A boolean indicating correctness.
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print(
            "The 'transformers' package is not installed. You can install it using the command: pip install transformers"
        )
    reference_answer = example.answer
    predicted_answer = pred.answer

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased", clean_up_tokenization_spaces=False
        )

    if model is None:
        model = AutoModel.from_pretrained("bert-base-uncased")

    # Calculate the cosine similarity
    cosine_sim = calculate_bert_score_with_transformers(
        predicted_answer, reference_answer, tokenizer, model
    )

    # Consider the answer correct if the similarity score is above threshold
    is_correct = cosine_sim >= threshold
    return is_correct
