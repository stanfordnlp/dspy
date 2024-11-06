import torch
import transformers
import tiktoken
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

# Goal is to take a sequence of tokens, and return the topk tokens at position i
# Such that the tokens maximize the negative gradient of the loss function across the entire sequence

# Initial sequence
sequence = "What is the capital of France?"

def setup_model_and_tokenizer():
    # Initialize model and tokenizer
    model_name = "lmsys/vicuna-7b-v1.5"
    chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ system_message }}{% endif %}{% if message['role'] == 'user' %}{{ ' USER: ' + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ ' ASSISTANT: ' + message['content'].strip() + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' ASSISTANT:' }}{% endif %}"
    tokenizer = AutoTokenizer.from_pretrained(model_name, chat_template=chat_template)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def get_gradient_tokens(model, tokenizer, sequence, position, top_k=5):
    # Tokenize the input sequence
    messages = [{"role": "user", "content": sequence}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    input_ids = inputs.to(model.device)
    
    # Create a copy of input_ids that requires gradient
    input_embeds = model.get_input_embeddings()(input_ids)
    input_embeds.requires_grad_(True)
    input_embeds.retain_grad()
    
    # Forward pass
    outputs = model(inputs_embeds=input_embeds)
    logits = outputs.logits
    
    # Calculate loss (we'll use the next token prediction loss)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                          shift_labels.view(-1))
    
    # Calculate gradients
    loss.backward()
    
    # Get gradients at the specified position
    position_gradients = input_embeds.grad[0, position]
    
    # Get the token embeddings
    token_embeddings = model.get_input_embeddings().weight
    
    # Calculate similarity between position gradients and all token embeddings
    similarities = torch.matmul(token_embeddings, position_gradients)
    
    # Get top-k tokens that maximize the negative gradient
    top_values, top_indices = torch.topk(-similarities, k=top_k)
    
    # Convert token ids to tokens
    top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices]

    loss.zero_grad()
    
    return top_tokens, list(map(lambda x: -x, top_values.tolist()))

def main():
    print(f"Analyzing sequence: {sequence}")
    
    # Setup
    model, tokenizer = setup_model_and_tokenizer()
    
    print("Converting to chatml format")
    messages = [{"role": "user", "content": sequence}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    print(inputs)
    # Get tokenized sequence length
    tokens = tokenizer.encode(inputs, return_tensors="pt")[0]
    seq_length = len(tokens)
    
    print(f"\nTokenized sequence length: {seq_length}")
    print("Analyzing gradient-based importance for each position...")
    
    # Analyze each position
    for i in range(seq_length):
        tokens_at_pos = tokenizer.decode([tokens[i]])
        print(f"\nPosition {i} (current token: '{tokens_at_pos}')")
        
        top_tokens, importance_scores = get_gradient_tokens(model, tokenizer, sequence, i)
        
        print("Top replacement tokens by gradient magnitude:")
        for token, score in zip(top_tokens, importance_scores):
            print(f"  Token: '{token}', Score: {score:.4f}")

if __name__ == "__main__":
    main()

