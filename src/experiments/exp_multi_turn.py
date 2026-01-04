import os
import json
import time

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():

    print("Starting multi-turn experiment")
    print("=" * 20)
    start = time.time()
    MAX_TURNS = 10
    # Load dataset
    lmsys = load_dataset("lmsys/lmsys-chat-1m", split="train")
    turns = list(lmsys["turn"])

    # Filter dataset 
    valid_indices = [i for i in range(len(turns)) if turns[i] == MAX_TURNS]
    conversations = lmsys[valid_indices[:1000]]  # For testing, limit to 1000 conversations
    num_conversations = len(conversations["conversation"])

    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", device_map="auto")

    results = []

    for i, conversation in enumerate(num_conversations):

        # Skip overly long conversations due to GPU constraints
        if sum([len(message["content"]) for message in conversations["conversation"][i]]) > 100000:
            continue

        conv_results = dict()
        conversation_id = ""
        conv_results["conversation_id"] = conversation_id
        # Last answer is always by model which we do not care about
        conversation = conversation[:-1]

        for j, num_turns in enumerate(range(1, MAX_TURNS + 1)):
            
            conversation_subset = conversation[-num_turns * 2:]

            input_ids = tokenizer.apply_chat_template(
                conversation_subset[:-1],
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )

            output_ids = tokenizer.apply_chat_template(
            [conversation_subset[-1]],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
            )

            complete_sequence = torch.cat((input_ids["input_ids"], output_ids["input_ids"]), dim = -1).to(device)
            # Get model logits
            with torch.no_grad():
                logits = model(complete_sequence).logits

            # Compute logprobs for output tokens
            output_logits = logits[:, -output_ids["input_ids"].shape[1]-1:-1, :]
            log_probs = torch.nn.functional.log_softmax(output_logits, dim=-1)

            target_tokens = output_ids["input_ids"].to(device)

            assert target_tokens.shape[1] == log_probs.shape[1]

            cum_logprob = 0.0
            for k in range(target_tokens.shape[1]):
                token_id = target_tokens[0, k].item()
                logprob = log_probs[0, k, token_id].item()
                cum_logprob += logprob
            
            conv_results[f'logprob_turns_{num_turns}'] = cum_logprob
        results.append(conv_results)

    # Save results
    output_path = f"results/exp_multi_{MAX_TURNS}_turn_logprobs.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    end = time.time()
    print(f"Experiment completed in {end - start:.2f} seconds.")