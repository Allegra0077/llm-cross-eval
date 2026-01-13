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
    HUMAN_FIRST_TURN = True
    HUMAN_ONLY = True
    MODEL_ONLY = False
    TEST_HUMAN_MESSAGE = True
    TEST_MODEL_MESSAGE = False

    assert not (HUMAN_ONLY and MODEL_ONLY), "Cannot set both HUMAN_ONLY and MODEL_ONLY to True"
    assert not (TEST_HUMAN_MESSAGE and TEST_MODEL_MESSAGE), "Cannot set both TEST_HUMAN_MESSAGE and TEST_MODEL_MESSAGE to True"
    assert not (HUMAN_FIRST_TURN and (TEST_MODEL_MESSAGE or MODEL_ONLY)), "Cannot set HUMAN_FIRST_TURN when TEST_MODEL_MESSAGE or MODEL_ONLY is True"

    print(f"Setting HUMAN_FIRST_TURN to {HUMAN_FIRST_TURN}")
    print(f"Setting HUMAN_ONLY to {HUMAN_ONLY}")
    print(f"Setting MODEL_ONLY to {MODEL_ONLY}")
    print(f"Setting TEST_HUMAN_MESSAGE to {TEST_HUMAN_MESSAGE}")
    print(f"Setting TEST_MODEL_MESSAGE to {TEST_MODEL_MESSAGE}")

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
    for i in range(num_conversations):

        # Skip overly long conversations due to GPU constraints
        if sum([len(message["content"]) for message in conversations["conversation"][i]]) > 100000:
            continue

        conv_results = dict()
        conv_results["conversation_id"] = conversations["conversation_id"][i]
        
        if TEST_HUMAN_MESSAGE:
            # Last answer is always by model which we do not care about
            conversation = conversations["conversation"][i][:-1]

            final_human_input = conversation[-1]
            final_formatted_message = "<|im_start|>user\n" + final_human_input["content"] + "<|im_end|>\n"

        elif TEST_MODEL_MESSAGE:
            # Need full conversation
            conversation = conversations["conversation"][i]

            final_model_input = conversation[-1]
            final_formatted_message = "<|im_start|>assistant\n" + final_model_input["content"] + "<|im_end|>\n"

        # Don't care about padding, attn_mask since no batch processing
        output_ids = tokenizer(final_formatted_message, return_tensors="pt")

        for num_turns in range(1, MAX_TURNS + 1):

            # Use last num_turns turns of conversation as conditioning
            # conversation_subset has structure [M, H] * num_turns, at MAX_TURNS it flips to full conversation [H, M] * MAX_TURNS
            conversation_subset = conversation[-num_turns * 2:]

            if HUMAN_ONLY or MODEL_ONLY:
                role = "user" if HUMAN_ONLY else "assistant"
                # Filter to only human or model messages
                conversation_subset = [msg for msg in conversation_subset if msg["role"] == role]

            if HUMAN_FIRST_TURN and conversation_subset[0]["role"] == "assistant":
                # Remove the first message by model
                conversation_subset = conversation_subset[1:]
                assert conversation_subset[0]["role"] == "user"

            if len(conversation_subset) > 1:
                input_ids = tokenizer.apply_chat_template(
                    conversation_subset[:-1],
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                )

                complete_sequence = torch.cat((input_ids["input_ids"], output_ids["input_ids"]), dim = -1).to(device)

            else:
                system_instruction = {'content': 'You are a helpful assistant.', 'role': 'system'}
                input_ids = tokenizer.apply_chat_template(
                    [system_instruction],
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
            probs = torch.nn.functional.softmax(output_logits, dim=-1)
            log_probs = torch.log(probs)

            target_tokens = output_ids["input_ids"].to(device)

            assert target_tokens.shape[1] == log_probs.shape[1]

            cum_logprob = 0.0
            for k in range(target_tokens.shape[1]):
                token_id = target_tokens[0, k].item()
                logprob = log_probs[0, k, token_id].item()
                cum_logprob += logprob

            entropy = -torch.sum(probs * torch.log(probs), dim=-1).sum().item()
            
            conv_results[f'logprob_turns_{num_turns}'] = cum_logprob
            conv_results[f'entropy_turns_{num_turns}'] = entropy

        results.append(conv_results)
        
    # Save results
    print(f"Recorded {len(results)} results")
    input_setting = "HO" if HUMAN_ONLY else "MO" if MODEL_ONLY else "all"
    output_setting = "TH" if TEST_HUMAN_MESSAGE else "TM"
    output_path = f"results/exp_multi_{input_setting}_{output_setting}_{MAX_TURNS}_turn.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    end = time.time()
    print(f"Experiment completed in {end - start:.2f} seconds.")

if __name__ == "__main__":
    main()