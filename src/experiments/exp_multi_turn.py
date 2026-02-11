import os
import json
import time
import random

from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

random.seed(42)
load_dotenv()
DEBUG_MODE = os.getenv("MODE") == "DEBUG"
INVESTIGATOR_PROMPT = "You are an expert investigator. As you hold a helpful conversation with the user, you should focus on extracting as much detailed information out of them as possible. Your final goal is to predict the user's question as they are asking it."

# seq is Qwen3 Tok of "<think>\n\n</think>\n\n"
def remove_sequence_list(data, seq=[151667, 271, 151668, 271]):
    n = len(seq)
    # Iterate through the list
    for i in range(len(data) - n + 1):
        # Check if the slice matches our target sequence
        if data[i : i + n] == seq:
            # Return list without the sequence
            return data[:i] + data[i + n:]
    return data

def apply_template(conversation, tokenizer, template_type = None, remove_think_tokens = False):
    # First message is system prompt
    assert conversation[0]["role"] == "system"
    assert template_type in [None, "A", "B", "C"], "template_type must be None, 'A' or 'B'"

    if template_type is None:
        # default chat template
        inputs = tokenizer.apply_chat_template(
                        conversation,
                        add_generation_prompt=False,
                        tokenize=True,
                        return_dict=True,
                        enable_thinking=False
                )
    
    if template_type == "A":
        formatted_text = f"<|im_start|>system\n{conversation[0]['content']}<|im_end|>\n"
    elif template_type == "B" or template_type == "C":
        formatted_text = f"SYSTEM: {conversation[0]['content']}\n"

    for msg in conversation[1:]:
        if msg["role"] == "user":
            if template_type == "A" or template_type == "C":
                formatted_text += f"USER: {msg['content']}\n"
            if template_type == "B":
                formatted_text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
        else:
            if template_type == "A" or template_type == "C":
                formatted_text += f"ASSISTANT: {msg['content']}\n"
            if template_type == "B":
                formatted_text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"

    if template_type != None:
        inputs = tokenizer(formatted_text) 
    # Assume BS 1
    inputs["input_ids"] = torch.tensor([remove_sequence_list(inputs["input_ids"])]) if remove_think_tokens else torch.tensor([inputs["input_ids"]])
    return inputs

def main(max_turns):

    print("Starting multi-turn experiment")
    print("=" * 20)
    start = time.time()

    human_first_turn = not MODEL_ONLY # This later enforces turn definition of human always starting a turn, implicitly forces turn 1 for test_model_message to be [H_1, M_1, H_2]

    assert not (HUMAN_ONLY and MODEL_ONLY), "Cannot set both HUMAN_ONLY and MODEL_ONLY to True"
    assert not (TEST_HUMAN_MESSAGE and TEST_MODEL_MESSAGE), "Cannot set both TEST_HUMAN_MESSAGE and TEST_MODEL_MESSAGE to True"
    assert not (human_first_turn and MODEL_ONLY), "Cannot set human_first_turn and MODEL_ONLY to True"

    print(f"Setting human_first_turn to {human_first_turn}")
    print(f"Setting HUMAN_ONLY to {HUMAN_ONLY}")
    print(f"Setting MODEL_ONLY to {MODEL_ONLY}")
    print(f"Setting TEST_HUMAN_MESSAGE to {TEST_HUMAN_MESSAGE}")
    print(f"Setting TEST_MODEL_MESSAGE to {TEST_MODEL_MESSAGE}")

    # Load dataset
    if DEBUG_MODE:
        lmsys = load_dataset("/home/giosue/.cache/huggingface/datasets/lmsys___lmsys-chat-1m/default/0.0.0/200748d9d3cddcc9d782887541057aca0b18c5da", split="train")
    else:
        lmsys = load_dataset("lmsys/lmsys-chat-1m", split="train")
    turns = list(lmsys["turn"])

    # Filter dataset 
    valid_indices = [i for i in range(len(turns)) if turns[i] == max_turns]
    conversations = lmsys[valid_indices[:1000]]  # For testing, limit to 1000 conversations
    num_conversations = len(conversations["conversation"])

    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    if DEBUG_MODE:
        print("In DEBUG MODE")
        model_name = "Qwen/Qwen2.5-0.5B"
    elif NATIVE_MODEL:
        # Most used model with around 0.6 of all conversations, validated in exp_multi_turn_visualize.ipynb
        # Paper: https://arxiv.org/pdf/2309.11998
        model_name = "lmsys/vicuna-13b-v1.5"
    else:
        model_name = "Qwen/Qwen3-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto") if not DEBUG_MODE else None

    debug_conversation = []

    results = []
    for i in range(num_conversations):

        if sum([len(message["content"]) for message in conversations["conversation"][i]]) > 100000:
            # Skip overly long conversations due to GPU constraints
            continue

        if NATIVE_MODEL and conversations["model"][i] != "vicuna-13b":
            # Skip non-native model conversations
            continue

        conv_results = dict()
        conv_results["conversation_id"] = conversations["conversation_id"][i]
        conversation = conversations["conversation"][i]
        
        if LOW_TOKENS:
            content = "".join([message["content"] for message in conversation])
            if len(tokenizer(content)["input_ids"]) > 4000:
                continue

        # Model answer is always last element
        test_index = -2 if TEST_HUMAN_MESSAGE else -1
        test_message = conversation[test_index]

        if TEST_HUMAN_MESSAGE:
            assert test_message["role"] == "user"
            if INJECT_RANDOM_TOPIC:
                content = "That's great to hear! Can you now give me a recipe for cooking a carbonara, true Italian style?"
            else:
                content = test_message["content"]

            if NATIVE_MODEL:
                final_formatted_message = "\nUSER: " + content + "\n"
            elif TEMPLATE_TYPE == "A" or TEMPLATE_TYPE == "C":
                final_formatted_message = f"USER: {content}\n"
            else:
                final_formatted_message = "<|im_start|>user\n" + content + "<|im_end|>\n"

        elif TEST_MODEL_MESSAGE:

            final_model_input = conversation[-1]
            if INJECT_RANDOM_TOPIC:
                content = "To make authentic Carbonara, first brown 100g of sliced guanciale in a pan until crispy, then set the pan aside. Whisk 3 egg yolks and 1 whole egg with 50g of finely grated Pecorino Romano and plenty of freshly cracked black pepper to form a thick paste. Boil 200g of pasta in salted water until al dente, reserving a small cup of the pasta water before draining. Toss the hot pasta into the pan with the guanciale fat, then—with the heat strictly turned off—pour in the egg mixture and a splash of pasta water. Stir vigorously and continuously until the residual heat creates a glossy, creamy emulsion. Serve immediately with an extra dusting of cheese and pepper."
            else:
                content = final_model_input["content"]

            if NATIVE_MODEL:
                final_formatted_message = "ASSISTANT: " + content + "<s>"
            elif TEMPLATE_TYPE == "C":
                final_formatted_message = f"ASSISTANT: {content}\n"
            else:
                final_formatted_message = "<|im_start|>assistant\n" + content + "<|im_end|>\n"

        conversation = conversation[:test_index]

        if MIX_INPUTS:
            assert HUMAN_ONLY, "Mixing inputs only supported for HUMAN_ONLY setting"
            random.shuffle(conversation)

        if EMPTY_ASSISTANT:
            assert not (MODEL_ONLY or HUMAN_ONLY), "Empty assistant setting expects full conversation input"
            for msg in conversation:
                if msg["role"] == "assistant":
                    msg["content"] = ""
            # == conversation = [msg if msg["role"] == "user" else {DIRECT OBJECT} for msg in conversation]

        # Don't care about padding, attn_mask since no batch processing
        output_ids = tokenizer(final_formatted_message, return_tensors="pt")
        conv_results["test_message_length"] = len(output_ids["input_ids"][0])

        for num_turns in range(0, max_turns):

            # Use last num_turns turns of conversation as conditioning
            # conversation_subset has structure [M, H] * num_turns, at MAX_TURNS it flips to full conversation [H, M] * MAX_TURNS
            # Ensure that at max_turns entire conversation is covered.
            # This basically ensure that if num_turns = 1, we use [H1, M1, H2] to predict M2 and so on. Idk I just chose this design
            if num_turns == 0:
                conversation_subset = []
            else:
                conversation_subset = conversation[-num_turns * 2:] if TEST_HUMAN_MESSAGE else conversation[(-num_turns * 2 - 1):]

            if HUMAN_ONLY or MODEL_ONLY:
                role = "user" if HUMAN_ONLY else "assistant"
                # Filter to only human or model messages
                conversation_subset = [msg for msg in conversation_subset if msg["role"] == role]

            if INVESTIGATOR_SETTING:
                assert TEST_HUMAN_MESSAGE, "Investigator prompt is specifically designed to estimate user input."
                if NATIVE_MODEL:
                    raise NotImplementedError("Investigator prompt not implemented for vicuna yet.")
                
                system_instruction = {'content': INVESTIGATOR_PROMPT, 'role': 'system'}
            else:
                system_instruction = {'content': 'You are a helpful assistant.', 'role': 'system'}

            if human_first_turn and len(conversation_subset) > 0 and conversation_subset[0]["role"] == "assistant":
                # This should never be triggered after new definition of conversation_subset
                print("This should not happen.")
                # Remove the first message by model
                conversation_subset = conversation_subset[1:]
                assert conversation_subset[0]["role"] == "user"
            
            if NATIVE_MODEL:
                # Vicuna doesn't support apply_chat_template...
                # https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md
                full_message = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
                if len(conversation_subset) > 1:
                    for msg in conversation_subset:
                        if msg["role"] == "user":
                            full_message += f"\nUSER: {msg['content']}\n"
                        else:
                            full_message += f"ASSISTANT: {msg['content']}<s>"
                input_ids = tokenizer(full_message, return_tensors="pt")

            else:
                conversation_subset = [system_instruction] + conversation_subset
                # TODO: Thinking removal doesn't work with EA setting, fix this or is this the same for all experiments and so doesnt matter? -> just run for now and check differences after, worst case rerun everything
                input_ids = apply_template(conversation_subset, tokenizer, TEMPLATE_TYPE)

            complete_sequence = torch.cat((input_ids["input_ids"], output_ids["input_ids"]), dim = -1).to(device)
            # For debugging
            if len(debug_conversation) != 3 and num_turns in [0, 1, max_turns - 1]:
                debug_conversation.append(str(tokenizer.batch_decode(complete_sequence)))
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
            absolute_rankings = []
            for k in range(target_tokens.shape[1]):
                token_id = target_tokens[0, k].item()
                logprob = log_probs[0, k, token_id].item()
                cum_logprob += logprob

                absolute_rankings.append(log_probs[0, k].argsort(descending=True).tolist().index(token_id)) # 0-based index

            entropy = -torch.sum(probs * torch.log(probs), dim=-1).sum().item()
            
            conv_results[f'logprob_turns_{num_turns + 1}'] = cum_logprob
            conv_results[f'entropy_turns_{num_turns + 1}'] = entropy
            # conv_results[f'absolute_rankings_turns_{num_turns + 1}'] = absolute_rankings
            conv_results[f'mean_rank_turns_{num_turns + 1}'] = np.mean(absolute_rankings) 
            conv_results[f'median_rank_turns_{num_turns + 1}'] = np.median(absolute_rankings)

        results.append(conv_results)
        
    # Save results
    print(f"Recorded {len(results)} results")
    input_setting = "HO" if HUMAN_ONLY else "MO" if MODEL_ONLY else "all"
    output_dir = f"results/exp_multi_turn/{input_setting}"
    os.makedirs(output_dir, exist_ok=True)

    output_setting = "TH" if TEST_HUMAN_MESSAGE else "TM"
    inject_message = "_RT" if INJECT_RANDOM_TOPIC else ""
    investigator_setting = "_INV" if INVESTIGATOR_SETTING else ""
    native_model_setting = "_NATIVE" if NATIVE_MODEL else ""
    mix_inputs_setting  = "_MIX" if MIX_INPUTS else ""
    empty_assistant_setting = "_EA" if EMPTY_ASSISTANT else ""
    template_setting = f"_T{TEMPLATE_TYPE}" if TEMPLATE_TYPE else ""
    low_token_setting = "_LT" if LOW_TOKENS else ""
    output_path = f"{output_dir}/exp_multi_{input_setting}_{output_setting}{inject_message}{investigator_setting}{native_model_setting}{mix_inputs_setting}{empty_assistant_setting}{template_setting}{low_token_setting}_{max_turns}_turn.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    end = time.time()
    print(f"Experiment completed in {end - start:.2f} seconds.")

    os.makedirs(f"debug/{input_setting}", exist_ok=True)

    with open(f'debug/{input_setting}/{output_path.split("/")[3].split(".")[0]}.txt', "w") as f:
        debug_string = "\n\n\n".join(debug_conversation)
        f.write(debug_string)
    print("=" * 20)

if __name__ == "__main__":

    INJECT_RANDOM_TOPIC = False
    INVESTIGATOR_SETTING = False
    MIX_INPUTS = False
    EMPTY_ASSISTANT = False
    HUMAN_ONLY = False
    MODEL_ONLY = False
    TEST_HUMAN_MESSAGE = False
    TEST_MODEL_MESSAGE = True
    TEMPLATE_TYPE = None
    LOW_TOKENS = True

    for NATIVE_MODEL in [True]:
        main(max_turns=5)

        # Short setting explanations:
        # 1. What message to score: Set either TEST_HUMAN_MESSAGE or TEST_MODEL_MESSAGE to True to test last human input or last model response respectively.
        # 2. What history to use: HUMAN_ONLY = True uses only previous human input, MODEL_ONLY = True uses only previous model responses. If both are set to False, the entire conversation history is utilized.
        # 3. Experiment variation parameters:
        # 3.a INJECT_RANDOM_TOPIC: If True, set the output message to score to a random request / completion based on setup specified in 1.
        # 3.b INVESTIGATOR_SETTING: If True, replace the system prompt with a description of the task of predicting the last human input. Requires TEST_HUMAN_MESSAGE = True.
        # 3.c NATIVE_MODEL: If True, use the model that gave most of the original responses in LMSYS dataset (vicuna-13b). Filter dataset to only include those conversations.
        # 3.d MIX_INPUTS: If True, randomly permute all input messages once. Keep this order fixed as the given context of previous turns changes.
        # 3.e EMPTY_ASSISTANT: If True, set the content of the model respones to empty strings. Only works with the entire conversation, so MODEL_ONLY = HUMAN_ONLY = False