from email.policy import default
from encodings.punycode import T
import json
from pyexpat.errors import messages
import os
import random
import re
from datasets import load_dataset
import uuid
import argparse
from typing import List, Dict, Optional
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#----------------------
# Configuration
#----------------------

# FIXME: for current outputs, used "Qwen/Qwen-4B-Instruct-2507" for both models
USER_MODEL_NAME = "Qwen/Qwen-4B-Instruct-2507" 
ASSISTANT_MODEL_NAME = "Qwen/Qwen-4B-Instruct-2507"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()

    parser.add_argument("--user_model", type=str, default=USER_MODEL_NAME, required=True, help="User model name",)
    parser.add_argument("--assistant_model", type=str, default=ASSISTANT_MODEL_NAME, required=True, help="Assistant model name",)
    parser.add_argument("--condition", type=str, choices=["no_persona", "hidden_persona", "both"], default="both", help="Which user condition to generate",)
    parser.add_argument("--num_conversations", type=int, default=20, help="Number of conversations to generate per condition",)  # PER CONDITION!!!
    parser.add_argument("--num_turns", type=int, default=6, help="Number of turns per conversation",)

    # prompt-only dataset seed config
    parser.add_argument("--seed_dataset", type=str, required=True, help="HF dataset name containing user prompts only (single-turn)",)
    parser.add_argument("--seed_split", type=str, default="train")
    parser.add_argument("--seed_column", type=str, required=True, help="Column name containing the prompt text (ex: prompt, instruction, question...)",)
    parser.add_argument("--seed_max_words", type=int, default=200)
    parser.add_argument("--seed_min_words", type=int, default=5)
    parser.add_argument("--seed_limit", type=int, default=5000, help="How many prompts to load then sample from (for speed)",)
    parser.add_argument("--seed_shuffle", action="store_true")
    parser.add_argument("--seed_seed", type=int, default=0)

    parser.add_argument("--output_path", type=str, default=f"src/data/conversations/llm_vs_llm_conversations_{timestamp}.jsonl", help="Path to save generated conversations",)
    return parser.parse_args()

#----------------------
# Prompts
#----------------------

ASSISTANT_SYSTEM_PROMPT = "You are a helpful assistant."

GENERATION_GUIDE = (
    "The user is discussing a topic with the assistant."
    "The conversation should feel natural and coherent."
)

USER_SYSTEM_PROMPT = (
    "Respond as a user reacting naturally to the assistant’s last message.\n\n"
    "Your response should be short (1–2 sentences) and conversational.\n"
    "You may:\n"
    "- ask for clarification\n"
    "- request a change or refinement\n"
    "- express agreement or disagreement\n"
    "- ask a follow-up question\n\n"
    "Do NOT:\n"
    "- continue the assistant’s answer\n"
    "- add new content or solutions\n"
    "- restate the original task\n\n"
    "Write only the user’s next message."
)

#----------------------
# Utilities
#----------------------

def load_seed_prompts(args) -> List[str]:
    ds = load_dataset(args.seed_dataset, split=args.seed_split)

    prompts = []
    n = 0
    for item in ds:
        if args.seed_column not in item:
            continue
        text = item[args.seed_column]
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue

        # filter by length
        w = text.split()
        if len(w) < args.seed_min_words or len(w) > args.seed_max_words:
            continue

        prompts.append(text)
        n += 1
        if n >= args.seed_limit:
            break

    if args.seed_shuffle:
        random.shuffle(prompts)

    if len(prompts) < args.num_conversations:
        raise ValueError(f"Not enough prompts after filtering: got {len(prompts)}")

    return prompts

def load_persona_dataset() -> List[str]:
    """Loads Persona-Chat and returns a list of joined persona strings."""
    print("Loading Persona-Chat dataset...")
    # used the PersonaChat dataset by Zhang et al. (2018)
    ds = load_dataset("AlekseyKorshuk/persona-chat", split="train")
    return [" ".join(item["personality"]) for item in ds]

def strip_reasoning(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Also remove if <think> appears without closing tag
    if '<think>' in text:
        text = text.split('<think>')[0]
    return text.strip()

def generate_reply(
        model,
        tokenizer,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 150,
        temperature: float = 0.7
) -> str:
    """ Generate one reply given a chat history. """
    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    # some tokenizers return a dict, some return a tensor directly
    if isinstance(enc, dict):
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device) 
    else:
        input_ids = enc.to(model.device)
        attention_mask = None

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )
    decoded = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    return strip_reasoning(decoded)

def load_model(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return model, tokenizer

def generate_conversation(
        user_model,
        user_tokenizer,
        assistant_model,
        assistant_tokenizer,
        condition: str,
        persona_text: Optional[str],
        seed_prompt: str,
) -> Dict:

    # visible transcript (saved in final output)
    messages = [
        {"role": "user", "content": seed_prompt},
    ]

    # Build the user system prompt ONCE, but it will be SENT every turn because
    # we rebuild user_messages each loop iteration (persona reminder every turn).
    if condition == "hidden_persona" and persona_text is not None:
          user_context_prompt = (
        f"{USER_SYSTEM_PROMPT}\n\n"
        "=== INTERNAL CHARACTER NOTES (DO NOT MENTION THESE) ===\n"
        f"{persona_text}\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- These are background facts about you as a person\n"
        "- They should SUBTLY influence your tone, interests, and reactions\n"
        "- DO NOT explicitly reference these facts unless the assistant directly asks about them\n"
        "- NEVER say things like 'my girlfriend' or 'I make 50k' unprompted\n"
        "- Stay focused on reacting to what the ASSISTANT just said\n"
        )
    else:
        user_context_prompt = USER_SYSTEM_PROMPT

    # FIRST assistant reply to seed prompt
    assistant_messages = [{"role": "system", "content": ASSISTANT_SYSTEM_PROMPT}]
    assistant_messages.extend(messages)
    assistant_reply = generate_reply(assistant_model, assistant_tokenizer, assistant_messages)
    messages.append({"role": "assistant", "content": assistant_reply})

    for _ in range(NUM_TURNS - 1):
        # User turn - only sees system prompt (with persona if applicable) + last assistant message
        user_messages = [
            {"role": "system", "content": user_context_prompt},
            {"role": "assistant", "content": messages[-1]["content"]},  # FIXED
        ]

        user_reply = generate_reply(
            user_model,
            user_tokenizer,
            user_messages,
            max_new_tokens=60,
            temperature=0.8
        )
        messages.append({"role": "user", "content": user_reply})

        # Assistant turn (assistant sees only visible transcript)
        assistant_messages = [{"role": "system", "content": ASSISTANT_SYSTEM_PROMPT}]
        assistant_messages.extend(messages)

        assistant_reply = generate_reply(assistant_model, assistant_tokenizer, assistant_messages)
        messages.append({"role": "assistant", "content": assistant_reply})

    return {
        "conversation_id": str(uuid.uuid4()),
        "condition": condition,
        "persona_text": persona_text,
        "seed_prompt": seed_prompt,
        "messages": messages,
    }

def main():
    args = parse_args()

    # ensure reproducible sampling/shuffle/persona choice
    random.seed(args.seed_seed)

    global NUM_TURNS
    NUM_TURNS = args.num_turns

    user_model, user_tokenizer = load_model(args.user_model)
    assistant_model, assistant_tokenizer = load_model(args.assistant_model)

    seeds = load_seed_prompts(args)
    seeds = seeds[: args.num_conversations]  # same seed for both conditions

    personas = load_persona_dataset()

    conversations = []

    # Generate conversations for both conditions
    for seed_prompt in seeds:

        if args.condition in ["no_persona", "both"]:
            conv_no_persona = generate_conversation(
                user_model, user_tokenizer,
                assistant_model, assistant_tokenizer,
                condition="no_persona",
                persona_text=None,
                seed_prompt=seed_prompt
            )
            conversations.append(conv_no_persona)

        if args.condition in ["hidden_persona", "both"]:
            persona_text = random.choice(personas)
            conv_hidden_persona = generate_conversation(
                user_model, user_tokenizer,
                assistant_model, assistant_tokenizer,
                condition="hidden_persona",
                persona_text=persona_text,
                seed_prompt=seed_prompt
            )
            conversations.append(conv_hidden_persona)

    with open(args.output_path, "w") as f:
        for conv in conversations:
            conv["seed_dataset"] = args.seed_dataset
            conv["seed_column"] = args.seed_column
            f.write(json.dumps(conv) + "\n")

if __name__ == "__main__":
    main()