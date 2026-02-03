import json 
import random
import uuid
import argparse
from typing import List, Dict, Optional
from datetime import datetime
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

#----------------------
# Configuration
#----------------------

# FIXME: for current outputs, used "Qwen/Qwen-4B-Instruct-2507" for both models, thinking models were hallucinating/ user model acting as assistant too...
USER_MODEL_NAME = "Qwen/Qwen3-8B" #reasoning llm, can switch between thinking and non-thinking mode
ASSISTANT_MODEL_NAME = "Qwen/Qwen3-8B"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FIXME: 
#sometimes, I see "<think>" appear in output when using thinking models --> might be source of pollution if dataset and a reason for "user" model to act as assistant too? 
# check if a "no_think" setting exists ? 

# Method used for seed: "persona filtered" seed
# use seed as "Thought" but have the user model phrase it based on their persona 

def parse_args():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()

    # model config
    parser.add_argument("--user_model", type=str, default=USER_MODEL_NAME, help="User model name",)
    parser.add_argument("--assistant_model", type=str, default=ASSISTANT_MODEL_NAME, help="Assistant model name",)
    
    # experiment config
    parser.add_argument("--condition", type=str, choices=["no_persona", "hidden_persona", "both"], default="both", help="Which user condition to generate",)
    parser.add_argument("--num_conversations", type=int, default=20, help="Number of conversations to generate per condition",) #PER CONIDITION!!! so if N = 6, then total of 12 convs (ie 6 w persona, 6 w/o)
    parser.add_argument("--num_turns", type=int, default=6, help="Number of turns per conversation",)
    
    # seed dataset config
    parser.add_argument("--seed_dataset", type=str, required=True, help="HF dataset name containing user prompts only (single-turn)",)
    parser.add_argument("--seed_split", type=str, default="train")
    parser.add_argument("--seed_column", type=str, required=True, help="Column name containing the prompt text (ex: prompt, instruction, question...)",)
    parser.add_argument("--seed_max_words", type=int, default=200)
    parser.add_argument("--seed_min_words", type=int, default=5)
    parser.add_argument("--seed_limit", type=int, default=5000, help="How many prompts to load then sample from (for speed)",)
    parser.add_argument("--seed_shuffle", action="store_true")
    
    parser.add_argument("--output_path", type=str, default=f"src/data/conversations/llm_vs_llm_conversations_{timestamp}.jsonl", help="Path to save generated conversations",)
    return parser.parse_args()

#----------------------
# Prompts
#----------------------

ASSISTANT_SYSTEM_PROMPT = "You are a helpful assistant. Reply with ONLY the final answer. No analysis, no planning, no meta-commentary."

"""
USER_NEUTRAL_PROMPT = (
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
"""
USER_SYSTEM_BASE = (
    "You are roleplaying the USER in a natural conversation.\n"
    "You are NOT the assistant.\n"
    "Write ONLY what the user would say next.\n"
    "Constraints:\n"
    "- 1–3 sentences\n"
    "- No reasoning, no meta-commentary (do not write 'the user wants...', 'let me think', etc.)\n"
    "- Do not solve the whole task; only react, ask, clarify, request a revision, or provide preferences\n"
    "- Do NOT output any prefixes like 'User:' or 'Assistant:'\n"
)

USER_TURN_INSTRUCTION = (
    "Now reply as the USER to the assistant's last message.\n"
    "Output ONLY the user's next message."
)

#----------------------
# Model I/O
#----------------------
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

def strip_reasoning(text: str) -> str:
    """
    Robustly remove <think>...</think> blocks, and also remove stray <think> tags
    without blanking the entire answer.
    """
    if not text:
        return ""

    # Remove full think blocks
    text = THINK_BLOCK.sub("", text)

    # Remove stray tags if present
    text = text.replace("<think>", "").replace("</think>", "")

    return text.strip()

def load_model(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="auto",)
    return model, tokenizer

def generate_reply(
        model,
        tokenizer,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
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
            attention_mask = attention_mask.to(model.device) # ensure on correct device, had runtime error once
    else:
        input_ids = enc.to(model.device)
        attention_mask = None

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    return strip_reasoning(decoded)

def safe_generate(model, tokenizer, messages, max_new_tokens=150):
    text = generate_reply(model, tokenizer, messages, max_new_tokens=max_new_tokens)
    text = strip_reasoning(text)
    return text if text else "[EMPTY]"

    
#----------------------
# Data Loading
#----------------------

def load_persona_dataset() -> List[str]:
    """Loads Persona-Chat and returns a list of joined persona strings."""
    print("Loading Persona-Chat dataset...")
    # used true-cased version of the PersonaChat dataset by Zhang et al. (2018)
    ds = load_dataset("AlekseyKorshuk/persona-chat", split="train")
    return [" ".join(item['personality']) for item in ds]

def load_seed_prompts(args) -> List[str]:
    print(f"Loading dataset: {args.seed_dataset}...")
    ds = load_dataset(args.seed_dataset, split=args.seed_split)
    prompts = []

    for item in ds:
       # Check if we are dealing with the nested LMSYS structure
        if args.seed_column == "nested_lmsys":
            # Access first turn of conversation_a
            try:
                text = item['conversation_a'][0]['content'].strip()
            except (KeyError, IndexError, TypeError):
                continue
        else:
            # Standard flat column access
            text = item.get(args.seed_column, "")
            if not isinstance(text, str): continue
            text = text.strip()

        if not text: continue
        
        words = text.split()
        if args.seed_min_words <= len(words) <= args.seed_max_words:
            prompts.append(text)
        if len(prompts) >= args.seed_limit: break
    if args.seed_shuffle: random.shuffle(prompts)
    return prompts

#----------------------
# Conversation generation
#----------------------

def build_user_system(persona_text: Optional[str]) -> str:
    if persona_text:
        return USER_SYSTEM_BASE + f"\nUser background facts (stay consistent): {persona_text}\n"
    return USER_SYSTEM_BASE

def generate_conversation(
        user_model,
        user_tokenizer, 
        assistant_model,    
        assistant_tokenizer,
        condition, 
        persona_text, 
        seed_topic,
        num_turns,
) -> Dict:
    """
    Generates a conversation where:
    - The user model produces user messages (persona-conditioned)
    - The assistant model responds normally
    Persona reminder is persistent via the user system prompt every user turn.
    """
    user_sys = build_user_system(persona_text)
    
    messages: List[Dict[str, str]] = [{"role": "user", "content": seed_topic.strip()}]
    
    # ---- Alternating turns ----
    for _ in range(num_turns):
        # 1) Assistant turn
        assistant_ctx = [{"role": "system", "content": ASSISTANT_SYSTEM_PROMPT}] + messages
        assistant_reply = safe_generate(assistant_model, assistant_tokenizer, assistant_ctx)
        if assistant_reply == "[EMPTY]":
            assistant_reply = "Can you clarify what you want me to do?"
        messages.append({"role": "assistant", "content": assistant_reply})

        # 2) User turn (FULL HISTORY to user model)
        # IMPORTANT: append a *user* instruction message so apply_chat_template generates an assistant continuation
        # which we interpret as "the user's next utterance" per instruction.
        user_ctx = (
            [{"role": "system", "content": user_sys}]
            + messages
            + [{"role": "user", "content": USER_TURN_INSTRUCTION}]
        )
        user_reply = safe_generate(user_model, user_tokenizer, user_ctx)
        if user_reply == "[EMPTY]":
            user_reply = "Could you clarify that?"

        messages.append({"role": "user", "content": user_reply})

    return {
        "conversation_id": str(uuid.uuid4()),
        "condition": condition,
        "persona_text": persona_text,
        "seed_topic": seed_topic,
        "messages": messages,
    }

def main():
    args = parse_args()

    #global NUM_TURNS
    #NUM_TURNS = args.num_turns 

    user_model, user_tokenizer = load_model(args.user_model)
    assistant_model, assistant_tokenizer = load_model(args.assistant_model)

    seeds = load_seed_prompts(args)[:args.num_conversations] #same seed for both conditions
    personas = load_persona_dataset()

    conversations : List[Dict] = []
    # Generate conversations for both conditions
    for seed in seeds:

        if args.condition in ["no_persona", "both"]:
        
            conv_no_persona = generate_conversation(
            user_model, user_tokenizer,
            assistant_model, assistant_tokenizer,
            condition="no_persona",
            persona_text=None, 
            seed_topic=seed,
            num_turns=args.num_turns
            )
            conversations.append(conv_no_persona)
        
        if args.condition in ["hidden_persona", "both"]:
            selected_persona = random.choice(personas)
            conv_hidden_persona = generate_conversation(
                user_model, user_tokenizer,
                assistant_model, assistant_tokenizer,
                "hidden_persona",
                selected_persona, 
                seed, 
                args.num_turns
            )
            conversations.append(conv_hidden_persona)

    with open(args.output_path, "w") as f:
        for conv in conversations:
            conv["seed_dataset"] = args.seed_dataset
            conv["seed_column"] = args.seed_column
            f.write(json.dumps(conv) + "\n")
    print(f"Generated {len(conversations)} conversations to {args.output_path}")
if __name__ == "__main__":
    main()