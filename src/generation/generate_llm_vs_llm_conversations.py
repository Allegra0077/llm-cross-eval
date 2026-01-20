from email.policy import default
from encodings.punycode import T
import json 
from pyexpat.errors import messages
import os
import random
from datasets import load_dataset   
import uuid
import argparse
from typing import List, Dict
from datetime import datetime

import torch
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

def parse_args():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()

    parser.add_argument("--user_model", type=str, default=USER_MODEL_NAME, required=True, help="User model name",)
    parser.add_argument("--assistant_model", type=str, default=ASSISTANT_MODEL_NAME, required=True, help="Assistant model name",)
    parser.add_argument("--condition", type=str, choices=["no_persona", "hidden_persona", "both"], default="both", help="Which user condition to generate",)
    parser.add_argument("--num_conversations", type=int, default=20, help="Number of conversations to generate per condition",) #PER CONIDITION!!! so if N = 6, then total of 12 convs (ie 6 w persona, 6 w/o)
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

SYSTEM_PROMPT = "You are a helpful assistant."

GENERATION_GUIDE = ("The user is discussing a topic with the assistant."
               "The conversation should feel natural and coherent.")

#USER_NEUTRAL_PROMPT = "You are a normal user interacting naturally with an assisant." --> prompt was not strong enough, user acted as an assistant...
USER_NEUTRAL_PROMPT = (
    "You are a HUMAN USER interacting with an AI assistant.\n"
    "You MUST behave like a real user, NOT like an assistant or writer.\n\n"
    "You are a HUMAN USER. You do NOT write the content.\n"
    "You only give feedback on the assistant’s answer.\n\n"
    "Hard rules:\n"
    "- Do NOT continue the assistant’s text\n"
    "- Do NOT add new tips, steps, or paragraphs\n"             
    "- Do NOT rewrite sections yourself\n"
    "- ONLY: comment, critique, ask for changes, ask questions (1–2 sentences)\n\n"
    
    "Rules:\n"
    "- NEVER explain how to solve the task\n"
    "- NEVER restate the task\n"
    "- NEVER write the final answer yourself\n\n"
    "What you SHOULD do:\n"
    "- React to what the assistant just said\n"
    "- Ask for clarifications, edits, or changes\n"
    "- Express opinions (like/dislike)\n"
    "- Ask follow-up questions\n"
    "- Push back or disagree if needed\n\n"
    "Your replies should be SHORT and conversational (1–3 sentences).\n"
    "You are not helpful. You are being helped.\n"
    "Every message you send must contain either a question or a request for a change."
)

#FIXME: maybe construct a better dataset of personas that are classified by type (ex: "common" personas vs "complex" personas...)
USER_PERSONA_PROMPTS = {
    "p01": "You are a very concise, efficiency-focused user.",
    "p02": "You are a very talkative and friendly user.",
    "p03": "You are a skeptical user who often asks questions to the assistant.",
    "p04": "You are a humorous user who likes to make jokes."
}

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

def strip_reasoning(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()

def generate_reply(
        model,
        tokenizer,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 150,
        temperature: float = 0.7
) -> str:
    """ Generate one reply given a chat history. """
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )
    decoded = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    return decoded.strip() 

def safe_generate(model, tokenizer, messages, max_new_tokens=150, temperature=0.7, retries=2):
    for attempt in range(retries):
        raw = generate_reply(model, tokenizer, messages, max_new_tokens, temperature)
        clean = strip_reasoning(raw)
        if clean:
            return clean
        return "[EMPTY]"
    
def load_model(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="auto",)
    return model, tokenizer

def generate_conversation(
        user_model,
        user_tokenizer, 
        assistant_model,    
        assistant_tokenizer,
        condition: str, 
        persona_id: str | None, 
        seed_prompt: str, 
) -> Dict:
    
    # visible transcript (saved in final output)
    messages = [
        {"role": "user", "content": seed_prompt},
    ]

    if condition == "hidden_persona":
        persona_prompt = USER_PERSONA_PROMPTS[persona_id]
        user_context_prompt = USER_NEUTRAL_PROMPT + " " + persona_prompt
    else: 
        user_context_prompt = USER_NEUTRAL_PROMPT

    # FIRST assistant reply to seed prompt
    assistant_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    assistant_messages.extend(messages)
    assistant_reply = safe_generate(assistant_model, assistant_tokenizer, assistant_messages)
    messages.append({"role": "assistant", "content": assistant_reply})

    for _ in range(NUM_TURNS - 1):
        # User turn (reacts to assistant)
        user_messages = [{"role": "system", "content": user_context_prompt}]
        user_messages.extend(messages)

        user_reply = safe_generate(user_model, user_tokenizer, user_messages, max_new_tokens=60, temperature=0.8) #FIXME: wanted shorter replies for user, since user model was generating very long answers --> but here I'm essentially "cutting" the answers, not prompting the model to be concise.. 
        messages.append({"role": "user", "content": user_reply})

        # Assistant turn (assistant sees only visible transcript)
        assistant_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        assistant_messages.extend(messages)
        
        assistant_reply = safe_generate(assistant_model, assistant_tokenizer, assistant_messages)
        messages.append({"role": "assistant", "content": assistant_reply})  

    return {
        "conversation_id": str(uuid.uuid4()),
        "condition": condition,
        "persona_id": persona_id,
        "seed_prompt": seed_prompt,
        "messages": messages,
    }

def main():
    args = parse_args()

    global NUM_TURNS
    NUM_TURNS = args.num_turns 

    user_model, user_tokenizer = load_model(args.user_model)
    assistant_model, assistant_tokenizer = load_model(args.assistant_model)

    seeds = load_seed_prompts(args)
    seeds = seeds[: args.num_conversations] #same seed for both conditions
    
    conversations = []

    # Generate conversations for both conditions
    for i, seed_prompt in enumerate(seeds):

        if args.condition in ["no_persona", "both"]:
        
            conv_no_persona = generate_conversation(
            user_model, user_tokenizer,
            assistant_model, assistant_tokenizer,
            condition="no_persona",
            persona_id=None, 
            seed_prompt=seed_prompt
            )
            conversations.append(conv_no_persona)
        
        if args.condition in ["hidden_persona", "both"]:
            persona_id = list(USER_PERSONA_PROMPTS.keys())[i % len(USER_PERSONA_PROMPTS)]
            conv_hidden_persona = generate_conversation(
                user_model, user_tokenizer,
                assistant_model, assistant_tokenizer,
                condition="hidden_persona",
                persona_id=persona_id, 
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