import json
import os
import random
import re
import uuid
import argparse
from typing import List, Dict, Optional
from datetime import datetime

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------
# Configuration
# ----------------------
USER_MODEL_NAME = "Qwen/Qwen3-8B"
ASSISTANT_MODEL_NAME = "Qwen/Qwen3-8B"


# ----------------------
# Args
# ----------------------
def parse_args():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()

    parser.add_argument("--user_model", type=str, default=USER_MODEL_NAME, required=True)
    parser.add_argument("--assistant_model", type=str, default=ASSISTANT_MODEL_NAME, required=True)

    # For stitching: you almost surely want hidden_persona (and maybe also no_persona baseline)
    parser.add_argument("--condition", type=str, choices=["no_persona", "hidden_persona", "both"],
                        default="both")

    parser.add_argument("--num_conversations", type=int, default=20,
                        help="Number of STITCHED conversations per condition")
    parser.add_argument("--num_turns", type=int, default=6,
                        help="Number of USER turns per segment (including that segment's seed prompt)")
    parser.add_argument("--num_segments", type=int, default=2,
                        help="How many segments to stitch. Your request = 2.")

    # prompt-only dataset seed config
    parser.add_argument("--seed_dataset", type=str, required=True)
    parser.add_argument("--seed_split", type=str, default="train")
    parser.add_argument("--seed_column", type=str, required=True)
    parser.add_argument("--seed_max_words", type=int, default=200)
    parser.add_argument("--seed_min_words", type=int, default=5)
    parser.add_argument("--seed_limit", type=int, default=20000)
    parser.add_argument("--seed_shuffle", action="store_true")
    parser.add_argument("--seed_seed", type=int, default=0)

    # generation knobs
    parser.add_argument("--assistant_max_new_tokens", type=int, default=180)
    parser.add_argument("--assistant_temp", type=float, default=0.7)
    parser.add_argument("--user_max_new_tokens", type=int, default=60)
    parser.add_argument("--user_temp", type=float, default=0.8)

    parser.add_argument("--output_path", type=str,
                        default=f"src/data/conversations/stitched_llm_vs_llm_{timestamp}.jsonl")
    return parser.parse_args()


# ----------------------
# Prompts
# ----------------------
ASSISTANT_SYSTEM_PROMPT = "You are a helpful assistant."

TASK_PROMPT_USER = """You are the USER in this conversation (not the assistant).
Write ONLY the user's next message responding to the assistant.

Constraints:
- 1–2 sentences (max 40 words).
- You may ask for clarification, request a change/refinement, accept/reject, or ask a follow-up question.
- Do NOT continue the assistant’s answer.
- Do NOT provide solutions, plans, or multi-step output.
- Do NOT write role tags like "Assistant:" or "User:".
- Output ONLY the user message text.
"""

USER_SYSTEM_PROMPT = (
    "Respond as a user reacting naturally to the assistant’s last message.\n"
    "Keep it short and conversational.\n"
)


# ----------------------
# Utilities
# ----------------------
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

        w = text.split()
        if len(w) < args.seed_min_words or len(w) > args.seed_max_words:
            continue

        prompts.append(text)
        n += 1
        if n >= args.seed_limit:
            break

    if args.seed_shuffle:
        random.shuffle(prompts)

    return prompts


def load_persona_dataset() -> List[str]:
    print("Loading Persona-Chat dataset...")
    ds = load_dataset("AlekseyKorshuk/persona-chat", split="train")
    return [" ".join(item["personality"]) for item in ds]


def strip_reasoning(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = text.split("<think>")[0]
    return text.strip()


def load_model(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    return model, tok


def generate_reply(model, tokenizer, messages: List[Dict[str, str]],
                   max_new_tokens: int, temperature: float) -> str:
    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
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


def build_user_system_prompt(condition: str, persona_text: Optional[str]) -> str:
    base = USER_SYSTEM_PROMPT + "\n" + TASK_PROMPT_USER

    if condition == "hidden_persona" and persona_text:
        return (
            base
            + "\n\n=== INTERNAL CHARACTER NOTES (DO NOT MENTION THESE) ===\n"
            + persona_text
            + "\n\nCRITICAL INSTRUCTIONS:\n"
            + "- These are background facts about you as a person.\n"
            + "- They should SUBTLY influence your tone, interests, and reactions.\n"
            + "- Do NOT explicitly reference these facts unless the assistant directly asks.\n"
            + "- Stay focused on reacting to what the assistant just said.\n"
        )
    return base


def generate_stitched_conversation(
    user_model, user_tok,
    assistant_model, assistant_tok,
    condition: str,
    persona_text: Optional[str],
    seed_prompts: List[str],          # length = num_segments
    num_turns_per_segment: int,
    user_max_new_tokens: int,
    user_temp: float,
    assistant_max_new_tokens: int,
    assistant_temp: float,
) -> Dict:
    """
    Creates ONE transcript by stitching num_segments segments.
    Each segment starts with its own seed prompt (a user message),
    then continues with user/assistant alternating for (num_turns_per_segment - 1) more user turns.
    """

    assert len(seed_prompts) >= 1
    messages: List[Dict[str, str]] = []
    seed_user_msg_indices: List[int] = []

    user_system_prompt = build_user_system_prompt(condition, persona_text)

    for seg_i, seed in enumerate(seed_prompts):
        # Add seed user message for this segment
        seed_user_msg_indices.append(len(messages))
        messages.append({"role": "user", "content": seed})

        # Assistant replies to seed, seeing full transcript so far
        assistant_messages = [{"role": "system", "content": ASSISTANT_SYSTEM_PROMPT}] + messages
        a_reply = generate_reply(
            assistant_model, assistant_tok, assistant_messages,
            max_new_tokens=assistant_max_new_tokens,
            temperature=assistant_temp
        )
        messages.append({"role": "assistant", "content": a_reply})

        # Now add the remaining (num_turns_per_segment - 1) user turns for this segment
        for _ in range(num_turns_per_segment - 1):
            user_messages = [
                {"role": "system", "content": user_system_prompt},
                {"role": "assistant", "content": messages[-1]["content"]},
            ]
            u_reply = generate_reply(
                user_model, user_tok, user_messages,
                max_new_tokens=user_max_new_tokens,
                temperature=user_temp
            )
            messages.append({"role": "user", "content": u_reply})

            assistant_messages = [{"role": "system", "content": ASSISTANT_SYSTEM_PROMPT}] + messages
            a_reply = generate_reply(
                assistant_model, assistant_tok, assistant_messages,
                max_new_tokens=assistant_max_new_tokens,
                temperature=assistant_temp
            )
            messages.append({"role": "assistant", "content": a_reply})

    return {
        "conversation_id": str(uuid.uuid4()),
        "stitched": True,
        "num_segments": len(seed_prompts),
        "num_turns_per_segment": num_turns_per_segment,

        "condition": condition,
        "persona_text": persona_text,
        "seed_prompts": seed_prompts,
        "seed_user_msg_indices": seed_user_msg_indices,  # IMPORTANT for scoring

        "assistant_system_prompt": ASSISTANT_SYSTEM_PROMPT,
        "user_system_prompt": user_system_prompt,

        "messages": messages,
    }


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    random.seed(args.seed_seed)

    user_model, user_tok = load_model(args.user_model)
    assistant_model, assistant_tok = load_model(args.assistant_model)

    # We need enough seeds: num_conversations * num_segments, for each condition we generate.
    # Easiest: just grab a big pool and sample without replacement.
    seed_pool = load_seed_prompts(args)
    if len(seed_pool) < args.num_conversations * args.num_segments:
        raise ValueError(
            f"Not enough seed prompts: need at least {args.num_conversations * args.num_segments}, got {len(seed_pool)}"
        )

    personas = load_persona_dataset()
    if len(personas) < args.num_conversations:
        raise ValueError(f"Not enough personas: need {args.num_conversations}, got {len(personas)}")

    conversations = []

    # Sample personas for stitched convs (one persona per stitched convo)
    chosen_personas = random.sample(personas, k=args.num_conversations)

    # Sample seed prompts in pairs (or num_segments-tuples)
    # We sample WITHOUT replacement so each stitched convo gets distinct seeds.
    chosen_seeds = random.sample(seed_pool, k=args.num_conversations * args.num_segments)

    def seeds_for_i(i: int) -> List[str]:
        start = i * args.num_segments
        return chosen_seeds[start:start + args.num_segments]

    for i in range(args.num_conversations):
        seg_seeds = seeds_for_i(i)

        if args.condition in ["no_persona", "both"]:
            conversations.append(
                generate_stitched_conversation(
                    user_model, user_tok,
                    assistant_model, assistant_tok,
                    condition="no_persona",
                    persona_text=None,
                    seed_prompts=seg_seeds,
                    num_turns_per_segment=args.num_turns,
                    user_max_new_tokens=args.user_max_new_tokens,
                    user_temp=args.user_temp,
                    assistant_max_new_tokens=args.assistant_max_new_tokens,
                    assistant_temp=args.assistant_temp,
                )
            )

        if args.condition in ["hidden_persona", "both"]:
            persona_text = chosen_personas[i]
            conversations.append(
                generate_stitched_conversation(
                    user_model, user_tok,
                    assistant_model, assistant_tok,
                    condition="hidden_persona",
                    persona_text=persona_text,
                    seed_prompts=seg_seeds,
                    num_turns_per_segment=args.num_turns,
                    user_max_new_tokens=args.user_max_new_tokens,
                    user_temp=args.user_temp,
                    assistant_max_new_tokens=args.assistant_max_new_tokens,
                    assistant_temp=args.assistant_temp,
                )
            )

    with open(args.output_path, "w", encoding="utf-8") as f:
        for conv in conversations:
            conv["seed_dataset"] = args.seed_dataset
            conv["seed_column"] = args.seed_column
            f.write(json.dumps(conv) + "\n")

    print(f"Wrote {len(conversations)} stitched conversations -> {args.output_path}")


if __name__ == "__main__":
    main()
