import json
import os
import random
import re
import uuid
import argparse
from typing import List, Dict
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------
# Defaults
# ----------------------
USER_MODEL_NAME = "Qwen/Qwen3-8B"
ASSISTANT_MODEL_NAME = "Qwen/Qwen3-8B"


# ----------------------
# Args
# ----------------------
def parse_args():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser()

    p.add_argument("--experiments_path", type=str, required=True,
                   help="Path to experiments.jsonl (your teammate's file)")
    p.add_argument("--user_model", type=str, default=USER_MODEL_NAME)
    p.add_argument("--assistant_model", type=str, default=ASSISTANT_MODEL_NAME)

    p.add_argument("--num_conversations", type=int, default=50,
                   help="How many experiment rows to sample")
    p.add_argument("--num_turns", type=int, default=6,
                   help="Number of user turns (incl. init_user_message)")

    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--output_path", type=str,
                   default=f"src/data/conversations/with_cat_persona_{timestamp}.jsonl")

    # generation knobs (optional, but nice to control)
    p.add_argument("--assistant_max_new_tokens", type=int, default=200)
    p.add_argument("--assistant_temp", type=float, default=0.7)
    p.add_argument("--user_max_new_tokens", type=int, default=80)
    p.add_argument("--user_temp", type=float, default=0.8)

    return p.parse_args()


# ----------------------
# Utilities
# ----------------------
def strip_reasoning(text: str) -> str:
    # Remove Qwen-style reasoning blocks if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = text.split("<think>")[0]
    return text.strip()

def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

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


# ----------------------
# Core generation
# ----------------------
def validate_experiment_row(exp: Dict):
    required = ["persona_id", "profile", "system_llm1", "system_llm2", "init_user_message"]
    missing = [k for k in required if k not in exp]
    if missing:
        raise ValueError(f"Experiment row missing keys: {missing}")

    if not isinstance(exp["profile"], dict):
        raise ValueError("profile must be a dict")

    if not isinstance(exp["system_llm1"], str) or not exp["system_llm1"].strip():
        raise ValueError("system_llm1 must be a non-empty string")
    if not isinstance(exp["system_llm2"], str) or not exp["system_llm2"].strip():
        raise ValueError("system_llm2 must be a non-empty string")
    if not isinstance(exp["init_user_message"], str) or not exp["init_user_message"].strip():
        raise ValueError("init_user_message must be a non-empty string")


def generate_conversation_with_persona(
    user_model, user_tok,
    assistant_model, assistant_tok,
    exp_row: Dict,
    num_turns: int,
    user_max_new_tokens: int,
    user_temp: float,
    assistant_max_new_tokens: int,
    assistant_temp: float,
) -> Dict:

    validate_experiment_row(exp_row)

    persona_id = exp_row["persona_id"]
    profile = exp_row["profile"]
    system_llm1 = exp_row["system_llm1"].strip()    # user system prompt (persona/style)
    system_llm2 = exp_row["system_llm2"].strip()    # assistant system prompt
    seed_prompt = exp_row["init_user_message"].strip()

    # Visible transcript saved
    messages = [{"role": "user", "content": seed_prompt}]

    # First assistant reply (assistant sees system_llm2 + full visible transcript)
    assistant_messages = [{"role": "system", "content": system_llm2}] + messages
    assistant_reply = generate_reply(
        assistant_model, assistant_tok, assistant_messages,
        max_new_tokens=assistant_max_new_tokens,
        temperature=assistant_temp
    )
    messages.append({"role": "assistant", "content": assistant_reply})


    TASK_PROMPT_USER = """\
    You are the USER in this conversation (not the assistant).
    Write ONLY the user's next message responding to the assistant.

    Constraints:
    - 1–2 sentences (max 40 words).
    - You may ask a question, accept/reject, request changes, or clarify.
    - Do NOT write an answer/solution or continue the assistant's output.
    - Do NOT write "Assistant:" or anything except the user message.
    """
    # Alternate user/assistant until we have num_turns user messages total
    # We already have 1 user message (seed_prompt). We need (num_turns - 1) more user turns
    for _ in range(num_turns - 1):
        # User reacts based on persona instructions (systemllm1) + task prompt + last assistant message only
        user_messages = [
            {"role": "system", "content": system_llm1},  # persona/style
            {"role": "user", "content": (
                TASK_PROMPT_USER
                + "\n\nAssistant just said:\n"
                + messages[-1]["content"]
                + "\n\nNow write the user's next message:"
            )},
        ]
        user_reply = generate_reply(
            user_model, user_tok, user_messages,
            max_new_tokens=user_max_new_tokens,
            temperature=user_temp
        )
        messages.append({"role": "user", "content": user_reply})

        # Assistant sees system_llm2 + full transcript
        assistant_messages = [{"role": "system", "content": system_llm2}] + messages
        assistant_reply = generate_reply(
            assistant_model, assistant_tok, assistant_messages,
            max_new_tokens=assistant_max_new_tokens,
            temperature=assistant_temp
        )
        messages.append({"role": "assistant", "content": assistant_reply})

    # fetch investigator mode, might use it later 
    inv_mode = None
    try:
        inv_mode = persona_id.split("__")[-1].replace("inv_", "")
    except Exception:
        inv_mode = profile.get("investigator_mode")

    return {
        "conversation_id": str(uuid.uuid4()),
        "condition": "with_persona",
        "persona_id": persona_id,
        "investigator_mode": inv_mode,
        "profile": profile,

        "seed_prompt": seed_prompt,
        "system_llm1": system_llm1,
        "system_llm2": system_llm2,

        
        "persona_text": system_llm1, #called like this in scoring 

        "messages": messages,
    }


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    random.seed(args.seed)

    rows = read_jsonl(args.experiments_path)
    if args.shuffle:
        random.shuffle(rows)

    rows = rows[: args.num_conversations]
    if len(rows) == 0:
        raise ValueError("No experiment rows loaded")

    user_model, user_tok = load_model(args.user_model)
    assistant_model, assistant_tok = load_model(args.assistant_model)

    out = []
    for exp in rows:
        try:
            conv = generate_conversation_with_persona(
                user_model, user_tok,
                assistant_model, assistant_tok,
                exp_row=exp,
                num_turns=args.num_turns,
                user_max_new_tokens=args.user_max_new_tokens,
                user_temp=args.user_temp,
                assistant_max_new_tokens=args.assistant_max_new_tokens,
                assistant_temp=args.assistant_temp,
            )
            out.append(conv)
        except Exception as e:
            print(f"Skipping row due to error: {e}")

    with open(args.output_path, "w", encoding="utf-8") as f:
        for conv in out:
            conv["experiments_path"] = args.experiments_path
            conv["user_model_name"] = args.user_model
            conv["assistant_model_name"] = args.assistant_model
            f.write(json.dumps(conv) + "\n")

    print(f"Wrote {len(out)} conversations -> {args.output_path}")


if __name__ == "__main__":
    main()
