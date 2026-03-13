import argparse
import json
import random
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

USER_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
ASSISTANT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
SPECIAL_TOKENS_RE = re.compile(r"(<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|<\|eot_id\|>)")
INV_TAIL_RE = re.compile(r'<INVESTIGATION\s+guess="([^"]+)"\s+confidence="(\d{1,3})"\s*/>\s*$')


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def strip_reasoning(text: str) -> str:
    if not text:
        return text
    text = THINK_RE.sub("", text)
    if "<think>" in text:
        text = text.split("<think>")[0]
    text = SPECIAL_TOKENS_RE.sub("", text)
    return text.strip()


def load_model(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    return model, tok


def generate_text(model, tokenizer, messages: List[Dict[str, str]], max_new_tokens: int, temperature: float) -> str:
    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = enc.to(model.device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out[0][input_ids.shape[1]:]
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
    if not decoded:
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return strip_reasoning(decoded)


def generate_json(model, tokenizer, prompt: str, max_new_tokens: int = 300, temperature: float = 0.7) -> Dict[str, Any]:
    text = generate_text(
        model,
        tokenizer,
        [{"role": "user", "content": prompt}],
        max_new_tokens,
        temperature,
    )
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Could not extract JSON from output:\n{text}")
    return json.loads(text[start:end + 1])


def parse_assistant(text: str) -> Tuple[str, Optional[Dict]]:
    t = text.strip()
    m = INV_TAIL_RE.search(t)
    if not m:
        return t, None
    guess = m.group(1).strip()
    conf = max(0, min(100, int(m.group(2))))
    clean = t[:m.start()].rstrip()
    return clean, {"guess": guess, "confidence": conf, "raw_line": m.group(0).strip()}


def format_recent_history(messages: List[Dict[str, str]], max_messages: int = 8) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages[-max_messages:])


def init_state(seed_row: Dict) -> Dict[str, Any]:
    return {
        "topic": seed_row.get("topic", ""),
        "goal": seed_row.get("conversation_goal", ""),
        "revealed_preferences": [],
        "revealed_constraints": [],
        "decisions_made": [],
        "open_questions": [],
        "tone": "neutral",
    }


def render_state_summary(state: Dict[str, Any]) -> str:
    return (
        f"Topic: {state.get('topic', '')}\n"
        f"Goal: {state.get('goal', '')}\n"
        f"Revealed preferences: {state.get('revealed_preferences', [])}\n"
        f"Revealed constraints: {state.get('revealed_constraints', [])}\n"
        f"Decisions made: {state.get('decisions_made', [])}\n"
        f"Open questions: {state.get('open_questions', [])}\n"
        f"Tone: {state.get('tone', 'neutral')}"
    )


def update_state_heuristic(state: Dict[str, Any], user_msg: str, assistant_msg: str):
    if "?" in user_msg:
        state["open_questions"].append(user_msg[:120])
        state["open_questions"] = state["open_questions"][-5:]
    if any(word in user_msg.lower() for word in ["prefer", "want", "need", "don't want", "cannot", "can't"]):
        state["revealed_constraints"].append(user_msg[:120])
        state["revealed_constraints"] = state["revealed_constraints"][-5:]
    if any(word in assistant_msg.lower() for word in ["plan", "option", "recommend", "suggest"]):
        state["decisions_made"] = state["decisions_made"][-5:]


def build_assistant_system_prompt(prompts_cfg: Dict, condition: str, style_profiles: Dict[str, Dict]) -> str:
    if condition == "with_persona_investigator_guided":
        style_ids = list(style_profiles.keys())
        style_names = [style_profiles[sid]["name"] for sid in style_ids]
        return prompts_cfg["system_llm2_investigator_guided"]["prompt"].format(
            STYLE_ID_LIST=", ".join(style_ids),
            STYLE_NAME_LIST=", ".join(style_names),
        )
    if condition == "with_persona_investigator_unguided":
        return prompts_cfg["system_llm2_investigator_unguided"]["prompt"]
    return prompts_cfg["system_llm2_base"]["prompt"]


def generate_first_user_message(model, tok, prompts_cfg: Dict, user_row: Dict, seed_row: Dict, temperature: float) -> str:
    prompt = prompts_cfg["generation_prompt_init_user_prompt"]["prompt"].format(
        BASE_PERSONA_JSON=json.dumps(user_row["base_persona"], ensure_ascii=False),
        STYLE_JSON=json.dumps(user_row["style"], ensure_ascii=False),
        SEED_JSON=json.dumps(seed_row, ensure_ascii=False),
    )
    obj = generate_json(model, tok, prompt, max_new_tokens=220, temperature=temperature)
    return obj["init_user_message"].strip()


def generate_conversation(
    user_model, user_tok,
    assistant_model, assistant_tok,
    prompts_cfg: Dict,
    style_profiles: Dict[str, Dict],
    user_row: Optional[Dict],
    seed_row: Dict,
    condition: str,
    num_user_turns: int,
    user_temp: float,
    assistant_temp: float,
    user_max_new_tokens: int,
    assistant_max_new_tokens: int,
) -> Tuple[Dict, List[Dict]]:
    topic = seed_row["topic"]
    scenario = seed_row["scenario"]
    goal = seed_row["conversation_goal"]
    state = init_state(seed_row)
    messages: List[Dict[str, str]] = []
    inv_meta: List[Dict] = []

    if condition == "baseline_no_persona":
        first_user_prompt = f"""
Write the FIRST user message in a conversation.

Scenario:
{scenario}

Conversation goal:
{goal}

Constraints:
- 1 to 3 sentences.
- Natural and realistic.
- No greetings or sign-offs.
- Do not mention metadata or act self-consciously.

Return ONLY valid JSON:
{{{{"init_user_message":"..."}}}}
""".strip()
        first_user_message = generate_json(user_model, user_tok, first_user_prompt)["init_user_message"].strip()
        user_system_llm1 = None
        persona_text = None
        user_id = None
    else:
        first_user_message = generate_first_user_message(
            user_model, user_tok, prompts_cfg, user_row, seed_row, user_temp
        )
        user_system_llm1 = user_row["system_llm1"]
        persona_text = user_row["system_llm1"]
        user_id = user_row["user_id"]

    messages.append({"role": "user", "content": first_user_message})

    assistant_system = build_assistant_system_prompt(prompts_cfg, condition, style_profiles)
    assistant_messages = [
        {
            "role": "system",
            "content": assistant_system + f"\n\nConversation topic: {topic}\nConversation goal: {goal}",
        }
    ] + messages

    assistant_raw = generate_text(
        assistant_model,
        assistant_tok,
        assistant_messages,
        assistant_max_new_tokens,
        assistant_temp,
    )
    assistant_clean, inv = parse_assistant(assistant_raw) if "investigator" in condition else (assistant_raw, None)
    messages.append({"role": "assistant", "content": assistant_clean})

    if inv is not None:
        inv_meta.append({"turn_idx": 1, **inv})

    for turn_idx in range(2, num_user_turns + 1):
        recent_history = format_recent_history(messages, max_messages=8)
        state_summary = render_state_summary(state)

        if condition == "baseline_no_persona":
            user_prompt = f"""
You are simulating a user in a conversation.

Conversation topic:
{topic}

Scenario:
{scenario}

Conversation goal:
{goal}

Conversation state:
{state_summary}

Recent dialogue:
{recent_history}

Write ONLY the user's next message.

Constraints:
- Continue the conversation naturally.
- Stay coherent with what has already been said.
- 1 to 3 sentences, max 80 words.
- Do not write the assistant's response.
""".strip()
            user_messages = [{"role": "user", "content": user_prompt}]
        else:
            user_prompt = f"""
Stable user profile:
{user_system_llm1}

Conversation topic:
{topic}

Scenario:
{scenario}

Conversation goal:
{goal}

Conversation state:
{state_summary}

Recent dialogue:
{recent_history}

Write ONLY the user's next message.

Constraints:
- Stay consistent with the stable profile.
- Respond naturally to the assistant.
- Continue the conversation rather than ending it too early.
- Reveal preferences or constraints naturally when relevant.
- 1 to 3 sentences, max 80 words.
- Do not write the assistant's response.
""".strip()
            user_messages = [{"role": "user", "content": user_prompt}]

        user_reply = generate_text(
            user_model,
            user_tok,
            user_messages,
            user_max_new_tokens,
            user_temp,
        ).strip()
        messages.append({"role": "user", "content": user_reply})

        assistant_messages = [
            {
                "role": "system",
                "content": assistant_system + f"\n\nConversation topic: {topic}\nConversation goal: {goal}",
            }
        ] + messages

        assistant_raw = generate_text(
            assistant_model,
            assistant_tok,
            assistant_messages,
            assistant_max_new_tokens,
            assistant_temp,
        )
        assistant_clean, inv = parse_assistant(assistant_raw) if "investigator" in condition else (assistant_raw, None)
        messages.append({"role": "assistant", "content": assistant_clean})

        if inv is not None:
            inv_meta.append({"turn_idx": turn_idx, **inv})

        update_state_heuristic(state, user_reply, assistant_clean)

    conv = {
        "conversation_id": str(uuid.uuid4()),
        "condition": condition,
        "user_id": user_id,
        "base_persona_id": None if user_row is None else user_row["base_persona_id"],
        "style_id": None if user_row is None else user_row["style_id"],
        "seed_id": seed_row["seed_id"],
        "topic": topic,
        "scenario": scenario,
        "conversation_goal": goal,
        "num_user_turns": num_user_turns,
        "persona_text": persona_text,
        "profile": None if user_row is None else {
            "base_persona": user_row["base_persona"],
            "style": user_row["style"],
        },
        "messages": messages,
    }
    return conv, inv_meta


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--users_path", type=str, required=True)
    p.add_argument("--seeds_path", type=str, required=True)
    p.add_argument("--prompts_config", type=str, required=True)
    p.add_argument("--persona_config", type=str, required=True)
    p.add_argument("--user_model", type=str, default=USER_MODEL_NAME)
    p.add_argument("--assistant_model", type=str, default=ASSISTANT_MODEL_NAME)
    p.add_argument(
        "--condition",
        type=str,
        default="with_persona",
        choices=[
            "baseline_no_persona",
            "with_persona",
            "with_persona_investigator_guided",
            "with_persona_investigator_unguided",
        ],
    )
    p.add_argument("--users_limit", type=int, default=None)
    p.add_argument("--conversations_per_user", type=int, default=3)
    p.add_argument("--num_user_turns", type=int, default=20)
    p.add_argument("--user_max_new_tokens", type=int, default=120)
    p.add_argument("--assistant_max_new_tokens", type=int, default=220)
    p.add_argument("--user_temp", type=float, default=0.8)
    p.add_argument("--assistant_temp", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle_users", action="store_true")
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--inv_output_path", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    prompts_cfg = load_yaml(Path(args.prompts_config))
    persona_cfg = load_yaml(Path(args.persona_config))
    style_profiles = persona_cfg["profiles"]["style_id"]

    seeds = read_jsonl(Path(args.seeds_path))
    users = [] if args.condition == "baseline_no_persona" else read_jsonl(Path(args.users_path))

    if args.users_limit is not None and users:
        users = users[: args.users_limit]
    if args.shuffle_users and users:
        random.shuffle(users)

    user_model, user_tok = load_model(args.user_model)
    assistant_model, assistant_tok = load_model(args.assistant_model)

    out_path = Path(args.output_path)
    inv_path = Path(args.inv_output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    inv_path.parent.mkdir(parents=True, exist_ok=True)

    num_written = 0
    num_inv = 0

    with out_path.open("w", encoding="utf-8") as f_out, inv_path.open("w", encoding="utf-8") as f_inv:
        if args.condition == "baseline_no_persona":
            chosen_seeds = seeds[: args.conversations_per_user]
            for seed_row in chosen_seeds:
                conv, inv_meta = generate_conversation(
                    user_model, user_tok,
                    assistant_model, assistant_tok,
                    prompts_cfg, style_profiles,
                    None, seed_row, args.condition,
                    args.num_user_turns, args.user_temp, args.assistant_temp,
                    args.user_max_new_tokens, args.assistant_max_new_tokens,
                )
                f_out.write(json.dumps(conv, ensure_ascii=False) + "\n")
                num_written += 1
        else:
            for user_idx, user_row in enumerate(users):
                chosen_seeds = random.sample(seeds, k=min(args.conversations_per_user, len(seeds)))
                for rep_idx, seed_row in enumerate(chosen_seeds):
                    t0 = time.time()
                    conv, inv_meta = generate_conversation(
                        user_model, user_tok,
                        assistant_model, assistant_tok,
                        prompts_cfg, style_profiles,
                        user_row, seed_row, args.condition,
                        args.num_user_turns, args.user_temp, args.assistant_temp,
                        args.user_max_new_tokens, args.assistant_max_new_tokens,
                    )
                    conv["user_model_name"] = args.user_model
                    conv["assistant_model_name"] = args.assistant_model
                    conv["user_index"] = user_idx
                    conv["replicate_index"] = rep_idx
                    f_out.write(json.dumps(conv, ensure_ascii=False) + "\n")
                    num_written += 1

                    for rec in inv_meta:
                        rec_out = {
                            "conversation_id": conv["conversation_id"],
                            "user_id": conv["user_id"],
                            "base_persona_id": conv["base_persona_id"],
                            "style_id": conv["style_id"],
                            "seed_id": conv["seed_id"],
                            **rec,
                        }
                        f_inv.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                        num_inv += 1

                    print(
                        f"[DONE] user={conv['user_id']} seed={conv['seed_id']} "
                        f"turns={args.num_user_turns} elapsed={time.time()-t0:.1f}s",
                        flush=True,
                    )

    print(f"Wrote {num_written} conversations -> {out_path}")
    print(f"Wrote {num_inv} investigation records -> {inv_path}")


if __name__ == "__main__":
    main()