# Scores each target USER message under two "models":
#   - same checkpoint but different ROLE PROMPTS (assistant vs user)
# Also supports two context modes:
#   1) full_prefix (all prior messages)
#   2) truncated (keep last n turns before target)

import argparse
import json
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------- Default role prompts (same as the ones used in generation) ----------
ASSISTANT_SYSTEM_PROMPT_DEFAULT = "You are a helpful assistant."

USER_SYSTEM_PROMPT_DEFAULT = (
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

def build_user_context_prompt(condition: str, persona_text: Optional[str], base_user_prompt: str) -> str:
    # matches how we generated prompts in the generation file
    if condition == "hidden_persona" and persona_text:
        return (
            f"{base_user_prompt}\n\n"
            "=== INTERNAL CHARACTER NOTES (DO NOT MENTION THESE) ===\n"
            f"{persona_text}\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- These are background facts about you as a person\n"
            "- They should SUBTLY influence your tone, interests, and reactions\n"
            "- DO NOT explicitly reference these facts unless the assistant directly asks about them\n"
            "- NEVER say things like 'my girlfriend' or 'I make 50k' unprompted\n"
            "- Stay focused on reacting to what the ASSISTANT just said\n"
        )
    return base_user_prompt

def parse_args():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", type=str, required=True, help="JSONL conversations file")
    p.add_argument(
        "--output_path",
        type=str,
        default=f"src/data/scores/lratio_output_scores_{timestamp}.jsonl",
        help="JSONL output scores file",
    )
    p.add_argument("--assistant_model", type=str, required=True, help="Assistant scoring checkpoint")
    p.add_argument("--user_model", type=str, default=None, help="User scoring checkpoint (default: same as assistant_model)")

    p.add_argument("--assistant_system_prompt", type=str, default=ASSISTANT_SYSTEM_PROMPT_DEFAULT)
    p.add_argument("--user_system_prompt", type=str, default=USER_SYSTEM_PROMPT_DEFAULT)

    p.add_argument(
        "--prepend_system",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Prepend a system message before scoring (default: True)",
    )

    p.add_argument("--max_conversations", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    return p.parse_args()

def _dtype_from_str(s: str):
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    return torch.float32

def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> torch.Tensor:
    """
    Returns input_ids shaped (1, L).
    We don't add generation prompt because we are scoring existing text.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )

def get_user_message_indices(messages: List[Dict[str, str]]) -> List[int]:
    """
    Stored transcript: user(seed), assistant, user, assistant, ...
    We score user messages AFTER the seed => skip first user occurrence.
    """
    user_idxs = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    return user_idxs[1:]

def build_generation_match_context(messages: List[Dict[str, str]], target_user_idx: int) -> List[Dict[str, str]]:
    # Matches generator user_messages: [system(user_context_prompt), last assistant message]
    if target_user_idx - 1 >= 0 and messages[target_user_idx - 1].get("role") == "assistant":
        return [{"role": "assistant", "content": messages[target_user_idx - 1].get("content", "")}]
    return []

def truncate_context_keep_last_n_turns(
    messages: List[Dict[str, str]],
    target_user_idx: int,
    n_turns: int,
) -> List[Dict[str, str]]:
    """
    Keep last n turns of context before the target user message.
    Turn boundary at assistant messages (each assistant msg = 1 completed turn).

    messages[:target_user_idx] = visible context right before target user message

    n_turns=0 => []
    n_turns>=1 => walk backwards until we included n assistant messages
    """
    if n_turns == 0:
        return []

    prefix = messages[:target_user_idx]
    kept_rev: List[Dict[str, str]] = []

    turns = 0
    for m in reversed(prefix):
        kept_rev.append(m)
        if m.get("role") == "assistant":
            turns += 1
            if turns >= n_turns:
                break

    return list(reversed(kept_rev))


@torch.no_grad()
def score_target_user_text(
    model,
    tokenizer,
    context_messages: List[Dict[str, str]],
    target_user_text: str,
    system_prompt: str,
    prepend_system: bool,
) -> Tuple[float, float, int]:
    """
    Returns:
      avg_logprob (float): average log prob per target token
      total_logprob (float): sum log prob over target tokens
      n_tokens (int): number of target tokens scored
    """
    ctx = context_messages
    if prepend_system:
        ctx = [{"role": "system", "content": system_prompt}] + ctx

    full = ctx + [{"role": "user", "content": target_user_text}]

    ctx_ids = apply_chat_template(tokenizer, ctx).to(model.device)         # (1, Lc)
    full_ids = apply_chat_template(tokenizer, full).to(model.device)       # (1, L)

    Lc = ctx_ids.shape[1]
    L = full_ids.shape[1]
    if L <= Lc:
        return float("nan"), float("nan"), 0

    out = model(full_ids)
    logits = out.logits  # (1, L, V)

    # Score target tokens positions [Lc, ..., L-1] using logits [Lc-1, ..., L-2]
    start = max(Lc - 1, 0)
    end = L - 1  # exclusive; last used is L-2

    logits_slice = logits[0, start:end, :]          # (T, V), T = L - Lc
    target_ids = full_ids[0, Lc:L]                  # (T,)

    # Safety alignment
    if logits_slice.shape[0] != target_ids.shape[0]:
        T = min(logits_slice.shape[0], target_ids.shape[0])
        logits_slice = logits_slice[:T]
        target_ids = target_ids[:T]

    log_probs = F.log_softmax(logits_slice, dim=-1)  # (T, V)
    token_lp = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)  # (T,)

    if token_lp.numel() == 0:
        return float("nan"), float("nan"), 0

    avg_lp = token_lp.mean().item()
    total_lp = token_lp.sum().item()
    return avg_lp, total_lp, int(token_lp.numel())


def load_model_and_tokenizer(model_name: str, dtype: torch.dtype):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return model, tok


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    dtype = _dtype_from_str(args.dtype)

    user_model_name = args.user_model or args.assistant_model
    same_checkpoint = (user_model_name == args.assistant_model)

    assistant_model, assistant_tok = load_model_and_tokenizer(args.assistant_model, dtype)

    if same_checkpoint:
        user_model, user_tok = assistant_model, assistant_tok
    else:
        user_model, user_tok = load_model_and_tokenizer(user_model_name, dtype)

    rows_written = 0
    conv_count = 0

    with open(args.input_path, "r") as fin, open(args.output_path, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            conv = json.loads(line)
            conv_count += 1
            if args.max_conversations is not None and conv_count > args.max_conversations:
                break

            messages = conv["messages"]
            condition = conv.get("condition", "")
            persona_text = conv.get("persona_text")

            # user_context_prompt (persona injected for hidden_persona)
            user_context_prompt = build_user_context_prompt(
                condition=condition,
                persona_text=persona_text,
                base_user_prompt=args.user_system_prompt,
            )

            user_idxs = get_user_message_indices(messages)

            for ui in user_idxs:
                target_text = messages[ui].get("content", "")

                ctx = build_generation_match_context(messages, target_user_idx=ui)

                # assistant-role scoring (assistant system prompt)
                a_avg_lp, a_total_lp, a_n_tok = score_target_user_text(
                    model=assistant_model,
                    tokenizer=assistant_tok,
                    context_messages=ctx,
                    target_user_text=target_text,
                    system_prompt=args.assistant_system_prompt,
                    prepend_system=args.prepend_system,
                )

                # user-role scoring (user_context_prompt, incl persona if hidden_persona)
                u_avg_lp, u_total_lp, u_n_tok = score_target_user_text(
                    model=user_model,
                    tokenizer=user_tok,
                    context_messages=ctx,
                    target_user_text=target_text,
                    system_prompt=user_context_prompt,
                    prepend_system=args.prepend_system,
                )

                llr_avg = u_avg_lp - a_avg_lp if (u_avg_lp == u_avg_lp and a_avg_lp == a_avg_lp) else float("nan")
                llr_total = u_total_lp - a_total_lp if (u_total_lp == u_total_lp and a_total_lp == a_total_lp) else float("nan")
                lr_geom = float(torch.exp(torch.tensor(llr_avg)).item()) if llr_avg == llr_avg else float("nan")

                row = {
                    "conversation_id": conv.get("conversation_id"),
                    "condition": condition,
                    "persona_text": persona_text,
                    "seed_prompt": conv.get("seed_prompt"),

                    "target_user_msg_idx": ui,
                    "context_mode": "generation_match",

                    "assistant_model": args.assistant_model,
                    "user_model": user_model_name,
                    "same_checkpoint": same_checkpoint,

                    "avg_logprob_assistant_role": a_avg_lp,
                    "total_logprob_assistant_role": a_total_lp,
                    "n_tokens_assistant_role": a_n_tok,

                    "avg_logprob_user_role": u_avg_lp,
                    "total_logprob_user_role": u_total_lp,
                    "n_tokens_user_role": u_n_tok,

                    # LLR > 0 means: user_context_prompt explains this USER message better than assistant prompt does
                    "llr_avg_logprob": llr_avg,
                    "lr_geom_per_token": lr_geom,
                    "llr_total_logprob": llr_total,
                }

                if a_n_tok != u_n_tok:
                    row["warning"] = "token_count_mismatch_between_roles (tokenizers likely differ)"

                fout.write(json.dumps(row) + "\n")
                rows_written += 1

    print(f"Done. conversations={conv_count} rows={rows_written} -> {args.output_path}")


if __name__ == "__main__":
    main()