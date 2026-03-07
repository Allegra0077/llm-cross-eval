import argparse
import json
import os
from typing import Dict, List, Tuple
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


FALLBACK_SYSTEM_PROMPT = "You are a helpful assistant."


def parse_args():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=f"src/data/scores/stitched_scores_{timestamp}.jsonl")
    parser.add_argument("--assistant_model", type=str, required=True)

    parser.add_argument("--n_turns_list", type=int, nargs="+", default=[0, 1, 2, 3, 4, 6])
    parser.add_argument("--prepend_system", action="store_true")

    parser.add_argument("--max_conversations", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def _dtype_from_str(s: str):
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    return torch.float32


def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> torch.Tensor:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )


def get_scored_user_message_indices(messages: List[Dict[str, str]], seed_user_msg_indices: List[int]) -> List[int]:
    """
    Return user message indices excluding all seed indices.
    Backward compatible:
      - if seed_user_msg_indices missing, default to skipping the first user message (idx 0 if it's user).
    """
    seed_set = set(seed_user_msg_indices or [])
    user_idxs = [i for i, m in enumerate(messages) if m["role"] == "user"]
    return [i for i in user_idxs if i not in seed_set]


def truncate_context_keep_last_n_turns(messages: List[Dict[str, str]], target_user_idx: int, n_turns: int) -> List[Dict[str, str]]:
    if n_turns == 0:
        return []

    prefix = messages[:target_user_idx]
    kept_rev: List[Dict[str, str]] = []

    turns = 0
    for m in reversed(prefix):
        kept_rev.append(m)
        if m["role"] == "assistant":
            turns += 1
            if turns >= n_turns:
                break

    return list(reversed(kept_rev))


@torch.no_grad()
def score_target_user_text(
    model,
    tokenizer,
    assistant_system_prompt: str,
    context_messages: List[Dict[str, str]],
    target_user_text: str,
    prepend_system: bool,
) -> Tuple[float, int]:

    ctx = context_messages
    if prepend_system:
        ctx = [{"role": "system", "content": assistant_system_prompt or FALLBACK_SYSTEM_PROMPT}] + ctx

    full = ctx + [{"role": "user", "content": target_user_text}]

    ctx_ids = apply_chat_template(tokenizer, ctx).to(model.device)
    full_ids = apply_chat_template(tokenizer, full).to(model.device)

    Lc = ctx_ids.shape[1]
    L = full_ids.shape[1]
    if L <= Lc:
        return float("nan"), 0

    out = model(full_ids)
    logits = out.logits  # (1, L, V)

    start = max(Lc - 1, 0)
    end = L - 1  # exclusive

    logits_slice = logits[0, start:end, :]
    target_ids = full_ids[0, Lc:L]

    if logits_slice.shape[0] != target_ids.shape[0]:
        T = min(logits_slice.shape[0], target_ids.shape[0])
        logits_slice = logits_slice[:T]
        target_ids = target_ids[:T]

    log_probs = F.log_softmax(logits_slice, dim=-1)
    token_lp = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)

    avg_lp = token_lp.mean().item() if token_lp.numel() > 0 else float("nan")
    return avg_lp, int(token_lp.numel())


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.assistant_model)
    dtype = _dtype_from_str(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.assistant_model,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()

    rows_written = 0
    conv_count = 0

    with open(args.input_path, "r", encoding="utf-8") as fin, open(args.output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            conv = json.loads(line)
            conv_count += 1
            if args.max_conversations is not None and conv_count > args.max_conversations:
                break

            messages = conv["messages"]

            # For stitched convs: skip BOTH seeds. For non-stitched convs: you can store [0] or omit.
            seed_user_msg_indices = conv.get("seed_user_msg_indices")
            if seed_user_msg_indices is None:
                # backward-compatible: skip the first user message in the transcript
                # (assumes transcript starts with a user seed)
                seed_user_msg_indices = [0]

            scored_user_idxs = get_scored_user_message_indices(messages, seed_user_msg_indices)
            assistant_system_prompt = conv.get("assistant_system_prompt", FALLBACK_SYSTEM_PROMPT)

            for ui in scored_user_idxs:
                target_text = messages[ui]["content"]

                for n_turns in args.n_turns_list:
                    ctx = truncate_context_keep_last_n_turns(messages, target_user_idx=ui, n_turns=n_turns)

                    avg_lp, n_tok = score_target_user_text(
                        model=model,
                        tokenizer=tok,
                        assistant_system_prompt=assistant_system_prompt,
                        context_messages=ctx,
                        target_user_text=target_text,
                        prepend_system=args.prepend_system,
                    )

                    row = {
                        "conversation_id": conv.get("conversation_id"),
                        "stitched": conv.get("stitched", False),
                        "num_segments": conv.get("num_segments"),
                        "num_turns_per_segment": conv.get("num_turns_per_segment"),

                        "condition": conv.get("condition"),
                        "persona_text": conv.get("persona_text"),
                        "seed_prompts": conv.get("seed_prompts"),
                        "seed_user_msg_indices": seed_user_msg_indices,

                        "target_user_msg_idx": ui,
                        "n_turns_context": n_turns,
                        "avg_logprob": avg_lp,
                        "n_tokens": n_tok,
                    }
                    fout.write(json.dumps(row) + "\n")
                    rows_written += 1

    print(f"Done. conversations={conv_count} rows={rows_written} -> {args.output_path}")


if __name__ == "__main__":
    main()
