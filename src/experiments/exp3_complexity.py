from __future__ import annotations

import csv
import time
import argparse
import random
import re
import time
from pathlib import Path

#import torch
import numpy as np
from datasets import load_dataset
from transformers import set_seed

from .model_loading import load_two_models_same_family
from .logprob_utils import score_continuation_tokens


PROMPTS = [
    "The role of large language models in modern society is",
    "In machine learning, a surprising property of neural networks is",
    "A simple explanation of why reinforcement learning is hard is",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_a", required=True, help="Generator / Model A")
    ap.add_argument("--model_b", required=True, help="Comparison / Model B")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max_new_tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--out_dir",
    default="/Data/allegra-maria-pia.boustany/llm_cross_eval/results",
    help="Where to write CSV outputs",
    ) #avoid hitting disk quota, change as needed
    ap.add_argument("--prompts_file", type=str, default=None, help="Path to a .txt file with one prompt per line") # use to override prompts/to test for non reasoning models 
    ap.add_argument("--num_prompts", type=int, default=None,
                help="If set, only use first N prompts")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                help="Random seeds for generation (to get multiple samples per prompt)")
    ap.add_argument("--dataset", type=str, default=None, help='HF dataset name, e.g. "math-ai/aime25"')
    ap.add_argument("--split", type=str, default="test", help='Dataset split, e.g. "test"')
    ap.add_argument("--text_field", type=str, default="problem", help="Column to use as prompt text")
    ap.add_argument("--max_examples", type=int, default=None, help="Limit number of dataset examples")
    ap.add_argument("--do_sample", action="store_true", help="If set, sample decoding (needed for multiple samples)")
    ap.add_argument("--math_by_level", action="store_true",
                help="If set, sample from MATH by level and attach bucket/level metadata")
    ap.add_argument("--n_per_level", type=int, default=3,
                help="How many examples to sample per MATH level (1..5)")
    ap.add_argument("--n_per_bucket", type=int, default=None, help="If set, sample this many examples per difficulty bucket (easy/medium/hard) from MATH")
    return ap.parse_args()

def parse_level(level_field) -> int:
    """
    MATH 'level' is often like 'Level 3'. Make it robust.
    """
    if isinstance(level_field, int):
        return level_field
    m = re.search(r"(\d+)", str(level_field))
    return int(m.group(1)) if m else -1

def bucket_from_level(level: int) -> str:
    if level in (1, 2):
        return "easy"
    if level in (3, 4):
        return "medium"
    if level == 5:
        return "hard"
    return "unknown"

def sample_math_prompts(dataset_name: str, split: str, n_per_level: int, n_per_bucket: int, seed: int):
    
    """
    If n_per_bucket is set, sample balanced buckets: 
    easy (l1-2), medium (l3-4), hard (l5).
    Otherwise (default), sample n_per_level from each level 1..5.
    """
    ds = load_dataset(dataset_name, split=split)
    rng = random.Random(seed)

    # group indices by level
    by_level = {1: [], 2: [], 3: [], 4: [], 5: []}
    for i, ex in enumerate(ds):
        lvl = parse_level(ex.get("level", ex.get("difficulty", ex.get("lvl", None))))
        if lvl in by_level:
            by_level[lvl].append(i)

    if n_per_bucket is not None: 
        n = n_per_bucket
        n12 = n//2
        n34 = n//2
        per_level = {1: n12, 2: n - n12, 3: n34, 4: n - n34, 5: n} #in case of odd n 
    else:
        per_level = {lvl: n_per_level for lvl in [1, 2, 3, 4, 5]}

    samples = []
    for lvl in [1, 2, 3, 4, 5]:
        need = per_level[lvl]
        idxs = by_level[lvl]
        if len(idxs) < need:
            raise ValueError(f"Not enough items for level {lvl}: have {len(idxs)}, need {need}")
        chosen = rng.sample(idxs, need)
        for j in chosen:
            ex = ds[j]
            samples.append({
                "id": ex.get("unique_id", ex.get("id", f"{split}_{j}")),
                "level": lvl,
                "bucket": bucket_from_level(lvl),
                "problem": ex["problem"],
                "answer": ex.get("answer", ex.get("final_answer", ex.get("solution", ""))),
            })
    return samples

def truncate_at_final_answer(text: str) -> str:
    key = "Final answer:"
    i = text.find(key)
    if i == -1:
        return text
    # keep up to end of that line
    j = text.find("\n", i)
    return text[: j if j != -1 else len(text)]

def main():

    args = parse_args()
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = [{"id": i, "prompt": p, "level": None, "bucket": None} for i, p in enumerate(PROMPTS)] # fallback
    
    if args.prompts_file is not None:
        prompts = [ln.strip() for ln in Path(args.prompts_file).read_text().splitlines() if ln.strip()]
        examples = [{"id": i, "prompt": p, "level": None, "bucket": None} for i, p in enumerate(file_prompts)]

    if args.dataset is not None:
        # sample MATH by level
        if args.math_by_level:
            sampled = sample_math_prompts(
                dataset_name=args.dataset,
                split=args.split,
                n_per_level=args.n_per_level if args.n_per_bucket is None else 0,
                n_per_bucket=args.n_per_bucket,
                seed=args.seeds[0],  # deterministic sample given first seed
            )
            examples = [{"id": ex["id"], "prompt": ex["problem"], "level": ex["level"], "bucket": ex["bucket"]} for ex in sampled]
        else:
            ds = load_dataset(args.dataset, split=args.split)
            prompts = list(ds[args.text_field])
            if args.max_examples is not None:
                prompts = prompts[:args.max_examples]
            examples = [{"id": i, "prompt": p, "level": None, "bucket": None} for i, p in enumerate(prompts)]

    if args.num_prompts is not None:
        examples = examples[: args.num_prompts]

    #debugging 
    print("Num examples:", len(examples))
    print("First prompt preview:", examples[0]["prompt"][:120])
    print("First meta:", {"id": examples[0]["id"], "level": examples[0]["level"], "bucket": examples[0]["bucket"]})


    tokenizer, model_a, model_b = load_two_models_same_family(
        model_name_1=args.model_a,
        model_name_2=args.model_b,
        device=device,
        dtype=args.dtype,
    )
    print("Loaded models:")
    print("  A:", model_a.name_or_path)
    print("  B:", model_b.name_or_path)

    # Reproducibility for generation
    # set_seed(42)

    safe_a = args.model_a.split("/")[-1].replace(".", "_")
    safe_b = args.model_b.split("/")[-1].replace(".", "_")
    ts = int(time.time())

    out_csv = out_dir / f"exp1_turn1_{safe_a}_vs_{safe_b}_{ts}.csv"

    print("Saving results to:", out_csv, flush=True)


    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        
        w.writerow([
            "seed",
            "example_id",
            "bucket",
            "level",
            "prompt",
            "continuation_text",
            "token_idx", "token_id", "token_str",
            "p_a", "p_b",
            "ratio_pA_over_pB",
        ])
        for seed in args.seeds:
            set_seed(seed)

            for ex_i, ex in enumerate(examples):
                problem_text = ex["prompt"]

                #debug
                print(problem_text[:400])

                prompt = (
                    "Solve the following math problem. Show your work.\n"
                    "End with a line of the form: Final answer: <your answer>\n\n"
                    f"Problem:\n{problem_text}\n\nSolution:\n"
                )
                enc = tokenizer(prompt, return_tensors="pt")
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)

                # Generate 1 continuation from model A (turn-1 only)
                gen = model_a.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample, 
                    temperature=args.temperature if args.do_sample else None,
                    top_p=args.top_p if args.do_sample else None,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1 # prompt0 repeated same sentence over and over without this
                )
                
                # truncate final answer
                
                cont_ids = gen[0, input_ids.shape[1]:]
                continuation_text = tokenizer.decode(cont_ids, skip_special_tokens=True)

                print("=" * 80)
                print(f"Prompt {id}:", prompt)
                #print("Continuation (A):", continuation_text)

                scores = score_continuation_tokens(
                    tokenizer=tokenizer,
                    model_a=model_a,
                    model_b=model_b,
                    prompt=prompt,
                    continuation=continuation_text,
                    device=device,
                )

                for s in scores:
                    w.writerow([
                        seed,
                        ex["id"], 
                        ex["bucket"],
                        ex["level"],
                        prompt,
                        continuation_text,
                        s.idx, s.token_id, s.token_str,
                        s.p_a, s.p_b,
                        s.ratio_pA_over_pB,
                    ])

    print("\nSaved:", out_csv, flush=True)


if __name__ == "__main__":
    main()
