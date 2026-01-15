from __future__ import annotations

import csv
import time
import argparse
from pathlib import Path

import torch
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
    return ap.parse_args()

def main():

    args = parse_args()
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = PROMPTS  # fallback
    
    if args.prompts_file is not None:
        prompts = [ln.strip() for ln in Path(args.prompts_file).read_text().splitlines() if ln.strip()]

    if args.dataset is not None:
        from datasets import load_dataset
        ds = load_dataset(args.dataset, split=args.split)
        prompts = [x for x in ds[args.text_field]]
        if args.max_examples is not None:
            prompts = prompts[:args.max_examples]

    if args.num_prompts is not None:
        prompts = prompts[: args.num_prompts]

    #comment out after sanity check
    #print("Num prompts:", len(prompts))
    #print("First prompt preview:", prompts[0][:120])

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
            "prompt",
            "continuation_text",
            "token_idx", "token_id", "token_str",
            "p_a", "p_b",
            "ratio_pA_over_pB",
        ])
        for seed in args.seeds:
            set_seed(seed)

            for pid, prompt in enumerate(prompts):
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

                cont_ids = gen[0, input_ids.shape[1]:]
                continuation_text = tokenizer.decode(cont_ids, skip_special_tokens=True)

                print("=" * 80)
                print(f"Prompt {pid}:", prompt)
                print("Continuation (A):", continuation_text)

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
                        pid, 
                        prompt,
                        continuation_text,
                        s.idx, s.token_id, s.token_str,
                        s.p_a, s.p_b,
                        s.ratio_pA_over_pB,
                    ])

    print("\nSaved:", out_csv, flush=True)


if __name__ == "__main__":
    main()
