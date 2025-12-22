from __future__ import annotations

import csv
import time
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

OUT_DIR = Path("results")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cpu"

    tokenizer, model_a, model_b = load_two_models_same_family(device=device)
    print("Loaded models:")
    print("  A:", model_a.name_or_path)
    print("  B:", model_b.name_or_path)

    # Reproducibility for generation
    set_seed(42)

    ts = int(time.time())
    out_csv = OUT_DIR / f"exp1_turn1_{ts}.csv"

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        """
        w.writerow([
            "prompt_id", "prompt",
            "continuation_text",
            "token_idx", "token_id", "token_str",
            "p_a", "p_b",
            "logp_a", "logp_b",
            "delta_logp",
            "ratio_pA_over_pB",
            "log10_ratio",
        ])
        """
        w.writerow([
            "prompt_id", "prompt",
            "continuation_text",
            "token_idx", "token_id", "token_str",
            "p_a", "p_b",
            "ratio_pA_over_pB",
        ])

        for pid, prompt in enumerate(PROMPTS):
            enc = tokenizer(prompt, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            # Generate 1 continuation from model A (turn-1 only)
            gen = model_a.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
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
                    pid, prompt,
                    continuation_text,
                    s.idx, s.token_id, s.token_str,
                    s.p_a, s.p_b,
                    s.ratio_pA_over_pB,
                ])

    print("\nSaved:", out_csv)


if __name__ == "__main__":
    main()
