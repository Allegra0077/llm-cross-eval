from __future__ import annotations

import csv
import time
import re
from pathlib import Path

import torch

from .model_loading import load_two_models_same_family
from .logprob_utils import score_full_text_next_tokens


DATA_PATH = Path("data/canonical.txt")
OUT_DIR = Path("results")

def clean_pdf_text(text: str) -> str:
    """
    Clean text extracted from PDFs:
    - remove hyphenation at line breaks
    - collapse line breaks into spaces
    - normalize whitespace
    """
    # Remove hyphenation across line breaks: "Qualifi-\ncations" -> "Qualifications"
    text = re.sub(r"-\n", "", text)

    # Replace remaining newlines with spaces
    text = re.sub(r"\n", " ", text)

    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def main():
    raw_text = DATA_PATH.read_text(encoding="utf-8")
    text = clean_pdf_text(raw_text)

    device = "cpu"

    tokenizer, model_a, model_b = load_two_models_same_family(device=device)
    print("Loaded models:")
    print("  A:", model_a.name_or_path)
    print("  B:", model_b.name_or_path)

    # Score every next-token in the canonical text
    scores = score_full_text_next_tokens(
        tokenizer=tokenizer,
        model_a=model_a,
        model_b=model_b,
        text=text,
        device=device,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_csv = OUT_DIR / f"exp2_canonical_{ts}.csv"

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model_a", "model_b",
            "token_idx",
            "token_id", "token_str",
            "p_a", "p_b",
            "ratio_pA_over_pB",
        ])

        for s in scores:
            w.writerow([
                model_a.name_or_path, model_b.name_or_path,
                s.idx,
                s.token_id, s.token_str,
                s.p_a, s.p_b,
                s.ratio_pA_over_pB,
            ])

    print("Saved:", out_csv)


if __name__ == "__main__":
    main()

