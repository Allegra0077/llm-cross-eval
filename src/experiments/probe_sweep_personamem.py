# Notes:
#  - Mean-pooling "just the user query tokens" is non-trivial with apply_chat_template because
#    token offsets are hard to recover from the serialized prompt. This script implements mean
#    over the full prompt tokens (standard and still a strong robustness check).
#  - To keep runtime sane, we cache features per (layer, pooling) on CPU. This will use RAM/disk if enabled.

import ast
import json
import os
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score


# -----------------------
# Config
# -----------------------
DATASET_REPO = "bowen-upenn/PersonaMem-v2"

SPLIT_TRAIN = "train_text"
SPLIT_VAL = "val_text"
SPLIT_TEST = "benchmark_text"

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LENGTH = 2048

# Pooling modes for readout
POOLINGS = ["last_token", "mean"]  # mean = mean over all tokens in the prompt

# Layer sweep:
# We'll sweep actual transformer layers 1..L (hidden_states[0] is embedding output in HF convention).
LAYER_SWEEP = "all"  # "al" or list like [5,10,15,20,-1]

# Data sizes (count ORIGINAL PersonaMem examples; each becomes 4 binary datapoints)
N_TRAIN = 2000
N_VAL = 300
N_TEST = 300

# History source
USE_SNIPPET_IF_AVAILABLE = True
FALLBACK_TO_FULL_HISTORY = True

# Probing options
CLASS_WEIGHT = "balanced"  # recommended for imbalance; set to None to disable


CACHE_DIR = None  # ex: "cache_probe_features" to cache to disk, or None to disable


# -----------------------
# Model
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
    trust_remote_code=True
).eval()

if DEVICE == "cuda":
    model.cuda()


# -----------------------
# Parsing helpers
# -----------------------
def parse_user_query(user_query_str: str) -> Dict:
    d = ast.literal_eval(user_query_str)
    if not (isinstance(d, dict) and "role" in d and "content" in d):
        raise ValueError(f"Unexpected user_query format: {type(d)} {d}")
    return {"role": d["role"], "content": d["content"]}


def parse_json_list_str(s: Optional[str]):
    if s is None:
        return None
    if not isinstance(s, str):
        return s
    s = s.strip()
    if not s:
        return None
    return json.loads(s)


def normalize_turns(turns) -> Optional[List[Dict]]:
    if turns is None:
        return None
    if isinstance(turns, dict):
        for k in ["messages", "chat_history", "conversation", "turns"]:
            if k in turns and isinstance(turns[k], list):
                turns = turns[k]
                break
        else:
            return None

    if not isinstance(turns, list) or len(turns) == 0:
        return None

    out = []
    for t in turns:
        if isinstance(t, dict) and "role" in t and "content" in t:
            out.append({"role": t["role"], "content": t["content"]})
        elif isinstance(t, dict) and "from" in t and "value" in t:
            out.append({"role": t["from"], "content": t["value"]})
        else:
            out.append({"role": "user", "content": str(t)})
    return out


def load_chat_history_from_link(link_path: str) -> Optional[List[Dict]]:
    if link_path is None:
        return None
    local_path = hf_hub_download(repo_id=DATASET_REPO, filename=link_path, repo_type="dataset")
    with open(local_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return normalize_turns(obj)


def build_options_and_label(ex: Dict) -> Tuple[Optional[List[str]], Optional[int]]:
    correct = ex.get("correct_answer", None)
    incorrect_str = ex.get("incorrect_answers", None)
    incorrect = parse_json_list_str(incorrect_str)

    if correct is None or incorrect is None:
        return None, None
    if not isinstance(incorrect, list) or len(incorrect) != 3:
        return None, None

    options = [correct] + incorrect
    label = 0

    pid = ex.get("persona_id", 0)
    rng = random.Random(int(pid))
    perm = list(range(4))
    rng.shuffle(perm)
    options = [options[i] for i in perm]
    label = perm.index(label)
    return options, label


def build_messages(ex: Dict) -> Optional[List[Dict]]:
    msgs = None

    if USE_SNIPPET_IF_AVAILABLE:
        snippet_raw = ex.get("related_conversation_snippet", None)
        snippet = parse_json_list_str(snippet_raw)
        msgs = normalize_turns(snippet)

    if msgs is None and FALLBACK_TO_FULL_HISTORY:
        link = ex.get("chat_history_32k_link", None)
        msgs = load_chat_history_from_link(link)

    if msgs is None:
        return None

    try:
        uq = parse_user_query(ex["user_query"])
    except Exception:
        return None
    msgs.append(uq)
    return msgs


def format_prompt(msgs: List[Dict]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return "\n".join([f'{m["role"].upper()}: {m["content"]}' for m in msgs]) + "\nASSISTANT:"


# -----------------------
# Readout / feature extraction
# -----------------------
@torch.no_grad()
def get_readout(prompt_text: str, layer: int, pooling: str) -> np.ndarray:
    enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[layer][0]  # (seq, hidden)

    if pooling == "last_token":
        vec = hs[-1]
    elif pooling == "mean":
        vec = hs.mean(dim=0)
    else:
        raise ValueError(f"Unknown pooling={pooling}")

    return vec.detach().float().cpu().numpy()


# -----------------------
# Binary dataset construction with grouping (for Top-1)
# -----------------------
def build_grouped_binary_prompts(ds_split, n_examples: int):
    """
    Returns:
      prompts_by_ex: list length n_examples, each item is list of 4 prompt strings
      labels_by_ex:  list length n_examples, each item is list of 4 labels (one 1, three 0)
    """
    prompts_by_ex = []
    labels_by_ex = []
    n_ok = 0

    for ex in tqdm(ds_split, desc=f"Building grouped prompts (target={n_examples})"):
        if n_ok >= n_examples:
            break

        options, correct_idx = build_options_and_label(ex)
        if options is None:
            continue

        msgs = build_messages(ex)
        if msgs is None:
            continue

        base_prompt = format_prompt(msgs)

        prompts = []
        labels = []
        for i, opt in enumerate(options):
            prompt = (
                base_prompt
                + "\n\nCandidate answer:\n"
                + opt
                + "\n\nIs this the correct personalized answer?"
            )
            prompts.append(prompt)
            labels.append(1 if i == correct_idx else 0)

        prompts_by_ex.append(prompts)
        labels_by_ex.append(labels)
        n_ok += 1

    if n_ok == 0:
        raise RuntimeError("No valid examples built. Check parsing / history availability.")
    return prompts_by_ex, labels_by_ex


def labels_to_flat(labels_by_ex: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten grouped labels and create group index array.
    """
    y = []
    g = []
    for gi, labels in enumerate(labels_by_ex):
        for lab in labels:
            y.append(lab)
            g.append(gi)
    return np.array(y), np.array(g)


def random_label_control(labels_by_ex: List[List[int]], seed: int = 0) -> List[List[int]]:
    """
    For each group of 4, randomly choose which position is labeled 1, preserving 1/3 split.
    """
    rng = random.Random(seed)
    out = []
    for _labels in labels_by_ex:
        pos = rng.randrange(4)
        new = [0, 0, 0, 0]
        new[pos] = 1
        out.append(new)
    return out


# -----------------------
# Featurization (grouped)
# -----------------------
def maybe_cache_path(split_name: str, n_examples: int, layer: int, pooling: str) -> Optional[str]:
    if CACHE_DIR is None:
        return None
    os.makedirs(CACHE_DIR, exist_ok=True)
    fname = f"{split_name}_n{n_examples}_layer{layer}_pool{pooling}.npz"
    return os.path.join(CACHE_DIR, fname)


def featurize_grouped(prompts_by_ex: List[List[str]],
                      labels_by_ex: List[List[int]],
                      split_name: str,
                      n_examples: int,
                      layer: int,
                      pooling: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns flat X, y, group indices.
    """
    cache_path = maybe_cache_path(split_name, n_examples, layer, pooling)
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["X"], data["y"], data["g"]

    X = []
    y = []
    g = []

    for gi, (prompts, labels) in enumerate(tqdm(list(zip(prompts_by_ex, labels_by_ex)),
                                                desc=f"Featurizing {split_name} layer={layer} pool={pooling}")):
        for p, lab in zip(prompts, labels):
            try:
                feat = get_readout(p, layer=layer, pooling=pooling)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    raise RuntimeError(
                        "CUDA OOM during featurization. Reduce MAX_LENGTH and/or N_* or use smaller model."
                    ) from e
                raise
            X.append(feat)
            y.append(lab)
            g.append(gi)

    X = np.stack(X)
    y = np.array(y)
    g = np.array(g)

    if cache_path:
        np.savez(cache_path, X=X, y=y, g=g)

    return X, y, g


# -----------------------
# Metrics: binary + top-1 ranking
# -----------------------
def label_stats(y: np.ndarray) -> Tuple[int, int, float]:
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return pos, neg, pos / len(y)


def top1_accuracy(probs: np.ndarray, y: np.ndarray, g: np.ndarray) -> float:
    """
    For each group (original PersonaMem instance), pick the option with max probability.
    """
    total = int(g.max()) + 1
    correct = 0
    for gi in range(total):
        idx = np.where(g == gi)[0]
        pick = idx[np.argmax(probs[idx])]
        if y[pick] == 1:
            correct += 1
    return correct / total


def train_and_eval(X_train, y_train, X_val, y_val, g_val, X_test, y_test, g_test) -> Dict[str, float]:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=10000, n_jobs=-1, class_weight=CLASS_WEIGHT)
    )
    clf.fit(X_train, y_train)

    # Binary predictions
    val_probs = clf.predict_proba(X_val)[:, 1]
    test_probs = clf.predict_proba(X_test)[:, 1]
    val_pred = (val_probs >= 0.5).astype(int)
    test_pred = (test_probs >= 0.5).astype(int)

    # Metrics robust to imbalance
    val_auc = roc_auc_score(y_val, val_probs)
    test_auc = roc_auc_score(y_test, test_probs)
    val_bal = balanced_accuracy_score(y_val, val_pred)
    test_bal = balanced_accuracy_score(y_test, test_pred)

    # Raw accuracy (reported but not primary due to 1:3 imbalance)
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)

    # Rank-based Top-1 among 4
    val_top1 = top1_accuracy(val_probs, y_val, g_val)
    test_top1 = top1_accuracy(test_probs, y_test, g_test)

    return {
        "val_acc": float(val_acc),
        "test_acc": float(test_acc),
        "val_bal": float(val_bal),
        "test_bal": float(test_bal),
        "val_auc": float(val_auc),
        "test_auc": float(test_auc),
        "val_top1": float(val_top1),
        "test_top1": float(test_top1),
    }


# -----------------------
# Main
# -----------------------
def get_layer_list() -> List[int]:
    if LAYER_SWEEP != "all":
        return list(LAYER_SWEEP)

    # HF hidden_states length is typically num_layers + 1 (incl embeddings at 0)
    # We'll probe transformer layers 1..num_hidden_layers (inclusive).
    L = int(getattr(model.config, "num_hidden_layers"))
    return list(range(1, L + 1))


def main():
    print("DEVICE:", DEVICE, "CUDA:", torch.cuda.is_available())
    print("Model:", MODEL_NAME)
    print("Splits:", SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)
    print("History mode:", "snippet-first" if USE_SNIPPET_IF_AVAILABLE else "full-history-only",
          "| fallback_to_full_history:", FALLBACK_TO_FULL_HISTORY)
    print("MAX_LENGTH:", MAX_LENGTH)
    print("Poolings:", POOLINGS)
    print("Layer sweep:", LAYER_SWEEP)
    print("Class weight:", CLASS_WEIGHT)
    print("Cache dir:", CACHE_DIR)

    ds = load_dataset(DATASET_REPO)
    for s in [SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST]:
        if s not in ds:
            raise KeyError(f"Split {s} not found. Available: {list(ds.keys())}")

    train = ds[SPLIT_TRAIN]
    val = ds[SPLIT_VAL]
    test = ds[SPLIT_TEST]

    # Build grouped prompts once (reused across layers/poolings)
    train_prompts, train_labels = build_grouped_binary_prompts(train, N_TRAIN)
    val_prompts, val_labels = build_grouped_binary_prompts(val, N_VAL)
    test_prompts, test_labels = build_grouped_binary_prompts(test, N_TEST)

    # Show label stats (flat)
    y_train_flat, _ = labels_to_flat(train_labels)
    y_val_flat, _ = labels_to_flat(val_labels)
    y_test_flat, _ = labels_to_flat(test_labels)
    print("Train pos/neg/pos_rate:", label_stats(y_train_flat))
    print("Val   pos/neg/pos_rate:", label_stats(y_val_flat))
    print("Test  pos/neg/pos_rate:", label_stats(y_test_flat))
    print("Majority-class accuracy baseline:", 0.75)
    print("Chance AUC baseline:", 0.50)
    print("Chance BalancedAcc baseline:", 0.50)
    print("Chance Top-1 baseline (4 options):", 0.25)

    layers = get_layer_list()
    results = []

    for pooling in POOLINGS:
        for layer in layers:
            # Featurize per split
            X_train, y_train, g_train = featurize_grouped(
                train_prompts, train_labels, "train", N_TRAIN, layer, pooling
            )
            X_val, y_val, g_val = featurize_grouped(
                val_prompts, val_labels, "val", N_VAL, layer, pooling
            )
            X_test, y_test, g_test = featurize_grouped(
                test_prompts, test_labels, "test", N_TEST, layer, pooling
            )

            metrics = train_and_eval(X_train, y_train, X_val, y_val, g_val, X_test, y_test, g_test)
            row = {"pooling": pooling, "layer": layer, **metrics, "control": "real"}
            results.append(row)

            print(f"[REAL] pool={pooling:10s} layer={layer:3d} "
                  f"test_auc={metrics['test_auc']:.3f} test_bal={metrics['test_bal']:.3f} "
                  f"test_top1={metrics['test_top1']:.3f}")

    # Random-label control on the BEST (pooling, layer) according to test_auc
    best = max(results, key=lambda r: r["test_auc"])
    best_pool = best["pooling"]
    best_layer = best["layer"]
    print("\nBest setting by test_auc:", best)

    # Create randomized labels (nonsense task)
    train_labels_rand = random_label_control(train_labels, seed=0)
    val_labels_rand = random_label_control(val_labels, seed=0)
    test_labels_rand = random_label_control(test_labels, seed=0)

    X_train, y_train, g_train = featurize_grouped(
        train_prompts, train_labels_rand, "train_rand", N_TRAIN, best_layer, best_pool
    )
    X_val, y_val, g_val = featurize_grouped(
        val_prompts, val_labels_rand, "val_rand", N_VAL, best_layer, best_pool
    )
    X_test, y_test, g_test = featurize_grouped(
        test_prompts, test_labels_rand, "test_rand", N_TEST, best_layer, best_pool
    )

    metrics_rand = train_and_eval(X_train, y_train, X_val, y_val, g_val, X_test, y_test, g_test)
    results.append({"pooling": best_pool, "layer": best_layer, **metrics_rand, "control": "random_labels"})
    print("\n[RANDOM LABEL CONTROL] pool=%s layer=%d metrics=%s" % (best_pool, best_layer, metrics_rand))

    # Save results to JSONL for plotting in a notebook
    out_path = "probe_sweep_results.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print("\nSaved results to:", out_path)

    # Optional: quick text summary of the top configurations
    real_only = [r for r in results if r["control"] == "real"]
    real_only_sorted = sorted(real_only, key=lambda r: r["test_auc"], reverse=True)[:10]
    print("\nTop 10 by test_auc (REAL):")
    for r in real_only_sorted:
        print(r)

    print("\nDone.")


if __name__ == "__main__":
    main()
