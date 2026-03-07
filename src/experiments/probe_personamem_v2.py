# probe_personamem_v2_corrected.py
# End-to-end: load PersonaMem-v2, build prompts, extract hidden states, train 4-way linear probe.
# Robust to missing snippet fields; can use snippet-only (fast) or full-history files (slow).

import ast
import json
import random
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score


# -----------------------
# Config
# -----------------------
DATASET_REPO = "bowen-upenn/PersonaMem-v2"

# Choose text splits (simplest)
SPLIT_TRAIN = "train_text"
SPLIT_VAL = "val_text"
SPLIT_TEST = "benchmark_text"

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"   # swap to your assistant model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tokenization / extraction
MAX_LENGTH = 2048          # increase if you have room
READOUT = "last_token"     # "last_token" or "mean"
LAYER = -1                 # last layer; later sweep

# Data sizes for a first run
N_TRAIN = 2000
N_VAL = 300
N_TEST = 300

# History source
USE_SNIPPET_IF_AVAILABLE = True   # use related_conversation_snippet when present (fast)
FALLBACK_TO_FULL_HISTORY = True   # if snippet missing, download chat_history_32k_link


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


# -----------------------
# Parsing helpers
# -----------------------
def parse_user_query(user_query_str: str) -> Dict:
    # user_query is like "{'role': 'user', 'content': '...'}"
    d = ast.literal_eval(user_query_str)
    if not (isinstance(d, dict) and "role" in d and "content" in d):
        raise ValueError(f"Unexpected user_query format: {type(d)} {d}")
    return {"role": d["role"], "content": d["content"]}


def parse_json_list_str(s: Optional[str]):
    """Parse a field that is stored as a JSON-encoded string, guarding None/empty."""
    if s is None:
        return None
    if not isinstance(s, str):
        # Sometimes already parsed; accept list/dict
        return s
    s = s.strip()
    if not s:
        return None
    return json.loads(s)


def normalize_turns(turns) -> Optional[List[Dict]]:
    """Normalize a chat history object into list[{'role','content'}]."""
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
            # fallback
            out.append({"role": "user", "content": str(t)})
    return out


def load_chat_history_from_link(link_path: str) -> Optional[List[Dict]]:
    """
    Download the chat history JSON file referenced by chat_history_32k_link and parse it.
    """
    if link_path is None:
        return None
    local_path = hf_hub_download(repo_id=DATASET_REPO, filename=link_path, repo_type="dataset")
    with open(local_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return normalize_turns(obj)


def build_options_and_label(ex: Dict) -> Tuple[Optional[List[str]], Optional[int]]:
    """
    Options = correct_answer + incorrect_answers (3). We shuffle deterministically to avoid
    position bias and return the corresponding label index.
    """
    correct = ex.get("correct_answer", None)
    incorrect_str = ex.get("incorrect_answers", None)
    incorrect = parse_json_list_str(incorrect_str)

    if correct is None or incorrect is None:
        return None, None
    if not isinstance(incorrect, list) or len(incorrect) != 3:
        return None, None

    options = [correct] + incorrect
    label = 0

    # Deterministic shuffle by persona_id (stable across runs)
    pid = ex.get("persona_id", 0)
    rng = random.Random(int(pid))
    perm = list(range(4))
    rng.shuffle(perm)
    options = [options[i] for i in perm]
    label = perm.index(label)
    return options, label


def build_messages(ex: Dict) -> Optional[List[Dict]]:
    """
    Build the message list: (history/snippet) + final user_query.
    Uses snippet when available; optionally falls back to full chat history file.
    """
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

    # Append the final user query
    try:
        uq = parse_user_query(ex["user_query"])
    except Exception:
        return None
    msgs.append(uq)
    return msgs


def format_prompt(msgs: List[Dict]) -> str:
    """
    Format messages using chat template if available, else fallback string format.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return "\n".join([f'{m["role"].upper()}: {m["content"]}' for m in msgs]) + "\nASSISTANT:"


@torch.no_grad()
def get_readout(prompt_text: str) -> np.ndarray:
    """
    Return a single feature vector from the chosen layer and readout pooling.
    """
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    out = model(**inputs, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[LAYER][0]  # (seq, hidden)

    if READOUT == "last_token":
        vec = hs[-1]
    elif READOUT == "mean":
        vec = hs.mean(dim=0)
    else:
        raise ValueError(f"Unknown READOUT={READOUT}")

    return vec.detach().float().cpu().numpy()


def featurize_split(ds_split, n_target: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Iterate the dataset and collect n_target valid examples (skipping malformed/missing).
    """
    X, y = [], []
    n_ok = 0

    for ex in tqdm(ds_split, desc=f"Featurizing (target={n_target})"):
        if n_ok >= n_target:
            break

        options, label = build_options_and_label(ex)
        if label is None:
            continue

        msgs = build_messages(ex)
        if msgs is None:
            continue

        prompt = format_prompt(msgs)

        try:
            feat = get_readout(prompt)
        except RuntimeError as e:
            # If you hit OOM, suggest decreasing MAX_LENGTH / batch sizes.
            if "out of memory" in str(e).lower():
                raise RuntimeError(
                    "CUDA OOM while extracting features. "
                    "Reduce MAX_LENGTH and/or N_TRAIN/N_VAL/N_TEST, or use a smaller model."
                ) from e
            raise

        X.append(feat)
        y.append(label)
        n_ok += 1

    if n_ok == 0:
        raise RuntimeError("No valid examples were featurized. Check dataset fields / parsing.")
    return np.stack(X), np.array(y)

def featurize_split_binary(ds_split, n_target_examples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each original example, create 4 datapoints:
      prompt = (history + user_query) + candidate answer text
      y = 1 if candidate is correct else 0
    """
    X, y = [], []
    n_ok = 0

    for ex in tqdm(ds_split, desc=f"Featurizing-binary (target_examples={n_target_examples})"):
        if n_ok >= n_target_examples:
            break

        options, correct_idx = build_options_and_label(ex)
        if options is None:
            continue

        msgs = build_messages(ex)
        if msgs is None:
            continue

        base_prompt = format_prompt(msgs)

        # Build 4 candidate prompts
        for i, opt in enumerate(options):
            prompt = (
                base_prompt
                + "\n\nCandidate answer:\n"
                + opt
                + "\n\nIs this the correct personalized answer?"
            )
            label = 1 if i == correct_idx else 0

            try:
                feat = get_readout(prompt)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    raise RuntimeError(
                        "CUDA OOM while extracting features. Reduce MAX_LENGTH and/or N_TRAIN/N_VAL/N_TEST."
                    ) from e
                raise

            X.append(feat)
            y.append(label)

        n_ok += 1

    return np.stack(X), np.array(y)


def main():
    print("DEVICE:", DEVICE, "CUDA:", torch.cuda.is_available())
    print("Splits will be:", SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)
    print("History mode:", "snippet-first" if USE_SNIPPET_IF_AVAILABLE else "full-history-only",
          "| fallback_to_full_history:", FALLBACK_TO_FULL_HISTORY)
    print("MAX_LENGTH:", MAX_LENGTH, "LAYER:", LAYER, "READOUT:", READOUT)

    ds = load_dataset(DATASET_REPO)

    # Guard splits
    for s in [SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST]:
        if s not in ds:
            raise KeyError(f"Split {s} not found. Available splits: {list(ds.keys())}")

    train = ds[SPLIT_TRAIN]
    val = ds[SPLIT_VAL]
    test = ds[SPLIT_TEST]

    X_train, y_train = featurize_split_binary(train, N_TRAIN)  # N_TRAIN counts original examples
    X_val, y_val     = featurize_split_binary(val, N_VAL)
    X_test, y_test   = featurize_split_binary(test, N_TEST)

    def label_stats(y):
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        pos_rate = pos / len(y)
        return pos, neg, pos_rate

    print("Train pos/neg/pos_rate:", label_stats(y_train))
    print("Val   pos/neg/pos_rate:", label_stats(y_val))
    print("Test  pos/neg/pos_rate:", label_stats(y_test))

    clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=10000, n_jobs=-1)  # no multi_class arg
    )
    clf.fit(X_train, y_train)


    val_acc = accuracy_score(y_val, clf.predict(X_val))
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    print(f"Val acc:  {val_acc:.4f}")
    print(f"Test acc: {test_acc:.4f}")
    print("4-way baseline:", 0.50)

    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    # For AUC we need probabilities for the positive class
    val_probs = clf.predict_proba(X_val)[:, 1]
    test_probs = clf.predict_proba(X_test)[:, 1]

    print("Val  balanced acc:", balanced_accuracy_score(y_val, val_pred))
    print("Test balanced acc:", balanced_accuracy_score(y_test, test_pred))
    print("Val  AUC:", roc_auc_score(y_val, val_probs))
    print("Test AUC:", roc_auc_score(y_test, test_probs))



if __name__ == "__main__":
    main()
