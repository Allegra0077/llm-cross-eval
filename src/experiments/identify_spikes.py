import pandas as pd, numpy as np
from pathlib import Path

path = sorted(Path("/Data/allegra-maria-pia.boustany/llm_cross_eval/results").glob("exp1_turn1_*.csv"))[-1]
df = pd.read_csv(path)

df["log_ratio"] = np.log(df["ratio_pA_over_pB"].clip(lower=1e-12))
df["abs_log_ratio"] = df["log_ratio"].abs()

top = (df.sort_values("abs_log_ratio", ascending=False)
         .groupby(["example_id","seed"])
         .head(10))
top[["example_id","seed","token_idx","token_str","p_a","p_b","ratio_pA_over_pB","log_ratio"]].head(20)

def show_spike_context(df, example_id, seed, token_idx, window=10):
    g = df[(df.example_id==example_id) & (df.seed==seed)].sort_values("token_idx")
    lo, hi = token_idx - window, token_idx + window
    ctx = g[(g.token_idx >= lo) & (g.token_idx <= hi)].copy()
    ctx["mark"] = np.where(ctx.token_idx==token_idx, "<<<SPIKE<<<", "")
    return ctx[["token_idx","token_str","ratio_pA_over_pB","log_ratio","mark"]]

# Take the biggest spike overall
row = top.iloc[0]
show_spike_context(df, row.example_id, row.seed, row.token_idx, window=12)

def text_snippet_around_spike(df, example_id, seed, token_idx, char_window=160):
    g = df[(df.example_id==example_id) & (df.seed==seed)].sort_values("token_idx")
    tokens = g["token_str"].astype(str).tolist()
    
    # Reconstruct progressively (token_str often includes leading spaces; that's fine)
    prefix = "".join(tokens[:token_idx])
    spike_tok = tokens[token_idx] if token_idx < len(tokens) else ""
    suffix = "".join(tokens[token_idx+1:])
    
    # Build marked text
    text = prefix + "[[[" + spike_tok + "]]]" + suffix
    
    # Slice around marker
    m = text.find("[[[")
    start = max(0, m - char_window)
    end = min(len(text), m + char_window)
    return text[start:end]

row = top.iloc[0]
print(text_snippet_around_spike(df, row.example_id, row.seed, row.token_idx))
