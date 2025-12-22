from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import math

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

@dataclass
class TokenScore:
    """
    scores for 1 token under both models 
    stores both log-prob and prob
    """
    idx: int
    token_id: int
    token_str: str

    logp_a: float
    p_a: float
    logp_b: float
    p_b: float

    #idk yet whether to use ratio of probs or logprobs for comparison??
    
    @property
    def delta_logp(self) -> float:
        # log pA - log pB = log (pA / pB) --> to use for comparison step
        return self.logp_a - self.logp_b

    @property
    def ratio_pA_over_pB(self) -> float:
        # pA/pB, computed stably via logs
        return math.exp(self.delta_logp)

    @property
    def log10_ratio(self) -> float:
        # log10(pA/pB) so 10^k = ratio
        return self.delta_logp / math.log(10.0)
    
#experiment 1 scoring

@torch.no_grad()
def _logprobs_next_token_matrix(model: PreTrainedModel, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Returns log-probs for next-token at each position.
    Shape: (seq_len - 1, vocab)
    Where row i corresponds to distribution for token at position i+1.
    """
    attention_mask = torch.ones_like(input_ids)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (1, seq_len, vocab)
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)  # (1, seq_len-1, vocab)
    return log_probs[0]  # (seq_len-1, vocab)


def _tok_str(tokenizer: PreTrainedTokenizerBase, token_id: int) -> str:
    # take into account raw token form (with spaces, punctuation, etc)
    # !! GPT-2 encodes spacing as part of the token -> token-level likelihood comparisosn can be sensitive to this
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)


@torch.no_grad()
def score_continuation_tokens(
    tokenizer: PreTrainedTokenizerBase,
    model_a: PreTrainedModel,
    model_b: PreTrainedModel,
    prompt: str,
    continuation: str,
    device: str,
) -> List[TokenScore]:
    """
    Experiment 1 helper:
    Score each token of 'continuation' given 'prompt' under both models.
    Returns per-token scores aligned to the continuation tokens.
    """
    # Encode full prompt+continuation
    enc_full = tokenizer(prompt + continuation, return_tensors="pt")
    input_ids_full = enc_full["input_ids"].to(device)  # (1, L)

    # Encode prompt only to get prompt length in tokens
    enc_prompt = tokenizer(prompt, return_tensors="pt")
    prompt_len = enc_prompt["input_ids"].shape[1]

    # Next-token log-prob matrices
    lp_a = _logprobs_next_token_matrix(model_a, input_ids_full)  # (L-1, V)
    lp_b = _logprobs_next_token_matrix(model_b, input_ids_full)  # (L-1, V)

    # Continuation token ids start at prompt_len
    cont_token_ids = input_ids_full[0, prompt_len:]  # (n,)

    scores: List[TokenScore] = []
    for i, tok_id_tensor in enumerate(cont_token_ids):
        tok_id = int(tok_id_tensor.item())

        # token at absolute position (prompt_len + i) is predicted from row (prompt_len + i - 1)
        pred_row = prompt_len + i - 1
        logp_a_tok = float(lp_a[pred_row, tok_id].item())
        logp_b_tok = float(lp_b[pred_row, tok_id].item())

        # Convert to probs (see if useful to have both)
        p_a_tok = math.exp(logp_a_tok)
        p_b_tok = math.exp(logp_b_tok)

        scores.append(
            TokenScore(
                idx=i,
                token_id=tok_id,
                token_str=_tok_str(tokenizer, tok_id),
                logp_a=logp_a_tok,
                logp_b=logp_b_tok,
                p_a=p_a_tok,
                p_b=p_b_tok,
            )
        )

    return scores

#experiment 2 scoring

@torch.no_grad()
def score_full_text_next_tokens(
    tokenizer: PreTrainedTokenizerBase,
    model_a: PreTrainedModel,
    model_b: PreTrainedModel,
    text: str,
    device: str,
) -> List[TokenScore]:
    """
    Experimnt 2 helper:
    Score every next-token in 'text' under both models.
    For token positions 1...L-1, score token t_i given prefix t_<i.

    Returns a list of TokenScore where idx corresponds to the token position in the text.
    """
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)  # (1, L)

    lp_a = _logprobs_next_token_matrix(model_a, input_ids)  # (L-1, V)
    lp_b = _logprobs_next_token_matrix(model_b, input_ids)  # (L-1, V)

    next_token_ids = input_ids[0, 1:]  # tokens being predicted at each step

    scores: List[TokenScore] = []
    for row_i, tok_id_tensor in enumerate(next_token_ids):
        tok_id = int(tok_id_tensor.item())

        logp_a_tok = float(lp_a[row_i, tok_id].item())
        logp_b_tok = float(lp_b[row_i, tok_id].item())
        p_a_tok = math.exp(logp_a_tok)
        p_b_tok = math.exp(logp_b_tok)

        # idx here is the token position in the text (1..L-1)
        scores.append(
            TokenScore(
                idx=row_i + 1,
                token_id=tok_id,
                token_str=_tok_str(tokenizer, tok_id),
                logp_a=logp_a_tok,
                logp_b=logp_b_tok,
                p_a=p_a_tok,
                p_b=p_b_tok,
            )
        )

    return scores