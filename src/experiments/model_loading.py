from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_two_models_same_family(
        model_name_1: str = 'gpt2',
        model_name_2: str = 'gpt2-medium', 
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        dtype: str = "bfloat16",  
): 

    """
    Load two language models and their tokenizers.

    """
    toarch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_name_1, use_fast=True)

    # Make generation safe if pad token is missing
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    
    model_1 = AutoModelForCausalLM.from_pretrained(
        model_name_1, torch_dtype=torch_dtype, device_map="auto" if device == "cuda" else None
    )
    model_2 = AutoModelForCausalLM.from_pretrained(
        model_name_2, torch_dtype=torch_dtype, device_map="auto" if device == "cuda" else None
    )

    model_1.eval()
    model_2.eval()

    return tokenizer, model_1, model_2