import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Any, Dict

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    return model, tok


def generate_json(model, tokenizer, prompt: str, max_new_tokens: int = 300, temperature: float = 0.7) -> Dict[str, Any]:
    messages = [{"role": "user", "content": prompt}]
    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = enc.to(model.device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Could not extract JSON from model output:\n{text}")
    return json.loads(text[start:end + 1])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--persona_config", type=str, required=True)
    p.add_argument("--prompts_config", type=str, required=True)
    p.add_argument("--model_name", type=str, default=MODEL_NAME)
    p.add_argument("--max_users", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_path", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    persona_path = Path(args.persona_config)
    prompts_path = Path(args.prompts_config)
    out_path = Path(args.output_path)

    persona_cfg = load_yaml(persona_path)
    prompts_cfg = load_yaml(prompts_path)

    base_profiles = persona_cfg["profiles"]["base_persona_id"]
    style_profiles = persona_cfg["profiles"]["style_id"]

    all_pairs = list(itertools.product(base_profiles.items(), style_profiles.items()))
    random.shuffle(all_pairs)

    if args.max_users is not None:
        all_pairs = all_pairs[: args.max_users]

    model, tok = load_model(args.model_name)
    template = prompts_cfg["generation_prompt_system_llm1"]["prompt"]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for idx, ((base_id, base_json), (style_id, style_json)) in enumerate(all_pairs):
            prompt = template.format(
                BASE_PERSONA_JSON=json.dumps(base_json, ensure_ascii=False),
                STYLE_JSON=json.dumps(style_json, ensure_ascii=False),
            )
            result = generate_json(model, tok, prompt)
            row = {
                "user_id": f"user_{idx:04d}",
                "base_persona_id": base_id,
                "style_id": style_id,
                "base_persona": base_json,
                "style": style_json,
                "system_llm1": result["system_llm1"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote users -> {out_path}")


if __name__ == "__main__":
    main()