from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt = "Large language models are"
    inputs = tokenizer(prompt, return_tensors="pt")

    output_ids = model.generate(**inputs, max_new_tokens=20)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("Prompt:", prompt)
    print("Completion:", text)


if __name__ == "__main__":
    main()