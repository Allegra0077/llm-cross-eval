from transformers import AutoTokenizer, AutoModelForCausalLM

def load_models(
        model_name_1: str = 'gpt2',
        model_name_2: str = 'gpt2-medium', 
        device: str = 'cpu', 
): 
    """
    Load two language models and their tokenizers.

    """

    tokenizer = AutoTokenizer.from_pretrained(model_name_1)
    model_1 = AutoModelForCausalLM.from_pretrained(model_name_1).to(device)
    model_2 = AutoModelForCausalLM.from_pretrained(model_name_2).to(device)

    return tokenizer, model_1, model_2