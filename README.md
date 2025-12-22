# LLM Cross-Model Evaluation

## Experiments

**Experiment 1 – Cross-evaluation on generated continuations**

- Choose a prompt (turn 1).
- Model A generates a continuation.
- We compute: 
  - The token-level log-likelihood of that continuation under Model A
  - The token-lebel log-likelihood of the same continuation under Model B 
- Compare likelihoods

**Experiment 2 – Cross-evaluation on fixed canonical text**
- Take a fixed text that both models almost surely saw during training (we take first page of US constitution) 
- For windows of text, we compte token-leve log-likekihood under both models A and B 
- Compare likelihoods

## Code layout 

- `src/experiments/model_loading.py` - load models and tokenizer
- `src/experiments/logprob_utils.py` - compute token-level log-probabilities
- `src/experiments/exp1_turn1.py` - run exp 1 
- `src/experiments/exp2_turn1.py` - run exp 2
