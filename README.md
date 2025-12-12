# LLM Cross-Model Evaluation

## Experiments

**Experiment 1 – Cross-evaluation on generated continuations**

- Choose a prompt (turn 1).
- Model A generates a continuation.
- We compute: 
  - The token-level log-likelihoodof that continuation under Model A
  - The token-lebel log-likelihoodof the same continuation under ModelB 
- We compare average log-likelihoods

**Experiment 2 – Cross-evaluation on fixed canonical text**
- Take a fixed text that both models almost surely saw during training(we take first page of US constitution) 
For windows of text, we compte token-leve log-likekihood under both models A and B 
- Compare 
