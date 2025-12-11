# LLM Cross-Model Evaluation 

## Experiments 

**Experiment 1 - Cross-evaluation on generated continuations**

- Choose a prompt (turn 1)
- Model A generates a continuation 
- we compute: 
    - token-level log-likelihood of continuation under model A
    - token-level log-likelihood of continuation under model B
- compare likelihhods 

**Experiment 2 - Cross-evaluation on fixed canonical text**

- take a fixed text both models surely saw during training (we take first page of US constitution)
- for sliding windows of text, compute token-level likelihoods under both models 
- compare average likelihoods ?

## Code layout 

- `src/experiments/model_loading.py` - load models and tokenizer
- `src/experiments/logprob_utils.py` - compute token-level log-probabilities
- `src/experiments/run_experiment_01_cross_eval.py` - run exp 1 
- `src/experiments/run_experiment_02_cross_eval.py` - run exp 2