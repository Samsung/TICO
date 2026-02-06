from lm_eval.models.huggingface import HFLM 
from lm_eval import evaluator

def evaluate_llm_on_tasks(model, tokenizer, tasks: str)->dict[str, any]:
    model_to_evaluate = HFLM(model, "causal", tokenizer=tokenizer)
    tasks=tasks.split(",")
    return evaluator.simple_evaluate(model_to_evaluate,tasks=tasks)
