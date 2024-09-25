import transformers
import torch


model_id = "meta-llama/Llama-3.1-8B"

pipleline = transformers.pipeline('text-generation', model=model_id,
                                  model_kwargs={"torch_dtype": torch.bfloat16},
                                  device="cuda")

