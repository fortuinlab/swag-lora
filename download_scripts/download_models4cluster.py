from transformers import (
    AutoModelForSequenceClassification, 
    AutoModelForMultipleChoice, 
    AutoTokenizer, 
    AutoModelForCausalLM
)
import os
import torch

def download_model(model_name, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Downloading the model and tokenizer
    cache_dir = os.path.join(save_directory, model_name)
    if model_name in ['roberta-base', 'roberta-large']:
        AutoModelForSequenceClassification.from_pretrained(model_name).save_pretrained(cache_dir)
        AutoModelForMultipleChoice.from_pretrained(model_name).save_pretrained(cache_dir)
        AutoTokenizer.from_pretrained(model_name).save_pretrained(cache_dir)
    
    elif model_name == 'meta-llama/Llama-2-7b-hf':
        device = "cuda"
        dtype = torch.bfloat16
     
        trash_dir = '/p/scratch/hai_baylora/trash'
        AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, cache_dir=trash_dir).save_pretrained(cache_dir)
        AutoTokenizer.from_pretrained(model_name).save_pretrained(cache_dir)
    
    elif model_name == 'meta-llama/Llama-2-7b-chat-hf':
        device = "cuda"
        dtype = torch.bfloat16
   
        AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).save_pretrained(cache_dir)
        AutoTokenizer.from_pretrained(model_name).save_pretrained(cache_dir)
    
    else:
        # Add other model download steps if needed″
        pass

    print(f"Model {model_name} downloaded and saved to {save_directory}")

# Example usage
#download_model('roberta-base', '/p/scratch/hai_baylora/models')
#download_model('roberta-large', '/p/scratch/hai_baylora/models')
download_model('meta-llama/Llama-2-7b-hf', '/p/scratch/hai_baylora/models2')
