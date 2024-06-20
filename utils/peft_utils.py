import os
from data import TASK_TYPE_DICT, N_CLASSES_DICT
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSequenceClassification
import torch
from peft import (
    get_peft_model,
    LoraConfig,
)
import evaluate
from transformers import AutoModelForSequenceClassification
from transformers import (
    AutoModelForMultipleChoice,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LlamaConfig,
    LlamaForSequenceClassification,
    LlamaForCausalLM
)
from peft.tuners.lora import LoraConfig
import torch

def create_transformer(config, num_classes=2):
    task_type = TASK_TYPE_DICT[config.experiment.task]
    model_name_or_path = os.path.join(config.experiment.model_path, config.model.model_name) if config.experiment.offline else config.model.model_name
    if task_type == 'SEQ_CLS':
        if 'meta-llama' in config.model.model_name:
            cache_dir = '/p/scratch/hai_baylora/trash'
            configuration = LlamaConfig() 
            model = LlamaForSequenceClassification(configuration).from_pretrained(model_name_or_path, return_dict=True, num_labels=num_classes, cache_dir=cache_dir)
            model.config.pad_token_id = model.config.eos_token_id
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, num_labels=num_classes)

    elif task_type == 'MCQA':
        if 'meta-llama' in config.model.model_name:
            cache_dir = '/p/scratch/hai_baylora/trash'
            configuration = LlamaConfig()
            model = LlamaForCausalLM(configuration).from_pretrained(model_name_or_path, return_dict=True, cache_dir=cache_dir)
            model.config.pad_token_id = model.config.eos_token_id
        else:
            model = AutoModelForMultipleChoice.from_pretrained(model_name_or_path, return_dict=True)
    else:
        raise Exception('Only SEQ_CLS and MCQA task types implemented.')
    return model

# trim output space of LLM for MCQA
def trim_head_for_mcqa(model, task, peft_config, tokenizer):
    # modify LM head for MCQA
    if task in ['obqa', 'mmlu', 'arc-e', 'arc-c']:
        id_list = [tokenizer.encode('A')[1], tokenizer.encode('B')[1], tokenizer.encode('C')[1], tokenizer.encode('D')[1]]
    elif task == 'cqa':
        id_list = [tokenizer.encode('A')[1], tokenizer.encode('B')[1], tokenizer.encode('C')[1], tokenizer.encode('D')[1], tokenizer.encode('E')[1]]

    # old head
    original_lm_head = model.lm_head
    original_weights = original_lm_head.weight[id_list, :].clone()
    # new head
    new_lm_head = torch.nn.Linear(in_features=original_weights.shape[1], out_features=len(id_list), bias=False)
    new_lm_head.weight.data = original_weights
    model.lm_head = new_lm_head
    
    if 'lm_head' not in peft_config.target_modules: # lora fine-tuning
        peft_config.modules_to_save.append('lm_head') # full fine-tuning

def create_peft_model(config, peft_config, verbose = False, num_classes=2, tokenizer=None):
    task_type = TASK_TYPE_DICT[config.experiment.task]
    model = create_transformer(config, num_classes)   

    if 'meta-llama' in config.model.model_name and task_type == 'MCQA':
        trim_head_for_mcqa(model, config.experiment.task, peft_config, tokenizer)
        
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


def load_peft_model(peft_model_id, task, is_trainable=False, tokenizer=None, base_model_path=None):
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    task_type = TASK_TYPE_DICT[task]
    if base_model_path is None: 
        model_name_or_path = peft_config.base_model_name_or_path
    else: # base_load_path lets you load a compatible new base backbone with PEFT weights (e.g. loading 32-bit llama when the PEFT model was trained on 16-bit llama)
        model_name_or_path = base_model_path
    print(f'Loading base model {model_name_or_path}')


    if task_type == 'SEQ_CLS':
        if 'llama' in model_name_or_path:
            print('setup Llama model')
            trash_dir = '/p/scratch/hai_baylora/trash'
            configuration = LlamaConfig() 
            base_model = LlamaForSequenceClassification(configuration).from_pretrained(model_name_or_path, return_dict=True, num_labels=N_CLASSES_DICT[task], cache_dir=trash_dir)
            base_model.config.pad_token_id = base_model.config.eos_token_id
        else:
            base_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, num_labels=N_CLASSES_DICT[task])

    elif task_type == 'MCQA':
        if 'llama' in model_name_or_path:
            print('loading causal LM llama for MCQA')
            trash_dir = '/p/scratch/hai_baylora/trash'
            configuration = LlamaConfig()
            base_model = LlamaForCausalLM(configuration).from_pretrained(model_name_or_path, return_dict=True, cache_dir=trash_dir)
            base_model.config.pad_token_id = base_model.config.eos_token_id
            if not tokenizer:
                tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            trim_head_for_mcqa(base_model, task, peft_config, tokenizer)
        else:
            base_model = AutoModelForMultipleChoice.from_pretrained(model_name_or_path, return_dict=True)
    else:
        raise Exception('Only SEQ_CLS and MCQA task types implemented.')
        
    model = PeftModel.from_pretrained(base_model, peft_model_id, is_trainable=is_trainable)
    return model

def get_peft_config(config, accelerator=None):
    task_type = TASK_TYPE_DICT[config.experiment.task]

    if 'meta-llama' in config.model.model_name and task_type == 'MCQA':
        task_type = 'CAUSAL_LM'
    else:
        task_type = 'SEQ_CLS'

    print_fn = accelerator.print if accelerator else print
    print_fn(f'Creating LoRA config. Lora-target modules {config.model.target_modules}')
    peft_config = LoraConfig(task_type=task_type, inference_mode=False, r=config.experiment.lora_r, lora_alpha=config.experiment.lora_alpha,
                            lora_dropout=config.experiment.lora_dropout, bias=config.experiment.train_biases,
                            target_modules=list(config.model.target_modules),
                            modules_to_save=list(config.model.modules_to_save))
    return peft_config


def load_peft_ensemble(peft_model_id, adapter_names, config, is_trainable=False):
    peft_ensemble = load_peft_model(peft_model_id, config.experiment.task, is_trainable=is_trainable)

    for adapter_name in adapter_names:
        peft_ensemble.load_adapter(os.path.join(peft_model_id, adapter_name), adapter_name, is_trainable=is_trainable)
    
    return peft_ensemble

def get_modules_with_grad(model):
    modules_with_grad = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            has_grad_params = False
            for param_name, param in module.named_parameters():
                if param.requires_grad:
                    has_grad_params = True
                    break
            if has_grad_params:
                modules_with_grad.append((name, module))
    return modules_with_grad
