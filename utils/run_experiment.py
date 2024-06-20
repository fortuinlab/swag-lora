import torch
from SWAG import SWAG
import numpy, random

from train import train_swag
from utils.peft_utils import create_peft_model, get_peft_config
from utils.config_utils import set_save_path
from data import load_glue_data, load_mcqa_data, MCQA_task_to_context_keys, GLUE_task_to_keys
from accelerate import Accelerator, DeepSpeedPlugin


def run_experiment(config):
    device = None
    # deepspeed and accelerate initialization
    deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=config.experiment.gradient_accumulation_steps)
    accelerator = Accelerator(split_batches=True, 
                            log_with="wandb",
                            deepspeed_plugin=deepspeed_plugin)  #mixed_precision='fp16')

    
    method = config.method.method_name
    task = config.experiment.task
    wandb_group = config.experiment.wandb_group
    model_name = config.model.model_name

    active_tags = [method, 'swag_task' if task == 'swag' else task, model_name]
    
    accelerator.init_trackers(project_name="baylora", 
                                init_kwargs={"wandb": {
                                                "entity": "baylora-team",
                                                "group": wandb_group,
                                                "tags": active_tags
                                                }
                                             }) 

    set_save_path(config, accelerator)

    seed = config.experiment.seed
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    
    batch_size = config.experiment.batch_size
    task = config.experiment.task

    print('offline loading = ', config.experiment.offline)

    ood_dataloader = None

    ## Load data
    if task in GLUE_task_to_keys:
        tokenizer, train_dataloader, eval_dataloader, test_dataloader, num_classes = load_glue_data(config, task, accelerator, batch_size=batch_size, subtask=config.experiment.subtask, offline=config.experiment.offline, data_path=config.experiment.data_path)
    elif task in MCQA_task_to_context_keys:
        tokenizer, train_dataloader, eval_dataloader, test_dataloader, num_classes = load_mcqa_data(config, task, accelerator, batch_size=batch_size, subtask=config.experiment.subtask, offline=config.experiment.offline, data_path=config.experiment.data_path)
    else:
        raise Exception('Only GLUE tasks and MCQA tasks implemented')

    if config.experiment.ood_task != "":
        if task in GLUE_task_to_keys:
            _, _, _, ood_dataloader, _ = load_glue_data(config, config.experiment.ood_task, accelerator, batch_size=config.experiment.ood_batch_size, subtask=config.experiment.ood_subtask, offline=config.experiment.offline, data_path=config.experiment.data_path)
        elif task in MCQA_task_to_context_keys:
            _, _, _, ood_dataloader, _ = load_mcqa_data(config, config.experiment.ood_task, accelerator, batch_size=config.experiment.ood_batch_size, subtask=config.experiment.ood_subtask, offline=config.experiment.offline, data_path=config.experiment.data_path)
        else:
            raise Exception('Only GLUE tasks and MCQA tasks implemented')

    peft_config = get_peft_config(config, accelerator=accelerator)

    accelerator.print(f'Method: {config.method.method_name}')

    # create peft model 
    model = create_peft_model(config, peft_config, num_classes=num_classes, tokenizer=tokenizer)

    # SWAG
    if config.method.method_name == 'swag':
        swag_model = SWAG(
            model,
            no_cov_mat=not config.method.swag_cov_mat,
            max_num_models=config.method.swag_max_num_models,
            modules_to_swag=config.method.modules_to_swag,
        )
        swag_model.train()
        
        swag_model = train_swag(model, swag_model, train_dataloader, eval_dataloader, test_dataloader, config, accelerator, tokenizer, num_classes=num_classes, peft_config=peft_config, ood_dataloader=ood_dataloader)

    else:
        raise Exception('Method not implemented')
    

    accelerator.end_training()
