import os
from datetime import datetime
import wandb
import torch
import accelerate

def set_save_path(config, accelerator):       
    # generate experiment name     
    exp_name = config.experiment.exp_name
    if exp_name == '':
        exp_name = '^'.join(['lr_'+str(config.experiment.learning_rate),
                             'bsz_'+str(config.experiment.batch_size),
                             'sch_'+config.experiment.scheduler,
                             'drop_'+str(config.experiment.lora_dropout),
                             'epc_'+str(config.experiment.num_epochs),
                             'lorar_'+str(config.experiment.lora_r),
                             'loraa_'+str(config.experiment.lora_alpha),
                             'loramod_'+str(config.model.target_modules).replace('_', '')
                            ])
        
        
        if config.method.method_name == 'wgd':
            exp_name += '^nparticles_' + str(config.method.n_particles)

        elif config.method.method_name == 'ensemble':
            exp_name += '^nparticles_' + str(config.method.n_particles)

        elif config.method.method_name == 'f-wgd':
            exp_name += '^nparticles_' + str(config.method.n_particles)
            
        elif config.method.method_name == 'swag':
            exp_name += '^'+ '^'.join(['cov_'+str(config.method.swag_cov_mat),
                                       'swagstart_'+str(config.method.swag_start),
                                       'lr_'+str(config.method.swag_learning_rate),
                                       'mod_'+config.method.modules_to_swag,
                                       'sch_'+config.method.swag_scheduler,
                                       'startbestmodel_'+str(config.method.swag_start_with_lowest_loss),
                                       'maxmodels_'+str(config.method.swag_max_num_models),
                                       'cepochs_'+str(config.method.swag_c_epochs),    
                                       ])
        else:
            raise Exception('only swag, wgd, f-wgd and ensemble methods supported')
    
    # Generate a timestamp
    timestamp = "_to_be_replaced"
    timestamp_tensor = torch.ByteTensor(list(timestamp.encode("utf-8"))).to(accelerator.device)
    if accelerator.is_main_process:
        timestamp = str(datetime.now().strftime("%Y%m%d-%H%M%S"))
        timestamp_tensor = torch.ByteTensor(list(timestamp.encode("utf-8"))).to(accelerator.device)
    timestamp_tensor = accelerate.utils.broadcast(timestamp_tensor).to('cpu')

    timestamp = bytes(timestamp_tensor.to(torch.uint8).tolist()).decode("utf-8")

    # get filepath
    save_folder = config.experiment.save_folder
    task = config.experiment.task
    if config.experiment.task == 'mmlu':
        subtask = '_all' if config.experiment.subtask == '' else '_'+config.experiment.subtask
        config.experiment.subtask = subtask.strip('_')
    elif config.experiment.task == 'mnli':
        subtask = '_matched' if config.experiment.subtask == '' else '_'+config.experiment.subtask
        config.experiment.subtask = subtask.strip('_')
    else:
        subtask = ''
    model_name = config.model.model_name
    method_name = config.method.method_name

    # Print the components for debugging
    accelerator.print("Save Folder:", save_folder)
    accelerator.print("Task:", task)
    accelerator.print("Subtask:", subtask)
    accelerator.print("Model Name:", model_name)
    accelerator.print("Method Name:", method_name)
    accelerator.print("Experiment Name:", exp_name)
    accelerator.print("Timestamp:", timestamp)

    save_path = os.path.join(save_folder, config.experiment.task + subtask, config.model.model_name, config.method.method_name, exp_name, timestamp)

    print('save_path', save_path)
    config.experiment.save_path = save_path

    if accelerator.is_main_process:
        if config.experiment.overwrite:
            os.makedirs(save_path, exist_ok=True)
        else:
            try:
                os.makedirs(save_path)
            except FileExistsError:
                raise Exception(f'Path {save_path} already exists. Ensure that config is correct or set config.experiment.overwrite=True to overwrite this experiment.')

        # save config
        with open(os.path.join(save_path, 'saved_config.yaml'), 'w') as f:
            f.write(str(config))

        wandb.config.update(dict(config), allow_val_change=True)

