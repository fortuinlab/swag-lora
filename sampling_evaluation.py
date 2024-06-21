import json
import os
import os
import random
from data import TASK_TYPE_DICT, GLUE_task_to_keys, MCQA_task_to_context_keys, load_glue_data, load_mcqa_data
from SWAG import SWAG
import numpy as np
import torch
import time
from omegaconf import DictConfig
import hydra
import os
from utils.eval_utils import compute_metrics, evaluate_task, ood_metrics_entropy
from utils.peft_utils import load_peft_model
from accelerate import Accelerator

# path given in config.evaluation.eval_model_path must be path to folder containing the model
# within this folder there should be a folder for the base LoRA model saved as "base_model"
# if SWAG has been trained, there will also be a swag_model.pt file

@hydra.main(version_base="1.3.2", config_path="conf", config_name="config")
def main(cfg: DictConfig):
  
  accelerator = Accelerator(split_batches=True)  #mixed_precision='fp16')

  config = cfg
  seed = config.evaluation.seed
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.cuda.manual_seed_all(seed)
  
  task = config.experiment.task

  task_type = TASK_TYPE_DICT[task]

  only_ood = config.evaluation.only_ood
  swag_cov_mat = (config.evaluation.eval_method == 'swag' and config.method.swag_cov_mat)
  accelerator.print(f'SWAG COV MAT = {swag_cov_mat}')
  
  num_samples = config.evaluation.num_samples 
  swag_sample_scale = config.evaluation.swag_sample_scale

  if 'meta-llama' in config.model.model_name and task_type == 'MCQA':
    causal_lm = True


  config.experiment.gradient_accumulation_steps = 1
  path = config.evaluation.eval_model_path
  accelerator.print(f'Model path: {path}')
  
  if config.experiment.ood_task != '':
      tokenizer, _, _, ood_dataloader, _ = load_mcqa_data(config, config.experiment.ood_task, accelerator, batch_size=config.experiment.ood_batch_size, subtask=config.experiment.ood_subtask, offline=config.experiment.offline, data_path=config.experiment.data_path)
      
  if task in GLUE_task_to_keys:
    tokenizer, train_dataloader, eval_dataloader, test_dataloader, num_classes = load_glue_data(config, task, accelerator, batch_size=config.experiment.batch_size, subtask=config.experiment.subtask, offline=config.experiment.offline, data_path=config.experiment.data_path)
  elif task in MCQA_task_to_context_keys:
    tokenizer, train_dataloader, eval_dataloader, test_dataloader, num_classes = load_mcqa_data(config, task, accelerator, batch_size=config.experiment.batch_size, subtask=config.experiment.subtask, offline=config.experiment.offline, data_path=config.experiment.data_path)
  else:
    raise Exception('Only GLUE tasks and MCQA tasks implemented')

  model_name_or_path = os.path.join(config.experiment.model_path, config.model.model_name)
  base_model = load_peft_model(os.path.join(path, 'base_model'), config.experiment.task, is_trainable=True, tokenizer=None, base_model_path = model_name_or_path)

  if config.evaluation.eval_method == 'swag':
    swag_model = SWAG(
        base_model,
        no_cov_mat=not swag_cov_mat,
        max_num_models=5,
        modules_to_swag='grad_only',
    )
    swag_model.eval()

    #load state dict
    state_dict = torch.load(os.path.join(path, 'swag_model.pt'), map_location='cpu')
    swag_model.load_state_dict(state_dict, strict=True)
    swag_model.to(accelerator.device)
    # prepare accelerator
    if not only_ood:
      
      train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(train_dataloader, eval_dataloader, test_dataloader)
    
      start = time.time()
      accelerator.print('====================================')
      accelerator.print(f'=== SWAG_SAMPLE_SCALE {swag_sample_scale} ===')
      accelerator.print(f'SWAG samples = {num_samples}')

      summary_metrics = evaluate_task(swag_model, 'swag', train_dataloader, eval_dataloader, test_dataloader, accelerator=accelerator, causal_lm=causal_lm, log_summary=False, swag_samples = num_samples, swag_sample_scale = swag_sample_scale, swag_cov_mat = swag_cov_mat)
      accelerator.print(summary_metrics)
      accelerator.print('TIME TAKEN = ', time.time()-start)
      accelerator.print('------------------------------------')

      accelerator.print('SAMPLING EVAL FINISHED')

    accelerator.wait_for_everyone()
    if config.experiment.ood_task != '':
      #_, _, _, ood_dataloader, _ = load_mcqa_data(config, config.experiment.ood_task, accelerator, batch_size=config.experiment.ood_batch_size, subtask=config.experiment.ood_subtask, offline=config.experiment.offline, data_path=config.experiment.data_path)
      ood_dataloader = accelerator.prepare(ood_dataloader)
      if only_ood:
          test_dataloader = accelerator.prepare(test_dataloader)

      accelerator.print('=========== OOD eval ==============')
      accelerator.print(f'Evaluating SWAG model trained on {config.experiment.task} on OOD task {config.experiment.ood_task + config.experiment.ood_subtask} )')
      
      start = time.time()
      accelerator.print('====================================')
      accelerator.print(f'=== SWAG_SAMPLE_SCALE {swag_sample_scale} ===')
      accelerator.print(f'SWAG samples = {num_samples}')
      
      ood_metrics, ood_report = compute_metrics(swag_model, ood_dataloader, 'OOD', 'swag', accelerator=accelerator, causal_lm=causal_lm, swag_samples=num_samples, swag_sample_scale=swag_sample_scale, swag_cov_mat=swag_cov_mat)
      id_metrics, id_report = compute_metrics(swag_model, test_dataloader, 'ID', 'swag', accelerator=accelerator, causal_lm=causal_lm, swag_samples=num_samples, swag_sample_scale=swag_sample_scale, swag_cov_mat=swag_cov_mat)
      if accelerator.is_main_process:
        print(id_report)
        print(ood_report)
        print('--- Detection (entropies) ---')
        auroc_entropy, aupr_in_entropy, aupr_ood_entropy = ood_metrics_entropy(id_metrics['entropies'], ood_metrics['entropies'])
        print(f'AUROC score (entropies) = {auroc_entropy}')
        print(f'AUPR_id score (entropies) = {aupr_in_entropy}')
        print(f'AUPR_ood score (entropies) = {aupr_ood_entropy}')

        print('--- Detection (MIs) ---')
        auroc_MI, aupr_in_MI, aupr_ood_MI = ood_metrics_entropy(id_metrics['MIs'], ood_metrics['MIs'])
        print(f'AUROC score (MIs) = {auroc_MI}')
        print(f'AUPR_id score (MIs) = {aupr_in_MI}')
        print(f'AUPR_ood score (MIs) = {aupr_ood_MI}')

        print('--- Detection (across model entropies) ---')
        auroc_cm_entropy, aupr_in, aupr_ood = ood_metrics_entropy(id_metrics['across_model_entropies'], ood_metrics['across_model_entropies'])
        print(f'AUROC score (across model entropies) = {auroc_cm_entropy}')
        print(f'AUPR_id score (across model entropies) = {aupr_in}')
        print(f'AUPR_ood score (across model entropies) = {aupr_ood}')

        print('--- Detection (disagrement ratio) ---')
        auroc_disagreement, aupr_in_disagreement, aupr_ood_disagreement = ood_metrics_entropy(id_metrics['disagreement_ratios'], ood_metrics['disagreement_ratios'])
        print(f'AUROC score (disagreement_ratios) = {auroc_disagreement}')
        print(f'AUPR_id score (disagreement_ratios) = {aupr_in_disagreement}')
        print(f'AUPR_ood score (disagreement_ratios) = {aupr_ood_disagreement}')
        

      accelerator.print('TIME TAKEN = ', time.time()-start)
      accelerator.print('------------------------------------')
  

  elif config.evaluation.eval_method == 'dropout':
    base_model.to(accelerator.device)
    if not only_ood:
      train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(train_dataloader, eval_dataloader, test_dataloader)
    
      start = time.time()
      accelerator.print('====================================')
      accelerator.print(f'Dropout samples = {num_samples}')
      summary_metrics = evaluate_task(base_model, 'base', train_dataloader, eval_dataloader, test_dataloader, accelerator=accelerator, causal_lm=causal_lm, log_summary=False, dropout_samples = num_samples)
      accelerator.print(summary_metrics)
      accelerator.print('TIME TAKEN = ', time.time()-start)
      accelerator.print('------------------------------------')

      accelerator.print('SAMPLING EVAL FINISHED')

    accelerator.wait_for_everyone()
    if config.experiment.ood_task != '':
      ood_dataloader = accelerator.prepare(ood_dataloader)
      if only_ood:
          test_dataloader = accelerator.prepare(test_dataloader)

      accelerator.print('=========== OOD eval ==============')
      accelerator.print(f'Evaluating MC dropout on best base model trained on {config.experiment.task} on OOD task {config.experiment.ood_task + config.experiment.ood_subtask} )')
      
      start = time.time()
      accelerator.print('====================================')
      
      accelerator.print(f'Dropout samples = {num_samples}')
      
      ood_metrics, ood_report = compute_metrics(base_model, ood_dataloader, 'OOD', 'base', accelerator=accelerator, causal_lm=causal_lm, dropout_samples=num_samples)
      id_metrics, id_report = compute_metrics(base_model, test_dataloader, 'ID', 'base', accelerator=accelerator, causal_lm=causal_lm, dropout_samples=num_samples)
      if accelerator.is_main_process:
        print(id_report)
        print(ood_report)
        print('--- Detection (entropies) ---')
        auroc_entropy, aupr_in_entropy, aupr_ood_entropy = ood_metrics_entropy(id_metrics['entropies'], ood_metrics['entropies'])
        print(f'AUROC score (entropies) = {auroc_entropy}')
        print(f'AUPR_id score (entropies) = {aupr_in_entropy}')
        print(f'AUPR_ood score (entropies) = {aupr_ood_entropy}')

        print('--- Detection (MIs) ---')
        auroc_MI, aupr_in_MI, aupr_ood_MI = ood_metrics_entropy(id_metrics['MIs'], ood_metrics['MIs'])
        print(f'AUROC score (MIs) = {auroc_MI}')
        print(f'AUPR_id score (MIs) = {aupr_in_MI}')
        print(f'AUPR_ood score (MIs) = {aupr_ood_MI}')

        print('--- Detection (across model entropies) ---')
        auroc_cm_entropy, aupr_in, aupr_ood = ood_metrics_entropy(id_metrics['across_model_entropies'], ood_metrics['across_model_entropies'])
        print(f'AUROC score (across model entropies) = {auroc_cm_entropy}')
        print(f'AUPR_id score (across model entropies) = {aupr_in}')
        print(f'AUPR_ood score (across model entropies) = {aupr_ood}')

        print('--- Detection (disagrement ratio) ---')
        auroc_disagreement, aupr_in_disagreement, aupr_ood_disagreement = ood_metrics_entropy(id_metrics['disagreement_ratios'], ood_metrics['disagreement_ratios'])
        print(f'AUROC score (disagreement_ratios) = {auroc_disagreement}')
        print(f'AUPR_id score (disagreement_ratios) = {aupr_in_disagreement}')
        print(f'AUPR_ood score (disagreement_ratios) = {aupr_ood_disagreement}')
        
      accelerator.print('TIME TAKEN = ', time.time()-start)
      accelerator.print('------------------------------------')
  else:
    raise Exception('eval script only built for eval_method == swag/dropout')

  if accelerator.is_main_process:
    save_name = config.evaluation.eval_method
    if save_name == 'swag':
      save_name += '_scale'+str(swag_sample_scale)
      save_name += '_cov'+str(swag_cov_mat)
    save_name += '_'+str(num_samples)

    if not only_ood:
      os.makedirs(os.path.dirname(os.path.join(path, save_name+'.json')), exist_ok=True)
      with open(os.path.join(path, save_name+'.json'), 'w') as f:
        json.dump(summary_metrics, f)

    if config.experiment.ood_task != '':
      
      ood_metrics['auroc_entropy'] = auroc_entropy
      ood_metrics['auroc_disagreement'] = auroc_disagreement
      ood_metrics['auroc_mi'] = auroc_MI
      ood_metrics['auroc_crossmodel_entropy'] = auroc_cm_entropy

      for key in ood_metrics:
        if isinstance(ood_metrics[key], torch.Tensor):
            ood_metrics[key] = ood_metrics[key].cpu().numpy().tolist()
      ood_save_name = save_name + '_ood-'+config.experiment.ood_task + config.experiment.ood_subtask
      os.makedirs(os.path.dirname(os.path.join(path, ood_save_name+'.json')), exist_ok=True)
      with open(os.path.join(path, ood_save_name+'.json'), 'w') as f:
        json.dump(ood_metrics, f)
      

if __name__ == "__main__":
    main()