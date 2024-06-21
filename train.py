import copy
import json
import os
import time
from data import TASK_TYPE_DICT
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.swa_utils import SWALR
from tqdm import tqdm
import tabulate
import pandas as pd
import wandb
from utils.peft_utils import load_peft_model
import json
import accelerate
from utils.eval_utils import evaluate_task, compute_metrics, ood_metrics_entropy

from transformers import (
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)

# num_batches = len(train_dataloader) 
def get_lr_scheduler(optimizer, num_batches, config):
  num_steps = num_batches * config.experiment.num_epochs
  if config.experiment.scheduler == 'constant':
    lr_scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.experiment.warmup_length * num_steps,
      )
  elif config.experiment.scheduler == 'linear':
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.experiment.warmup_length * num_steps,
        num_training_steps=num_steps,
      )
  elif config.experiment.scheduler == 'cosine':
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.experiment.warmup_length * num_steps,
        num_training_steps=num_steps,
        num_cycles=config.experiment.num_lr_cycles
      )
  elif config.experiment.scheduler == 'cosine_hard_restarts':
    if not float(config.experiment.num_lr_cycles).is_integer():
      raise Exception('num_lr_cycles must be an integer for hard restarts.')
    lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.experiment.warmup_length * num_steps,
        num_training_steps=num_steps,
        num_cycles=config.experiment.num_lr_cycles
      )
  else:
    raise Exception('Unimplemented learning rate scheduler given.')
  
  if config.method.method_name == 'swag':
    swag_scheduler = SWALR(
        optimizer,
        anneal_strategy=config.method.swag_anneal_strategy, 
        anneal_epochs=config.method.swag_anneal_epochs * num_batches, 
        swa_lr = config.method.swag_learning_rate
      )
    return lr_scheduler, swag_scheduler
  return lr_scheduler

def train_swag(model, swag_model, train_dataloader, eval_dataloader, test_dataloader, config, accelerator, tokenizer, num_classes=2, peft_config=None, ood_dataloader=None):
  n_gpus = torch.cuda.device_count()
  print(f"Number of GPUs available: {n_gpus}")

  causal_lm = False
  task_type = TASK_TYPE_DICT[config.experiment.task]
  if 'meta-llama' in config.model.model_name and task_type == 'MCQA':
    causal_lm = True

  swag_start = config.method.swag_start
  num_epochs = config.experiment.num_epochs
  save_path = config.experiment.save_path
  print('save_path in train_swag is:', save_path)

  optimizer = AdamW(
      params=model.parameters(), 
      lr=config.experiment.learning_rate, 
      weight_decay=config.experiment.weight_decay
    )
  loss_fn = CrossEntropyLoss()
  
  lr_scheduler, swag_scheduler = get_lr_scheduler(
      optimizer=optimizer, 
      num_batches=len(train_dataloader), 
      config=config
    )

  # prepare model, data, optimizer, scheduler
  swag_model, optimizer, lr_scheduler, swag_scheduler, train_dataloader, eval_dataloader, test_dataloader \
   = accelerator.prepare(swag_model, optimizer, lr_scheduler, swag_scheduler, train_dataloader, eval_dataloader, test_dataloader)
  if ood_dataloader is not None:
    ood_dataloader = accelerator.prepare(ood_dataloader)

  
  best_base_eval_loss = np.inf
  best_swag_eval_loss = np.inf
  base_save_info = {}
  swag_save_info = {}
  swag_eval_metrics = {}
  
  columns = ["epc", "lr_e", "swag_lr_e", "tr_loss", "tr_acc", "val_loss", "val_acc", "swag_val_loss", "swag_val_acc", 'nll', 'swag_nll', 'ece', 'swag_ece', 'brier', 'swag_brier', "time"]
  print('')
  print('-------------------------'+'Start training'+'-------------------------')
  
  learning_rates = []
  logged_values = []
  t_start = time.time()
  avg_time = 0

  counter=-1
  for epoch in range(num_epochs):

    # load base model with lowest loss when starting SWA(G)
    if epoch == swag_start:
      print('Beginning SWAG collection')
      if config.method.swag_start_with_lowest_loss:
        print('Loading model from training with lowest validation loss')
        model_load = load_peft_model(os.path.join(save_path, 'base_model'), config.experiment.task, is_trainable=True, tokenizer=tokenizer)
        
        unwrapped_model = accelerator.unwrap_model(swag_model)
        unwrapped_model.base.load_state_dict(model_load.state_dict(), strict=False)

    start_time = time.time()
    eval_acc = 0
    swag_eval_acc = 0
    eval_loss = np.inf
    swag_eval_loss = np.inf
    
    model.train()
    swag_model.train()
    total_loss = 0.0

    all_preds = torch.tensor([], dtype=torch.long, device=accelerator.device)
    all_labels = torch.tensor([], dtype=torch.long, device=accelerator.device)

    correct = torch.tensor(0, dtype=torch.long, device=accelerator.device)
    total = torch.tensor(0, dtype=torch.long, device=accelerator.device)
    
    for step, batch in enumerate(tqdm(train_dataloader, disable=not accelerator.is_main_process)):
      with accelerator.accumulate(model):
        output = model(**{'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']})
        logits = output.logits[:, -1, :] if causal_lm else output.logits # if causal lm we use last representation's logits

        loss = loss_fn(logits, batch['labels'])
        total_loss += loss.detach()
        #loss.backward()
        accelerator.backward(loss)


        optimizer.step()
        optimizer.zero_grad()
      
        preds = logits.argmax(axis=-1)
        all_preds = torch.cat((all_preds, preds), dim=0)
        all_labels = torch.cat((all_labels, batch['labels']), dim=0)

        correct += preds.eq(batch['labels'].view_as(preds)).sum()
        total += len(batch['labels'])
      
        if epoch < swag_start:
          lr_scheduler.step()
        else:
          swag_scheduler.step()
        learning_rates.append(optimizer.param_groups[0]['lr'])
      
      if accelerator.is_main_process and step % config.experiment.gradient_accumulation_steps == 0:
        counter+=1
        accelerator.log({'step': counter, 'epoch': epoch, 'lr': optimizer.param_groups[0]['lr']})
    
    collected_total_loss = accelerator.gather_for_metrics(total_loss).sum().item()
    train_loss = collected_total_loss / (len(train_dataloader) / config.experiment.gradient_accumulation_steps)

    collected_correct, collected_total = accelerator.gather_for_metrics((correct, total))
    collected_correct = collected_correct.sum().item()
    collected_total = collected_total.sum().item()
    if accelerator.is_main_process:
      train_acc = collected_correct/collected_total
    accelerator.wait_for_everyone()

    if epoch >= swag_start and (epoch - swag_start) % config.method.swag_c_epochs == 0:
      # collect model to SWAG
      swag_model.collect_model(model)
      # eval swag on validation
      model_cpu = swag_model.to('cpu')
      swag_state = copy.deepcopy(model_cpu.state_dict())
      swag_model.to(accelerator.device)

      swag_model.sample(0.0)
      #utils.bn_update (not needed unless we have batch norm)

      swag_eval_metrics, _ = compute_metrics(swag_model, eval_dataloader, 'eval', 'swag', swag_samples=1, swag_sample_scale=0.0, swag_cov_mat=config.method.swag_cov_mat, accelerator=accelerator, causal_lm=causal_lm)

      swag_eval_acc = swag_eval_metrics['acc']
      swag_eval_loss = swag_eval_metrics['loss']
      swag_model.load_state_dict(swag_state, strict=True)
      del swag_state
      

    eval_metrics, _ = compute_metrics(model, eval_dataloader, 'eval', 'base', accelerator=accelerator, causal_lm=causal_lm)
    accelerator.print('eval_metrics', {k:v for (k,v) in eval_metrics.items() if 'entrop' not in k})
    eval_loss = eval_metrics['loss']
    eval_acc = eval_metrics['acc']

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
      # save LoRA model if it improves upon best validation loss thus far
      if config.method.swag_save_base_model and eval_loss < best_base_eval_loss:
        print('Saving (lowest loss) base model checkpoint')
        best_base_eval_loss = eval_loss
        base_save_info['saved_epoch'] = epoch
        model.save_pretrained(os.path.join(save_path, 'base_model'))
        print('saved base model to path ', os.path.join(save_path, 'base_model'))
    
      # save swa model
      if swag_eval_loss < best_swag_eval_loss:
        print('Saving (lowest loss) SWAG model checkpoint')
        best_swag_eval_loss = swag_eval_loss
        swag_save_info['saved_epoch'] = epoch
        unwrapped_swag_model = accelerator.unwrap_model(swag_model)
        torch.save(unwrapped_swag_model.state_dict(), os.path.join(save_path, 'swag_model.pt'))
        print('saved swag model to path ', os.path.join(save_path, 'swag_model.pt'))
      elif epoch == swag_start + config.method.force_save:
        print('Force saving SWAG model checkpoint')
        swag_save_info['saved_epoch'] = epoch
        unwrapped_swag_model = accelerator.unwrap_model(swag_model)
        torch.save(unwrapped_swag_model.state_dict(), os.path.join(save_path, 'swag_model.pt'))
        print('force saved swag model to path ', os.path.join(save_path, 'swag_model.pt'))

    
    accelerator.wait_for_everyone()   
    time_diff = time.time() - start_time

    nll_value = eval_metrics.get('nll', None).item() if isinstance(eval_metrics.get('nll', None), torch.Tensor) else eval_metrics.get('nll', None)
    swag_nll_value = swag_eval_metrics.get('nll', None).item() if isinstance(swag_eval_metrics.get('nll', None), torch.Tensor) else swag_eval_metrics.get('nll', None)
    brier_value = eval_metrics.get('brier', None).item() if isinstance(eval_metrics.get('brier', None), torch.Tensor) else eval_metrics.get('brier', None)
    swag_brier_value = swag_eval_metrics.get('brier', None).item() if isinstance(swag_eval_metrics.get('brier', None), torch.Tensor) else swag_eval_metrics.get('brier', None)

    # Update values list
    if accelerator.is_main_process:
      values = [
          epoch,
          lr_scheduler.get_last_lr()[0],
          swag_scheduler.get_last_lr()[0],
          train_loss,
          train_acc,
          eval_loss,
          eval_acc,
          swag_eval_loss,
          swag_eval_acc,
          nll_value,
          swag_nll_value,
          eval_metrics.get('ece', None),
          swag_eval_metrics.get('ece', None),
          brier_value,
          swag_brier_value,
          time_diff,
      ]
    
      table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
      print(table)
      logged_values.append(values)

      # Log each metric individually for graphing
      metrics_to_log = {col: val for col, val in zip(columns, values)}
      counter +=1
      accelerator.log(metrics_to_log, step=counter) #here
    accelerator.wait_for_everyone()

  if accelerator.is_main_process:
    counter +=1
    accelerator.log({"Epoch Summary": wandb.Table(data=logged_values, columns=columns)}, step=counter)#here

    train_log = pd.DataFrame(logged_values, columns=columns)
    train_log = train_log.round(2)

    print('Training complete.')
  

    csv_file = os.path.join(save_path, "training_log.csv")
    train_log.to_csv(csv_file, index=False)
    print(f"Training metrics saved to {csv_file}")
    lr_file = os.path.join(save_path, "learning_rates.npy")
    np.save(lr_file, learning_rates)
    print(f'Learning rates saved to {lr_file}')
        
  if config.method.swag_save_base_model:
    base_path = os.path.join(save_path, "base_model")
    if accelerator.is_main_process:
      print(f'Saving LoRA base model saving info to {os.path.join(base_path, "save_info.json")}')
      with open(os.path.join(base_path, "base_save_info.json"), 'w') as f:
        json.dump(base_save_info, f)
    
    if config.experiment.eval_upon_save:
      if accelerator.is_main_process:
        accelerator.print(f'Evaluating best loss LoRA model (using model from epoch {base_save_info["saved_epoch"]})')
      
      unwrapped_model = accelerator.unwrap_model(swag_model)

      # change buffer shape of base model (we know when it was saved & swag_start)
      if accelerator.is_main_process:
        base_save_tensor = torch.tensor(base_save_info['saved_epoch'], device=accelerator.device)
      else:
        base_save_tensor = torch.tensor([1], device=accelerator.device)
      base_save_tensor = accelerate.utils.broadcast(base_save_tensor)
      base_save_epoch = base_save_tensor.item()

      for name, buffer in unwrapped_model.named_buffers():
        if 'weight_cov' in name: 
          if base_save_epoch >= swag_start:
            new_buffer_size = (min(config.method.swag_max_num_models, base_save_epoch - swag_start + 1),) + buffer.size()[1:] 
            new_buffer = torch.empty(*new_buffer_size)
            
            buffer.data = new_buffer
          else:
            new_buffer_size = (0,) + buffer.size()[1:] 
            new_buffer = torch.empty(*new_buffer_size)
            
            buffer.data = new_buffer

      unwrapped_model.base.load_adapter(base_path, 'default', is_trainable=False)


      evaluate_task(model, 'base', train_dataloader, eval_dataloader, test_dataloader, save_path=base_path, prefix='base_', accelerator=accelerator, causal_lm=causal_lm)

      if config.experiment.ood_task != "":
        print('=========== OOD eval ==============')
        if accelerator.is_main_process:
          accelerator.print(f'Evaluating best loss LoRA model on OOD task {config.experiment.ood_task + config.experiment.ood_subtask} )')
        ood_metrics, ood_report = compute_metrics(model, ood_dataloader, 'OOD', 'base', accelerator=accelerator, causal_lm=causal_lm)
        id_metrics, id_report = compute_metrics(model, test_dataloader, 'ID', 'base', accelerator=accelerator, causal_lm=causal_lm)
        if accelerator.is_main_process:
          print(id_report)
          print(ood_report)
        
          print('--- Detection (entropies) ---')
          auroc_score, aupr_in, aupr_ood = ood_metrics_entropy(id_metrics['entropies'], ood_metrics['entropies'])
          print(f'AUROC score (entropies) = {auroc_score}')
          print(f'AUPR_in score (entropies) = {aupr_in}')
          print(f'AUPR_ood score (entropies) = {aupr_ood}')

          print('--- Detection (MIs) ---')
          auroc_score, aupr_in, aupr_ood = ood_metrics_entropy(id_metrics['MIs'], ood_metrics['MIs'])
          print(f'AUROC score (MIs) = {auroc_score}')
          print(f'AUPR_in score (MIs) = {aupr_in}')
          print(f'AUPR_ood score (MIs) = {aupr_ood}')


          print('--- Detection (across model entropies) ---')
          auroc_score, aupr_in, aupr_ood = ood_metrics_entropy(id_metrics['across_model_entropies'], ood_metrics['across_model_entropies'])
          print(f'AUROC score (across model entropies) = {auroc_score}')
          print(f'AUPR_in score (across model entropies) = {aupr_in}')
          print(f'AUPR_ood score (across model entropies) = {aupr_ood}')


          print('--- Detection (disagrement ratio) ---')
          auroc_score, aupr_in, aupr_ood = ood_metrics_entropy(id_metrics['disagreement_ratios'], ood_metrics['disagreement_ratios'])
          print(f'AUROC score (disagreement_ratios) = {auroc_score}')
          print(f'AUPR_in score (disagreement_ratios) = {aupr_in}')
          print(f'AUPR_ood score (disagreement_ratios) = {aupr_ood}')
        accelerator.wait_for_everyone()

  if accelerator.is_main_process:
    print(f'Saving SWAG model saving info to {save_path}')
    with open(os.path.join(save_path, 'swag_save_info.json'), 'w') as f:
      json.dump(swag_save_info, f)
  accelerator.wait_for_everyone()
    
  if config.experiment.eval_upon_save:
    if accelerator.is_main_process:
      if "saved_epoch" in swag_save_info:
        print(f'Evaluating best loss SWAG model (using model from epoch {swag_save_info["saved_epoch"]})')
      else:
        print('\n\n[!!!] Warning: Saved epoch information is not available. No loading possible!')
        return None
    accelerator.wait_for_everyone()

    # change buffer shape of base model (we know when it was saved & swag_start)
    if accelerator.is_main_process:
      swag_save_tensor = torch.tensor(swag_save_info['saved_epoch'], device=accelerator.device)
    else:
      swag_save_tensor = torch.tensor([1], device=accelerator.device)
    swag_save_tensor = accelerate.utils.broadcast(swag_save_tensor)
    swag_save_epoch = swag_save_tensor.item()

    unwrapped_swag_model = accelerator.unwrap_model(swag_model)

    # change buffer shape of base model (we know when it was saved & swag_start)
    for name, buffer in unwrapped_swag_model.named_buffers():
      if 'weight_cov' in name: 
          new_buffer_size = (min(config.method.swag_max_num_models, swag_save_epoch - swag_start + 1),) + buffer.size()[1:] 
          new_buffer = torch.empty(*new_buffer_size)
          buffer.data = new_buffer

    state_dict = torch.load(os.path.join(save_path, 'swag_model.pt'), map_location='cpu')
    if "n_models" not in state_dict:
      print('n_models not in state_dict WTF')
      print(state_dict)
    unwrapped_swag_model.load_state_dict(state_dict, strict=True)
    del state_dict
    swag_model.sample(0.0)
    
    evaluate_task(swag_model, 'swag', train_dataloader, eval_dataloader, test_dataloader, save_path = save_path, accelerator=accelerator, causal_lm=causal_lm)
    if config.experiment.ood_task != "":
      print('=========== OOD eval ==============')
      if accelerator.is_main_process:
        accelerator.print(f'Evaluating best loss LoRA model on OOD task {config.experiment.ood_task + config.experiment.ood_subtask} )')
      ood_metrics, ood_report = compute_metrics(swag_model, ood_dataloader, 'OOD', 'swag', accelerator=accelerator, causal_lm=causal_lm, swag_samples=15, swag_sample_scale=2.0, swag_cov_mat=True)
      id_metrics, id_report = compute_metrics(swag_model, test_dataloader, 'ID', 'swag', accelerator=accelerator, causal_lm=causal_lm, swag_samples=15, swag_sample_scale=2.0, swag_cov_mat=True)
      if accelerator.is_main_process:
        print(id_report)
        print(ood_report)
        print('--- Detection (entropies) ---')
        auroc_score, aupr_in, aupr_ood = ood_metrics_entropy(id_metrics['entropies'], ood_metrics['entropies'])
        print(f'AUROC score (entropies) = {auroc_score}')
        print(f'AUPR_in score (entropies) = {aupr_in}')
        print(f'AUPR_ood score (entropies) = {aupr_ood}')

        print('--- Detection (MIs) ---')
        auroc_score, aupr_in, aupr_ood = ood_metrics_entropy(id_metrics['MIs'], ood_metrics['MIs'])
        print(f'AUROC score (MIs) = {auroc_score}')
        print(f'AUPR_in score (MIs) = {aupr_in}')
        print(f'AUPR_ood score (MIs) = {aupr_ood}')

        print('--- Detection (across model entropies) ---')
        auroc_score, aupr_in, aupr_ood = ood_metrics_entropy(id_metrics['across_model_entropies'], ood_metrics['across_model_entropies'])
        print(f'AUROC score (across model entropies) = {auroc_score}')
        print(f'AUPR_in score (across model entropies) = {aupr_in}')
        print(f'AUPR_ood score (across model entropies) = {aupr_ood}')

        print('--- Detection (disagrement ratio) ---')
        auroc_score, aupr_in, aupr_ood = ood_metrics_entropy(id_metrics['disagreement_ratios'], ood_metrics['disagreement_ratios'])
        print(f'AUROC score (disagreement_ratios) = {auroc_score}')
        print(f'AUPR_in score (disagreement_ratios) = {aupr_in}')
        print(f'AUPR_ood score (disagreement_ratios) = {aupr_ood}')
  return None
  
