import os
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score, average_precision_score
import wandb

def set_dropout_training(module):
  if type(module) == torch.nn.modules.dropout.Dropout:
      module.train()

def mean_preds(predictions):
    return torch.stack([p.logits for p in predictions]).mean(dim=0)

def mean_logits(logits):
    return torch.stack(logits).mean(dim=0)

def compute_nll(probabilities, labels, normalize=True):
    labels = labels.long()

    n_classes = probabilities.shape[-1]
    y_true_one_hot = F.one_hot(labels.squeeze(), num_classes=n_classes)

    nll = -torch.sum(y_true_one_hot * torch.log(probabilities + 1e-9))
    if normalize:
        nll /= len(labels)
    return nll

def compute_brier(probs, labels, normalize=False):
    labels = labels.long()

    # Ensure labels are on the same device as probs
    if labels.device != probs.device:
        labels = labels.to(probs.device)
        
    n_classes = probs.size(-1)

    y_true_one_hot = F.one_hot(labels.squeeze(), num_classes=n_classes)

    if normalize:
        brier_score = torch.mean(torch.sum((probs - y_true_one_hot) ** 2, dim=-1))
    else:
        brier_score = torch.sum(torch.sum((probs - y_true_one_hot) ** 2, dim=-1))
    return brier_score

def compute_ece(probabilities, labels, num_bins=20, div_factor=None):

    labels = torch.as_tensor(labels)
    confidences = torch.max(probabilities, dim=1)[0]

    if labels.device != probabilities.device:
        labels = labels.to(probabilities.device)

    denom = confidences.shape[0]
    if div_factor is not None:
       denom = div_factor

    predictions = torch.argmax(probabilities, dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, num_bins+1)
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        # samples in current bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = torch.sum(in_bin).item()
        if bin_size > 0:
            accuracy_in_bin = torch.mean(accuracies[in_bin].float()).item()
            confidence_in_bin = torch.mean(confidences[in_bin]).item()
            ece += abs(accuracy_in_bin - confidence_in_bin) * (bin_size / denom)
    return ece

def compute_mcc(all_preds, all_labels):
    if isinstance(all_preds, torch.Tensor):
        all_preds = all_preds.cpu()
    if isinstance(all_preds, torch.Tensor):
        all_labels = all_labels.cpu()
        
    mcc = matthews_corrcoef(all_labels, all_preds)
    return mcc


### evaluation
# kwargs include: n_particles=None, swag_samples=None, swag_sample_scale=0.0, swag_cov_mat=None, dropout_samples=None, ensemble=None, P=None, accelerator=None, causal_lm=False
def evaluate_task(model, method, train_dataloader, eval_dataloader, test_dataloader, prefix='', save_path=None, accelerator=None, log_summary=True, **kwargs):
  results = ''
  summary_metrics = {}
  data = [('Train', train_dataloader), ('Eval', eval_dataloader), ('Test', test_dataloader)]
  for split, dataloader in data:
    metrics, split_report = compute_metrics(model, dataloader, split, method, accelerator=accelerator, **kwargs)
    line = f"{split} accuracy = {metrics['acc']:.3f}, loss = {metrics['loss']:.3f}, nll = {metrics['nll']:.3f}, ece = {metrics['ece']:.3f}, brier = {metrics['brier']:.3f}, mcc = {metrics['mcc']:.3f}"
    results += line+'\n'
    accelerator.print(line)
    pre = prefix + split
    summary_metrics[f'{pre}_accuracy'] = metrics['acc']
    summary_metrics[f'{pre}_loss'] = metrics['loss']
    summary_metrics[f'{pre}_nll'] = metrics['nll']
    summary_metrics[f'{pre}_ece'] = metrics['ece']
    summary_metrics[f'{pre}_brier'] = metrics['brier']
    summary_metrics[f'{pre}_mcc'] = metrics['mcc']

  if accelerator.is_main_process:
    if save_path:
        print(f'Saving evaluation results to path {save_path}/eval.txt')
        with open(os.path.join(save_path, "eval.txt"), "w") as text_file:
            text_file.write(results)
    if log_summary:
        wandb.run.summary.update(summary_metrics)
  accelerator.wait_for_everyone()

  return summary_metrics

# ensemble_probs is [num_samples , num_models, classes]
def compute_metrics_given_probs(ensemble_probs, labels, accelerator):
    if not accelerator.is_main_process:
        raise Exception('needs to be called on main process')

    probs = ensemble_probs.mean(dim=1) # mean over models; [num_samples, classes]
    preds = probs.argmax(dim=-1) # [num_samples]
    member_preds = ensemble_probs.argmax(dim=-1) # [num_samples, num_models]

    if torch.any(probs < 0) or torch.any(probs > 1):
        print('out of [0,1] range in probs')
    if torch.any(ensemble_probs < 0) or torch.any(ensemble_probs > 1):
        print('out of [0,1] range in ensemble_probs')
    
    
    labels = torch.tensor(labels, device = ensemble_probs.device)
    
    # inits
    correct = 0
    total = 0
    nll = 0
    ece = 0
    brier = 0
    entropy = 0
    MI = 0
    across_model_entropy = 0

    # computations
    correct = preds.eq(labels).view_as(preds).sum().item()
    
    total = len(labels)

    entropies = -torch.sum(torch.log(probs + 1e-20) * probs, 1) # [num_samples]
    entropy += torch.sum(entropies) / len(labels)

    # ensemble_probs is [num_samples , num_models, classes]
    MIs = entropies - torch.mean(torch.sum(-ensemble_probs * torch.log(ensemble_probs + 1e-20), axis=-1), axis=1) # (num_samples)
    MI = torch.sum(MIs) / len(labels)

    across_model_entropies_per_model = -torch.sum(torch.log(ensemble_probs + 1e-20) * ensemble_probs, dim=-1) # [num_samples, num_models]
    across_model_entropies = torch.mean(across_model_entropies_per_model, dim = 1) # [num_samples]
    across_model_entropy = torch.sum(across_model_entropies_per_model) / len(labels)

    # member preds is [num_samples, num_models]
    disagreement_ratios = 1 - torch.sum(member_preds.permute(1,0) == preds, axis=0) / ensemble_probs.size(1) # (num_samples)

    ece = compute_ece(probs, labels)
    mcc = compute_mcc(preds, labels)
    nll = compute_nll(probs, labels, normalize=False)
    brier = compute_brier(probs, labels)
    nll /= len(labels)
    brier /= len(labels)

    metrics = {}
    metrics = {} 
    metrics['acc'] = correct / total
    metrics['ece'] = ece
    metrics['brier'] = brier
    metrics['nll'] = nll
    metrics['mcc'] = mcc
    metrics['entropy'] = entropy.item()
    metrics['across_model_entropy'] = across_model_entropy.item()
    metrics['MI'] = MI.item()

    report_str = 'multimodel eval metrics:\n'+str(metrics)
    metrics['entropies'] = entropies
    metrics['disagreement_ratios'] = disagreement_ratios
    metrics['across_model_entropies'] = across_model_entropies
    metrics['MIs'] = MIs

    return metrics, report_str


def compute_metrics(model, dataloader, split, method, n_particles=None, swag_samples=None, swag_sample_scale=0.0, swag_cov_mat=True, dropout_samples=None, ensemble=None, P=None, accelerator=None, causal_lm=False, return_probs=False):
   
    model.eval()
    loss_fn = CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0
    nll = 0
    ece = 0
    brier = 0
    all_preds = []
    all_labels = []
    all_probs_list = []
    all_ensemble_probs_list = []
    entropies_list = []
    across_model_entropies_list = []
    disagreement_ratio_list = []
    entropy = 0
    MI = 0
    MI_list =[]
    across_model_entropy = 0

    for batch in tqdm(dataloader, disable=not accelerator.is_main_process):
        with torch.no_grad():
            if method == 'swag':
                losses = []
                swag_logits = []

                if swag_samples is None:
                    swag_samples = 1
                    swag_sample_scale = 0.0
                for i in range(swag_samples):

                    model.sample(scale=swag_sample_scale, cov=swag_cov_mat)
                    model.eval()
                    
                    output = model(**{'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']})
                    logits = output.logits[:, -1, :] if causal_lm else output.logits # if causal lm we use last representation's logits
                    swag_logits.append(logits)
                    loss = loss_fn(logits, batch['labels'])
                    losses.append(loss)
                total_loss += sum(tensor for tensor in losses) / len(losses)

                probs = torch.stack([F.softmax(logs, dim=-1) for logs in swag_logits]) # (num_models, bsz, classes)

                ensemble_probs = probs.permute(1, 0, 2) #(bsz , num_models, classes)

                member_preds = probs.argmax(dim=-1) # all models' predictions (num_models, bsz)
                member_preds = member_preds.permute(1,0) # permute to (bsz, num_models)
                preds = mean_logits(swag_logits).argmax(dim=-1) # swag ensemble predictions (bsz)

                probs = probs.mean(dim=0) # get swag 'ensemble' probs

            elif method == 'base':
                losses = []
                dropout_logits = []

                model.eval()
                if dropout_samples is None or dropout_samples==1:
                    dropout_samples = 1
                else:
                    model.apply(set_dropout_training)
                for i in range(dropout_samples):
                    output = model(**{'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']})
                    logits = output.logits[:, -1, :] if causal_lm else output.logits # if causal lm we use last representation's logits
                    dropout_logits.append(logits)
                    loss = loss_fn(logits, batch['labels'])
                    losses.append(loss)
                total_loss += sum(tensor for tensor in losses) / len(losses)

                probs = torch.stack([F.softmax(logs, dim=-1) for logs in dropout_logits]) # (num_models, bsz, classes)

                ensemble_probs = probs.permute(1, 0, 2) #(bsz , num_models, classes)

                member_preds = probs.argmax(dim=-1) # all models' predictions (num_models, bsz)
                member_preds = member_preds.permute(1,0) # permute to (bsz, num_models)
                preds = mean_logits(dropout_logits).argmax(dim=-1) # ensemble predictions (bsz)
                
                probs = probs.mean(dim=0) # get mc dropout 'ensemble' probs (bsz, classes)

            else:
                raise Exception('No such method implemented: ', method)

            collected_preds, collected_member_preds, collected_labels, collected_probs, collected_ensemble_probs = accelerator.gather_for_metrics((preds, member_preds, batch['labels'], probs, ensemble_probs))
            if accelerator.is_main_process:
                all_preds += collected_preds.tolist()
                all_labels += collected_labels.tolist()
                all_probs_list.append(collected_probs) #[(bsz, classes)...]
                all_ensemble_probs_list.append(collected_ensemble_probs) # [(bsz , num_models, classes) .. ]
            accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            nll += compute_nll(collected_probs, collected_labels, normalize=False)
            
            brier += compute_brier(collected_probs, collected_labels)
            correct += collected_preds.eq(collected_labels.view_as(collected_preds)).sum().item()
            total += len(collected_labels)
            

            batch_entropies = -torch.sum(torch.log(collected_probs + 1e-20) * collected_probs, 1)
            entropies_list.append(batch_entropies)
            
            entropy += torch.sum(batch_entropies)
            
            batch_MIs = batch_entropies - torch.mean(torch.sum(-collected_ensemble_probs * torch.log(collected_ensemble_probs + 1e-20), axis=-1), axis=1) # (bsz)
            MI_list.append(batch_MIs)
            MI += torch.sum(batch_MIs)

            #collected_ensemble_probs is (bsz , num_models, classes)
            collected_ensemble_probs = collected_ensemble_probs.permute(0, 2, 1)  # Adjusts to (bsz, classes, num_models)

            across_model_entropies = -torch.sum(torch.log(collected_ensemble_probs + 1e-20) * collected_ensemble_probs, dim=-1)
            
            
            across_model_entropies_per_sample = torch.mean(across_model_entropies, dim=1)
            across_model_entropies_list.append(across_model_entropies_per_sample)
            
            across_model_entropy += torch.sum(across_model_entropies)

            
            collected_member_preds = collected_member_preds.permute(1,0) # permute back to (num_models, bsz)
            num_samples = dropout_samples if method == 'base' else swag_samples
            disagreement_ratio = 1 - torch.sum(collected_member_preds == collected_preds, axis=0) / num_samples # (bsz)
            disagreement_ratio_list.append(disagreement_ratio)
        accelerator.wait_for_everyone()

    collected_loss = accelerator.gather_for_metrics(total_loss).sum().item()

    if accelerator.is_main_process:
        entropies = torch.cat(entropies_list, dim=0)
        disagreement_ratios = torch.cat(disagreement_ratio_list, dim=0)
        across_model_entropies = torch.cat(across_model_entropies_list, dim=0)
        MIs = torch.cat(MI_list, dim=0)

        all_ensemble_probs = torch.cat(all_ensemble_probs_list, dim=0)
        
    else:
        entropies = torch.empty(1, device=accelerator.device)
        disagreement_ratios = torch.empty(1, device=accelerator.device)
        across_model_entropies = torch.empty(1, device=accelerator.device)
        MIs = torch.empty(1, device=accelerator.device)

        all_ensemble_probs = torch.empty(1, device=accelerator.device)

    metrics = {
        'loss': torch.tensor(0.0, device=accelerator.device),
        'acc': torch.tensor(0.0, device=accelerator.device),
        'ece': torch.tensor(0.0, device=accelerator.device),
        'brier': torch.tensor(0.0, device=accelerator.device),
        'nll': torch.tensor(0.0, device=accelerator.device),
        'mcc': torch.tensor(0.0, device=accelerator.device),
        'entropy': torch.tensor(0.0, device=accelerator.device),
        'across_model_entropy': torch.tensor(0.0, device=accelerator.device),
        'MI': torch.tensor(0.0, device=accelerator.device)
    }

    if accelerator.is_main_process:
        all_probs = torch.cat(all_probs_list, dim=0) #(all_samples, classes)
        ece = compute_ece(all_probs, all_labels)
        mcc = compute_mcc(all_preds, all_labels)
        nll /= len(dataloader.dataset)
        brier /= len(dataloader.dataset)
        MI /= len(dataloader.dataset)
        across_model_entropy /= len(dataloader.dataset)

        metrics = {} 
        metrics['loss'] = collected_loss / len(dataloader)
        metrics['acc'] = correct / total
        metrics['ece'] = ece
        metrics['brier'] = brier
        metrics['nll'] = nll
        metrics['mcc'] = mcc
        metrics['entropy'] = entropy.item()
        metrics['across_model_entropy'] = across_model_entropy.item()
        metrics['MI'] = MI.item()
    accelerator.wait_for_everyone()

    report_str = split+':\n'+str(metrics)

    metrics['entropies'] = entropies
    metrics['disagreement_ratios'] = disagreement_ratios
    metrics['across_model_entropies'] = across_model_entropies
    metrics['MIs'] = MIs

    if return_probs:
        return metrics, report_str, all_ensemble_probs, all_labels
    
    return metrics, report_str


def ood_metrics_entropy(in_dis, out_dis):
    """ Compute AUROC, AUPRC and FPR80 score.
    Args:
        in_dis: the average entropy of the in-distribution data.
        out_dis: the average entropy of the out-of-distribution data.
    Return:
        The AUROC, AUPR IN, AUPR OUT, FPR80, FPR95 and the corresponding
        detection errors. Formulas from https://arxiv.org/pdf/1706.02690.pdf.
    """
    with torch.no_grad():
        y_true = np.concatenate([np.zeros(in_dis.numel(), dtype=np.int64),
                                 np.ones(out_dis.numel(), dtype=np.int64)])
        y_scores = in_dis.flatten().tolist()+ out_dis.flatten().tolist()

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name='example estimator')
        display.plot()
        plt.show()

        roc_auc = roc_auc_score(y_true, y_scores)
        return roc_auc, \
                average_precision_score(y_true, y_scores, pos_label=0), \
                average_precision_score(y_true, y_scores, pos_label=1)