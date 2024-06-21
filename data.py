import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from dataclasses import dataclass
from datasets import load_from_disk
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    DataCollatorWithPadding
)
from datasets import load_dataset, DatasetDict
from typing import Optional, Union
from torch.utils.data import ConcatDataset, DataLoader


GLUE_task_to_keys = {
  "cola": ("sentence", None),
  "mnli": ("premise", "hypothesis"),
  "mrpc": ("sentence1", "sentence2"),
  "qnli": ("question", "sentence"),
  "qqp": ("question1", "question2"),
  "rte": ("sentence1", "sentence2"),
  "sst2": ("sentence", None),
  "wnli": ("sentence1", "sentence2")
}

MCQA_task_to_context_keys = {
  'swag':'sent1',
  'cqa':'question',
  'mmlu':'question',
  'obqa':'question_stem',
  'arc-c':'question',
  'arc-e':'question',
}

N_CLASSES_DICT = {'mrpc':2, 'cola':2, 'mnli':3, 'qnli':2, 'qqp':2, 'rte':2, 'sst2':2, 'wnli':2}
TASK_TYPE_DICT = {key:'SEQ_CLS' for key in GLUE_task_to_keys}
TASK_TYPE_DICT.update({key:'MCQA' for key in MCQA_task_to_context_keys})

## GLUE
def load_glue_data(config, task, accelerator, batch_size=32, subtask=None, offline=False, data_path='./data'):

  batch_size_per_accum = batch_size // config.experiment.gradient_accumulation_steps

  def tokenize_function(examples):
    args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )

    outputs = tokenizer(*args, truncation=True, max_length=None)
    return outputs

  def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")

  if task in GLUE_task_to_keys:
    sentence1_key, sentence2_key = GLUE_task_to_keys[task]
  else:
    raise Exception('No other dataset implemented; Must be GLUE dataset')

  model_name_or_path = os.path.join(config.experiment.model_path, config.model.model_name) if config.experiment.offline else config.model.model_name
    
  tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='right')
  if getattr(tokenizer, "pad_token_id") is None:
      tokenizer.pad_token_id = tokenizer.eos_token_id
      tokenizer.pad_token = tokenizer.eos_token
  
  if 'meta-llama' in config.model.model_name:
    cache_dir = '/p/scratch/hai_baylora/cache_dir'
    Llama_tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', add_prefix_space=True, cache_dir=cache_dir)
    assert getattr(Llama_tokenizer, "pad_token_id") is None
    Llama_tokenizer.pad_token = Llama_tokenizer.eos_token
    Llama_tokenizer.pad_token_id = Llama_tokenizer.eos_token_id
    tokenizer = Llama_tokenizer

  print('data_path load_glue', data_path)
  if offline:
      datasets = load_from_disk(os.path.join(data_path, f"data_{task}"))
  elif task in GLUE_task_to_keys:
      datasets = load_dataset("glue", task)
  else:
      raise Exception(f'Dataset {task} not implemented')

  # get number of classes
  train_features = datasets['train'].features
  num_classes = len(train_features['label'].names)
  
  remove_columns_list = ["idx", sentence1_key, sentence2_key] if sentence2_key else ["idx", sentence1_key]
  existing_columns = datasets["train"].column_names
  remove_columns = [col for col in remove_columns_list if col in existing_columns]
  datasets = datasets.rename_column('label', 'labels')

  test_name = 'test_'+subtask if task =='mnli' else 'test'
  val_name = 'validation_'+subtask if task=='mnli' else 'validation'


  with accelerator.main_process_first():
    # split train into train and val if test labels not available
    if datasets[test_name][0]['labels'] == -1:
      train_val_split = datasets['train'].train_test_split(test_size=config.experiment.val_split_size, seed=config.experiment.seed)
      
      datasets = DatasetDict({
        'train': train_val_split['train'],
        val_name: train_val_split['test'],
        test_name: datasets[val_name]
      })
    
    tokenized_datasets = datasets.map(
      tokenize_function,
      batched=True,
      remove_columns=remove_columns,
    )
  accelerator.wait_for_everyone()

  # Instantiate dataloaders
  train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size_per_accum)

  eval_dataloader = DataLoader(
  tokenized_datasets[val_name], shuffle=False, collate_fn=collate_fn, batch_size=batch_size_per_accum
    )
  test_dataloader = DataLoader(tokenized_datasets[test_name], shuffle=False, collate_fn=collate_fn, batch_size=batch_size_per_accum)
  return tokenizer, train_dataloader, eval_dataloader, test_dataloader, num_classes

## MCQA collator
# from huggingface MCQA docs: https://huggingface.co/docs/transformers/en/tasks/multiple_choice:
@dataclass
class DataCollatorForMultipleChoice: 
  """
  Data collator that will dynamically pad the inputs for multiple choice received.
  """
  tokenizer: PreTrainedTokenizerBase
  padding: Union[bool, str, PaddingStrategy] = True
  max_length: Optional[int] = None
  pad_to_multiple_of: Optional[int] = None

  def __call__(self, features):
      label_name = "label" if "label" in features[0].keys() else "labels"
      labels = [feature.pop(label_name) for feature in features]
      batch_size = len(features)
      num_choices = len(features[0]["input_ids"])
      flattened_features = [
          [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
      ]
      flattened_features = sum(flattened_features, [])

      batch = self.tokenizer.pad(
          flattened_features,
          padding=self.padding,
          max_length=self.max_length,
          pad_to_multiple_of=self.pad_to_multiple_of,
          return_tensors="pt",
      )

      batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
      batch["labels"] = torch.tensor(labels, dtype=torch.int64)
      return batch

def load_mcqa_data(config, task, accelerator, batch_size=32, subtask=None, verbose=False, offline=False, data_path='./data'):
  num_choices_dict = {'swag':4, 'cqa':5, 'mmlu':4, 'obqa':4, 'arc-e':4, 'arc-c':4}
  label_map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, '':'no_label', '1':0, '2':1, '3':2, '4':3}

  subtask_groups = {'law': ['international_law', 'jurisprudence', 'professional_law'],
                     'cs':['college_computer_science', 'computer_security', 'high_school_computer_science', 'machine_learning'], 
                     'eng':['electrical_engineering'], 
                     'health':['anatomy', 'clinical_knowledge', 'college_medicine', 'human_aging', 'nutrition', 'professional_medicine', 'virology'],
                     'ss':['econometrics', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_microeconomics', 'high_school_psychology', 'human_sexuality', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy'],
                     'stem':['abstract_algebra', 'anatomy', 'astronomy', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_physics', 'computer_security', 'conceptual_physics', 'electrical_engineering', 'elementary_mathematics', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_mathematics', 'high_school_physics', 'high_school_statistics', 'machine_learning']}


  causal_lm = False
  if 'meta-llama' in config.model.model_name:
    causal_lm = True

  batch_size_per_accum = batch_size // config.experiment.gradient_accumulation_steps


  if task not in MCQA_task_to_context_keys:
    raise Exception(f'Dataset {task} not implemented.')
    
  def tokenize_function(examples):
      num_choices = num_choices_dict[task]
      first_sentences = [[context] * num_choices for context in examples[MCQA_task_to_context_keys[task]]]

      if task == 'swag':
        second_sentences = [ [f"{header} {examples[end][i]}" for end in ["ending0", "ending1", "ending2", "ending3"]] for i, header in enumerate(examples["sent2"]) ]
      elif task in ['cqa', 'obqa', 'arc-c', 'arc-e']:
        second_sentences = [ [choice for choice in examples['choices'][i]['text']] for i in range(len(examples[MCQA_task_to_context_keys[task]]))]
      elif task == 'mmlu':
        second_sentences = [ [choice for choice in examples['choices'][i]] for i in range(len(examples['question'])) ]

      first_sentences = sum(first_sentences, [])
      second_sentences = sum(second_sentences, [])

      if len(first_sentences) != len(second_sentences):
        for i, choice_set in enumerate(examples['choices']):
          if len(choice_set['text']) != num_choices:
            print(f"Mismatch in example {i}: Expected {num_choices} choices, found {len(choice_set['text'])}")
            print(choice_set['text'])
            print(choice_set)

      if len(examples['choices']) != len(examples[MCQA_task_to_context_keys[task]]):
        raise Exception("Mismatch in number of questions and choice sets")
      if len(first_sentences) != len(second_sentences):
        raise ValueError(f"Length mismatch: first_sentences ({len(first_sentences)}) vs second_sentences ({len(second_sentences)})")

      tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
      tokenized_output = {k: [v[i : i + num_choices] for i in range(0, len(v), num_choices)] for k, v in tokenized_examples.items()}

      if task in ['cqa','obqa', 'arc-c', 'arc-e']:
        tokenized_output['labels'] = [label_map[label] for label in examples['answerKey']]
      if task == 'mmlu':
        tokenized_output['labels'] = examples['answer']
      return tokenized_output
  
  def preprocess_for_mcqa(example):
    # need to account for arc-c/e having numerical labels if we are doing generative mcqa
    if task in ['cqa', 'obqa', 'arc-c', 'arc-e']:
      if task == 'obqa':
        example['choices']['text'] = [text if text.endswith('.') else text + '.' for text in example['choices']['text']]
      example['choices']['text'] = [text[0].upper() + text[1:] if text else text for text in example['choices']['text']]

    elif task in ['swag']:
      # Format endings
      for i in range(4):
          ending_key = f'ending{i}'
          ending = example[ending_key]

          # Ensure ending is a complete sentence (ends with a period)
          if not ending.endswith('.'):
              ending += '.'

          # Capitalize the first letter of the ending
          if ending:
              ending = ending[0].upper() + ending[1:]

          example[ending_key] = ending

      # Combine sentences to form the complete scenario
      example['complete_scenario'] = [f"{example['sent1']} {example['sent2']} {example[f'ending{i}']}" for i in range(4)]
    else:
      # mmlu
      #print('example keys: ', example.keys())
      #print(example)
      example['choices'] = [text if text.endswith('.') else text + '.' for text in example['choices']]
      example['choices'] = [text[0].upper() + text[1:] if text else text for text in example['choices']]
    return example

  def tokenize_function_for_causal_lm(examples):
    if task in ['obqa', 'arc-c', 'arc-e']:
      choices_list = [' '.join(f'{label}. {text}' for label, text in zip(['A', 'B', 'C', 'D'], choices['text'])) for choices in examples['choices']]
      texts = [f"Select one of the choices that answers the following question: {question} Choices: {choices} Answer:" for question, choices in zip(examples[MCQA_task_to_context_keys[task]], choices_list)]
      tokenized_output = tokenizer(texts, padding=False, truncation=True)
      tokenized_output["labels"] = [label_map[label] for label in examples["answerKey"]]
    elif task  == 'cqa':
      choices_list = [' '.join(f'{label}. {text}' for label, text in zip(['A', 'B', 'C', 'D', 'E'], choices['text'])) for choices in examples['choices']]
      texts = [f"Select one of the choices that answers the following question: {question} Choices: {choices} Answer:" for question, choices in zip(examples[MCQA_task_to_context_keys[task]], choices_list)]
      tokenized_output = tokenizer(texts, padding=False, truncation=True)
      tokenized_output["labels"] = [label_map[label] for label in examples["answerKey"]]
    elif task == 'mmlu':
      choices_list = [' '.join(f'{label}. {text}' for label, text in zip(['A', 'B', 'C', 'D'], choices)) for choices in examples['choices']]
      texts = [f"Select one of the choices that answers the following question: {question} Choices: {choices} Answer:" for question, choices in zip(examples[MCQA_task_to_context_keys[task]], choices_list)]
      tokenized_output = tokenizer(texts, padding=False, truncation=True)
      tokenized_output["labels"] = [label_map.get(label, label) for label in examples["answer"]]
    else:
      raise Exception('Only mmlu, cqa, obqa, arc-e/c MCQA tokenization is implemented in data.py')
    return tokenized_output


  def prepare_dataset(config, dataset, task, train_name, causal_lm):
    if task in ['arc-c', 'arc-e']:
      filtered_train = dataset["train"].filter(lambda example: len(example['choices']['label']) == 4)
      filtered_valid = dataset["validation"].filter(lambda example: len(example['choices']['label']) == 4)
      filtered_test = dataset["test"].filter(lambda example: len(example['choices']['label']) == 4)
      dataset["train"] = filtered_train
      dataset["validation"] = filtered_valid
      dataset["test"] = filtered_test

    # obqa, mmlu have their test set labels
    if task not in ['obqa', 'mmlu', 'arc-c', 'arc-e']:
      # need to create our own test split
      train_val_split = dataset[train_name].train_test_split(test_size=config.experiment.val_split_size, seed=config.experiment.seed)
      dataset = DatasetDict({
          train_name: train_val_split['train'],
          'validation': train_val_split['test'],
          'test': dataset['validation']
        })
      
    if causal_lm:
      dataset[train_name] = dataset[train_name].map(preprocess_for_mcqa)
      dataset['validation'] = dataset['validation'].map(preprocess_for_mcqa)
      dataset['test'] = dataset['test'].map(preprocess_for_mcqa)


    signature_columns = ['input_ids', 'attention_mask', 'label', 'labels']
    remove_columns = list(set(dataset[train_name].column_names) - set(signature_columns))

    with accelerator.main_process_first():
      tokenized_dataset = dataset.map(
        tokenize_function if not causal_lm else tokenize_function_for_causal_lm,
        batched=True,
        remove_columns=remove_columns,
      )
    accelerator.wait_for_everyone()

    return tokenized_dataset
  

  model_name_or_path = os.path.join(config.experiment.model_path, config.model.model_name) if config.experiment.offline else config.model.model_name
  tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
  if getattr(tokenizer, "pad_token_id") is None:
      tokenizer.pad_token_id = tokenizer.eos_token_id
      tokenizer.pad_token = tokenizer.eos_token

  if 'meta-llama' in config.model.model_name:
    trash_dir = '/p/scratch/hai_baylora/trash3'
    Llama_tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', add_prefix_space=True, cache_dir=trash_dir)
    assert getattr(Llama_tokenizer, "pad_token_id") is None
    Llama_tokenizer.pad_token = Llama_tokenizer.eos_token
    Llama_tokenizer.pad_token_id = Llama_tokenizer.eos_token_id
    tokenizer = Llama_tokenizer

  train_name = 'train'
  if offline:
    if task =='mmlu':
      train_name = 'train' if subtask == 'auxiliary_train' else 'dev' # no train set for non-aux-train subtasks
      if subtask in subtask_groups: # concatenate several MMLU subtasks (need same splits)
        datasets = [load_from_disk(os.path.join(data_path, f"data_{task}_{subtask_name}")) for subtask_name in subtask_groups[subtask]]

      else: # use single MMLU subtask
        dataset = load_from_disk(os.path.join(data_path, f"data_{task}_{subtask}"))
    else:
      dataset = load_from_disk(os.path.join(data_path, f"data_{task}"))
  else:
    if task == 'swag':
      dataset = load_dataset("swag", "regular")
    elif task == 'cqa':
      dataset = load_dataset('commonsense_qa')
    elif task == 'obqa':
      dataset = load_dataset('openbookqa')
    elif task == 'mmlu':
      train_name = 'train' if subtask == 'auxiliary_train' else 'dev' # no train set for non-aux-train subtasks
      if subtask in subtask_groups: # concatenate several MMLU subtasks (need same splits)
        datasets = [load_dataset('cais/mmlu', task_name) for task_name in subtask_groups[subtask]]
      else: # use single MMLU subtask
        dataset = load_dataset('cais/mmlu', subtask)
    elif task == 'arc-c':
      dataset = load_dataset('allenai/ai2_arc', 'ARC-Challenge')
    elif task == 'arc-e':
      dataset = load_dataset('allenai/ai2_arc', 'ARC-Easy')
    

  #datasets = [load_from_disk(os.path.join(data_path, f"data_{task}_{subtask_name}")) for subtask_name in subtask_groups[subtask]]
  if task == 'mmlu' and subtask in subtask_groups:
    tokenized_dataset = DatasetDict()
    tokenized_subtasks = [prepare_dataset(config, subtask_data, task, train_name, causal_lm) for subtask_data in datasets]
    for split in ['dev', 'validation', 'test']:
      tokenized_dataset[split] = ConcatDataset([x[split] for x in tokenized_subtasks])
        # for split in ['dev', 'validation', 'test']:
        #   dataset[split] = ConcatDataset([data[split] for data in datasets])
  else:
    tokenized_dataset = prepare_dataset(config, dataset, task, train_name, causal_lm)

  if causal_lm:
    collate_fn = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))
  else:
    collate_fn = DataCollatorForMultipleChoice(tokenizer=tokenizer)
  train_dataloader = DataLoader(tokenized_dataset[train_name], shuffle=True, collate_fn=collate_fn, batch_size=batch_size_per_accum)
  eval_dataloader = DataLoader(tokenized_dataset["validation"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size_per_accum)
  test_dataloader = DataLoader(tokenized_dataset["test"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size_per_accum)

  return tokenizer, train_dataloader, eval_dataloader, test_dataloader, num_choices_dict[task]

