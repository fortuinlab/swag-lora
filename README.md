# Gaussian Stochastic Weight Averaging for Bayesian Low-Rank Adaptation of Large Language Models

This repo contains the implementation for the paper ["Gaussian Stochastic Weight Averaging for Bayesian Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2405.03425). Please refer to the paper for a detailed description of our approach and experimental information. 

## Installation
```
pip install -r requirements.txt (??)
source sc_venv_template/activate.sh # activate environment
```

## Running experiments
We use Accelerate and Hydra to run our experiments.

### Running training script

#### Running training using Hydra
To launch a training run locally:
```
accelerate launch launch_exp_hydra.py \
model=llama_7b_hf \
method=swag \
experiment.task=obqa \
method.force_save=5 \
method.swag_start=8 \
method.swag_anneal_epochs=5 \
method.swag_learning_rate=1e-4 \
experiment.learning_rate=6e-4 \
experiment.num_epochs=18 \
experiment.batch_size=16
```

To run our code offline, use the download scripts from download_models4cluster.py and download_datatsets4cluster.py to locally store models and datasets. Add the following flags to your hydra call:
``` 
experiment.offline=True \
experiment.data_path=/path/to/your/data/folder \
experiment.model_path=/path/to/your/models/folder \
```

#### Running training via job script 
After setting up the training job information and hyperparameters in train:
```
sbatch train.bash # slurm
bash train.bash # local execution
```
 ### Running evaluation scripts
We have created scripts for evaluating base / MAP ([evaluate_base.bash](./evaluate_base.bash)), MC dropout ([evaluate_dropout.bash](./evaluate_dropout.bash)), Ensemble ([evaluate_ensemble.bash](./evaluate_ensemble.bash)), SWAG ([evaluate_swag.bash](./evaluate_swag.bash)), MultiSWAG ([evaluate_multiswag.bash](./evaluate_multiswag.bash)).

```
# e.g.
sbatch evaluate_swag.bash # slurm
bash evaluate_swag.bash # local execution
```

The scripts are set up to evaluate both ID and OOD performance, which are set using the task, subtask, ood_task and ood_subtask variables in the evaluation scripts.

The trained model path(s) to be evaluated have to be specified in the script's MODEL_PATH variable for single-model methods (base/dropout/SWAG) and the paths variable for multi-model methods (ensemble/MultiSWAG). Ensembles and MultiSWAG are implemented post-hoc, requiring paths to several (independently trained) models to use as ensemble members.

In order to evaluate SWA (or MultiSWA), use [evaluate_swag.py](./evaluate_swag.bash) (or [evaluate_multiswag.py](./evaluate_multiswag.bash)) and set ``NUM_SAMPLES=1`` (default) and ``SWAG_SAMPLE_SCALE=0.0``. This only samples one network from SWAG and incorporates no noise, directly setting the single network to the mean of the posterior over the weights.

``NUM_SAMPLES`` determines the number of samples to use when evaluating Dropout or (Multi)SWAG. ``SWAG_SAMPLE_SCALE`` scales the covariance of the learned SWAG distribution we are sampling from. 

### Notable flags/hyperparameters

Important hydra flags/hyperparameters are discussed below. See a more comprehensive description of hyperparameters in [config.yaml](./conf/config.yaml) and [swag.yaml](./conf/method/swag.yaml). SWAG-specific flags are represented as `method.<var_name>` as in Hydra.



| Hyperparameter                     | Description                                                                 | Possible Values                     |
|------------------------------------|-----------------------------------------------------------------------------|-------------------------------------|
| `model`                            | Model to use                                              | `llama_7b_hf`, `roberta-base`, `roberta-large`   |
| `experiment.learning_rate`         | The learning rate for training                                       | Any float (e.g., `0.001`, `0.01`) |
| `method`                           | The method configuration to use                                           | Only `swag`     |
| `method.swag_learning_rate`        | Learning rate for the SWAG training                                          | Any float (e.g., `1e-4`)      |
| `method.swag_anneal_epochs`        | Number of epochs for annealing in SWAG                                     | Any integer (e.g., `5`)             |
| `method.swag_anneal_strategy`      | Strategy for annealing in SWAG                                             | `constant`, `linear`, `cosine`, `cosine_hard_restarts`                             |
| `method.swag_start`                | The epoch to start SWAG collection                                                    | Any integer (e.g., `8`)             |
| `method.modules_to_swag`           | Modules over which to learn the SWAG distribution (supports only LoRA layers, only layers with gradients, and all)                                                   | `grad_only`, `lora_only`, `all`     |
| `method.swag_max_num_models`       | Maximum number of models to maintain for covariance approximation in SWAG                               | Any integer (e.g., `5`)             |
| `method.swag_cov_mat`              | Whether to learn a covariance matrix in SWAG (if `False`, only a diagonal covariance is used when sampling)                               | `True`, `False`                     |
| `method.force_save`                | Force save epoch for early stopping SWAG training; SWAG model will be saved `method.force_save` epochs after `method.swag_start`                                            | Any integer (e.g. 5)             |
| `experiment.task`                  | The (ID) task for training/evaluation                                           | `obqa`, `cqa`, `swag`, `mmlu`, `arc-e`, `arc-c`, `cola`, `mnli`, `mrpc`, (other GLUE tasks...)     |
| `experiment.subtask`               | (ID) Subtask for training/evaluation                                       | Subtask name (e.g. `experiment.task=mmlu`, `experiment.subtask=anatomy`)                    |
| `experiment.ood_task`              | (OOD) task for evaluation                      | `obqa`, `cqa`, `swag`, `mmlu`, `arc-e`, `arc-c`, `cola`, `mnli`, `mrpc`, (other GLUE tasks...)                  |
| `experiment.ood_subtask`           | (OOD) subtask for evaluation task.                                               | Subtask name (e.g. `experiment.ood_task=mmlu`, `experiment.ood_subtask=anatomy`)                |
| `experiment.ood_batch_size`        | Batch size for the OOD  task                                                | Any integer (e.g., `32`)            |
| `experiment.num_epochs`            | Total number of epochs for training                                        | Any integer (e.g., `20`)            |
| `experiment.batch_size`            | Batch size for the training                                              | Any integer (e.g., `16`)            |
| `experiment.overwrite`             | Whether to overwrite existing experiments                                  | `True`, `False`                     |
| `experiment.data_path`             | Path to the data folder for the tasks (only required if `experiment.offline=True`)                                        | `/path/to/your/data/folder`                 |
| `experiment.model_path`            | Path to the model folder  (only required if `experiment.offline=True`)                                       | `/path/to/your/models/folder`                 |
| `experiment.wandb_path`            | Path to the Weights and Biases logging folder                               | `/path/to/your/wandb/folder`                |
| `experiment.wandb_group`           | Group name for Weights and Biases tracking                                 | Any group name                      |
| `experiment.exp_name`              | (Optional) Name of the experiment                                                     | Any experiment name                 |


### Cite this work
If you are using this codebase in your work, please cite it as:
```
@misc{onal2024gaussian,
      title={Gaussian Stochastic Weight Averaging for Bayesian Low-Rank Adaptation of Large Language Models}, 
      author={Emre Onal and Klemens Flöge and Emma Caldwell and Arsen Sheverdin and Vincent Fortuin},
      year={2024},
      eprint={2405.03425},
      archivePrefix={arXiv},
}
```
