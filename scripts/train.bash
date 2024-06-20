#!/bin/bash
export ACCELERATE_CONFIG_FILE="accelerate_train_config.yaml"

jutil env activate -p hai_baylora
USER="onal1"
WANDB_GROUP="default"
CODE_PATH="/p/project/hai_baylora/"${USER}"/SWAG-LoRA"
WANDB_PATH="/p/project/hai_baylora/"${USER}"/SWAG-LoRA"
OUT_PATH="/p/project/hai_baylora/"${USER}"/SWAG-LoRA/outputs"
DATA_PATH="/p/scratch/hai_baylora/data_${USER}"
MODELS_PATH="/p/scratch/hai_baylora/models2"

TIME="02:00:00"

LR=3e-4
BSZ=16
EPOCHS=2
SWAG_START=1
OOD_BSZ=8

method="swag"
task="obqa"
subtask=""
ood_task="" # specify a task here for OOD evaluation after training
ood_subtask=""
FORCE_SAVE=3
SWAG_DIV_FACTOR=10
swag_anneal_epochs=5

JOB_NAME="0TEST${method}_ID${task}${subtask}_OOD${ood_task}${ood_subtask}_LR${LR}_BSZ${BSZ}_DIV${SWAG_DIV_FACTOR}"
EXP_NAME="0TEST${method}_ID${task}${subtask}_OOD${ood_task}${ood_subtask}_LR${LR}_BSZ${BSZ}_DIV${SWAG_DIV_FACTOR}"


SWAG_LR=$(awk -v lr="$LR" 'BEGIN {print lr / '$SWAG_DIV_FACTOR'}')

# Check if the directory exists
if [ ! -d "$OUT_PATH" ]; then
    # Directory does not exist, so create it
    mkdir -p "$OUT_PATH"
    echo "Directory created: $OUT_PATH"
else
    echo "Directory already exists: $OUT_PATH"
fi

METHODS=($method)
TASKS=($task)
for method in ${METHODS[@]}; do
for task in ${TASKS[@]}; do
        sleep 0.1
        job_file="${OUT_PATH}/${JOB_NAME}.cmd"
echo "#!/bin/bash
#SBATCH --account=hai_baylora
#SBATCH --nodes=1
# ---------------------
#SBATCH --ntasks-per-node=4
# ---------------------
#SBATCH -J ${JOB_NAME}
#SBATCH --output=${OUT_PATH}/${JOB_NAME}.out
#SBATCH --error=${OUT_PATH}/${JOB_NAME}.err
#SBATCH --partition=develbooster
#SBATCH --time=$TIME
#SBATCH --gres=gpu:4
source ~/.bash_profile
module load git
# conda activate mvTCR
source sc_venv_template/activate.sh
accelerate launch ${CODE_PATH}/launch_exp_hydra.py model=llama_7b_hf experiment.learning_rate=$LR method=${method} method.force_save=${FORCE_SAVE} method.swag_learning_rate=$SWAG_LR method.swag_anneal_epochs=$swag_anneal_epochs method.swag_start=$SWAG_START experiment.task=${task} experiment.subtask=$subtask experiment.ood_task=$ood_task experiment.ood_subtask=$ood_subtask experiment.ood_batch_size=$OOD_BSZ experiment.num_epochs=$EPOCHS experiment.batch_size=$BSZ experiment.overwrite=True experiment.data_path=${DATA_PATH} experiment.model_path=${MODELS_PATH} experiment.wandb_path=${WANDB_PATH} experiment.wandb_group=${WANDB_GROUP} experiment.exp_name=$EXP_NAME
" > ${job_file}
        sbatch $job_file
done
done