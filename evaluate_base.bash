#!/bin/bash
export ACCELERATE_CONFIG_FILE="accelerate_inference_config.yaml"

jutil env activate -p hai_baylora

USER="onal1"
WANDB_GROUP="default"
CODE_PATH="/p/project/hai_baylora/"${USER}"/SWAG-LoRA"
WANDB_PATH="/p/project/hai_baylora/"${USER}"/SWAG-LoRA"
OUT_PATH="/p/project/hai_baylora/"${USER}"/SWAG-LoRA/outputs"
DATA_PATH="/p/scratch/hai_baylora/data_${USER}"
MODELS_PATH="/p/scratch/hai_baylora/models2"


### Job specs ### ------------------------------------------
TIME="02:00:00"
PARTITION="develbooster"

model="llama_7b_hf"
task="obqa"
subtask=""
bsz=16

ood_task="mmlu"
ood_subtask="anatomy"
ood_bsz=16
only_ood=True

#NUM_SAMPLES=1  # by default num samples = 1 for base model (i.e. no MC dropout)

RUN_ID=0 # 0, 1, 2, .... for repeat exps
MODEL_PATH="results/obqa/meta-llama/Llama-2-7b-hf/swag/trainbaseswag_obqa_LR3e-4_BSZ16_DIV10_${RUN_ID}/20240326-180433"
JOB_NAME="evalbase_${task}${subtask}-${ood_task}${ood_subtask}_${RUN_ID}"

### --------------------------------------------------------
### ========================================================

# Check if the directory exists
if [ ! -d "$OUT_PATH" ]; then
    # Directory does not exist, so create it
    mkdir -p "$OUT_PATH"
    echo "Directory created: $OUT_PATH"
else
    echo "Directory already exists: $OUT_PATH"
fi

#================================================================


for method in "_"; do
for task_name in $task; do
    sleep 0.1
    job_file="$OUT_PATH/$JOB_NAME.cmd"
echo "#!/bin/bash
#SBATCH --account=hai_baylora
#SBATCH --nodes=1
# ------------------------
#SBATCH --ntasks-per-node=4
# ------------------------
#SBATCH -J $JOB_NAME
#SBATCH --output=$OUT_PATH/$JOB_NAME.out
#SBATCH --error=$OUT_PATH/$JOB_NAME.err
#SBATCH --partition=$PARTITION
#SBATCH --time=$TIME
#SBATCH --gres=gpu:4
source ~/.bash_profile
module load git
# conda activate mvTCR
source sc_venv_template/activate.sh
accelerate launch $CODE_PATH/sampling_evaluation.py model=$model experiment.task=$task_name experiment.batch_size=$bsz experiment.ood_task=$ood_task experiment.ood_subtask=$ood_subtask experiment.ood_batch_size=$ood_bsz \
evaluation.eval_method=dropout evaluation.only_ood=$only_ood evaluation.num_samples=1 experiment.data_path=$DATA_PATH experiment.model_path=${MODELS_PATH} evaluation.seed=$RUN_ID evaluation.eval_model_path=$MODEL_PATH
" > ${job_file}
        sbatch $job_file
done
done
