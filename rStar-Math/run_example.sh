# In our main table, generating 64 trajectories requires running MCTS inference 64 times, with only the highest-scoring response selected each time. 
# Therefore, I am providing an example script here that takes the number of GPUs and the total number of runs as inputs.
# Using 4 GPUs and running a total of 8 times, the command is: 
# bash run_example.sh 4 8

MODEL="policy model dir"
RM="reward model dir" 
QAF="test set path"
CFG="config/sft_eval_mcts.yaml"
# CFG="config/sft_eval_bs.yaml"

NUM_GPU=$1
NUM_RUNS=$2

# Calculate the number of runs required for each GPU.
RUNS_PER_GPU=$(( (NUM_RUNS + NUM_GPU - 1) / NUM_GPU ))

# Define a function to execute the task and repeat the process.
run_task() {
    local gpu_id=$1
    local runs=$2
    for ((i=1; i<=runs; i++))
    do
        echo "Starting run $i on GPU $gpu_id"
        CUDA_VISIBLE_DEVICES=$gpu_id python main.py --qaf $QAF --custom_cfg $CFG --model_dir $MODEL --reward_model_dir $RM
    done
}

# Launch background tasks on each GPU.
for ((gpu_id=0; gpu_id<NUM_GPU; gpu_id++))
do
    run_task $gpu_id $RUNS_PER_GPU &
done


# Our final score is determined by a majority vote on the top K highest-reward trajectories.
# K is usually set between 4 and 8. Run the command to see the results. For other parameters, please refer to the code.
python eval_maj.py --path $QAF/rollout --top_n 4