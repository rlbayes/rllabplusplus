if [ ! $# == 2 ]; then
    echo "Usage: $0 <process_id> <exp_name>"
    exit
fi
process_id=$1
exp_name=$2

# define environment variables
export HOME=/home/sgu
export RLLABPATH=$HOME/rllab-private
export PYTHONPATH=$RLLABPATH:$RLLABPATH/rllab:$PYTHONPATH
export PATH="$HOME/anaconda3/bin:$PATH"

# set up to run rllab
source activate rllab3
cd $HOME/rllab-private/sandbox/rocky/tf/launchers

# execute hyperparam 
i=0
seeds=( 2 3 4 )
algo_names=( "qprop" )
env_names=( "Humanoid-v1" "HalfCheetah-v1" )
scale_rewards=( 10 1 0.1 0.01 )
for seed in "${seeds[@]}"; do
for algo_name in "${algo_names[@]}"; do
for env_name in "${env_names[@]}"; do
for scale_reward in "${scale_rewards[@]}"; do
    if [ $i == $process_id ]; then
        echo "Executing process_id=$process_id..."
        cmd=(python algo_gym_stub.py --exp=$exp_name --max_episode=100000 
            --seed=$seed 
            --algo_name=$algo_name
            --env_name=$env_name 
            --scale_reward=$scale_reward
	    --qf_updates_ratio=1
	    --step_size=0.01
            --batch_size=5000
            --qprop_eta_option=adapt1 
        )
        echo "${cmd[@]}"
        "${cmd[@]}"
        exit
    fi
    i=$((i+1))
done
done
done
done
echo "Warning: process_id=$1 is invalid."

# clean up rllab
source deactivate
