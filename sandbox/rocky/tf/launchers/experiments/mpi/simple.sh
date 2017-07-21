export HOME=/home/sgu
# added rllab
export RLLABPATH=$HOME/rllab-private
export PYTHONPATH=$RLLABPATH:$RLLABPATH/rllab:$PYTHONPATH
# added by Anaconda3 4.2.0 installer
export PATH="$HOME/anaconda3/bin:$PATH"
#printenv
echo "HOME=$HOME"
echo "PWD=$PWD"
echo "Source rllab3..."
source activate rllab3
echo "Cd to launchers folder..."
cd $HOME/rllab-private/sandbox/rocky/tf/launchers
echo "Executing script..."
python algo_gym_stub.py --exp=simple --overwrite=true --algo_name=trpo --seed=3
source deactivate
