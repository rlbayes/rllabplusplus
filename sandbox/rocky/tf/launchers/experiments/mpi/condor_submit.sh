if [ ! $# == 6 ]; then
    echo "Usage: $0 <num_process> <exp_name> <script> <memory> <cpu> <gpu>"
    exit
fi
num_process=$1
exp_name=$2
script=$3
memory=$4
cpus=$5
gpus=$6

# check/make necessary directories
if [ ! -e $script ]; then
    echo "Warning: not submitting, since script=$script does not exist."
    exit
fi
if [ -e $exp_name ]; then
    echo "Warning: not submitting, since exp folder=$exp_name exists."
    exit
fi
mkdir $exp_name
cp $script $exp_name/script.sh
cp $script $exp_name.sh

if [ "$gpus" -gt 0 ]; then
cat >$exp_name/test.sub << EOL
executable=/bin/bash
arguments="$script \$(Process) $exp_name"
error=$exp_name/test.\$(Process).err
output=$exp_name/test.\$(Process).out
log=$exp_name/test.\$(Process).log
request_memory=$memory
request_cpus=$cpus
request_gpus=$gpus
requirements=CUDAGlobalMemoryMb>4096
+MaxRunningPrice=5
queue $num_process
EOL
else
cat >$exp_name/test.sub << EOL
executable=/bin/bash
arguments="$script \$(Process) $exp_name"
error=$exp_name/test.\$(Process).err
output=$exp_name/test.\$(Process).out
log=$exp_name/test.\$(Process).log
request_memory=$memory
request_cpus=$cpus
+MaxRunningPrice=5
queue $num_process
EOL
fi
condor_submit $exp_name/test.sub
