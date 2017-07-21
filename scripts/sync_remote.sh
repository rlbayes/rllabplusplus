#!/bin/bash
folder_pattern="*"
RLLAB_PATH=$HOME/rllab-private

read -e -p "Please enter the remote address [username@ip_address]:" remote
read -e -p "Please enter the remote folder_pattern name to sync [$folder_pattern]:" folder_pattern

path=$RLLAB_PATH/data/local

if [ ! -d $path ]; then mkdir -p $path ;fi

rsync -av --progress --include "$folder_pattern/" --include "$folder_pattern/*/" --include "$folder_pattern/*/*.json" --include "$folder_pattern/*/*.csv" --include "$folder_pattern/*/params.pkl"  --exclude '*' --exclude '*.meta' --exclude '*npy' --exclude '*.log' --exclude '*.chk' $remote:~/rllab-private/data/local/ $path/ 
