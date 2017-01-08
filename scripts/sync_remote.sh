#!/bin/bash
save="remote"
RLLAB_PATH=$HOME/rllab-private

read -e -p "Please enter the remote address [username@ip_address]:" remote
read -e -p "Please enter the local folder name [$save]:" save

path=$RLLAB_PATH/data/$save

if [ ! -d $path ]; then mkdir -p $path ;fi

rsync -av --exclude '*.log' $remote:~/rllab-private/data/local/* $path/ 
