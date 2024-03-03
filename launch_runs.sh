#!/bin/bash


for i in $(eval echo {$1..$2})
do
    ./run_launcher.sh $i 1000 100 0
done
