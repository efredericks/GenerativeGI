RUN_NUM="$1"
GENS="$2"
POP_SIZE="$3"
TRT_NUM="$4"
parent_path=/home/moorejar/

echo $parent_path

python3 deap_main_no_tk.py --gens="$GENS" --pop_size="$POP_SIZE" --treatment="$TRT_NUM" --run_num="$RUN_NUM" --output_path="$parent_path"/research_results/generative_gi_cnn/ --lexicase --shuffle --clear_canvas > "$parent_path"/research_results/generative_gi_cnn/"$RUN_NUM".out
wait
#cd "$parent_path"/research_results/"$TRT_NUM"/"$RUN_NUM"/
#mv ../*"$RUN_NUM"*out ./
#python ../../../plotting.py --run_num=$RUN_NUM --tx_num=$TRT_NUM
# pause >nul
exit
