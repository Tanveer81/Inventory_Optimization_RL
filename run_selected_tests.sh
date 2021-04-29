#!/bin/sh
# script containing testing commands
# awk '{ sub("\r$", ""); print }' run_selected_tests.sh > run_selected_tests2.sh
# mv run_selected_tests2.sh run_selected_tests.sh
# bash run_selected_tests.sh
rm -r output_test/total_rewards.csv
python benchmarks.py
for material in 'Q115' 'B120BP' 'B120' 'Q120' 'TA2J6500' 'Q2100H' 'Q3015'
do

python dqn.py --experiment_name dqn_episod_100000_trgt1000_env1_hlyr64_32_lstm_32 --hidden_layers 64 32 --hidden_dim_lstm 32 --test True --material_name $material
python dqn.py --experiment_name dqn_episod_100000_trgt1000_env1_hlyr64_32_lstm_32 --hidden_layers 64 32 --hidden_dim_lstm 32 --test True --evaluate_train True --material_name $material

python dqn.py --experiment_name ep_100000_hlyr512_256_128_64_lstm_128_dmnd_12_6_gamma_.999_lr_1e-8_learn_every_16_batch_128_Q3015 --hidden_layers 512 256 128 64 --hidden_dim_lstm 128 --past_demand 12 --demand_embedding 6 --test True --material_name $material
python dqn.py --experiment_name ep_100000_hlyr512_256_128_64_lstm_128_dmnd_12_6_gamma_.999_lr_1e-8_learn_every_16_batch_128_Q3015 --hidden_layers 512 256 128 64 --hidden_dim_lstm 128 --past_demand 12 --demand_embedding 6 --test True --material_name $material --evaluate_train True

done
