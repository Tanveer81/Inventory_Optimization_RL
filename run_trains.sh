#!/bin/sh
# script containing training commands
python dqn.py --experiment_name dqn_epsd_100000_trgt1000_env1_hlyr16_8_lstm_8_dmnd_12_1_gamma_.999_lr_1e-8_learn_every_16_batch_64 --hidden_layers 16 8 --hidden_dim_lstm 8 --past_demand 12 --demand_embedding 1 --train True --material_name Q115 --immediate_action_train True --num_episodes 100000 --target_update 1000 --env stockManager-v1 --gamma .999 --learning_rate 1e-8 --learn_every 16 --batch_size 64

python dqn.py --experiment_name dqn_episod_100000_trgt1000_env1_hlyr64_32_lstm_32 --hidden_layers 64 32 --hidden_dim_lstm 32 --train True --material_name Q115 --immediate_action_train True --num_episodes 100000 --target_update 1000 --env stockManager-v1

python dqn.py --experiment_name dqn_epsd_100000_trgt1000_env1_hlyr512_256_128_64_lstm_128_dmnd_12_6_gamma_.999_lr_1e-8_learn_every_16_batch_128 --hidden_layers 512 256 128 64 --hidden_dim_lstm 128 --past_demand 12 --demand_embedding 6 --train True --material_name Q115 --immediate_action_train True --num_episodes 100000 --target_update 1000 --env stockManager-v1 --gamma .999 --learning_rate 1e-8 --learn_every 16 --batch_size 128

python dqn.py --experiment_name ep_100000_hlyr16_8_lstm_8_dmnd_12_1_gamma_.999_lr_1e-8_learn_every_16_batch_64_Q115 --hidden_layers 16 8 --hidden_dim_lstm 8 --past_demand 12 --demand_embedding 1 --train True --material_name Q115 --hack_test True --num_episodes 100000 --target_update 1000 --env stockManager-v1 --gamma .999 --learning_rate 1e-8 --learn_every 16 --batch_size 64

python dqn.py --experiment_name ep_100000_hlyr512_256_128_64_lstm_128_dmnd_12_6_gamma_.999_lr_1e-8_learn_every_16_batch_128_Q115 --hidden_layers 512 256 128 64 --hidden_dim_lstm 128 --past_demand 12 --demand_embedding 6 --train True --material_name Q115 --hack_test True --num_episodes 100000 --target_update 1000 --env stockManager-v1 --gamma .999 --learning_rate 1e-8 --learn_every 16 --batch_size 128

python dqn.py --experiment_name ep_100000_hlyr16_8_lstm_8_dmnd_12_1_gamma_.999_lr_1e-8_learn_every_16_batch_64_Q3015 --hidden_layers 16 8 --hidden_dim_lstm 8 --past_demand 12 --demand_embedding 1 --train True --material_name Q3015 --hack_test True --num_episodes 100000 --target_update 1000 --env stockManager-v1 --learn_every 16 --batch_size 64

python dqn.py --experiment_name ep_100000_hlyr16_8_lstm_8_dmnd_12_1_gma_.999_lr_1e-8_lrn_evry16_b64_Q3015 --hidden_layers 16 8 --hidden_dim_lstm 8 --past_demand 12 --demand_embedding 1 --train True --material_name Q3015 --hack_test True --num_episodes 100000 --target_update 1000 --env stockManager-v1 --gamma .999 --learning_rate 1e-8 --learn_every 16 --batch_size 64

python dqn.py --experiment_name ep_100000_hlyr512_256_128_64_lstm_128_dmnd_12_6_gamma_.999_lr_1e-8_learn_every_16_batch_128_Q3015 --hidden_layers 512 256 128 64 --hidden_dim_lstm 128 --past_demand 12 --demand_embedding 6 --train True --material_name Q3015 --hack_test True --num_episodes 100000 --target_update 1000 --env stockManager-v1 --gamma .999 --learning_rate 1e-8 --learn_every 16 --batch_size 128
