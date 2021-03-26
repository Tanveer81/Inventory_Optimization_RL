#!/bin/sh
# $1 -> train material
# $2 -> cuda device
#python dqn.py --experiment_name hlyr16_8_lstm_8_dmnd_12_1_gamma_.999_lr_1e-8_learn_every_16_batch_64_$1 --num_episodes 5000 --test True --train True --hack_test True --cuda_visible_device $2 --hidden_layer 16 8 --hidden_dim_lstm 8 --past_demand 12 --demand_embedding 1 --gamma .999 --learning_rate 1e-8 --learn_every 16 --batch_size 64 --material_name $1
python dqn.py --experiment_name hlyr512_256_128_64_lstm_128_dmnd_12_6_gamma_.999_lr_1e-8_learn_every_16_batch_64_$1 --num_episodes 5000  --env stockManager-v1 --train True --test True --hack_test True --cuda_visible_device $2 --hidden_layer 512 256 128 64 --hidden_dim_lstm 128 --past_demand 12 --demand_embedding 6 --gamma .999 --learning_rate 1e-8 --learn_every 16 --batch_size 128 --material_name $1

for material in 'B120BP' 'B120' 'Q120' 'TA2J6500' 'Q115' 'Q2100H' 'Q3015'
do
#  python dqn.py --experiment_name hlyr16_8_lstm_8_dmnd_12_1_gamma_.999_lr_1e-8_learn_every_16_batch_64_$1 --num_episodes 5000 --hack_test True --evaluate_train True --test True --cuda_visible_device $2 --hidden_layer 16 8 --hidden_dim_lstm 8 --past_demand 12 --demand_embedding 1 --gamma .999 --learning_rate 1e-8 --learn_every 16 --batch_size 64 --material_name $material
#  python dqn.py --experiment_name hlyr16_8_lstm_8_dmnd_12_1_gamma_.999_lr_1e-8_learn_every_16_batch_64_$1 --num_episodes 5000 --hack_test True --test True --cuda_visible_device $2 --hidden_layer 16 8 --hidden_dim_lstm 8 --past_demand 12 --demand_embedding 1 --gamma .999 --learning_rate 1e-8 --learn_every 16 --batch_size 64 --material_name $material
  python dqn.py --experiment_name hlyr512_256_128_64_lstm_128_dmnd_12_6_gamma_.999_lr_1e-8_learn_every_16_batch_64_$1 --num_episodes 5000  --env stockManager-v1 --test True --hack_test True --cuda_visible_device $2 --hidden_layer 512 256 128 64 --hidden_dim_lstm 128 --past_demand 12 --demand_embedding 6 --gamma .999 --learning_rate 1e-8 --learn_every 16 --batch_size 128 --material_name $material
  python dqn.py --experiment_name hlyr512_256_128_64_lstm_128_dmnd_12_6_gamma_.999_lr_1e-8_learn_every_16_batch_64_$1 --num_episodes 5000  --env stockManager-v1 --evaluate_train True --test True --hack_test True --cuda_visible_device $2 --hidden_layer 512 256 128 64 --hidden_dim_lstm 128 --past_demand 12 --demand_embedding 6 --gamma .999 --learning_rate 1e-8 --learn_every 16 --batch_size 128 --material_name $material

#  python dqn.py --experiment_name dqn_epsd_100000_trgt1000_env1_hlyr16_8_lstm_8_dmnd_12_1_gamma_.999_lr_1e-8_learn_every_16_batch_64 --hidden_layers 16 8 --hidden_dim_lstm 8 --past_demand 12 --demand_embedding 1 --test True --material_name $material
#  python dqn.py --experiment_name dqn_epsd_100000_trgt1000_env1_hlyr16_8_lstm_8_dmnd_12_1_gamma_.999_lr_1e-8_learn_every_16_batch_64 --hidden_layers 16 8 --hidden_dim_lstm 8 --past_demand 12 --demand_embedding 1 --test True --material_name $material --hack_test True
#  python dqn.py --experiment_name dqn_epsd_100000_trgt1000_env1_hlyr16_8_lstm_8_dmnd_12_1_gamma_.999_lr_1e-8_learn_every_16_batch_64 --hidden_layers 16 8 --hidden_dim_lstm 8 --past_demand 12 --demand_embedding 1 --test True --evaluate_train True --material_name $material --hack_test True
#  python dqn.py --experiment_name dqn_epsd_100000_trgt1000_env1_hlyr16_8_lstm_8_dmnd_12_1_gamma_.999_lr_1e-8_learn_every_16_batch_64 --hidden_layers 16 8 --hidden_dim_lstm 8 --past_demand 12 --demand_embedding 1 --test True --evaluate_train True --material_name $material

#  python dqn.py --experiment_name dqn_episod_100000_trgt1000_env1_hlyr64_32_lstm_32 --hidden_layers 64 32 --hidden_dim_lstm 32 --test True --material_name $material
#  python dqn.py --experiment_name dqn_episod_100000_trgt1000_env1_hlyr64_32_lstm_32 --hidden_layers 64 32 --hidden_dim_lstm 32 --test True --material_name $material --hack_test True
#  python dqn.py --experiment_name dqn_episod_100000_trgt1000_env1_hlyr64_32_lstm_32 --hidden_layers 64 32 --hidden_dim_lstm 32 --test True --evaluate_train True --material_name $material --hack_test True
#  python dqn.py --experiment_name dqn_episod_100000_trgt1000_env1_hlyr64_32_lstm_32 --hidden_layers 64 32 --hidden_dim_lstm 32 --test True --evaluate_train True --material_name $material

#  python dqn.py --experiment_name dqn_epsd_100000_trgt1000_env1_hlyr512_256_128_64_lstm_128_dmnd_12_6_gamma_.999_lr_1e-8_learn_every_16_batch_128 --hidden_layers 512 256 128 64 --hidden_dim_lstm 128 --past_demand 12 --demand_embedding 6 --test True --material_name $material
#  python dqn.py --experiment_name dqn_epsd_100000_trgt1000_env1_hlyr512_256_128_64_lstm_128_dmnd_12_6_gamma_.999_lr_1e-8_learn_every_16_batch_128 --hidden_layers 512 256 128 64 --hidden_dim_lstm 128 --past_demand 12 --demand_embedding 6 --test True --material_name $material --hack_test True
#  python dqn.py --experiment_name dqn_epsd_100000_trgt1000_env1_hlyr512_256_128_64_lstm_128_dmnd_12_6_gamma_.999_lr_1e-8_learn_every_16_batch_128 --hidden_layers 512 256 128 64 --hidden_dim_lstm 128 --past_demand 12 --demand_embedding 6 --test True --evaluate_train True --material_name $material --hack_test True
#  python dqn.py --experiment_name dqn_epsd_100000_trgt1000_env1_hlyr512_256_128_64_lstm_128_dmnd_12_6_gamma_.999_lr_1e-8_learn_every_16_batch_128 --hidden_layers 512 256 128 64 --hidden_dim_lstm 128 --past_demand 12 --demand_embedding 6 --test True --evaluate_train True --material_name $material
done
