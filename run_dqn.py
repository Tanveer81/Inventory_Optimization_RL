import os

for material in ['B120', 'Q120', 'TA2J6500', 'Q2100H', 'Q3015']:
    os.system("conda init bash")
    os.system("conda activate contrastive")
    os.system(f"python dqn.py --experiment_name dqn_epsd_100000_trgt1000_env1_hlyr16_8_lstm_8_dmnd_12_2_gamma_.999_lr_1e-8_learn_every_16_batch_64"
              f" --hidden_layers 16 8 --hidden_dim_lstm 8 --past_demand 12 --demand_embedding 1"
              f" --test True --material_name {material}")

    os.system(f"python dqn.py --experiment_name dqn_epsd_100000_trgt1000_env1_hlyr16_8_lstm_8_dmnd_12_2_gamma_.999_lr_1e-8_learn_every_16_batch_64"
              f" --hidden_layers 16 8 --hidden_dim_lstm 8 --past_demand 12 --demand_embedding 1"
              f" --test True --material_name {material} --hack_test True")

    os.system(f"python dqn.py --experiment_name dqn_epsd_100000_trgt1000_env1_hlyr16_8_lstm_8_dmnd_12_2_gamma_.999_lr_1e-8_learn_every_16_batch_64"
              f" --hidden_layers 16 8 --hidden_dim_lstm 8 --past_demand 12 --demand_embedding 1"
              f" --test True --evaluate_train True --material_name {material} --hack_test True")

    os.system(f"python dqn.py --experiment_name dqn_epsd_100000_trgt1000_env1_hlyr16_8_lstm_8_dmnd_12_2_gamma_.999_lr_1e-8_learn_every_16_batch_64"
              f" --hidden_layers 16 8 --hidden_dim_lstm 8 --past_demand 12 --demand_embedding 1"
              f" --test True --evaluate_train True --material_name {material}")