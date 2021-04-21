# gym_example

## Usage

Clone the repo and connect into its top level directory.

Create Conda Environemnt:
```
conda create -n inv_op python=3.6
conda activate inv_op
conda install pytorch==1.5.0 cudatoolkit=10.1 -c pytorch
```

To initialize and run the `gym` example:

```
pip install -r requirements.txt
pip install -e gym-example

```

Inside gym_example folder
```
pip install -e .
```

To run Ray RLlib to train a policy based on this environment:

```
python dqn.py
```



