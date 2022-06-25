# QMDP-net

Implementation of the NIPS 2017 paper: 

QMDP-Net: Deep Learning for Planning under Partial Observability  
Peter Karkus, David Hsu, Wee Sun Lee  
National University of Singapore  
https://arxiv.org/abs/1703.06692

The code implements the 2D grid navigation domain, and a QMDP-net with 2D state space in tensorflow.

### Requirement
  1. python3.10
  2. anconda
  3. good mood

### How to run the code
1. clone the repo
2. run the following command: conda create -n qmdp_env python=3.9
3. active the virtual environment by running: conda activate qmdp_env
4. install the requirements libraries with the command: pip install -r requirements.txt
4. run the code :)



### Train and evaluate a QMDP-net

The folder ./data/grid10 contains training and test data for the deterministic 10x10 grid navigation domain
(10,000 environments, 5 trajectories each for training, 500 environments, 1 trajectory each for testing).


Train network using only the first 4 steps of each training trajectory:
```
python train.py ./data/grid10/ --logpath ./data/grid10/output-lim4/ --lim_traj_len 4
```
The learned model will be saved to ./data/grid10/output-lim4/final.chk
 

Load the previously saved model and train further using the full trajectories:
```
python train.py ./data/grid10/ --logpath ./data/grid10/output-lim100/ --loadmodel ./data/grid10/output-lim4/final.chk --lim_traj_len 100
```

For help on arguments execute:
```
python train.py --help
```

### Evaluate a previously trained model
A model trained by the commands above is readily available in the folder: data/grid10/trained-model. You may load and evaluate this model using the following command: 
```
python train.py ./data/grid10/ --loadmodel ./data/grid10/trained-model/final.chk --epochs 0
```

The expected output:
```
Evaluating 100 samples, repeating simulation 1 time(s)
Expert
Success rate: 1.000  Trajectory length: 7.3  Collision rate: 0.000
QMDP-Net
Success rate: 0.990  Trajectory length: 7.1  Collision rate: 0.000
```

### Generate training and test data

You may generate data using the script grid.py.  
As an example, the command for the 18x18 deterministic grid navigation domain is: 
```
python grid.py ./data/grid18/ 10000 500 --N 18 --train_trajs 5 --test_trajs 1
```
This will generate 10,000 random environments for training, 500 for testing, 5 and 1 trajectories per environment.

For the stochastic variant use:
```
python grid.py ./data/grid18/ 10000 500 --N 18 --train_trajs 5 --test_trajs 1 --Pmove_succ 0.8 --Pobs_succ 0.9
```

For help on arguments execute:
```
python grid.py --help
```

