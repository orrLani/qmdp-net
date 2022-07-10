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


### Presentation


[Presentation (3) (1).pptx](https://github.com/orrLani/qmdp-net/files/8999851/Presentation.3.1.pptx)

### Definitions
  1. Pmove_succ: probability of transition succeeding, otherwise stays in place
  2. Pobs_succ: probability of correct observation, independent in each direction
   
### Demos
We created demos to show the movement of the robot.

qmdp example:

qmdp with Pmove_succ=1 and Pobs_succ=1

https://user-images.githubusercontent.com/54217343/175830057-0a68a8ef-6281-4a69-8682-fceae6a16054.mp4
 
qmdp with pmove_succ=0.9 and pobs_succ=0.9


https://user-images.githubusercontent.com/54217343/175919833-51faadf7-8a03-4daf-b893-a6f650f22f75.mp4




qmdp-net that train on 10*10 grid world example:

qmdp-net with pmove_succ=1 and pobs_succ=1

https://user-images.githubusercontent.com/54217343/175831651-622b3237-dd20-43b9-a6ef-5c98c1fcc00a.mp4

qmdp-net with pmove_succ=0.9 and pobs_succ=0.9

https://user-images.githubusercontent.com/54217343/175917134-c0e95a4d-9a64-4ef8-ac27-86a8c8c5f981.mp4

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

