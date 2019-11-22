# toh-gym
Open AI Gym discrete environment for Tower of Hanoi problem

The latest version can be installed using pip:
`
pip install -e git+git://github.com/xadahiya/toh-gym#egg=toh-gym
`


## Steps to use -
1. Import the environment using `from toh_gym.envs import TohEnv`
2. Create the environment using `env = TohEnv()`

NOTE: To change the number of disks pass initial_state and goal_state parameters like `initial_state = ((5,4,3,2,1,0), (), ())` and `goal_state = ((), (), (5,4,3,2,1,0)`  
