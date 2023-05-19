FAB Lifter Operation using Semi-MDP-RL
====================================================

This repository includes a PyTorch implementation of **deep reinforcement learning methods for semi-MDP** which is used to control FAB lifters.

## 1. Requirements


To run our code, the followings must be installed:

- **Python**

- **[Gym][gymlink]** (>= 0.26.2) 

- **[Pytorch][pytorchlink]** 

- **[pygame][pygamelink]**

## 2. Installation
To use the environment, the installation of the module `gym_lifter` is needed. The module includes an implementation of FAB lifter simulator.
A pygame-based simulator is provided to help users to monitor get key operational statistics, while providing real-time visualization of the environment.

Before starting the installation, creating a separate virtual environment, e.g., Anaconda, is recommended.
Then, the rest of the installation process is straightforward.
For example, once you activate your Conda environment, just run
```
$ cd gym_lifter && pip install -e .
$ python load_scenarios.py
```
to complete your installation.

Currently, 3 different environments are available:
- `Lifter-v0` : standard lifter control environment

- `LifterPOD-v0` : lifter control environment with more than one type of lots

- `LifterCAPA-v0` : lifter control environment with a larger lifter capacity

To personally use these environment, just use `gym.make(id, mode)` as follows:

```python
import gym
import gym_lifter

env = gym.make('Lifter-v0', mode=2)
observation, info = env.reset()

for _ in range(1000):
    a = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(a)
    env.render()
```
where `mode`, the number of operating floors, has to be chosen among `2, 3, 4, 6, 8`.

## 2. Training/Evaluation
We provide a comprehensible interface for applying algorithms to the lifter simulation environments.
For instance, to train a SAC agent in the environment without POD, you may run the following command:
```
$ cd semi-SAC
$ python semisac.py --env=Lifter-v0 --mode=2 --render
```

`--render` option invokes visualization of the environment by activating the simulators equipped in `gym_lifter` package. 
 You may specify the values of the hyperparameters of your agent.
 For example, to set the size of the actor learning rate to `1e-4`,
 add `--pi_lr` to your command as follows:
```
$ python semisac.py --env=Lifter-v0 --mode=2 --pi_lr=1e-4
```

 ## 3. Loading Training/Evaluation Data
During training, all of the training/evaluation logs are saved at the directory `log`.
If you run SAC on `Lifter-v0`, then the corresponding evaluation log will be written in `eval.csv` format,
and be located under `log/Lifter-v0/`.
The directory includes network weight file which ends with `.pth.tar` which can be used to test the learned result.

## 4. Test
We provide pre-trained weights which can be used to run evaluation of the learned agent on a single scenario. 
For example, to test the agent learned in the environment under the presence of POD, run
```
$ cd semi-SAC
$ python test.py --env=LifterPOD-v0 --mode=2 --pth=[path-to-log-directory]
```

## 5. Baseline Controller
As a baseline, MPC controller is implemented and located under directory `mpc`. The concise description of the baseline is given at [`mpc/README.md`][mpclink]
 To run the baseline, you need [gurobipy][gurobipylink] with full support license. To obtain and set up Gurobi license, please refer to the instruction at [Gurobi][gurobilink].

Running the controller is again straightforward; to run the MPC controller with horizon=3, run
```
$ cd mpc
$ python runner.py --mode=2 --steps=3 --file_count=0
```



[gymlink]: https://github.com/openai/gym/
[pytorchlink]: https://pytorch.org/
[pygamelink]: https://github.com/pygame/pygame/
[gurobipylink]: https://pypi.org/project/gurobipy/
[gurobilink]: https://www.gurobi.com/documentation/9.5/quickstart_mac/retrieving_and_setting_up_.html#section:RetrieveLicense
[mpclink]: https://github.com/CORE-SNU/Semi-MDP-RL/mpc/
