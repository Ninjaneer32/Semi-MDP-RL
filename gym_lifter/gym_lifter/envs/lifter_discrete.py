import gym
import numpy as np
from typing import List, Tuple
from gym_lifter.envs.fab.discrete_action_set import available_actions_no_wt
from gym_lifter.envs.fab.fab_discrete import DiscreteFAB

class DiscreteLifterEnv(gym.Env):
    def __init__(self):
        # super(gym.Env, self).__init__()
        self.dt = 3.
        self.fab = DiscreteFAB(mode='day')
        self.num_layers = self.fab.num_layers
        self.state_dim = 3 + 2 * self.num_layers
        # state variables : rack position, pod, lower_to, upper_to, con_i_to's, con_i_wt's
        # note that operating time is not regarded as a state variable
        # TODO : take POD into consideration
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float)
        # 24 actions in total
        self.action_space = gym.spaces.Discrete(9)
        self.state = None
        self.pos_to_flr = [2, 2, 2, 3, 3, 3, 3, 6, 6, 6]  # convert control point to floor

        self.capacities = self.fab.capacities
        return

    def reset(self):
        self.fab.reset()
        return self.get_obs()

    def step(self, action: int):
        # 0 : go up
        # 1 : go down
        # 2 : load lower
        # 3 : load upper
        # 4 : load both
        # 5 : unload lower
        # 6 : unload upper
        # 7 : unload both
        # 8 : stay

        assert self.action_space.contains(action)

        # wt = self.waiting_time
        # rew = -np.sum(wt) / 3600.
        # operate the fab
        info = self.fab.sim(action)
        done = False

        rew = -np.sum([(wt / 180.) ** 2 for wt in self.fab.waiting_time])

        return self.get_obs(), rew, done, info

    def render(self, mode='human'):
        return

    def get_obs(self) -> np.ndarray:
        # encode the current state into a point in $\mathbb{R}^n$ (n : state space dimension)
        ######################################################################################################
        # rack position | POD | lower destination | upper destination | queue destination | waiting quantity #
        ######################################################################################################
        rpos = self.rack_pos / 9.
        rack_flr = self.pos_to_flr[self.rack_pos]
        lower_to, upper_to = self.rack_destination
        # rack_info = [rpos, float(self.is_pod_loaded), (lower_to - rack_flr) / 4., (upper_to - rack_flr) / 4.]
        rack_info = [rpos, lower_to / 6., upper_to / 6.]
        # layer_info = self.travel_distance + self.normalized_wt
        destination = [d / 6. for d in self.destination]
        #waiting_quantity = [self.waiting_quantity[i] / self.capacities[i] for i in range(self.num_layers)]
        waiting_time = [(wt / 180.) for wt in self.fab.waiting_time]
        #layer_info = destination + waiting_quantity
        layer_info = destination + waiting_time
        obs = np.array(rack_info + layer_info)
        # obs = np.concatenate([rack_info, np.array(self.destination) / 6., np.array(self.waiting_quantity) / 30.])
        return obs

    def get_obs_tmp(self) -> np.ndarray:
        # encode the current state into a point in $\mathbb{R}^n$ (n : state space dimension)
        # print(self.rack_pos)
        # print(self.rack_destination)
        rpos = self.rack_pos / 9.
        rack_flr = self.pos_to_flr[self.rack_pos]
        lower_to, upper_to = self.rack_destination
        rack_info = np.array(
            [rpos, float(self.is_pod_loaded), (lower_to - rack_flr) / 4., (upper_to - rack_flr) / 4.])
        # rack_info = np.array([rpos, float(self.is_pod_loaded), lower_to / 6., upper_to / 6.])
        obs = np.concatenate([rack_info, self.travel_distance, self.normalized_wt])
        # obs = np.concatenate([rack_info, np.array(self.destination) / 6., np.array(self.waiting_quantity) / 30.])
        return obs


    @staticmethod
    def action_map(state) -> List[int]:
        return available_actions_no_wt(state)

    @property
    def operation_log(self):
        return self.fab.operation_log

    @property
    def waiting_time(self):
        return self.fab.waiting_time

    @property
    def normalized_wt(self):
        return self.waiting_time / 180.

    @property
    def rack_pos(self):
        return self.fab.rack_pos

    @property
    def destination(self):
        return self.fab.destination

    @property
    def rack_destination(self):
        return self.fab.rack_destination

    @property
    def is_pod_loaded(self):
        return self.fab.is_pod_loaded

    @property
    def travel_distance(self):
        return self.fab.travel_distance

    @property
    def waiting_quantity(self):
        return self.fab.waiting_quantity
