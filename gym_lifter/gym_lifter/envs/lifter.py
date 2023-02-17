import gym
import numpy as np
from typing import List, Tuple, Dict
# from gym_lifter_4F.envs.fab.action_set import action2operation, available_actions_no_wt, operation2action
from gym_lifter.envs.fab.fab import FAB


class LifterEnv(gym.Env):
    """
    Implementation of the lifter control system as a semi-Markov decision process (or semi-MDP).
    The process evolves by executing actions based on the state of the fab.
    """

    def __init__(self, mode):

        # super(gym.Env, self).__init__()

        # fab simulator
        self.fab = FAB(mode=mode)
        self.num_layers = self.fab.num_layers
        self.state_dim = 3 + 3 * self.fab.num_layers
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float)
        self.state = None
        self.pos_to_flr = self.fab.pos2floor
        self.capacities = self.fab.capacities

        pos2floor, action2operation, operation2action, label2pos = self.action_mapping()
        self.pos2floor = pos2floor
        self.action2operation = action2operation
        self.operation2action = operation2action
        self.label2pos = label2pos
        self.action_space = gym.spaces.Discrete(len(self.action2operation) + 1)
        return

    def reset(self):
        self.fab.reset()
        return self.get_obs(), {}

    def reset_fixed(self, rack_pos):
        self.fab.reset_fixed(rack_pos)
        return self.get_obs()

    def step(self, action: int):
        """
                Execute the given action, observe the next state, and receive the corresponding reward.

                :param action: Integer action input which is first transformed to the operation and then applied to the fab.
                               The transformation rule is predefined as a dictionary, which maps {0, 1, ..., |A| - 1} to the set of operations.

                """
        assert self.action_space.contains(action)
        rew = 0
        if action == len(self.action2operation):
            # The last action corresponds to 'STAY' action.
            operation = None
        else:
            # If the given action is not 'STAY', then transform the action into an operation and parse it.
            operation = self.action2operation[action]
            floor, pos, low_up, load_unload = operation
            if low_up == 2:
                rew += 1.
            else:
                rew += 0.5
        # execute the operation
        info = self.fab.sim(operation)
        # get result
        rew += info['lost']

        # defined to follow OpenAI gym syntax
        done = False
        truncated = False
        return self.get_obs(), rew, done, truncated, info

    def get_obs(self) -> np.ndarray:
        """
        encode the current state into a vector in R^n (n: state space dimension)
        -----------------------------------------------------------------------------------
        rack position | lower/upper destination | queue destination | waiting quantity/time
        -----------------------------------------------------------------------------------
        """
        rpos = (self.rack_pos) / (self.fab.NUM_FLOOR * 4 - 1)
        lower_to, upper_to = self.rack_destination

        rack_info = [rpos, lower_to / float(self.fab.NUM_FLOOR), upper_to / float(self.fab.NUM_FLOOR)]
        destination = [d / self.fab.NUM_FLOOR for d in self.destination]
        waiting_quantity = [(self.waiting_quantity[i] / self.capacities[i]) for i in range(self.num_layers)]
        waiting_time = [self.waiting_time[i] / 300. for i in range(len(self.waiting_time))]
        layer_info = destination + waiting_quantity + waiting_time
        obs = np.array(rack_info + layer_info)

        return obs



    def action_mapping(self):

        NUM_FLOOR = self.fab.NUM_FLOOR

        label_decoder = {}
        conveyor = 1
        for floor in range(1, NUM_FLOOR + 1):
            for layer in range(1, 4):
                label_decoder[conveyor] = (floor, layer)
                conveyor += 1
        pos2label = {}
        pos2floor = {}
        for floor in range(1, NUM_FLOOR + 1):
            for pos in range(1, 5):
                if (floor == 1 and pos == 1) or (floor == NUM_FLOOR and pos == 4):
                    continue
                if pos == 1:
                    pos2label[4 * (floor - 1) + pos] = (None, 3 * (floor - 1) + pos)
                elif pos == 4:
                    pos2label[4 * (floor - 1) + pos] = (3 * floor, None)
                else:
                    pos2label[4 * (floor - 1) + pos] = (3 * (floor - 1) + pos - 1, 3 * (floor - 1) + pos)
                pos2floor[4 * (floor - 1) + pos] = floor

        labels = list(np.arange(1, 3 * NUM_FLOOR + 1))
        capacities = 3 * np.ones_like(labels)

        action2operation: Dict[int, Tuple[int, int, int, int]] = {}
        idx = 0
        for pos in list(pos2label.keys()):
            # for a in range(4):
            # action2operation[idx]= (pos2floor[pos],pos,0,0)
            if pos2label[pos][0] != None and pos2label[pos][1] != None:
                action2operation[idx] = (pos2floor[pos], pos, 2, 0)
                idx += 1
                action2operation[idx] = (pos2floor[pos], pos, 2, 1)
                idx += 1

                action2operation[idx] = (pos2floor[pos], pos, 0, 0)
                idx += 1
                action2operation[idx] = (pos2floor[pos], pos, 0, 1)
                idx += 1
                action2operation[idx] = (pos2floor[pos], pos, 1, 0)
                idx += 1
                action2operation[idx] = (pos2floor[pos], pos, 1, 1)
                idx += 1
            elif pos2label[pos][0] != None:

                action2operation[idx] = (pos2floor[pos], pos, 0, 0)
                idx += 1
                action2operation[idx] = (pos2floor[pos], pos, 0, 1)
                idx += 1

            elif pos2label[pos][1] != None:

                action2operation[idx] = (pos2floor[pos], pos, 1, 0)
                idx += 1
                action2operation[idx] = (pos2floor[pos], pos, 1, 1)
                idx += 1

        label2pos: Dict[Tuple[int, int], int] = {op: a for a, op in pos2label.items()}
        operation2action: Dict[Tuple[int, int, int], int] = {op: a for a, op in action2operation.items()}

        return pos2floor, action2operation, operation2action, label2pos

    # @staticmethod
    def action_map(self, state) -> List[int]:

        NUM_FLOOR = self.fab.NUM_FLOOR
        actions = []


        current_pos = int(round(state[0] * (NUM_FLOOR * 4 - 1)))  # control point
        # rack position -> floor
        current_floor = self.pos2floor[current_pos]

        lower_floor = int(round(state[1] * NUM_FLOOR))
        upper_floor = int(round(state[2] * NUM_FLOOR))

        if lower_floor == 0:
            # lower fork is empty
            lower_occupied = False
        else:
            lower_occupied = True

        if upper_floor == 0:
            # upper fork is empty
            upper_occupied = False
        else:
            upper_occupied = True

        wq = state[-(3 * NUM_FLOOR):]

        if all([wq[layer] < 0.0001 for layer in range((3 * NUM_FLOOR))]) and (not lower_occupied) and (
                not upper_occupied):
            # if everything is empty, do nothing
            actions.append(len(self.action2operation))
            return actions

        for floor in range(1, NUM_FLOOR + 1):
            L1 = 3 * (floor - 1) + 1
            q = wq[L1 - 1:L1 + 2]
            if q[0] > 0:
                if not upper_occupied and floor != 1:
                    actions.append(self.operation2action[(floor, self.label2pos[(None, L1)], 1, 0)])
                if not lower_occupied:
                    actions.append(self.operation2action[(floor, self.label2pos[(L1, L1 + 1)], 0, 0)])
            if q[1] > 0:
                if not upper_occupied:
                    actions.append(self.operation2action[(floor, self.label2pos[(L1, L1 + 1)], 1, 0)])
                if not lower_occupied:
                    actions.append(self.operation2action[(floor, self.label2pos[(L1 + 1, L1 + 2)], 0, 0)])
            if q[2] > 0:
                if not upper_occupied:
                    actions.append(self.operation2action[(floor, self.label2pos[(L1 + 1, L1 + 2)], 1, 0)])
                if not lower_occupied and floor != NUM_FLOOR:
                    actions.append(self.operation2action[(floor, self.label2pos[(L1 + 2, None)], 0, 0)])
            if q[0] > 0 and q[1] > 0:
                if not upper_occupied and not lower_occupied:
                    actions.append(self.operation2action[(floor, self.label2pos[(L1, L1 + 1)], 2, 0)])
            if q[1] > 0 and q[2] > 0:
                if not upper_occupied and not lower_occupied:
                    actions.append(self.operation2action[(floor, self.label2pos[(L1 + 1, L1 + 2)], 2, 0)])

        ##if lower uc
        if lower_occupied:
            L1 = (lower_floor - 1) * 3 + 1
            actions.append(self.operation2action[(lower_floor, self.label2pos[(L1, L1 + 1)], 0, 1)])
            actions.append(self.operation2action[(lower_floor, self.label2pos[(L1 + 1, L1 + 2)], 0, 1)])
            if lower_floor != NUM_FLOOR:
                actions.append(self.operation2action[(lower_floor, self.label2pos[(L1 + 2, None)], 0, 1)])

        ##upper_floor
        if upper_occupied:
            L1 = (upper_floor - 1) * 3 + 1
            actions.append(self.operation2action[(upper_floor, self.label2pos[(L1, L1 + 1)], 1, 1)])
            actions.append(self.operation2action[(upper_floor, self.label2pos[(L1 + 1, L1 + 2)], 1, 1)])
            if upper_floor != 1:
                actions.append(self.operation2action[(upper_floor, self.label2pos[(None, L1)], 1, 1)])

        if upper_occupied and lower_occupied and lower_floor == upper_floor:
            # a ; action2operation[a] == [lower_floor, _ , 2, 1]
            L1 = (lower_floor - 1) * 3 + 1
            actions.append(self.operation2action[(lower_floor, self.label2pos[(L1, L1 + 1)], 2, 1)])
            actions.append(self.operation2action[(lower_floor, self.label2pos[(L1 + 1, L1 + 2)], 2, 1)])

        actions = list(set(actions))  # remove duplicates

        return actions

    def get_capacities(self):
        return self.capacities.copy()

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
    def destination(self) -> List[int]:
        return self.fab.destination

    @property
    def rack_destination(self) -> Tuple[int, int]:
        return self.fab.rack_destination

    @property
    def is_pod_loaded(self):
        return self.fab.is_pod_loaded

    @property
    def travel_distance(self):
        return self.fab.travel_distance

    @property
    def waiting_quantity(self) -> List[int]:
        return self.fab.waiting_quantity

    def render(self, mode='human'):
        self.fab.render()
        return

    def close(self):
        self.fab.close()
