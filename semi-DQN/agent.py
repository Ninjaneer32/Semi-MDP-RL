import copy
import torch
import numpy as np
from torch.optim import Adam
from torch.nn import MSELoss
from torch.nn import SmoothL1Loss
import gym
from utils import freeze
from buffer import SemiMDPReplayBuffer
from model import Critic#, Critic_target, DoubleCritic
from typing import Callable, List
import random
from replay import PrioritizedTransitionReplay, Transition
import torch.nn.functional as F
import random
import torch
import numpy as np

import time
import os
torch.backends.cudnn.benchmark=True

class SemiDQNAgent:
    def __init__(self,
                 env,
                 dimS,
                 nA,
                 action_map: Callable[..., List[int]],
                 num_layer,
                 hidden1,
                 hidden2,
                 hidden3,
                 gamma=0.999,
                 anneal_schedule: Callable=None,
                 lr=1e-4,
                 tau=1e-3,
                 buffer_size=int(1e6),
                 batch_size=32,
                 priority_exponent: Callable=None,
                 normalize_weights=True,
                 uniform_sample_prob=1e-3,
                 device='cuda',
                 render=False,
                 dueling=1,
                 n_atom=100
                 ):



        arg_dict = locals()
        print('agent spec')
        print('-' * 80)
        print(arg_dict)
        print('-' * 80)
        self.atoms = n_atom
        self.dimS = dimS
        self.nA = nA
        # set networks
        self.Q = Critic(dimS, nA, hidden_size1=hidden1, hidden_size2=hidden2, hidden_size3=hidden3, num_layer=num_layer, dueling=dueling, atoms=self.atoms).to(device)
        self.optimizer = Adam(self.Q.parameters(), lr=lr)


        self.target_Q = copy.deepcopy(self.Q).to(device)
        freeze(self.target_Q)


        # discount factor & polyak constant
        self.gamma = gamma
        self.tau = tau

        replay_structure = Transition(s_tm1=None, a_tm1=None, r_t=None, s_t=None, dt=None, d=None)

        # replay buffer for experience replay in semi-MDP
        self.buffer = SemiMDPReplayBuffer(dimS, buffer_size)
        self.batch_size = batch_size
        # prioritized experience replay for semi-DQN
        self.replay = PrioritizedTransitionReplay(capacity=buffer_size,
                                                  structure=replay_structure,
                                                  priority_exponent=priority_exponent,
                                                  importance_sampling_exponent=anneal_schedule,
                                                  uniform_sample_probability=uniform_sample_prob,
                                                  normalize_weights=normalize_weights,
                                                  random_state=np.random.RandomState(42),
                                                  encoder=None,
                                                  decoder=None)
        self.max_seen_priority = 1.
        self.schedule = anneal_schedule

        # function which returns the set of executable actions at a given state
        # expected return type : numpy array when 2nd arg = True / list when False
        self.action_map = action_map
        self.render = render
        self.device = device
        self.env = env




    def get_action_rule(self, state, rule = 0):
        #Rule = { 0: SJF-TFP, 1: LWF-TFP, 2: Random }

        possible_actions = self.action_map(state)  # return a set of indices instead of a mask vector
        possible_actions = random.sample(possible_actions, len(possible_actions))



        #Longest Waiting First(LWF) + Two-Forks Priority(TFP)
        if rule ==1:
            max_wait=-10.
            waiting_time=-10.
            a= None
            for action in possible_actions:

                if action == self.nA - 1:
                    operation = None
                else:
                    operation = self.env.action2operation[action]

                if operation is None:
                    waiting_time = 0.
                else:
                    floor, pos, low_up, load_unload = operation
                    operation_time = int(abs(pos - self.env.rack_pos)) * 0.5 + 2. + 3.
                    bonus = np.random.rand() ### Priority value to Two-Forks
                    if load_unload == 0:  # LOAD
                        if low_up == 0:
                            waiting_time = self.env.fab.t - self.env.fab.layers[self.env.fab.pos2label[pos][0]].cmd_time
                        elif low_up == 1:
                            waiting_time = self.env.fab.t - self.env.fab.layers[self.env.fab.pos2label[pos][1]].cmd_time
                        else:
                            waiting_time = max(self.env.fab.t - self.env.fab.layers[self.env.fab.pos2label[pos][0]].cmd_time,
                                               self.env.fab.t - self.env.fab.layers[self.env.fab.pos2label[pos][1]].cmd_time) + bonus
                    if load_unload == 1:  # UNLOAD
                        if low_up == 0:
                            waiting_time = self.env.fab.t - self.env.fab.rack.lower_fork.CMD_RACK
                        elif low_up == 1:
                            waiting_time = self.env.fab.t - self.env.fab.rack.upper_fork.CMD_RACK
                        else:
                            waiting_time = max(self.env.fab.t - self.env.fab.rack.lower_fork.CMD_RACK,
                                               self.env.fab.t  - self.env.fab.rack.upper_fork.CMD_RACK) + bonus


                if max_wait <= waiting_time:

                    max_wait = waiting_time
                    a = action




        #Shortest Job First (SJF) + Two-Forks Priority(TWF)
        elif rule ==0 :
            possible_actions = random.sample( possible_actions, len(possible_actions) )
            min_t = 1e+8
            for action in possible_actions:
                bonus = 0
                if action == self.nA-1:
                    operation = None
                else:
                    operation = self.env.action2operation[action]

                if operation is None:
                    operation_time = 1000.
                else:
                    floor, pos, low_up, load_unload = operation
                    if low_up == 2:
                        bonus = - np.random.rand() ### Priority value to Two-Forks
                    operation_time = int(abs(pos - self.env.rack_pos)) * 0.5 + 2. + 3. + bonus
                    # operation_time = distance_matrix[pos, self.env.rack_pos] + 3.
                    # operations.append(operation)

                if min_t >= operation_time:
                    min_t = operation_time
                    a = action

        ##Random
        else:
            prob = np.zeros_like(possible_actions) * 0.
            for i, action in enumerate(possible_actions):
                if action == self.nA-1:
                    prob[i] = 1e-8
                else:
                    prob[i] = 1.
                    operation = self.env.action2operation[action]
                    floor, pos, low_up, load_unload = operation
                    if low_up == 2:
                        prob[i] += 5.

            prob /= np.sum(prob)
            a = np.random.choice(possible_actions, p = prob)



        return a


    def get_action(self, state, eps):
        dimS = self.dimS
        possible_actions = self.action_map(state)  # return a set of indices instead of a mask vector

        u = np.random.rand()
        if u < eps:
            # random selection among executable actions

            a = random.choice(possible_actions)
            # print('control randomly selected : ', a)
        else:
            m = self.mask(possible_actions)
            # greedy selection among executable actions
            # non-admissible actions are not considered since their value corresponds to -inf
            s = torch.tensor(state, dtype=torch.float).view(1, dimS).to(self.device)
            action_value = self.Q(s).mean(dim=2)  # (N_ENVS, N_ACTIONS)
            a = int(np.argmax( action_value.data.cpu().numpy() + m) )

        return a




    def load_model(self, path):
        print('networks loading...')
        checkpoint = torch.load(path + 'model.pth.tar', map_location=self.device)
        self.Q.load_state_dict(checkpoint['critic'])
        self.target_Q.load_state_dict(checkpoint['target_critic'])
        self.optimizer.load_state_dict(checkpoint['critic_optimizer'])
        return


    def save_model(self, path):
        print('adding checkpoints...')
        checkpoint_path = path + 'model.pth.tar'
        torch.save(
                    {
                     'critic': self.Q.state_dict(),
                     'target_critic': self.target_Q.state_dict(),
                     'critic_optimizer': self.optimizer.state_dict()
                     },
                    checkpoint_path)

        return



    def target_update(self):
        for p, target_p in zip(self.Q.parameters(), self.target_Q.parameters()):
            target_p.data.copy_(self.tau * p.data + (1.0 - self.tau) * target_p.data)

        return


    def train(self, bs, update = False):
        torch.backends.cudnn.benchmark = True
        device = self.device
        gamma = self.gamma

        # transition samples with importance sampling weights
        transitions, indices, weights = self.replay.sample(bs)
        # TODO : unroll transitions
        state = transitions[0]
        state_next = transitions[3]


        # masking layer
        m = np.vstack([self.mask(self.action_map(state[i])) for i in range(bs)])
        m_next = np.vstack([self.mask(self.action_map(state_next[i])) for i in range(bs)])

        # unroll batch
        # each sample : (s, a, r, s^\prime, \Delta t)
        with torch.no_grad():
            s = torch.tensor(transitions[0], dtype=torch.float).to(device)
            a = torch.unsqueeze(torch.tensor(transitions[1], dtype=torch.long).to(device), 1)  # action type : discrete
            r = torch.tensor(transitions[2], dtype=torch.float).to(device)
            s_next = torch.tensor(transitions[3], dtype=torch.float).to(device)
            d = torch.tensor(transitions[4], dtype=torch.float).to(device)
            dt = torch.tensor(transitions[5], dtype=torch.float).to(device)


            m = torch.tensor(m, dtype=torch.float).to(device)
            m_next = torch.tensor(m_next, dtype=torch.float).to(device)


            w = torch.tensor(weights, dtype=torch.float).to(device)
            # compute $\max_{a^\prime} Q (s^\prime, a^\prime)$
            # note that the maximum MUST be taken over the set of admissible actions
            # this can be done via masking invalid entries
            # Be careful of shape of each tensor!



            # get next state value
            # target construction in semi-MDP case
            # see [Puterman, 1994] for introduction to the theory of semi-MDPs
            # $r\Delta t + \gamma^{\Delta t} \max_{a^\prime} Q (s^\prime, a^\prime)$
            # target = r + (gamma ** dt) * (1. - d) * q_next
            best_actions = (self.Q(s_next).detach().mean(dim=2) + m_next ).argmax(dim=1)
            q_next = self.target_Q(s_next).detach()  # (m, N_ACTIONS, N_QUANT)
            q_next = torch.stack([q_next[i].index_select(0, best_actions[i]) for i in range(bs)]).squeeze(1)
            q_target = r.unsqueeze(1) + gamma ** dt.unsqueeze(1) * (1. - d.unsqueeze(1)) * q_next
            q_target = q_target.unsqueeze(1)  # (m , 1, N_QUANT)








        ##############
        # action value distribution prediction
        q_eval = self.Q(s)  # (m, N_ACTIONS, N_QUANT)
        q_eval = torch.stack([q_eval[i].index_select(0, a[i]) for i in range(bs)]).squeeze(1)
        # (m, N_QUANT)
        q_eval = q_eval.unsqueeze(2)  # (m, N_QUANT, 1)
        # note that dim 1 is for present quantile, dim 2 is for next quantile
        ########
        u =  q_target.detach() - q_eval


        QUANTS = np.linspace(0.0, 1.0, self.atoms + 1)[1:]
        QUANTS_TARGET = (np.linspace(0.0, 1.0, self.atoms + 1)[:-1] + QUANTS) / 2
        tau = torch.FloatTensor(QUANTS_TARGET).view(1, -1, 1).to(device)
        weight = torch.abs(tau - u.le(0.).float())  # (m, N_QUANT, N_QUANT)
        td_loss = F.smooth_l1_loss(q_eval, q_target.detach(), reduction='none')
        td_loss = torch.mean(weight * td_loss, dim=1).mean(dim=1)
        loss = torch.mean(w * td_loss)


        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()


        # priority update
        new_priorities = np.abs(np.squeeze(td_loss.cpu().data.numpy()))
        max_priority = np.max(new_priorities)
        self.max_seen_priority = max([self.max_seen_priority, max_priority])
        self.replay.update_priorities(indices=indices, priorities=new_priorities)

        # soft target update
        self.target_update()

        return

    def eval(self, test_env, T=14400, eval_num=3):
        """
        evaluation of agent
        during evaluation, agent execute noiseless actions
        """

        print('evaluating on 24 hrs data...', end=' ')

        reward_log = np.zeros(eval_num)
        num_log = np.zeros(eval_num, dtype=int)
        total_log = np.zeros(eval_num, dtype=int)

        max_flow_log = np.zeros(eval_num)
        avg_flow_log = np.zeros(eval_num)
        max_wait_log = np.zeros(eval_num)
        avg_wait_log = np.zeros(eval_num)


        for ep in range(eval_num):
            num_step = 0
            state = test_env.reset()
            if len(state) == 2:
                state = state[0]
            step_count = 0
            ep_reward = 0.
            t = 0.
            # done = False
            info = None
            # while not done:
            while t < T:
                # half hr evaluation
                if self.render and ep == 0:
                    test_env.render()


                action = self.get_action(state, 0.)  # noiseless evaluation
                next_state, reward, done, _, info = test_env.step(action)

                step_count += 1
                state = next_state
                ep_reward += self.gamma ** t * reward
                t = info['elapsed_time']


            # save carried quantity at the end of the episode
            log = test_env.operation_log
            reward_log[ep] = ep_reward
            num_log[ep] = log['carried']
            total_log[ep] = sum(log['total'])
            max_flow_log[ep] = log['max_flow_time']
            avg_flow_log[ep] = log['average_flow_time']
            max_wait_log[ep] = log['max_waiting_time']
            avg_wait_log[ep] = log['average_waiting_time']


            if self.render and ep == 0:
                test_env.close()

        reward_avg = np.mean(reward_log)
        num_avg = np.mean(num_log)
        avg_ratio = np.mean(num_log / total_log)
        max_flow = np.mean(max_flow_log)
        avg_flow = np.mean(avg_flow_log)
        max_wait = np.mean(max_wait_log)
        avg_wait = np.mean(avg_wait_log)

        print('\naverage reward : {:.4f} || carried (ratio) : {} ({:.2f}%)'.format(reward_avg, num_avg, 100. * avg_ratio))
        print('max_flow_time : {:.4f} / avg_flow_time : {:.4f}'.format(max_flow, avg_flow))
        print('max_waitng_time : {:.4f} / avg_waiting_time : {:.4f}'.format(max_wait, avg_wait))

        return dict(reward_avg=reward_avg,
                    num_avg=num_avg,
                    avg_ratio=avg_ratio,
                    max_flow=max_flow,
                    avg_flow=avg_flow,
                    max_wait=max_wait,
                    avg_wait=avg_wait
                    )

    def mask(self, actions: List[int]) -> np.ndarray:
        # generate a mask representing the set
        m = np.full(self.nA, -np.inf)
        # 0 if admissible, -inf else
        m[actions] = 0.
        return m

