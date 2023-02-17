import time
import csv
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent
from torch.distributions.normal import Normal
from torch.optim import Adam
from itertools import chain
from memory import OnPolicyMemory
from utils import *
import gym
import argparse
import gym_lifter
from typing import List
import datetime


def ppo_train(env_id,
              gamma=0.99, lr=3e-4, lam=0.95, delta=1e-3,
              epsilon=0.2,
              steps_per_epoch=200, device='cpu', mode=2, T=300, num_ep=1000
              ):
    env = gym.make(env_id, mode=mode)
    dimS = env.observation_space.shape[0]   # dimension of state space
    nA = env.action_space.n                 # number of actions


    agent = PPOAgent(env, dimS=dimS, dimA=nA, device=device)


    memory = OnPolicyMemory(dimS, 1, gamma, lam, lim=steps_per_epoch)
    env = gym.make(env_id, mode=mode)
    test_env = gym.make(env_id, mode=mode)


    params = chain(agent.pi.parameters(), agent.V.parameters())
    optimizer = Adam(params, lr=lr)
    begin = time.time()




    ############################################
    # log setting
    arg_dict = locals()
    set_log_dir(env_id)


    current_time = time.strftime("%y_%m_%d_%H_%M_%S")
    FAB = {2: '8F', 3: '10F', 4: '12F', 6: '16F', 8: '20F'}
    pth = './log/' + env_id + '/' + FAB[mode] + '/' + current_time + '/'
    os.makedirs(pth, exist_ok=True)


    log_file = open(pth + 'progress.csv',
                    'w',
                    encoding='utf-8',
                    newline='')
    eval_log_file = open(pth + 'progress_eval.csv',
                         'w',
                         encoding='utf-8',
                         newline='')

    logger = csv.writer(log_file)
    eval_logger = csv.writer(eval_log_file)
    eval_logger = csv.DictWriter(eval_log_file, fieldnames=['episode',
                                                            'reward_avg',
                                                            'num_avg',
                                                            'avg_ratio',
                                                            'max_flow',
                                                            'avg_flow',
                                                            'max_wait',
                                                            'avg_wait']
                                 )
    eval_logger.writeheader()
    # save parameter configuration
    with open(pth + 'config.txt', 'w') as f:
        for key, val in arg_dict.items():
            print(key, '=', val, file=f)
    ####################################################################


    OPERATION_HOUR = T * num_ep * 4
    EVALUATION_INTERVAL = OPERATION_HOUR / 100
    CHECKPOINT_INTERVAL = OPERATION_HOUR / 10
    evaluation_count = 0
    checkpoint_count = 0
    global_t = 0.

    for i in range(num_ep):
        # start agent-env interaction
        state = env.reset()
        if len(state) == 2:
            state = state[0]
        info = None
        ep_reward = 0.
        t = 0.
        for j in range(steps_per_epoch):

            if evaluation_count * EVALUATION_INTERVAL <= global_t:
                log = evaluate(agent, test_env, num_episodes=1, gamma=gamma)
                log['episode'] = i
                eval_logger.writerow(log)
                evaluation_count += 1

            if checkpoint_count * CHECKPOINT_INTERVAL <= global_t:
                agent.save_model(pth)
                checkpoint_count += 1


            # collect transition samples by executing the policy
            action, log_prob, v = agent.act(state)
            next_state, reward, done, _, info = env.step(action)
            memory.append(state, action, reward, v, log_prob, info['dt'])


            ep_reward += gamma ** t * reward
            t += info['dt']
            global_t += info['dt']

            if j == steps_per_epoch - 1:
                s_last = torch.tensor(next_state, dtype=torch.float).to(device)
                v_last = agent.V(s_last).item()
                memory.compute_values(v_last)

            state = next_state


        # train agent at the end of each epoch
        ppo_update(agent, memory, optimizer, epsilon, num_updates=5, device=device)


        log_time = datetime.datetime.now(tz=None).strftime("%Y-%m-%d %H:%M:%S")
        op_log = env.operation_log
        # TODO : improve logging
        print('+' + '=' * 78 + '+')
        print('+' + '-' * 31 + 'TRAIN-STATISTICS' + '-' * 31 + '+')
        print(
            '{} (episode {} / epsilon = {:.2f}) reward = {:.4f}'.format(
                log_time,
                i, epsilon, ep_reward))
        print('+' + '-' * 32 + 'FAB-STATISTICS' + '-' * 32 + '+')
        print('carried = {}/{}\n'.format(op_log['carried'], sum(op_log['total'])) +
              'remain quantity : {}\n'.format(op_log['waiting_quantity']) +
              'visit_count : {}\n'.format(op_log['visit_count']) +
              'load_two : {}\n'.format(op_log['load_two']) +
              'unload_two : {}\n'.format(op_log['unload_two']) +
              'load_sequential : {}\n'.format(op_log['load_sequential']) +
              'average_waiting_time : {}\n'.format(op_log['average_waiting_time']) +
              'max_waiting_time : {}\n'.format(op_log['max_waiting_time']) +
              'average_flow_time : {}\n'.format(op_log['average_flow_time']) +
              'max_flow_time : {}'.format(op_log['max_flow_time'])
              )
        print('+' + '=' * 78 + '+')
        print('\n', end='')
        logger.writerow(
            [i, ep_reward, op_log['carried']]
            + op_log['waiting_quantity']
            + list(op_log['visit_count'])
            + [op_log['load_two'], op_log['unload_two'], op_log['load_sequential']]
            + list(op_log['total'])
            + [op_log['average_waiting_time']]
            + [op_log['max_waiting_time']]
            + [op_log['average_flow_time']]
            + [op_log['max_flow_time']]
        )



    print("saving...")
    log_file.close()
    eval_log_file.close()
    return


def ppo_update(agent, memory, optimizer, epsilon, num_updates=1, device='cpu'):
    batch = memory.load()
    target_v = torch.Tensor(batch['val']).to(device)
    A = torch.Tensor(batch['A']).to(device)
    old_log_probs = torch.Tensor(batch['log_prob']).to(device)

    for _ in range(num_updates):
        ################
        # train critic #
        ################
        m = np.vstack([mask_mul(agent.env.action_map(batch['state'][i]), num_actions=agent.dimA) for i in range(len(batch['state']))])
        m = torch.tensor(m, dtype=torch.float).to(device)
        states = torch.Tensor(batch['state']).to(device)
        # with torch.no_grad():
        probs = agent.pi(states)
        probs_new = m * (probs)


        probs_new = probs_new / torch.sum(probs_new, dim=1, keepdim=True)
        log_prob = torch.log(probs_new + 1e-10)

        ent = -torch.sum(probs_new * log_prob, dim=1, keepdim=True)
        log_probs = torch.stack([log_prob[i][batch['action'][i]] for i in range(len(batch['action']))]).to(device)



        # compute prob ratio
        # $\frac{\pi(a_t | s_t ; \theta)}{\pi(a_t | s_t ; \theta_\text{old})}$
        r = torch.exp(log_probs.squeeze(1) - old_log_probs)

        # construct clipped loss
        # $r^\text{clipped}_t(\theta) = \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)$
        clipped_r = torch.clamp(r, 1 - epsilon, 1 + epsilon)

        # surrogate objective for each $t$
        # $\min \{ r_t(\theta) \hat{A}_t, r^\text{clipped}_t(\theta) \hat{A}_t \}$
        single_step_obj = torch.min(r * A, clipped_r * A)


        pi_loss = -torch.mean(single_step_obj)
        v = agent.V(states)
        V_loss = torch.mean((v - target_v) ** 2)
        ent_bonus = torch.mean(ent)
        loss = pi_loss + 0.5 * V_loss - 0.0001 * ent_bonus
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return



class PPOAgent:
    def __init__(
                 self,
                 env,
                 dimS,
                 dimA,
                 hidden1=256,
                 hidden2=256,
                 hidden3=256,
                 device='cpu',
                 ):

        self.dimS = dimS
        self.dimA = dimA
        self.device = device
        self.action_map = env.action_map
        self.env = env
        self.pi = PPOActor(2, dimS, dimA, hidden1, hidden2, hidden3).to(device)
        self.V = ValueFunction(dimS, hidden1, hidden2).to(device)

    def act(self, state):

        m = mask_mul(self.action_map(state), num_actions=self.dimA)
        m = torch.tensor(m, dtype=torch.float).to(self.device)
        state = torch.tensor(state, dtype=torch.float).to(self.device)

        with torch.no_grad():

            probs = self.pi(state)
            probs_new = m * (probs)
            probs_new = probs_new / torch.sum(probs_new)


            p=np.array(probs_new.cpu())



            action = np.random.choice(self.dimA, p=p)
            log_prob = torch.log(probs_new + 1e-10)
            log_prob = log_prob[action]

            val = self.V(state)

        log_prob = log_prob.cpu().detach().numpy()
        val = val.cpu().detach().numpy()

        return action, log_prob, val



    def load_model(self, path):
        print('networks loading...')
        checkpoint = torch.load(path + 'model.pth.tar', map_location=self.device)
        self.pi.load_state_dict(checkpoint['actor'])
        self.V.load_state_dict(checkpoint['value'])
        return


    def save_model(self, path):
        print('adding checkpoints...')
        checkpoint_path = path + 'model.pth.tar'
        torch.save(
                    {
                     'actor': self.pi.state_dict(),
                     'value': self.V.state_dict(),
                     },
                    checkpoint_path)

        return



class PPOActor(nn.Module):
    def __init__(self, num_layer, dimS, nA, hidden1, hidden2, hidden3):
        super(PPOActor, self).__init__()
        self.num_layer = num_layer
        self.nA = nA
        self.fc1 = nn.Linear(dimS, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        if self.num_layer == 2:
            self.fc3 = nn.Linear(hidden2, nA)
        else:
            self.fc3 = nn.Linear(hidden2, hidden3)
            self.fc4 = nn.Linear(hidden3, nA)

    def forward(self, state):
        #state = F.normalize(state, dim=-1)
        x = F.elu(self.fc1(state))
        x = F.elu(self.fc2(x))
        if self.num_layer == 2:
            p = F.softmax(self.fc3(x), dim=-1)
            p = torch.exp(p)
        else:
            x = F.elu(self.fc3(x))
            x = F.elu(self.fc4(x))
            p = F.softmax(x, dim=-1)
        return p



class ValueFunction(nn.Module):
    # state value function
    def __init__(self, dimS, hidden1, hidden2):
        super(ValueFunction, self).__init__()
        self.fc1 = nn.Linear(dimS, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x



def evaluate(agent, test_env, num_episodes=3, T=86400,gamma=0.99):
    print('evaluating on 24 hrs data...', end=' ')
    eval_num = num_episodes
    reward_log = np.zeros(eval_num)
    num_log = np.zeros(eval_num, dtype=int)
    total_log = np.zeros(eval_num, dtype=int)
    max_flow_log = np.zeros(eval_num)
    avg_flow_log = np.zeros(eval_num)
    max_wait_log = np.zeros(eval_num)
    avg_wait_log = np.zeros(eval_num)

    for ep in range(eval_num):
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
            with torch.no_grad():
                action, _, _ = agent.act(state)
            next_state, reward, done, _, info = test_env.step(action)

            step_count += 1
            state = next_state

            ep_reward += gamma**t * reward
            t = info['elapsed_time']

            step_count += 1

        # save carried quantity at the end of the episode
        log = test_env.operation_log
        carried = log['carried']
        total = log['total']
        max_flow_log[ep] = log['max_flow_time']
        avg_flow_log[ep] = log['average_flow_time']
        max_wait_log[ep] = log['max_waiting_time']
        avg_wait_log[ep] = log['average_waiting_time']
        reward_log[ep] = ep_reward
        num_log[ep] = carried
        total_log[ep] = sum(total)


    reward_avg = np.mean(reward_log)
    num_avg = np.mean(num_log)
    avg_ratio = np.mean(num_log / total_log)
    max_flow = np.mean(max_flow_log)
    avg_flow = np.mean(avg_flow_log)
    max_wait = np.mean(max_wait_log)
    avg_wait = np.mean(avg_wait_log)

    # display
    print('average reward : {:.4f} || carried (ratio) : {} ({:.2f}%)'.format(reward_avg, num_avg, 100. * avg_ratio))
    print('max_flow_time : {:.4f} / avg_flow_time : {:.4f}'.format(max_flow, avg_flow))
    print('max_waitng_time : {:.4f} / avg_waiting_time : {:.4f}'.format(max_wait, avg_wait))
    # return [reward_avg, num_avg, avg_ratio, max_flow, avg_flow, max_wait, avg_wait]
    return dict(reward_avg=reward_avg,
                num_avg=num_avg,
                avg_ratio=avg_ratio,
                max_flow=max_flow,
                avg_flow=avg_flow,
                max_wait=max_wait,
                avg_wait=avg_wait
                )


def mask_mul(actions: List[int], num_actions: int) -> np.ndarray:
    # generate a multiplicative mask representing the set
    m = np.full(num_actions, 0.)
    # 1 if admissible, 0 else
    m[actions] = 1.
    return m



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    num_eps = {2: 250, 3: 500, 4: 500, 6: 1000, 8: 1000}
    parser.add_argument('--env', required=True)
    parser.add_argument('--num_trial', required=False, default=1, type=int)
    parser.add_argument('--gamma', required=False, default=0.99**11, type=float)
    parser.add_argument('--lr', required=False, default=3e-4, type=float)
    parser.add_argument('--num_steps', required=False, default=1024, type=int)
    parser.add_argument('--T', required=False, default=30*60, type=float)
    parser.add_argument('--mode', required=True, type=int)
    args = parser.parse_args()

    for _ in range(args.num_trial):
        ppo_train(args.env,
                  gamma=0.99, lr=args.lr, lam=0.95, delta=1e-3,
                  epsilon=0.2,
                  steps_per_epoch=args.num_steps, device=device, mode=args.mode, T=args.T, num_ep=num_eps[args.mode]
                  )

