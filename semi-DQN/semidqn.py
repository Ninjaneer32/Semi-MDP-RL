import csv
import time
import argparse
import gym
from agent import SemiDQNAgent
from replay import Transition
import torch
import gym_lifter
import datetime
import os
import numpy as np

import random


def run_prioritized(env_id='Lifter-v0',
                    gamma=0.94176,
                    lr=1e-5,
                    polyak=5e-4,
                    num_layer=3,
                    hidden1=256,
                    hidden2=256,
                    hidden3=256,
                    num_ep=2000,
                    buffer_size=int(1e5),
                    fill_buffer=2000,
                    batch_size=32,
                    train_interval=4,
                    eval_interval=100,
                    eval_num= 3,
                    T= 60*30,
                    start_train = 1000,
                    priority_exponent=.5,
                    importance_sampling_exponent_begin=.4,
                    importance_sampling_exponent_end=1.,
                    uniform_sample_prob=1e-3,
                    normalize_weights=True,
                    dueling=1,
                    pth=None,
                    device='cuda',
                    render=False,
                    mode = 2,
                    n_atom = 100,
                    ):



    arg_dict = locals()

    num_ep = int(num_ep)
    buffer_size = int(buffer_size)

    env = gym.make(env_id, mode=mode)
    test_env = gym.make(env_id, mode=mode)


    dimS = env.observation_space.shape[0]   # dimension of state space
    nA = env.action_space.n                 # number of actions





    # linearly scheduled $\epsilon$
    max_epsilon = 1.
    min_epsilon = 0.02
    exploration_schedule = LinearSchedule(begin_t=0,
                                          end_t= num_ep // 2,
                                          begin_value=max_epsilon,
                                          end_value=min_epsilon)


    # linearly scheduled importance sampling weight exponent
    anneal_schedule = LinearSchedule(begin_t=0,
                                     end_t=num_ep * 200,
                                     begin_value=importance_sampling_exponent_begin,
                                     end_value=1)

    exponent_anneal_schedule = LinearSchedule(begin_t=0,
                                     end_t=num_ep * 200,
                                     begin_value = priority_exponent,
                                     end_value = priority_exponent)


    agent = SemiDQNAgent(
                         env=env,
                         dimS=dimS,
                         nA=nA,
                         action_map=env.action_map,
                         gamma=gamma,
                         num_layer=num_layer,
                         hidden1=hidden1,
                         hidden2=hidden2,
                         hidden3=hidden3,
                         lr=lr,
                         tau=polyak,
                         buffer_size=buffer_size,
                         batch_size=batch_size,
                         priority_exponent=exponent_anneal_schedule,
                         anneal_schedule=anneal_schedule,
                         uniform_sample_prob=uniform_sample_prob,
                         normalize_weights=normalize_weights,
                         device=device,
                         render=render,
                         dueling=dueling,
                         n_atom = n_atom,
                         )



    # default location of directory for training log
    current_time = time.strftime("%y_%m_%d_%H_%M_%S")

    FAB = { 2 :'8F', 3:'10F', 4:'12F', 6:'16F', 8:'20F'}
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
    with open(pth + 'config.txt', 'w') as f:
        for key, val in arg_dict.items():
            print(key, '=', val, file=f)


    total_operation_hr = T * num_ep
    evaluation_interval = total_operation_hr / eval_interval
    CHECKPOINT_INTERVAL = total_operation_hr / 10

    evaluation_count = 0
    checkpoint_count = 0

    global_t = 0.
    counter = 0

    for i in range(num_ep):


        s = env.reset()
        if len(s) == 2:
            s = s[0]
        t = 0.  # physical elapsed time of the present episode
        ep_reward = 0.
        epsilon = exploration_schedule(i)
        if global_t >= total_operation_hr:
            break


        bottleneck = False
        while t < T:
            if evaluation_count * evaluation_interval <= global_t:
                log = agent.eval(test_env, T=60*60*24,  eval_num=eval_num)
                log['episode'] = i
                eval_logger.writerow(log)
                evaluation_count += 1

            if checkpoint_count * CHECKPOINT_INTERVAL <= global_t:
                agent.save_model(pth)
                print("saving...")
                checkpoint_count += 1

            while not bottleneck:
                s_next, _, _, _, info = env.step(nA-1)
                s = s_next
                if info['elapsed_time'] > 60.:  #during first 1 minutes, we do not store transitions
                    bottleneck = True

            if counter < fill_buffer:
                a = random.choice(env.action_map(s))
            else:
                a = agent.get_action(s, epsilon)
            s_next, r, d, _, info = env.step(action=a)

            ep_reward += gamma ** t * r
            dt = info['dt']
            t += dt

            global_t += dt
            counter += 1


            #Standard DQN case : const_dt[mode] ; const_dt = {2:11,3:12.5,4:13,5:14,6:15,8:16}
            transition = Transition(s_tm1=s, a_tm1=a, r_t=r, s_t=s_next, d=False, dt=dt)
            agent.replay.add(item=transition, priority=agent.max_seen_priority)



            s = s_next

            if counter > start_train and counter % train_interval == 0:
                agent.train(bs=batch_size)

        log_time = datetime.datetime.now(tz=None).strftime("%Y-%m-%d %H:%M:%S")
        replay_size = agent.replay.size


        op_log = env.operation_log
        # TODO : improve logging
        print('+' + '=' * 78 + '+')
        print('+' + '-' * 31 + 'TRAIN-STATISTICS' + '-' * 31 + '+')
        print('{} (episode {} / epsilon = {:.2f}) reward = {:.4f} \nmax_seen_priority = {:.2f} \nreplay size = {}'.format(log_time,
              i, epsilon, ep_reward, agent.max_seen_priority, replay_size))
        print('+' + '-' * 32 + 'FAB-STATISTICS' + '-' * 32 + '+')
        print('carried = {}/{}\n'.format(op_log['carried'], sum(op_log['total'])) +
              # 'carried_pod = {}/{}\n'.format(info['carried_pod'], info['pod_total']) +
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





class LinearSchedule:
    """Linear schedule, used for exploration epsilon in DQN agents."""
    # taken from https://github.com/deepmind/dqn_zoo/blob/master/dqn_zoo/parts.py
    def __init__(self,
                 begin_value,
                 end_value,
                 begin_t,
                 end_t=None,
                 decay_steps=None):
        if (end_t is None) == (decay_steps is None):
            raise ValueError('Exactly one of end_t, decay_steps must be provided.')
        self._decay_steps = decay_steps if end_t is None else end_t - begin_t
        self._begin_t = begin_t
        self._begin_value = begin_value
        self._end_value = end_value

    def __call__(self, t):
        """Implements a linear transition from a begin to an end value."""
        frac = min(max(t - self._begin_t, 0), self._decay_steps) / self._decay_steps
        return (1 - frac) * self._begin_value + frac * self._end_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(default_device)

    parser.add_argument('--env', required=True)
    parser.add_argument('--gamma', required=False, default=0.99, type=float)
    parser.add_argument('--q_lr', required=False, default=1e-4, type=float)
    parser.add_argument('--tau', required=False, default=1e-3, type=float)
    parser.add_argument('--num_layer', required=False, default=3, type=int)
    parser.add_argument('--hidden1', required=False, default=256, type=int)
    parser.add_argument('--hidden2', required=False, default=256, type=int)
    parser.add_argument('--hidden3', required=False, default=256, type=int)
    parser.add_argument('--num_ep', required=False, default=4e3, type=float)
    parser.add_argument('--buffer_size', required=False, default=int(1e6), type=int)
    parser.add_argument('--fill_buffer', required=False, default=1000, type=int)
    parser.add_argument('--batch_size', required=False, default=32, type=int)
    parser.add_argument('--train_interval', required=False, default=4, type=int)
    parser.add_argument('--start_train', required=False, default=10000, type=int)
    parser.add_argument('--eval_interval', required=False, default=100, type=int)
    parser.add_argument('--eval_num', required=False, default=1, type=int)
    parser.add_argument('--T', required=False, default=60*30, type=float)

    parser.add_argument('--prior_exp', required=False, default=0.5, type=float)
    parser.add_argument('--exp_begin', required=False, default=0.4, type=float)
    parser.add_argument('--exp_end', required=False, default=1., type=float)
    parser.add_argument('--unif_prob', required=False, default=1e-3, type=float)

    parser.add_argument('--dueling', required=False, default=1, type=int)
    parser.add_argument('--device', required=False, default=default_device, type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--mode', required=True, type=int)
    parser.add_argument('--num_trials', required=False, default=1, type=int)
    parser.add_argument('--num_atoms', required=False, default=100, type=int)


    args = parser.parse_args()
    for _ in range(args.num_trials):
        run_prioritized(args.env,
                        gamma=args.gamma,
                        lr=args.q_lr,
                        polyak=args.tau,
                        num_layer=args.num_layer,
                        hidden1=args.hidden1,
                        hidden2=args.hidden2,
                        hidden3=args.hidden3,
                        num_ep=args.num_ep,
                        buffer_size=args.buffer_size,
                        fill_buffer=args.fill_buffer,
                        batch_size=args.batch_size,
                        train_interval=args.train_interval,
                        start_train=args.start_train,
                        eval_interval=args.eval_interval,
                        eval_num=args.eval_num,
                        T=args.T,
                        priority_exponent=args.prior_exp,
                        importance_sampling_exponent_begin=args.exp_begin,
                        importance_sampling_exponent_end=args.exp_end,
                        uniform_sample_prob=args.unif_prob,
                        dueling=args.dueling,
                        device=args.device,
                        render=args.render,
                        mode= args.mode,
                        n_atom= args.num_atoms,
                        )
