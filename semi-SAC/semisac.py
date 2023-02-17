import time
from datetime import datetime
import os
import csv
import argparse
import random
import numpy as np
import torch
import gym
import gym_lifter
from sac_agent import SACAgent
from utils import get_env_spec, set_log_dir


def run_sac(
            env_id,
            num_ep=2e3,
            T=300,
            num_layer=2,
            eval_interval=100,
            start_train=10000,
            train_interval=50,
            buffer_size=1e6,
            fill_buffer=20000,
            gamma=0.99,
            pi_lr=3e-4,
            q_lr=3e-4,
            alpha_lr=3e-4,
            polyak=5e-3,
            adjust_entropy=False,
            alpha=0.2,
            target_entropy=-6.0,
            hidden1=128,
            hidden2=128,
            hidden3=128,
            batch_size=64,
            pth=None,
            device='cpu',
            render='False',
            dueling=False,
            mode=2
            ):

    arg_dict = locals()

    num_ep = int(num_ep)
    buffer_size = int(buffer_size)

    env = gym.make(env_id, mode=mode)
    test_env = gym.make(env_id, mode=mode)


    dimS = env.observation_space.shape[0]   # dimension of state space
    nA = env.action_space.n                 # number of actions

    # (physical) length of the time horizon of each truncated episode
    # each episode run for t \in [0, T)
    # set for RL in semi-MDP setting

    agent = SACAgent(dimS,
                     env,
                     nA,
                     num_layer,
                     env.action_map,
                     gamma,
                     pi_lr=pi_lr,
                     q_lr=q_lr,
                     alpha_lr=alpha_lr,
                     polyak=polyak,
                     adjust_entropy=adjust_entropy,
                     target_entropy=target_entropy,
                     alpha=alpha,
                     hidden1=hidden1,
                     hidden2=hidden2,
                     hidden3=hidden3,
                     buffer_size=buffer_size,
                     batch_size=batch_size,
                     device=device,
                     render=render,
                     dueling=dueling
                     )

    # log setting
    set_log_dir(env_id)

    current_time = time.strftime("%y_%m_%d_%H_%M_%S")
    FAB = {2: '8F', 3: '10F', 4: '12F', 6: '16F', 8: '20F'}
    pth = './log/' + env_id + '/' + FAB[mode] + '/' + current_time + '/'
    os.makedirs(pth, exist_ok=True)


    log_file = open(pth+'progress.csv',
                    'w',
                    encoding='utf-8',
                    newline='')
    eval_log_file = open(pth+'progress_eval.csv',
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

    OPERATION_HOUR = T * num_ep
    # 200 evaluations in total
    EVALUATION_INTERVAL = OPERATION_HOUR / 100
    CHECKPOINT_INTERVAL = OPERATION_HOUR / 10
    evaluation_count = 0
    checkpoint_count = 0
    # start environment roll-out
    global_t = 0.
    counter = 0
    t_unit = 1.


    for i in range(num_ep):
        if global_t >= OPERATION_HOUR:
            break

        # initialize an episode
        s = env.reset()
        if len(s) == 2:
            s = s[0]
        t = 0.  # physical elapsed time of the present episode
        info = None
        ep_reward = 0.

        bottleneck = False
        while t < T:
            # evaluation is done periodically
            if evaluation_count * EVALUATION_INTERVAL <= global_t:
                log = agent.eval(test_env, T=24*60*60, eval_num=1)
                log['episode'] = i
                eval_logger.writerow(log)
                evaluation_count += 1


            if checkpoint_count * CHECKPOINT_INTERVAL <= global_t:
                agent.save_model(pth)
                checkpoint_count += 1

            while not bottleneck:
                s_next, _, _, _, info = env.step(nA-1)
                s = s_next

                if info['elapsed_time'] > 60. :  # 1분(60초) 후부터 넘기기
                    bottleneck = True


            if counter < fill_buffer:
                a = random.choice(env.action_map(s))
            else:
                a = agent.get_action(s)#, waiting_time=env.waiting_time)

            s_next, r, d, _, info = env.step(action=a)


            ep_reward += gamma ** t * r
            dt = info['dt']
            t += dt

            global_t += dt
            counter += 1

            #const = {2: 11, 3: 12.5, 4: 13, 5:14, 6: 15, 8: 16}
            agent.buffer.append(s, a, r, s_next, False, dt)
            if counter >= start_train and counter % train_interval == 0:
                # training stage
                # single step per one transition observation
                for _ in range(1):
                    agent.train(i)

            s = s_next


        # save training statistics
        log_time = datetime.now(tz=None).strftime("%Y-%m-%d %H:%M:%S")
        op_log = env.operation_log
        print('+' + '=' * 78 + '+')
        print('+' + '-' * 31 + 'TRAIN-STATISTICS' + '-' * 31 + '+')
        print('{} (episode {}) reward = {:.4f}'.format(log_time, i, ep_reward))
        print('+' + '-' * 32 + 'FAB-STATISTICS' + '-' * 32 + '+')
        print('carried = {}/{}\n'.format(op_log['carried'], sum(op_log['total'])) +
              'remain_quantity : {}\n'.format(op_log['waiting_quantity']) +
              'visit_count : {}\n'.format(op_log['visit_count']) +
              'load_two : {}\n'.format(op_log['load_two']) +
              'unload_two : {}\n'.format(op_log['unload_two']) +
              'load_sequential : {}\n'.format(op_log['load_sequential']) +
              'total : ', op_log['total']
              )
        print('+' + '=' * 78 + '+')
        print('\n', end='')
        logger.writerow(
            [i, ep_reward, op_log['carried']]
            + op_log['waiting_quantity']
            + list(op_log['visit_count'])
            + [op_log['load_two'], op_log['unload_two'], op_log['load_sequential']]
            + list(op_log['total'])
        )
    log_file.close()
    eval_log_file.close()

    return


if __name__ == "__main__":
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True)
    parser.add_argument('--num_trial', required=False, default=1, type=int)
    parser.add_argument('--gamma', required=False, default=0.99, type=float)
    parser.add_argument('--tau', required=False, default=5e-3, type=float)
    parser.add_argument('--pi_lr', required=False, default=3e-4, type=float)
    parser.add_argument('--q_lr', required=False, default=3e-4, type=float)
    parser.add_argument('--alpha_lr', required=False, default=3e-4, type=float)
    parser.add_argument('--num_layer', required=False, default=2, type=int)
    parser.add_argument('--hidden1', required=False, default=256, type=int)
    parser.add_argument('--hidden2', required=False, default=256, type=int)
    parser.add_argument('--hidden3', required=False, default=256, type=int)
    parser.add_argument('--train_interval', required=False, default=4, type=int)
    parser.add_argument('--start_train', required=False, default=20000, type=int)
    parser.add_argument('--fill_buffer', required=False, default=0, type=int)
    parser.add_argument('--batch_size', required=False, default=32, type=int)
    parser.add_argument('--buffer_size', required=False, default=1e5, type=float)
    parser.add_argument('--num_ep', required=False, default=4e3, type=float)
    parser.add_argument('--T', required=False, default=30 * 60, type=float)
    parser.add_argument('--alpha', required=False, default=0.2, type=float)
    parser.add_argument('--adjust_entropy', action='store_true')
    parser.add_argument('--target_entropy', required=False, default=-4.0, type=float)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--pth', required=False, default=None)
    parser.add_argument('--device', required=False, default=default_device)
    parser.add_argument('--dueling', action='store_true')
    parser.add_argument('--mode', required=True, type=int)
    args = parser.parse_args()

    for _ in range(args.num_trial):
        run_sac(args.env,
                num_ep=args.num_ep,
                T=args.T,
                num_layer=args.num_layer,

                start_train=args.start_train,
                train_interval=args.train_interval,
                fill_buffer=args.fill_buffer,
                gamma=args.gamma,
                pi_lr=args.pi_lr,
                q_lr=args.q_lr,
                alpha_lr=args.alpha_lr,
                polyak=args.tau,
                adjust_entropy=args.adjust_entropy,
                target_entropy=args.target_entropy,
                alpha=args.alpha,
                hidden1=args.hidden1,
                hidden2=args.hidden2,
                hidden3=args.hidden3,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size,
                pth=args.pth,
                device=args.device,
                render=args.render,
                dueling=args.dueling,
                mode=args.mode
                )
