import time
import os
import csv
import math
import argparse
import random
import numpy as np
import torch
import gym
import gym_lifter
from semippo import PPOAgent



parser = argparse.ArgumentParser()


parser.add_argument('--env', required=True, type=str, choices=['Lifter-v0', 'LifterPOD-v0', 'LifterCAPA-v0'])
parser.add_argument('--mode', required=True, type=int) # 2F, 3F, 4F, 6F, 8F
parser.add_argument('--pth', required=False, default=None, type=str)
parser.add_argument('--num_eval', required=False, default=1, type=int)
args = parser.parse_args()

env_id = args.env
device = 'cuda' if torch.cuda.is_available() else 'cpu'


env = gym.make(env_id, mode=args.mode)
dimS = env.observation_space.shape[0]  # dimension of state space
nA = env.action_space.n  # number of actions


agent = PPOAgent(env, dimS=dimS, dimA=nA, device=device)


pth = args.pth
if pth is not None:
    agent.load_model(pth)


carrieds = 0
totals = 0
max_flows = 0
avg_flows = 0
max_waits = 0
avg_waits = 0


num_eval = args.num_eval
T = 24 * 60 * 60 #24 hrs
for i in range(num_eval):
    state = env.reset()
    if len(state) == 2:
        state = state[0]
    step_count = 0

    t = 0.
    info = None
    while t < T:
        if i == num_eval-1 and env_id == 'Lifter-v0':
            env.render()

        action, _, _ = agent.act(state)
        next_state, reward, done, _, info = env.step(action)

        step_count += 1
        state = next_state
        t = info['elapsed_time']

    env.close()

    log = env.operation_log
    carrieds += log['carried']
    totals += sum(log['total'])
    max_flows += log['max_flow_time']
    avg_flows += log['average_flow_time']
    max_waits += log['max_waiting_time']
    avg_waits += log['average_waiting_time']



print('env : ', env_id)
print('carried (ratio) : {} ({:.2f}%)'.format(carrieds/num_eval, 100. * (carrieds) / (totals) ))
print('max_flow_time : {:.4f} / avg_flow_time : {:.4f}'.format(max_flows/num_eval, avg_flows/num_eval))
print('max_waitng_time : {:.4f} / avg_waiting_time : {:.4f}'.format(max_waits/num_eval, avg_waits/num_eval))

