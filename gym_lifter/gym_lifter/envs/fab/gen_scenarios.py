import argparse
import os
from os import path
import numpy as np


def generate_scenarios(floors, seed, pod=False):
    """
    Data generation code for fab.
    To consider diverse fab architecture, we consider 5 different type of facilities: {8F, 10F, 12F, 16F, 20F}.
    In each case, multiple 24-hrs scenarios are generated, each of which are sampled from the same data distribution.
    The generated scenarios are used to simulate episodes of (semi-)MDP designed in the paper.
    """
    assert floors in [8, 10, 12, 16, 20]

    np.random.seed(seed)
    # ------------------------------ problem data ------------------------------ #
    sim_time = 86400.               # generation of arrival data for 24 hrs
    # number of floors used for operation
    num_working_floors_dict = {8: 2, 10: 3, 12: 4, 16: 6, 20: 8}

    # expected number of arrived lots during a single simulation
    num_lots_dict = {8: 6400, 10: 6000, 12: 5600, 16: 5200, 20: 4800}

    # list of floors engaged in fab operation in each case
    fab_floors_dict = {8:  [3, 7],
                  10: [2, 5, 9],
                  12: [3, 6, 9, 11],
                  16: [3, 5, 9, 10, 12, 15],
                  20: [3, 5, 6, 10, 12, 14, 16, 18]
                  }
    # -------------------------------------------------------------------------- #

    # TODO : move these to argparse
    # number of total scenarios
    # Every scenario (or episode) follows the same arrival distribution (same MDP)
    num_scenarios = 200
    start = 0
    pod_str = 'o' if pod else 'x'
    print('generating {} scenarios for {}-floor fab (pod {})...'.format(num_scenarios, floors, pod_str), end='')
    num_working_floors = num_working_floors_dict[floors]
    num_lots = num_lots_dict[floors]
    interested = fab_floors_dict[floors]

    # list of possible (origin, destinations) pairs of each lot
    # example: 10F -> {(2, 5), (2, 9). (5, 2), (5, 9), (9, 2), (9, 5)}
    missions = []
    for one in interested:
        for another in interested:
            if another is not one:
                missions.append((one, another))

    prob = []
    seed = np.random.rand(len(missions))
    for i in range(len(missions)):
        prob.append(seed[i] / sum(seed))
    num = [num_lots * p for p in prob]  # number of lots that have a given (origin, destination)
    """
    computation of the parameters for Poisson distribution & Exponential distribution
    lambda: parameter for Poisson process
    S_n : time when n-th lot arrival occurs
    Here we assume S_n = tau_1 + ... tau_n, where each tau_k is an i.i.d. sample drawn from Exponential(beta):
    P(tau_k > x) = e^{-beta x}, x > 0
    N_t : # of lots arrived until time t which is defined by
    N_t = max(n >= 0 : S_n <=> t)
    Since N_t >= n iff S_n <= t, we deduce that N_t follows Poisson(lambda * t) where lambda := 1 / beta.
    Thus, E(N_t) = lambda * t.
    See also Billingsley, 1979.
    """
    lam = [n_lots / sim_time for n_lots in num]     # Poisson
    beta = [1. / ell for ell in lam]                # Exponential

    # verification
    lam_floor = {f: sum([lam[i] for i, mission in enumerate(missions) if mission[0] == f]) / 3. for f in interested}
    # print(lam_floor)
    # assert abs(sum(lam_floor.values()) * sim_time - num_lots) < 1.

    # data generation loop
    for scenario in range(start, start+num_scenarios):

        data = {mission: [] for mission in missions}
        num_arrival = 0
        elapsed_t = 0.
        for i in range(len(missions)):
            mission = missions[i]
            mission_list = list(mission)
            b = beta[i]
            elapsed_t = 0.

            while elapsed_t < sim_time:
                # tau_k ~ Exponential(beta)
                dt = np.random.exponential(b)
                elapsed_t += dt
                data[mission].append([elapsed_t] + mission_list)
                num_arrival += 1

            # remove the final one, since its command time exceeds 24 hrs
            data[mission].pop()
            data[mission] = np.array(data[mission])
            num_arrival -= 1

        entire_episode = np.concatenate([data[mission] for mission in missions if data[mission].size > 0], axis=0)
        entire_episode = entire_episode[entire_episode[:, 0].argsort()]     # sort entire data by command time

        # save generated data
        data_cmd = entire_episode[:, 0]
        data_from = entire_episode[:, 1]
        data_to = entire_episode[:, 2]

        dir_path = str(num_working_floors)+'F'

        if pod:
            pod_data = np.random.choice([0., 1.], len(data_cmd), p=[0.9, 0.1])
            dir_path = path.join(path.dirname(__file__), 'assets', 'POD', dir_path, 'scenario{}'.format(scenario))
        else:
            dir_path = path.join(path.dirname(__file__), 'assets', dir_path, 'scenario{}'.format(scenario))

        os.makedirs(dir_path, exist_ok=True)

        if pod:
            np.save(dir_path + 'data_pod.npy', np.array(pod_data))

        np.save(path.join(dir_path, 'data_cmd.npy'), np.array(data_cmd))
        np.save(path.join(dir_path, 'data_from.npy'), np.array(data_from, dtype=np.int))
        np.save(path.join(dir_path, 'data_to.npy'), np.array(data_to, dtype=np.int))
    print('done!')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--floors', required=True, type=int, choices=[8, 10, 12, 16, 20])
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--pod', action='store_true')
    args = parser.parse_args()

    generate_scenarios(floors=args.floors, seed=args.seed, pod=args.pod)
