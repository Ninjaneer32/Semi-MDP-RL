import os
import gym
import numpy as np
import math
import argparse
import csv

import gym_lifter
from mpc import MPCController
from estimator import ParameterEstimator


DAY = 86400.

def main(mode=2, steps=3, file_count=0):
    """
    Simulation of the fab environment with MPC controller.
    -------------------------------------------------------------------------------------------------------------------
    :param mode: choice of the fab
                 chosen among {2, 3, 4, 6, 8} each of which corresponds to {8F, 10F, 12F, 16F, 20F}
    
    :param steps: number of steps considered in MPC

    :param file_count: index representing the log file (included in file name)
    -------------------------------------------------------------------------------------------------------------------
    """
    assert steps > 0

    os.makedirs('log', exist_ok=True)

    # operating floors in the fab
    # used to reduce the number of variables in MINLP
    # TODO: make it as an environment property
    valid_floors = {
        2: [3, 7],
        3: [2, 5, 9],
        4: [3, 6, 9, 11],
        6: [3, 5, 9, 10, 12, 15],
        8: [3, 5, 6, 10, 12, 14, 16, 18]
    }

    env = gym.make('Lifter-v0', mode=mode)

    floors = valid_floors[mode]

    # possible positions of the lifter
    points = [4 * (f - 1) + i for f in floors for i in range(1, 5)]

    # keys for denoting conveyor belts
    conveyors = [3 * (f - 1) + i for f in floors for i in range(1, 4)]
    capacities = env.get_capacities()
    capacities = {c: capacities[c-1] for c in conveyors}

    num_actions = env.action_space.n

    # -------------------------- problem parameter matrices --------------------------
    reward_vector = {}
    time_matrix = {p: {} for p in points}
    position_matrix = {p: {} for p in points}
    L_u = {c: {} for c in conveyors}
    L_ell = {c: {} for c in conveyors}
    U_u = {f: {} for f in floors}
    U_ell = {f: {} for f in floors}

    # initialization
    stay_action = num_actions - 1
    actions = [stay_action]
    reward_vector[stay_action] = 0.
    for pt in points:
        time_matrix[pt][stay_action] = 1.

    for pt in points:
        position_matrix[pt][stay_action] = 0

    for con in conveyors:
        L_u[con][stay_action] = 0
        L_ell[con][stay_action] = 0

    for flr in floors:
        U_u[flr][stay_action] = 0
        U_ell[flr][stay_action] = 0

    for a in range(num_actions - 1):
        # low_up = 0: lower / = 1: upper / = 2 : both
        # load_unload = 0: load / 1: unload
        f, pos, lower_upper, load_unload = env.action2operation[a]
        if pos in points:
            # if f == 3:
            #     print(lower_upper, load_unload)

            # The chosen action is valid for the current configuration.
            actions.append(a)
            # initialization
            for con in conveyors:
                L_ell[con][a], L_u[con][a] = 0, 0
            for flr in floors:
                U_ell[flr][a], U_u[flr][a] = 0, 0
            reward_vector[a] = 1. if lower_upper == 2 else .5   # reward vector construction
            for pt in points:
                time_matrix[pt][a] = .5 * abs(pt - pos) + 5.
                position_matrix[pt][a] = 0
            position_matrix[pos][a] = 1
            # print(f)
            # f = env.fab.pos2floor[pos]

            c_lower, c_upper = env.fab.pos2label[pos]

            if load_unload == 0:
                # load
                if lower_upper == 0:
                    # lower
                    L_ell[c_lower][a] = 1
                elif lower_upper == 1:
                    # upper
                    L_u[c_upper][a] = 1
                elif lower_upper == 2:
                    L_ell[c_lower][a] = 1
                    L_u[c_upper][a] = 1

            elif load_unload == 1:
                if lower_upper == 0:

                    U_ell[f][a] = 1
                elif lower_upper == 1:
                    U_u[f][a] = 1
                elif lower_upper == 2:
                    U_ell[f][a] = 1
                    U_u[f][a] = 1
    # --------------------------------------------------------------------------------

    param_estimator = ParameterEstimator(params=conveyors)

    controller = MPCController(actions=actions,
                               stay_action=stay_action,
                               points=points,
                               floors=floors,
                               num_steps=steps,
                               horizon_length=100,
                               conveyors=conveyors,
                               capacities=capacities,
                               reward_vector=reward_vector,
                               time_matrix=time_matrix,
                               position_matrix=position_matrix,
                               L_u=L_u,
                               L_ell=L_ell,
                               U_u=U_u,
                               U_ell=U_ell
                               )

    _ = env.reset()
    env.fab.rack_pos = np.random.choice(points)
    done = False

    t_unit = env.fab.t_unit

    controller.update_model_params(param_estimator.infer())
    
    log = env.operation_log

    fieldnames = ['tot', 'miss', 'missing', 'carried',
                  'load_two', 'unload_two', 'load_sequential',
                  'average_waiting_time', 'max_waiting_time',
                  'average_flow_time', 'max_flow_time',
                  'time', 'computation_time'
                  ]

    file = 'log/res{}_mode{}_steps{}.csv'.format(file_count, mode, steps)
    csvfile = open(file, 'w', newline='')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    save_interval = 5
    display_interval = 50

    count = 0

    mpc_runtime = 0.

    while env.fab.elapsed_time < DAY:
        if count % save_interval == 0 and  count > 0:
            log = env.operation_log
            log = {key: log[key] for key in fieldnames if key in log}
            log['time'] = env.fab.elapsed_time
            log['computation_time'] = mpc_runtime / count
            writer.writerow(log)

        if count % display_interval == 0 and count > 0:
            display_log(log, count, display_width=60)

        lower, upper = env.rack_destination
        num_lots = {c: env.waiting_quantity[c-1] for c in conveyors}
        destinations = {f: {c: {k: 0 for k in range(steps)} for c in conveyors} for f in floors}

        for c in conveyors:
            f = env.destination[c-1]
            if f > 0:
                destinations[f][c][0] = 1
            current_f = (c - 1) // 3 + 1
            candidate_destinations = [flr for flr in floors if flr != current_f]
            for k in range(1, steps):
                idx = np.random.choice(candidate_destinations)

                destinations[idx][c][k] = 1
        
        # computation of an action to be applied to the system
        open_loop, mpc_info = controller.act(lifter_pos=env.rack_pos, upper=upper, lower=lower, num_lots=num_lots, destinations=destinations)
        mpc_runtime += mpc_info['runtime']

        # In MPC, only the first action of the obtained sequence is used.
        a = open_loop[0]
        s, r, done, _, info = env.step(a)

        counts = {c: info['arrival_counts'][c] for c in conveyors}
        time_intervals = info['dt'] * t_unit

        # parameter update via MLE
        param_estimator.observe(time_intervals=time_intervals, counts=counts)
        params = param_estimator.infer()
        controller.update_model_params(params=params)
        count += 1

    csvfile.close()
    return


def display_log(log, count, display_width=60):
    digit = math.floor(math.log10(count)) + 1
    left = (display_width - (5 + digit)) // 2
    right = display_width - (left + 5 + digit)
    print('+' + left * '-' + 'step {}'.format(count) + right * '-' + '+')

    for key, val in log.items():
        print('{} : {}'.format(key, val))
    print('+' + display_width * '-' + '+')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=False, default=2, type=int, choices=[2, 3, 4, 6, 8])
    parser.add_argument('--steps', required=False, default=3, type=int)
    parser.add_argument('--file_count', required=False, default=0, type=int)
    args = parser.parse_args()
    main(mode=args.mode, steps=args.steps, file_count=args.file_count)
