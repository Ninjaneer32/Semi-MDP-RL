from typing import List, Union, Dict, Tuple
import numpy as np


# ordering
# FOUP < POD -> LOAD < UNLOAD -> FLOOR -> QUANTITY -> LAYER -> CONVEYOR (-> 2nd CONVEYOR) -> LOWER < UPPER
action_space_description = {
                            0: 'LOAD 1 FOUP 2FL2C2 LOWER',
                            1: 'LOAD 1 FOUP 2FL2C2 UPPER',
                            2: 'LOAD 1 FOUP 3FL2C2 LOWER',
                            3: 'LOAD 1 FOUP 3FL2C2 UPPER',
                            4: 'LOAD 1 FOUP 3FL3C2 LOWER',
                            5: 'LOAD 1 FOUP 3FL3C2 UPPER',
                            6: 'LOAD 2 FOUP 3FL2C2 3FL3C2',
                            7: 'LOAD 1 FOUP 6FL2C2 LOWER',
                            8: 'LOAD 1 FOUP 6FL2C2 UPPER',
                            9: 'LOAD 1 FOUP 6FL3C2 UPPER',
                            10: 'LOAD 2 FOUP 6FL2C2 6FL3C2',
                            11: 'UNLOAD 1 FOUP 2FL2C1 LOWER',
                            12: 'UNLOAD 1 FOUP 2FL2C1 UPPER',
                            13: 'UNLOAD 1 FOUP 2FL3C3 LOWER',
                            14: 'UNLOAD 1 FOUP 2FL3C3 UPPER',
                            15: 'UNLOAD 2 FOUP 2FL2C1 2FL3C3',
                            16: 'UNLOAD 1 FOUP 3FL2C1 LOWER',
                            17: 'UNLOAD 1 FOUP 3FL2C1 UPPER',
                            18: 'UNLOAD 1 FOUP 3FL3C1 LOWER',
                            19: 'UNLOAD 1 FOUP 3FL3C1 UPPER',
                            # 12: 'UNLOAD 1 FOUP 3FL3C3',
                            20: 'UNLOAD 2 FOUP 3FL2C1 3FL3C1',
                            # 14: 'UNLOAD 2 FOUP 3FL2C1 3FL3C3',
                            21: 'UNLOAD 1 FOUP 6FL2C1 LOWER',
                            22: 'UNLOAD 1 FOUP 6FL2C1 UPPER',
                            # 16: 'UNLOAD 1 FOUP 6FL2C3',
                            23: 'UNLOAD 1 FOUP 6FL3C1 UPPER',
                            24: 'UNLOAD 2 FOUP 6FL2C1 6FL3C1',
                            # 19: 'UNLOAD 2 FOUP 6FL2C3 6FL3C1',
                            25: 'LOAD 1 POD 2FL3C2',
                            26: 'LOAD 1 POD 6FL1C2',
                            27: 'UNLOAD 1 POD 2FL3C2',
                            28: 'UNLOAD 1 POD 6FL1C2',
                            29: 'STAY'
                            }


# action -> (rack position, lower / upper, load / unload, pod)
# note that POD is always loaded in the lower fork
action2operation: Dict[int, Tuple[int, int, int, int]] = {
                    0: (1, 0, 0, 0),
                    1: (0, 1, 0, 0),
                    2: (5, 0, 0, 0),
                    3: (4, 1, 0, 0),
                    4: (6, 0, 0, 0),
                    5: (5, 1, 0, 0),
                    6: (5, 2, 0, 0),
                    7: (9, 0, 0, 0),
                    8: (8, 1, 0, 0),
                    9: (9, 1, 0, 0),

                    10: (9, 2, 0, 0),
                    11: (1, 0, 1, 0),
                    12: (0, 1, 1, 0),
                    13: (2, 0, 1, 0),
                    14: (1, 1, 1, 0),
                    15: (1, 2, 1, 0),

                    16: (5, 0, 1, 0),
                    17: (4, 1, 1, 0),
                    18: (6, 0, 1, 0),
                    19: (5, 1, 1, 0),
                    20: (5, 2, 1, 0),

                    21: (9, 0, 1, 0),
                    22: (8, 1, 1, 0),
                    23: (9, 1, 1, 0),
                    24: (9, 2, 1, 0),

                    25: (2, 0, 0, 1),
                    26: (8, 0, 0, 1),
                    27: (2, 0, 1, 1),
                    28: (8, 0, 1, 1)
                    }

operation2action: Dict[Tuple[int, int, int, int], int] = {op: a for a, op in action2operation.items()}
operation2str = {op: action_space_description[a] for op, a in operation2action.items()}
operation2str[None] = 'STAY'

def available_actions(state: np.ndarray) -> List[int]:
    actions = []
    current_pos = int(round(state[0] * 9.))     # control point
    # rack position -> floor
    if current_pos < 3:
        current_floor = 2
    elif current_pos < 7:
        current_floor = 3
    else:
        current_floor = 6

    lower_floor = int(round(state[2] * 6.))
    upper_floor = int(round(state[3] * 6.))

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
    is_pod = bool(round(state[1]))
    wt = state[-7:]

    if all([wt[layer] < 0.0001 for layer in range(7)]) and (not lower_occupied) and (not upper_occupied):
        # if everything is empty, do nothing
        actions.append(29)



    if is_pod:
        if lower_floor == 2:
            actions.append(27)
        elif lower_floor == 6:
            actions.append(28)
    else:

        if lower_occupied:
            if lower_floor == 2:
                actions += [11, 13]
            elif lower_floor == 3:
                actions += [16, 18]
            elif lower_floor == 6:
                actions.append(21)
        else:
            candidates = [0, 2, 4, 7]
            corresponding_flrs = [0, 2, 3, 5]
            actions += [candidates[i] for i in range(4) if wt[corresponding_flrs[i]] > 0.]

        if upper_occupied:
            if upper_floor == 2:
                actions += [12, 14]
            elif upper_floor == 3:
                actions += [17, 19]
            elif upper_floor == 6:
                actions += [22, 23]
        else:
            candidates = [1, 3, 5, 8, 9]
            corresponding_flrs = [0, 2, 3, 5, 6]
            actions += [candidates[i] for i in range(5) if wt[corresponding_flrs[i]] > 0.]

        if not (upper_occupied or lower_occupied):
            if wt[1] > 0.:
                actions.append(25)
            if wt[4] > 0.:
                actions.append(26)
            if wt[2] > 0. and wt[3] > 0.:
                actions.append(6)
            if wt[5] > 0. and wt[6] > 0.:
                actions.append(10)
            # TODO : add POD

        if upper_occupied and lower_occupied:
            if upper_floor == 2 and lower_floor == 2:
                actions.append(15)
            elif upper_floor == 3 and lower_floor == 3:
                actions.append(20)
            elif upper_floor == 6 and lower_floor == 6:
                actions.append(24)
    # actions = list(set(actions))        # remove duplicates

    return actions


def available_actions_no_wt(state: np.ndarray) -> List[int]:
    actions = []
    lower_floor = int(round(6. * state[2]))
    upper_floor = int(round(6. * state[3]))


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
    is_pod = bool(round(state[1]))

    wq = state[-7:]
    if all([wq[layer] == 0 for layer in range(7)]) and (not lower_occupied) and (not upper_occupied):
        # if everything is empty, do nothing
        actions.append(29)

    if is_pod:
        if lower_floor == 2:
            actions.append(27)
        elif lower_floor == 6:
            actions.append(28)
    else:

        if lower_occupied:
            if lower_floor == 2:
                actions += [11, 13]
            elif lower_floor == 3:
                actions += [16, 18]
            elif lower_floor == 6:
                actions.append(21)
        else:
            candidates = [0, 2, 4, 7]
            corresponding_flrs = [0, 2, 3, 5]
            actions += [candidates[i] for i in range(4) if wq[corresponding_flrs[i]] > 0]

        if upper_occupied:
            if upper_floor == 2:
                actions += [12, 14]
            elif upper_floor == 3:
                actions += [17, 19]
            elif upper_floor == 6:
                actions += [22, 23]
        else:
            candidates = [1, 3, 5, 8, 9]
            corresponding_flrs = [0, 2, 3, 5, 6]
            actions += [candidates[i] for i in range(5) if wq[corresponding_flrs[i]] > 0]

        if not (upper_occupied or lower_occupied):
            if wq[1] > 0:
                actions.append(25)
            if wq[4] > 0:
                actions.append(26)

            if wq[2] > 0 and wq[3] > 0:
                actions.append(6)

            if wq[5] > 0. and wq[6] > 0:
                actions.append(10)

        if upper_occupied and lower_occupied:
            if upper_floor == 2 and lower_floor == 2:
                actions.append(15)
            elif upper_floor == 3 and lower_floor == 3:
                actions.append(20)
            elif upper_floor == 6 and lower_floor == 6:
                actions.append(24)

    return actions
