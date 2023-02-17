from typing import List, Union, Dict, Tuple
import numpy as np


# ordering
# FOUP < POD -> LOAD < UNLOAD -> FLOOR -> QUANTITY -> LAYER -> CONVEYOR (-> 2nd CONVEYOR) -> LOWER < UPPER

action_space_description = {
                            0: 'LOAD 1 FOUP 2FL2C2 LOWER',
                            1: 'LOAD 1 FOUP 2FL2C2 UPPER', # 2F L3 Load 추가하기
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
                            25: 'LOAD 1 FOUP 2FL3 UPPER', ##여기부터 분배해야해요 25: (1, 1, 0)
                            26: 'LOAD 1 FOUP 2FL3 LOWER', ## 26: (2, 0, 0)
                            27: 'LOAD 2 FOUP 2FL2 2FL3', ## 27: (1, 2, 0)
                            28: 'LOAD 1 FOUP 6FL1 UPPER', ##여기부터 6층 28: (7, 1, 0)
                            29: 'LOAD 1 FOUP 6FL1 LOWER', ##여기부터 6층 29: (8, 0, 0)
                            30: 'LOAD 2 FOUP 6FL1 6FL2', ##여기부터 6층 30: (8, 2, 0)
                            31: 'UNLOAD 1 FOUP 6FL1 UPPER', ##여기부터 6층 31: (7, 1, 1)
                            32: 'UNLOAD 1 FOUP 6FL1 LOWER', ##여기부터 6층 32: (8, 0, 1)
                            33: 'UNLOAD 2 FOUP 6FL1 6FL2', ##여기부터 6층 33: (8, 2, 1)

                            34:'UNLOAD 1 FOUP 5FL2 LOWER', # (9, 0, 1)
                            35:'UNLOAD 1 FOUP 5FL2 UPPER', # (8, 1, 1)
                            36: 'UNLOAD 1 FOUP 5FL3 LOWER',  # (10, 0, 1)
                            37: 'UNLOAD 1 FOUP 5FL3 UPPER',  # (9, 1, 1)
                            38: 'UNLOAD 2 FOUP 5FL2 5FL3',  # (9, 2, 1)

                            39: 'LOAD 1 FOUP 5FL2 LOWER', # (9, 0, 0)
                            40: 'LOAD 1 FOUP 5FL2 UPPER', # (8, 1, 0)
                            41: 'LOAD 1 FOUP 5FL3 LOWER', # (10, 0, 0)
                            42: 'LOAD 1 FOUP 5FL3 UPPER', # (9, 1, 0)
                            43: 'LOAD 1 FOUP 5FL2 5FL3',  # (9, 2, 0)
                            44: ' ',
                            45: ' ',
                            46: ' ',
                            47: ' ',
                            48: ' ',
                            49: ' ',
                            50: ' ',
                            51: ' ',
                            52: ' ',
                            53: ' ',
                            54: ' ',
                            55: ' ',
                            # 37:'LOAD 1 FOUP 6FL1 LOWER',
                            # 38:'LOAD 1 FOUP 6FL1 LOWER',
                            # 39:'LOAD 1 FOUP 6FL1 LOWER',

                            # 19: 'UNLOAD 2 FOUP 6FL2C3 6FL3C1',
                            # 25: 'LOAD 1 POD 2FL3C2',
                            # 26: 'LOAD 1 POD 6FL1C2',
                            # 27: 'UNLOAD 1 POD 2FL3C2',
                            # 28: 'UNLOAD 1 POD 6FL1C2',
                            56: 'STAY'
                            }




#pos, low_up, load_unload
NUM_FLOOR = 12


##### label_decoder={ conveyor(456) : (floor, layer) : (2,  123) }
label_decoder={}
conveyor = 1
for floor in range(1, NUM_FLOOR+1):
    for layer in range(1, 4):
        label_decoder[conveyor]=(floor, layer)
        conveyor += 1

##### pos2label = { pos(5678) : (lower_conv, upper_conv) : (x456, 456x) }
pos2label = {}
pos2floor = {}
for floor in range(1, NUM_FLOOR+1):
    for pos in range(1, 5):
        if (floor == 1 and pos == 1) or (floor == NUM_FLOOR and pos == 4):
            continue
        if pos == 1:
            pos2label[4 * (floor - 1) + pos] = (None, 3 * (floor - 1) + pos)
        elif pos == 4:
            pos2label[4 * (floor - 1) + pos] = (3 * floor , None)
        else:
            pos2label[4 * (floor - 1) + pos] = (3 * (floor-1) + pos -1,  3 * (floor-1) + pos)
        pos2floor[4 * (floor - 1) + pos] = floor



labels = list(np.arange(1, 3*NUM_FLOOR+1))
capacities = 3 * np.ones_like(labels)


action2operation : Dict[int, Tuple[int, int, int, int]]= {}
idx=0
for pos in list(pos2label.keys()):
    #for a in range(4):
    #action2operation[idx]= (pos2floor[pos],pos,0,0)
    if pos2label[pos][0] != None and pos2label[pos][1] != None :
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

#list(pos2floor.values())
#print(action2operation)

label2pos: Dict[Tuple[int, int], int] = {op: a for a, op in pos2label.items()}
operation2action: Dict[Tuple[int, int, int], int] = {op: a for a, op in action2operation.items()}

# action -> (rack position, lower / upper, load / unload)
# action2operation: Dict[int, Tuple[int, int, int, int]] = {
#                     0: (1, 0, 0),
#                     1: (0, 1, 0),
#                     2: (5, 0, 0),
#                     3: (4, 1, 0),
#                     4: (6, 0, 0),
#                     5: (5, 1, 0),
#                     6: (5, 2, 0),
#                     7: (13, 0, 0),
#                     8: (12, 1, 0),
#                     9: (13, 1, 0),
#                     10: (13, 2, 0),
#                     11: (1, 0, 1),
#                     12: (0, 1, 1),
#                     13: (2, 0, 1),
#                     14: (1, 1, 1),
#                     15: (1, 2, 1),
#                     16: (5, 0, 1),
#                     17: (4, 1, 1),
#                     18: (6, 0, 1),
#                     19: (5, 1, 1),
#                     20: (5, 2, 1),
#                     21: (13, 0, 1),
#                     22: (12, 1, 1),
#                     23: (13, 1, 1),
#                     24: (13, 2, 1),
#                     25: (1, 1, 0),
#                     26: (2, 0, 0),
#                     27: (1, 2, 0),
#                     28: (11, 1, 0),
#                     29: (12, 0, 0),
#                     30: (12, 2, 0),
#                     31: (11, 1, 1),
#                     32: (12, 0, 1),
#                     33: (12, 2, 1),
#                     34: (9, 0, 1),
#                     35: (8, 1, 1),
#                     36: (10, 0, 1),
#                     37: (9, 1, 1),
#                     38: (9, 2, 1),
#                     39: (9, 0, 0),
#                     40: (8, 1, 0),
#                     41: (10, 0, 0),
#                     42: (9, 1, 0),
#                     43: (9, 2, 0),
#
#                     44: (14, 0, 0), ## 6F     #
#                     45: (14, 0, 1),           #
#                     46: (13, 2, 2),           # dummy
#                     47: (13, 2, 1),           #
#
#                     48: (16, 1, 0), ## 8F   #
#                     49: (16, 1, 1),          #
#                     50: (17, 1, 0),          #
#                     51: (17, 0, 0),          #
#                     52: (17, 1, 1),          #
#                     53: (17, 0, 1),          #
#                     54: (17, 2, 0),          #
#                     55: (17, 2, 1),          #
#                     # 25: (2, 0, 0),
#                     # 26: (8, 0, 0),
#                     # 27: (2, 0, 1),
#                     # 28: (8, 0, 1)
#                     }

#operation2str = {op: action_space_description[a] for op, a in operation2action.items()}
#operation2str[None] = 'STAY'

def available_actions_no_wt(state: np.ndarray) -> List[int]:
    actions = []
    current_pos = int(round(state[0] *(NUM_FLOOR*4-1) ))     # control point
    # rack position -> floor
    current_floor = pos2floor[current_pos]
    #
    # if current_pos < 3:
    #     current_floor = 2
    # elif current_pos < 7:
    #     current_floor = 3
    # elif current_pos < 11:
    #     current_floor = 5
    # elif current_pos < 15:
    #     current_floor = 6
    # else:
    #     current_floor = 8

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
    #is_pod = bool(round(state[1]))


    #wq = state[-11:]

    wq=state[-(3 * NUM_FLOOR):]
    # wq=[]
    # pos2flr = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # idx = 0
    # for i in range(len(pos2flr)):
    #     if pos2flr[i]>0:
    #         wq.append(wq_imsi[idx])
    #         idx+=1
    #     else:
    #         wq.append(0)

    if all([wq[layer] < 0.0001 for layer in range((3 * NUM_FLOOR))]) and (not lower_occupied) and (not upper_occupied):
        # if everything is empty, do nothing
        actions.append(len(action2operation))
        return actions


    is_pod = False
    if is_pod:
        if lower_floor == 2:
            actions.append(27)
        elif lower_floor == 6:
            actions.append(28)
    else:
        # if lower_occupied:                      #### 층을 찾고 (포인트는 중요하지 않음) unload/load 정하고 아랫칸/윗칸 정하고
        #     if lower_floor == 2:
        #         actions += [11, 13]
        #     elif lower_floor == 3:
        #         actions += [16, 18]
        #     elif lower_floor == 6:
        #         actions += [21, 32,45] #45
        #     elif lower_floor == 5:
        #         actions += [34, 36]
        #     elif lower_floor == 8:
        #         actions += [53]
        # else:
        #     candidates = [0, 2, 4, 7, 26, 29, 39, 41, 51,44]
        #     corresponding_flrs = [0, 2, 3, 7, 1, 6, 4, 5, 9,8] #8
        #     actions += [candidates[i] for i in range(len(candidates)) if wq[corresponding_flrs[i]] > 0.]
        #
        # if upper_occupied:
        #     if upper_floor == 2:
        #         actions += [12, 14]
        #     elif upper_floor == 3:
        #         actions += [17, 19]
        #     elif upper_floor == 6:
        #         actions += [22, 23, 31]
        #     elif upper_floor == 5:
        #         actions += [35, 37]
        #     elif upper_floor == 8:
        #         actions += [         49, 52]
        # else:
        #     candidates = [1, 3, 5, 8, 9, 28, 25, 40, 42, 48, 50]
        #     corresponding_flrs = [0, 2, 3, 7, 8, 6, 1, 4, 5, 9, 10]
        #     actions += [candidates[i] for i in range(11) if wq[corresponding_flrs[i]] > 0.]
        #

        for floor in range(1, NUM_FLOOR+1):
            L1= 3*(floor-1)+1
            q = wq[L1-1:L1+2]
            if q[0] >0:
                if not upper_occupied and floor != 1:
                    actions.append(operation2action[(floor, label2pos[(None,L1)], 1, 0)])
                if not lower_occupied:
                    actions.append(operation2action[(floor, label2pos[(L1,L1+1)], 0, 0)])
            if q[1] > 0:
                if not upper_occupied:
                    actions.append(operation2action[(floor, label2pos[(L1, L1 + 1)], 1, 0)])
                if not lower_occupied:
                    actions.append(operation2action[(floor, label2pos[(L1+1, L1+2)], 0, 0)])
            if q[2] > 0:
                if not upper_occupied:
                    actions.append(operation2action[(floor, label2pos[(L1+1, L1+2)], 1, 0)])
                if not lower_occupied and floor != NUM_FLOOR:
                    actions.append(operation2action[(floor, label2pos[(L1+2, None)], 0, 0)])
            if q[0]>0 and q[1] > 0:
                if not upper_occupied and not lower_occupied:
                    actions.append(operation2action[(floor, label2pos[(L1, L1+1)], 2, 0)])
            if q[1]>0 and q[2] > 0:
                if not upper_occupied and not lower_occupied:
                    actions.append(operation2action[(floor, label2pos[(L1+1, L1+2)], 2, 0)])


        ##if lower uc
        if lower_occupied:
            L1= (lower_floor-1)*3 +1
            actions.append(operation2action[(lower_floor, label2pos[(L1, L1+1)], 0, 1)])
            actions.append(operation2action[(lower_floor, label2pos[(L1+1, L1+2)], 0, 1)])
            if lower_floor != NUM_FLOOR:
                actions.append(operation2action[(lower_floor, label2pos[(L1+2, None)], 0, 1)])


        ##upper_floor
        if upper_occupied:
            L1= (upper_floor-1)*3 +1
            actions.append(operation2action[(upper_floor, label2pos[(L1, L1+1)], 1, 1)])
            actions.append(operation2action[(upper_floor, label2pos[(L1+1, L1+2)], 1, 1)])
            if upper_floor != 1:
                actions.append(operation2action[(upper_floor, label2pos[(None, L1)], 1, 1)])


        if upper_occupied and lower_occupied and lower_floor == upper_floor:
            # a ; action2operation[a] == [lower_floor, _ , 2, 1]
            L1= (lower_floor-1) * 3 + 1
            actions.append(operation2action[(lower_floor, label2pos[(L1, L1+1)], 2, 1)])
            actions.append(operation2action[(lower_floor, label2pos[(L1+1, L1+2)], 2, 1)])



        # if not (upper_occupied or lower_occupied):
        #     #1) floor마다 둘다 싣기 가능한 후보가 잇음
        #     #2) 그 후보마다 가능한 액션을 찾아주면 됨
        #
        #     if upper_floor == lower_floor:
        #         # a ; action2operation[a] == [lower_floor, _ , 2, 1]
        #         if lower_floor == 1:
        #             start = 0
        #         else:
        #             start = 14 + 16 * (lower_floor-2)
        #
        #         while start < len(action2operation) and action2operation[start] == lower_floor:
        #             if action2operation[start][2] == 2 and action2operation[start][3] == 1:
        #                 actions.append(start)
        #             start +=1
        #
        #     if wq[0] > 0. and wq[1] > 0.:
        #         actions.append(27)
        #     if wq[2] > 0. and wq[3] > 0.:
        #         actions.append(6)
        #     if wq[6] > 0. and wq[7] > 0.:
        #         actions.append(30)
        #     if wq[7] > 0. and wq[8] > 0.:
        #         actions.append(10)
        #     if wq[4] > 0. and wq[5] > 0.:
        #         actions.append(43)
        #     if wq[9] > 0. and wq[10] > 0.:
        #         actions.append(54)
        #     # TODO : add POD


            #
            #     # 해당 플로어에 해당되는게 lower_floor = (0~13) (14~29) (30~45) ... (302~315)
            #
            # if upper_floor == 2 and lower_floor == 2:
            #     actions.append(15)
            # elif upper_floor == 3 and lower_floor == 3:
            #     actions.append(20)
            # elif upper_floor == 6 and lower_floor == 6:
            #     actions+= [24, 33,        47] ## 24 필요 없지
            # elif upper_floor == 5 and lower_floor == 5:
            #     actions+= [38] ## 24 필요 없지
            # elif upper_floor == 8 and lower_floor == 8:
            #     actions+= [           55] ## 24 필요 없지

        # #For 3F FAB
        # not_allowed =[36,41]
        # for e in not_allowed:
        #     if e in actions:
        #         actions.remove(e)
    actions = list(set(actions))        # remove duplicates
    return actions


def available_actions_no_wts(state: np.ndarray) -> List[int]:
    actions = []
    lower_floor = int(round(6. * state[1]))
    upper_floor = int(round(6. * state[2]))

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
    # is_pod = bool(round(state[1]))
    wq = state[-7:]
    if all([wq[layer] == 0 for layer in range(7)]) and (not lower_occupied) and (not upper_occupied):
        # if everything is empty, do nothing
        actions.append(34)

    # if is_pod:
    #     if lower_floor == 2:
    #         actions.append(27)
    #     elif lower_floor == 6:
    #         actions.append(28)

    if lower_occupied:
        if lower_floor == 2:
            actions += [11, 13]
        elif lower_floor == 3:
            actions += [16, 18]
        elif lower_floor == 6:
            actions += [21, 32]
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

