import gym
import numpy as np
from typing import List, Tuple
from gym_lifter.envs.fab.fab_capa import FAB


class LifterCAPAEnv(gym.Env):
	def __init__(self, mode):
		# super(gym.Env, self).__init__()
		self.fab = FAB(mode=mode)
		self.num_layers = self.fab.num_layers
		self.state_dim = 3 + 2 + 3 * self.fab.num_layers  # + 2 : additional fork

		#self.state_dim = 3 + 2 * 7  # 3: DUMMY
		# state variables : rack position, lower_to, upper_to, con_i_to's, con_i_wt's
		# note that operating time is not regarded as a state variable
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float)
		# 24 actions in total
		#self.goal_space = gym.spaces.Discrete(4)
		self.state = None
		self.pos_to_flr = self.fab.pos2floor #[2, 2, 2,  3, 3, 3, 3,    5, 5, 5, 5,     6, 6, 6, 6,   8,8,8]  # convert control point to floor
		self.capacities = self.fab.capacities

		pos2floor, action2operation, operation2action, label2pos = self.action_mapping()
		self.pos2floor = pos2floor
		self.action2operation = action2operation
		self.operation2action = operation2action
		self.label2pos = label2pos
		self.action_space = gym.spaces.Discrete(len(self.action2operation)+1)




		#self.action2operation = action2operation
		return



	def reset(self):
		self.fab.reset()
		return self.get_obs()





	def step(self, action: int):
		assert self.action_space.contains(action)
		rew=0
		if action == len(self.action2operation):
			operation = None
		else:
			operation = self.action2operation[action]
			floor, pos, low_up, load_unload, low_lr, up_lr = operation
			if low_up == 2:
				rew += 1.
			else:
				rew += 0.5
			# if load_unload == 1:
			# 	if low_up == 2:
			# 		rew += 2.
			# 	else:
			# 		rew += 1.


		# rew = 0 #np.zeros(self.fab.num_layers)
		# max = 0
		# for i, conveyors in enumerate(self.fab.layers.values()):
		# 	for j in range(conveyors.QUEUE_LEN):
		# 		if (self.fab.t - conveyors.QUEUE[j].cmd_time) > max :
		# 			max = (self.fab.t - conveyors.QUEUE[j].cmd_time)
		# 		rew -= (self.fab.t - conveyors.QUEUE[j].cmd_time)


		#if action in [11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 31, 32, 33, 34, 35, 36]:
		#	rew = 1.
		#elif action in [15, 20, 24, 33, 37]:
		#	rew = 2.
		#else:
		#	rew = 0.



		# elif action in [6, 10, 27, 30]:
		# 	rew = 1.
		# else:
		# 	rew = 0.5
		# if action == 34:
		# 	rew = 0.

		# operate the FAB

		info = self.fab.sim(operation)
		# rew=0
		# for layer in self.fab.layers:
		# 	if self.fab.layers[layer].QUEUE_LEN > 0:
		# 		for wafer in self.fab.layers[layer].QUEUE:
		# 			rew -= (self.fab.t - wafer.cmd_time)/300.
		# rew = -sum([self.waiting_time[i] / 300 for i in range(len(self.waiting_time))])
		rew += info['lost']
		done = False
		truncated = False
		return self.get_obs(), rew, done, truncated, info

	def get_obs(self) -> np.ndarray:
		# encode the current state into a point in $\mathbb{R}^n$ (n : state space dimension)
		######################################################################################################
		# rack position | POD | lower destination | upper destination | queue destination | waiting quantity #
		######################################################################################################
		rpos = (self.rack_pos) / (self.fab.NUM_FLOOR*4-1)
		lower_to, upper_to = self.rack_destination
		lower_to_, upper_to_ = self.rack_destination_
		# rack_info = [rpos, float(self.is_pod_loaded), lower_to / 6., upper_to / 6.]
		rack_info = [rpos, lower_to / float(self.fab.NUM_FLOOR), upper_to / float(self.fab.NUM_FLOOR), lower_to_ / float(self.fab.NUM_FLOOR), upper_to_ / float(self.fab.NUM_FLOOR)]
		destination = [d / self.fab.NUM_FLOOR for d in self.destination]
		waiting_quantity = [( self.waiting_quantity[i] / self.capacities[i]) for i in range(self.num_layers)]
		waiting_time = [self.waiting_time[i] / 300. for i in range(len(self.waiting_time))]
		layer_info = destination + waiting_quantity + waiting_time
		obs = np.array(rack_info + layer_info)

		return obs


	def action_mapping(self):
		# pos, low_up, load_unload
		NUM_FLOOR = self.fab.NUM_FLOOR

		##### label_decoder={ conveyor(456) : (floor, layer) : (2,  123) }
		label_decoder = {}
		conveyor = 1
		for floor in range(1, NUM_FLOOR + 1):
			for layer in range(1, 4):
				label_decoder[conveyor] = (floor, layer)
				conveyor += 1

		##### pos2label = { pos(5678) : (lower_conv, upper_conv) : (x456, 456x) }
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

		action2operation: Dict[int, Tuple[int, int, int, int, int, int]] = {}
		idx = 0
		for pos in list(pos2label.keys()): #########################POD NON
			# for a in range(4):
			# action2operation[idx]= (pos2floor[pos],pos,0,0)
			if pos2label[pos][0] != None and pos2label[pos][1] != None:
				action2operation[idx] = (pos2floor[pos], pos, 2, 0, 0, 0)
				idx += 1
				action2operation[idx] = (pos2floor[pos], pos, 2, 1, 0, 0)
				idx += 1

				action2operation[idx] = (pos2floor[pos], pos, 2, 0, 1, 0)
				idx += 1
				action2operation[idx] = (pos2floor[pos], pos, 2, 1, 1, 0)
				idx += 1

				action2operation[idx] = (pos2floor[pos], pos, 2, 0, 1, 1)
				idx += 1
				action2operation[idx] = (pos2floor[pos], pos, 2, 1, 1, 1)
				idx += 1

				action2operation[idx] = (pos2floor[pos], pos, 2, 0, 0, 1)
				idx += 1
				action2operation[idx] = (pos2floor[pos], pos, 2, 1, 0, 1)
				idx += 1
				#######
				action2operation[idx] = (pos2floor[pos], pos, 0, 0, 0, 0)
				idx += 1
				action2operation[idx] = (pos2floor[pos], pos, 0, 1, 0, 0)
				idx += 1
				action2operation[idx] = (pos2floor[pos], pos, 0, 0, 1, 0)
				idx += 1
				action2operation[idx] = (pos2floor[pos], pos, 0, 1, 1, 0)
				idx += 1


				action2operation[idx] = (pos2floor[pos], pos, 1, 0, 0, 0)
				idx += 1
				action2operation[idx] = (pos2floor[pos], pos, 1, 1, 0, 0)
				idx += 1
				action2operation[idx] = (pos2floor[pos], pos, 1, 0, 0, 1)
				idx += 1
				action2operation[idx] = (pos2floor[pos], pos, 1, 1, 0, 1)
				idx += 1
				####
			elif pos2label[pos][0] != None:

				action2operation[idx] = (pos2floor[pos], pos, 0, 0, 0, 0)
				idx += 1
				action2operation[idx] = (pos2floor[pos], pos, 0, 1, 0, 0)
				idx += 1
				action2operation[idx] = (pos2floor[pos], pos, 0, 0, 1, 0)
				idx += 1
				action2operation[idx] = (pos2floor[pos], pos, 0, 1, 1, 0)
				idx += 1

			elif pos2label[pos][1] != None:

				action2operation[idx] = (pos2floor[pos], pos, 1, 0, 0, 0)
				idx += 1
				action2operation[idx] = (pos2floor[pos], pos, 1, 1, 0, 0)
				idx += 1


				action2operation[idx] = (pos2floor[pos], pos, 1, 0, 0, 1)
				idx += 1
				action2operation[idx] = (pos2floor[pos], pos, 1, 1, 0, 1)
				idx += 1

		# list(pos2floor.values())
		# print(action2operation)

		label2pos: Dict[Tuple[int, int], int] = {op: a for a, op in pos2label.items()}
		operation2action: Dict[Tuple[int, int, int], int] = {op: a for a, op in action2operation.items()}
		#print(operation2action)
		#print(operation2action)
		return pos2floor, action2operation, operation2action, label2pos

	#@staticmethod
	def action_map(self, state) -> List[int]:

		NUM_FLOOR = self.fab.NUM_FLOOR
		actions = []
		current_pos = int(round(state[0] * (NUM_FLOOR * 4 - 1)))  # control point
		# rack position -> floor
		current_floor = self.pos2floor[current_pos]


		lower_floor = int(round(state[1] * NUM_FLOOR))
		upper_floor = int(round(state[2] * NUM_FLOOR))
		lower_floor_ = int(round(state[3] * NUM_FLOOR))
		upper_floor_ = int(round(state[4] * NUM_FLOOR))

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

		if lower_floor_ == 0:
			# lower fork is empty
			lower_occupied_ = False
		else:
			lower_occupied_ = True

		if upper_floor_ == 0:
			# upper fork is empty
			upper_occupied_ = False
		else:
			upper_occupied_ = True


		wq = state[-(3 * NUM_FLOOR):]


		if all([wq[layer] < 0.0001 for layer in range((3 * NUM_FLOOR))]) and (not lower_occupied) and (not upper_occupied) and (not lower_occupied_) and (not upper_occupied_):
			# if everything is empty, do nothing
			actions.append(len(self.action2operation))
			return actions


		for floor in range(1, NUM_FLOOR + 1):
			L1 = 3 * (floor - 1) + 1
			q = wq[L1 - 1:L1 + 2]
			if q[0] > 0:
				if not upper_occupied and floor != 1:
					actions.append(self.operation2action[(floor, self.label2pos[(None, L1)], 1, 0, 0, 0)])#
				if not upper_occupied_ and floor != 1:
					actions.append(self.operation2action[(floor, self.label2pos[(None, L1)], 1, 0, 0, 1)])
				if not lower_occupied:
					actions.append(self.operation2action[(floor, self.label2pos[(L1, L1 + 1)], 0, 0, 0, 0)])#
				if not lower_occupied_:
					actions.append(self.operation2action[(floor, self.label2pos[(L1, L1 + 1)], 0, 0, 1, 0)])
			if q[1] > 0:
				if not upper_occupied:
					actions.append(self.operation2action[(floor, self.label2pos[(L1, L1 + 1)], 1, 0, 0, 0)])#
				if not upper_occupied_:
					actions.append(self.operation2action[(floor, self.label2pos[(L1, L1 + 1)], 1, 0, 0, 1)])
				if not lower_occupied:
					actions.append(self.operation2action[(floor, self.label2pos[(L1 + 1, L1 + 2)], 0, 0, 0, 0)])#
				if not lower_occupied_:
					actions.append(self.operation2action[(floor, self.label2pos[(L1 + 1, L1 + 2)], 0, 0, 1, 0)])
			if q[2] > 0:
				if not upper_occupied:
					actions.append(self.operation2action[(floor, self.label2pos[(L1 + 1, L1 + 2)], 1, 0, 0, 0)])#
				if not upper_occupied_:
					actions.append(self.operation2action[(floor, self.label2pos[(L1 + 1, L1 + 2)], 1, 0, 0, 1)])
				if not lower_occupied and floor != NUM_FLOOR:
					actions.append(self.operation2action[(floor, self.label2pos[(L1 + 2, None)], 0, 0, 0, 0)])#
				if not lower_occupied_ and floor != NUM_FLOOR:
					actions.append(self.operation2action[(floor, self.label2pos[(L1 + 2, None)], 0, 0, 1, 0)])
			if q[0] > 0 and q[1] > 0:
				if not upper_occupied and not lower_occupied:
					actions.append(self.operation2action[(floor, self.label2pos[(L1, L1 + 1)], 2, 0, 0, 0)])#
				if not upper_occupied_ and not lower_occupied_:
					actions.append(self.operation2action[(floor, self.label2pos[(L1, L1 + 1)], 2, 0, 1, 1)])
				if not upper_occupied and not lower_occupied_:
					actions.append(self.operation2action[(floor, self.label2pos[(L1, L1 + 1)], 2, 0, 1, 0)])
				if not upper_occupied_ and not lower_occupied:
					actions.append(self.operation2action[(floor, self.label2pos[(L1, L1 + 1)], 2, 0, 0, 1)])
			if q[1] > 0 and q[2] > 0:
				if not upper_occupied and not lower_occupied:
					actions.append(self.operation2action[(floor, self.label2pos[(L1 + 1, L1 + 2)], 2, 0, 0, 0)])
				if not upper_occupied_ and not lower_occupied_:
					actions.append(self.operation2action[(floor, self.label2pos[(L1 + 1, L1 + 2)], 2, 0, 1, 1)])
				if not upper_occupied and not lower_occupied_:
					actions.append(self.operation2action[(floor, self.label2pos[(L1 + 1, L1 + 2)], 2, 0, 1, 0)])
				if not upper_occupied_ and not lower_occupied:
					actions.append(self.operation2action[(floor, self.label2pos[(L1 + 1, L1 + 2)], 2, 0, 0, 1)])

		##if lower uc
		if lower_occupied:
			L1 = (lower_floor - 1) * 3 + 1
			actions.append(self.operation2action[(lower_floor, self.label2pos[(L1, L1 + 1)], 0, 1, 0, 0)])#
			actions.append(self.operation2action[(lower_floor, self.label2pos[(L1 + 1, L1 + 2)], 0, 1, 0, 0)])#

			if lower_floor != NUM_FLOOR:
				actions.append(self.operation2action[(lower_floor, self.label2pos[(L1 + 2, None)], 0, 1, 0, 0)])#
		if lower_occupied_:
			L1_ = (lower_floor_ - 1) * 3 + 1
			actions.append(self.operation2action[(lower_floor_, self.label2pos[(L1_, L1_ + 1)], 0, 1, 1, 0)])
			actions.append(self.operation2action[(lower_floor_, self.label2pos[(L1_ + 1, L1_ + 2)], 0, 1, 1, 0)])
			if lower_floor_ != NUM_FLOOR:
				actions.append(self.operation2action[(lower_floor_, self.label2pos[(L1_ + 2, None)], 0, 1, 1, 0)])

		##upper_floor
		if upper_occupied:
			L1 = (upper_floor - 1) * 3 + 1
			actions.append(self.operation2action[(upper_floor, self.label2pos[(L1, L1 + 1)], 1, 1, 0, 0)])#
			actions.append(self.operation2action[(upper_floor, self.label2pos[(L1 + 1, L1 + 2)], 1, 1, 0, 0)])#
			if upper_floor != 1:
				actions.append(self.operation2action[(upper_floor, self.label2pos[(None, L1)], 1, 1, 0, 0)])#
		if upper_occupied_:
			L1_ = (upper_floor_ - 1) * 3 + 1
			actions.append(self.operation2action[(upper_floor_, self.label2pos[(L1_, L1_ + 1)], 1, 1, 0, 1)])
			actions.append(self.operation2action[(upper_floor_, self.label2pos[(L1_ + 1, L1_ + 2)], 1, 1, 0, 1)])
			if upper_floor_ != 1:
				actions.append(self.operation2action[(upper_floor_, self.label2pos[(None, L1_)], 1, 1, 0, 1)])

		if upper_occupied and lower_occupied and lower_floor == upper_floor:
			# a ; action2operation[a] == [lower_floor, _ , 2, 1]
			L1 = (lower_floor - 1) * 3 + 1
			actions.append(self.operation2action[(lower_floor, self.label2pos[(L1, L1 + 1)], 2, 1, 0, 0)])#
		if upper_occupied and lower_occupied_ and lower_floor_ == upper_floor:
			# a ; action2operation[a] == [lower_floor, _ , 2, 1]
			L1 = (lower_floor_ - 1) * 3 + 1
			actions.append(self.operation2action[(lower_floor_, self.label2pos[(L1, L1 + 1)], 2, 1, 1, 0)])
		if upper_occupied_ and lower_occupied and lower_floor == upper_floor_:
			# a ; action2operation[a] == [lower_floor, _ , 2, 1]
			L1 = (lower_floor - 1) * 3 + 1
			actions.append(self.operation2action[(lower_floor, self.label2pos[(L1, L1 + 1)], 2, 1, 0, 1)])
		if upper_occupied_ and lower_occupied_ and lower_floor_ == upper_floor_:
			# a ; action2operation[a] == [lower_floor, _ , 2, 1]
			L1 = (lower_floor_ - 1) * 3 + 1
			actions.append(self.operation2action[(lower_floor_, self.label2pos[(L1, L1 + 1)], 2, 1, 1, 1)])



		if upper_occupied and lower_occupied and lower_floor == upper_floor:
			# a ; action2operation[a] == [lower_floor, _ , 2, 1]
			L1 = (lower_floor - 1) * 3 + 1
			actions.append(self.operation2action[(lower_floor, self.label2pos[(L1 + 1, L1 + 2)], 2, 1, 0, 0)])#

		if upper_occupied_ and lower_occupied and lower_floor == upper_floor_:
			# a ; action2operation[a] == [lower_floor, _ , 2, 1]
			L1 = (lower_floor - 1) * 3 + 1
			actions.append(self.operation2action[(lower_floor, self.label2pos[(L1 + 1, L1 + 2)], 2, 1, 0, 1)])

		if upper_occupied and lower_occupied_ and lower_floor_ == upper_floor:
			# a ; action2operation[a] == [lower_floor, _ , 2, 1]
			L1 = (lower_floor_ - 1) * 3 + 1
			actions.append(self.operation2action[(lower_floor_, self.label2pos[(L1 + 1, L1 + 2)], 2, 1, 1, 0)])

		if upper_occupied_ and lower_occupied_ and lower_floor_ == upper_floor_:
			# a ; action2operation[a] == [lower_floor, _ , 2, 1]
			L1 = (lower_floor_ - 1) * 3 + 1
			actions.append(self.operation2action[(lower_floor_, self.label2pos[(L1 + 1, L1 + 2)], 2, 1, 1, 1)])


		actions = list(set(actions))  # remove duplicates


		return actions

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
	def rack_destination_(self) -> Tuple[int, int]:
		return self.fab.rack_destination_


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
