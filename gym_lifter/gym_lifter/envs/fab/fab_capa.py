from gym_lifter.envs.fab.wafer import Wafer
from gym_lifter.envs.fab.rack_capa import Rack
from gym_lifter.envs.fab.conveyor import ConveyorBelt
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from os import path
import pygame
from collections import deque


class FAB:
    # meta data for rendering
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    BLACK = (0, 0, 0)
    PINK = (255, 96, 208)
    size = [1200, 800]

    def __init__(self, mode=2):
        #self.socket = launch_server()
        # architecture description
        # family of InConveyors labelled by their floors
        # 7 In-Conveyor C2
        # 2F : L2 L3(POD) / 3F : L1 L2 L3 / 6F : L1(POD) L2 L3
        self.rack = Rack()
        #self.floors = [2,3,      5,       6,        8]

        self.tot = 0
        self.miss =0
        # label assigned to each conveyor belt
        #self.labels = [2, 3, 5, 6,          7, 8,     9,10,11,      12,13] #10 11
        NUM_FLOOR = 4 + 2 * mode
        self.NUM_FLOOR = NUM_FLOOR
        self.floors = list(np.arange(1, NUM_FLOOR + 1))
        self.labels = list(np.arange(1, 3 * NUM_FLOOR + 1))
        self.capacities = 3 * np.ones_like(self.labels)
        #self.capacities = [3, 2, 4, 3, 3, 2, 3, 2, 2, 3, 2]

        # self.max_capacity = max(self.capacities)
        self.num_layers = len(self.labels)
        self.layers: Dict[int, ConveyorBelt] = {
            label: ConveyorBelt(capacity=self.capacities[i]) for i, label in enumerate(self.labels)
        }

        # label -> (floor, layer)
        # self.label_decoder = {2: (2, 2), 3: (2, 3), 5: (3, 2), 6: (3, 3),     7:(5,2), 8:(5,3),       9: (6, 1), 10: (6, 2), 11: (6, 3),       12:(8,2), 13:(8,3) }
        # self.label2str = {
        #     label: '{}F L{}'.format(self.label_decoder[label][0], self.label_decoder[label][1]) for label in self.labels}
        #
        # self.label2floor = {label: self.label_decoder[label][0] for label in self.labels}
        #
        #
        #
        # self.pos2label = {0: (None, 2), 1: (2, 3), 2: (3, None),        3: (None, None), 4: (None, 5),  5: (5, 6), 6: (6, None),
        #                7: (None, None), 8: (None, 7),  9: (7, 8), 10:(8, None),              11:(None,9), 12:(9,10), 13:(10,11), 14:(11,None),         15:(None,None),16:(None,12),17:(12,13) }
        # self.pos2floor = [2, 2, 2, 3, 3, 3, 3,         5, 5, 5, 5,        6, 6, 6,     6,8,8,8]  # convert control point to floor


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
                if (floor == 1 and pos == 1) or (floor == 20 and pos == 4):
                    continue
                if pos == 1:
                    pos2label[4 * (floor - 1) + pos] = (None, 3 * (floor - 1) + pos)
                elif pos == 4:
                    pos2label[4 * (floor - 1) + pos] = (3 * floor, None)
                else:
                    pos2label[4 * (floor - 1) + pos] = (3 * (floor - 1) + pos - 1, 3 * (floor - 1) + pos)
                pos2floor[4 * (floor - 1) + pos] = floor

        self.pos2label = pos2label
        self.pos2floor = pos2floor#.values()
        #self.pos2floor = list(pos2floor.values())
        self.flow_time_log = None
        self.waiting_time_log = None

        self.loaded_time_log = None
        self.unloaded_time_log = None
        self.scene= 0
        # travel time of the rack master between two floors
        # d[i, j] = time consumed for rack master to move from (ctrl pt i) to (ctrl pt j)
        """
        self.distance_matrix = np.array([[0.,   2.5,  3.4,  6.87, 6.94, 7.01, 7.08, 8.85, 8.9,  8.96],
                                         [2.5,  0.,   2.5,  6.79, 6.87, 6.94, 7.01, 8.79, 8.85, 8.9],
                                         [3.4,  2.5,  0.,   6.72, 6.79, 6.87, 6.94, 8.73, 8.79, 8.85],
                                         [6.87, 6.79, 6.72, 0.,   2.5,  3.4,  4.02, 5.03, 5.11, 5.19],
                                         [6.94, 6.87, 6.79, 2.5,  0.,   2.5,  3.4,  4.95, 5.03, 5.11],
                                         [7.01, 6.94, 6.87, 3.4,  2.5,  0.,   2.5,  4.87, 4.95, 5.03],
                                         [7.08, 7.01, 6.94, 4.02, 3.4,  2.5,  0.,   4.80, 4.87, 4.95],
                                         [8.85, 8.79, 8.73, 5.03, 4.95, 4.87, 4.80, 0.,   2.5,  3.4],
                                         [8.9,  8.85, 8.79, 5.11, 5.03, 4.95, 4.87, 2.5,  0.,   2.5],
                                         [8.96, 8.9,  8.85, 5.19, 5.11, 5.03, 4.95, 3.4,  2.5,  0.]])
        """



        self.rack_pos = None        # takes value in [0, 10)
        self.mode = mode
        self.data_cmd = None
        self.data_from = None
        self.data_to = None
        self.num_data = None
        self.num_added = None
        self.end = None
        self.t = None
        self.t_unit = 1.
        self.visit_count = None
        # statistics
        self.num_carried = None
        self.load_two = None
        self.unload_two = None
        self.load_sequential = None
        self.total_amount = None
        self.arrival = None

        # attributes for rendering
        self.command_queue = deque(maxlen=5)    # store 5 latest rack master commands with their execution times

        self.screen = None
        self.clock = None
        self.framerate = 4
        self.pause = None
        self.quit = None

    def load_arrival_data(self):
        scenario = np.random.randint(low=0, high=200)

        # poisson

        dir = str(self.mode) + "F"
        if self.scene==0:
            dir_path = 'assets/'+dir+ '/scenario{}/'.format(scenario)
        if self.scene==1:
            dir_path = 'assets/'+dir+'/scenario{}/'.format(scenario)

        self.data_cmd = np.load(path.join(path.dirname(__file__), dir_path + "data_cmd.npy"))
        self.data_from = np.load(path.join(path.dirname(__file__), dir_path + "data_from.npy"))
        self.data_to = np.load(path.join(path.dirname(__file__), dir_path + "data_to.npy"))
        self.num_data = self.data_cmd.shape[0]


    def reset(self):
        self.flow_time_log = []
        self.waiting_time_log = []
        self.rack.reset()


        self.rack_pos = np.random.randint(low=2, high=self.NUM_FLOOR*4) #1이 없기 때문
        for conveyor in self.layers.values():
            conveyor.reset()

        self.load_arrival_data()
        self.end = 0
        self.t = 0.

        # FAB statistics
        self.num_carried = 0
        self.visit_count = np.zeros(4*self.NUM_FLOOR, dtype=int)
        self.total_amount = np.zeros(len(self.capacities), dtype=int)
        self.load_two = 0
        self.unload_two = 0
        self.load_sequential = 0
        self.missing = 0

        self.pause = False
        self.quit = False





    def sim(self, operation: Optional[Tuple[int, int, int]]) -> Dict[str, Any]:
        self.command_queue.append((self.t, operation))
        self.visit_count[self.rack_pos] += 1
        if operation is None:
            # no rack operation
            # action staying at the position is not chosen unless all facilities are empty
            # it is free to set its operation time to be small
            operation_time = 1.
        else:
            floor, pos, low_up, load_unload, low_lr, up_lr = operation #(pos, low, up)   (0,1,-1   ,0,1,-1,   )

            # operation : move to the desired position -> load or unload
            # travel_t + loading/unloading_t 0~6 7~14 15~17   4 8 4 0
            operation_time = int(abs(pos-self.rack_pos))*0.5 + 2. + 3. #noise + max(-1,np.random.normal(0.,0.))

            self.rack_pos = pos

            if low_up == 0:
                if load_unload == 0 and low_lr ==0:
                    self.load_lower()
                    self.rack.lower_fork.CMD_RACK = self.t + operation_time
                    self.waiting_time_log.append(self.t + operation_time - self.rack.lower_fork.cmd_time)
                    if self.rack.is_upper_loaded:
                        self.load_sequential += 1

                ###
                elif load_unload == 0 and low_lr == 1:
                    self.load_lower_()
                    self.rack.lower_fork_.CMD_RACK = self.t + operation_time
                    self.waiting_time_log.append(self.t + operation_time - self.rack.lower_fork_.cmd_time)
                    if self.rack.is_upper_loaded:
                        self.load_sequential += 1
                ###
                    #if self.t + operation_time - self.rack.lower_fork.cmd_time>330.:
                        #print(operation_time,self.t + operation_time- self.rack.lower_fork.cmd_time)
                elif load_unload == 1 and low_lr ==0:
                    assert self.rack.destination[0] == self.pos2floor[self.rack_pos]
                    self.num_carried += 1
                    released = self.rack.release_lower_fork()
                    self.flow_time_log.append(self.t + operation_time - released.cmd_time)

                elif load_unload == 1 and low_lr ==1:
                    assert self.rack.destination_[0] == self.pos2floor[self.rack_pos]
                    self.num_carried += 1
                    released = self.rack.release_lower_fork_()
                    self.flow_time_log.append(self.t + operation_time - released.cmd_time)



            elif low_up == 1:
                if load_unload == 0 and up_lr==0:
                    self.load_upper()
                    self.rack.upper_fork.CMD_RACK=self.t + operation_time
                    self.waiting_time_log.append(self.t + operation_time - self.rack.upper_fork.cmd_time)
                    if self.rack.is_lower_loaded:
                        self.load_sequential += 1

                elif load_unload == 0 and up_lr == 1:
                    self.load_upper_()
                    self.rack.upper_fork_.CMD_RACK = self.t + operation_time
                    self.waiting_time_log.append(self.t + operation_time - self.rack.upper_fork_.cmd_time)
                    if self.rack.is_lower_loaded:
                        self.load_sequential += 1

                elif load_unload == 1 and up_lr==0:
                    assert self.rack.destination[1] == self.pos2floor[self.rack_pos]
                    self.num_carried += 1
                    released = self.rack.release_upper_fork()
                    self.flow_time_log.append(self.t + operation_time - released.cmd_time)

                elif load_unload == 1 and up_lr==1:
                    assert self.rack.destination_[1] == self.pos2floor[self.rack_pos]
                    self.num_carried += 1
                    released = self.rack.release_upper_fork_()
                    self.flow_time_log.append(self.t + operation_time - released.cmd_time)


            elif low_up == 2 and low_lr==0 and up_lr ==0:
                if load_unload == 0:
                    self.load_lower(), self.load_upper()
                    self.rack.lower_fork.CMD_RACK=self.t + operation_time
                    self.rack.upper_fork.CMD_RACK=self.t + operation_time
                    self.waiting_time_log.append(self.t + operation_time - self.rack.lower_fork.cmd_time)
                    self.waiting_time_log.append(self.t + operation_time - self.rack.upper_fork.cmd_time)

                    self.load_two += 1
                elif load_unload == 1:
                    assert self.rack.destination[0] == self.pos2floor[self.rack_pos]
                    assert self.rack.destination[1] == self.pos2floor[self.rack_pos]
                    released1, released2 = self.rack.release_lower_fork(), self.rack.release_upper_fork()
                    self.num_carried += 2
                    self.unload_two += 1
                    self.flow_time_log.append(self.t + operation_time - released1.cmd_time)
                    self.flow_time_log.append(self.t + operation_time - released2.cmd_time)
            elif low_up == 2 and low_lr==1 and up_lr ==0:
                if load_unload == 0:
                    self.load_lower_(), self.load_upper()
                    self.rack.lower_fork_.CMD_RACK=self.t + operation_time
                    self.rack.upper_fork.CMD_RACK=self.t + operation_time
                    self.waiting_time_log.append(self.t + operation_time - self.rack.lower_fork_.cmd_time)
                    self.waiting_time_log.append(self.t + operation_time - self.rack.upper_fork.cmd_time)

                    self.load_two += 1
                elif load_unload == 1:
                    assert self.rack.destination_[0] == self.pos2floor[self.rack_pos]
                    assert self.rack.destination[1] == self.pos2floor[self.rack_pos]
                    released1, released2 = self.rack.release_lower_fork_(), self.rack.release_upper_fork()
                    self.num_carried += 2
                    self.unload_two += 1
                    self.flow_time_log.append(self.t + operation_time - released1.cmd_time)
                    self.flow_time_log.append(self.t + operation_time - released2.cmd_time)
            elif low_up == 2 and low_lr==0 and up_lr ==1:
                if load_unload == 0:
                    self.load_lower(), self.load_upper_()
                    self.rack.lower_fork.CMD_RACK=self.t + operation_time
                    self.rack.upper_fork_.CMD_RACK=self.t + operation_time
                    self.waiting_time_log.append(self.t + operation_time - self.rack.lower_fork.cmd_time)
                    self.waiting_time_log.append(self.t + operation_time - self.rack.upper_fork_.cmd_time)

                    self.load_two += 1
                elif load_unload == 1:
                    assert self.rack.destination[0] == self.pos2floor[self.rack_pos]
                    assert self.rack.destination_[1] == self.pos2floor[self.rack_pos]
                    released1, released2 = self.rack.release_lower_fork(), self.rack.release_upper_fork_()
                    self.num_carried += 2
                    self.unload_two += 1
                    self.flow_time_log.append(self.t + operation_time - released1.cmd_time)
                    self.flow_time_log.append(self.t + operation_time - released2.cmd_time)
            elif low_up == 2 and low_lr==1 and up_lr ==1:
                if load_unload == 0:
                    self.load_lower_(), self.load_upper_()
                    self.rack.lower_fork_.CMD_RACK=self.t + operation_time
                    self.rack.upper_fork_.CMD_RACK=self.t + operation_time
                    self.waiting_time_log.append(self.t + operation_time - self.rack.lower_fork_.cmd_time)
                    self.waiting_time_log.append(self.t + operation_time - self.rack.upper_fork_.cmd_time)

                    self.load_two += 1
                elif load_unload == 1:
                    assert self.rack.destination_[0] == self.pos2floor[self.rack_pos]
                    assert self.rack.destination_[1] == self.pos2floor[self.rack_pos]
                    released1, released2 = self.rack.release_lower_fork_(), self.rack.release_upper_fork_()
                    self.num_carried += 2
                    self.unload_two += 1
                    self.flow_time_log.append(self.t + operation_time - released1.cmd_time)
                    self.flow_time_log.append(self.t + operation_time - released2.cmd_time)

        # simulation of lots arrival
        # performed by reading the simulation data

        done, flag = self.sim_arrival(dt=operation_time)
        info = {
                'lost' : flag,
                'dt': operation_time / self.t_unit,
                'elapsed_time': self.elapsed_time / self.t_unit,
                }
        return info




    def sim_arrival(self, dt: float):
        # conveyor(456) : (floor, layer) : (2,  123)

        # read data for simulation
        assert dt > 0.
        begin = self.end
        next_t = self.t + dt
        while self.end < self.num_data:
            if self.t < self.data_cmd[self.end] <= next_t:
                self.end += 1
            else:
                break
        self.num_added = self.end - begin
        flag=0


        for layer in self.layers:
            if self.layers[layer].QUEUE_LEN > 0:
                wafers=[]
                for wafer in self.layers[layer].QUEUE:
                    if next_t - wafer.cmd_time > 300.:
                        wafers.append(wafer)
                        self.miss +=1
                        flag -= 1
                for wafer in wafers:
                    self.layers[layer].QUEUE.remove(wafer)
                    self.layers[layer].QUEUE_LEN -= 1

        for i in range(begin, self.end):
            # wafer generation from data
            wafer = Wafer(cmd_t=self.data_cmd[i], origin=self.data_from[i], destination=self.data_to[i])
            self.tot+=1

            # arrived lots are randomly distributed into several layers
            coin = np.random.rand()

            #Finding available layer
            candidate=[1,2,3]
            available=[]
            for e in candidate:
                if self.layers[3 * (self.data_from[i] - 1) + e].available() == True:
                    available.append(e)

            #Randomly select a layer
            if len(available) == 0:
                self.missing+=1
                flag -= 1
            else:
                for j, e in enumerate(available):
                    if coin <= (j+1)/len(available):
                        a = self.layers[3 * (self.data_from[i] - 1) + e].push(wafer)
                        self.total_amount[3 * (self.data_from[i] -1) + e] += 1
                        break


        if self.end == self.num_data:
            done = True
        else:
            done = False
        self.t = next_t
        return done, flag


    def render(self):
        # if self.t < 1e-5:
        #      self.socket = launch_server()
        #
        # message = '11111111'
        # hex_message = message.encode("utf-8").hex()
        # hex_message = "02" + hex_message[0:] + hex_message[:0] + "03"
        # binary = binascii.a2b_hex(hex_message)
        # self.socket.send(binary)

        #data = self.socket.recv(1024)
        #print(data)
        # while True:
        #     # 클라이언트가 보낸 메시지를 수신하기 위
        #     # 해 대기합니다.
        #     data = self.socket.recv(1024)
        #     print(data)
        #     # 수신받은 문자열을 출력합니다.
        #     print(data.decode())
        #
        #     # 빈 문자열을 수신하면 루프를 중지합니다.
        #     if data:
        #         break

        #b'\x00\x0e

        if self.quit:
            return
        capacities = [0, 3, 2, 0, 4, 2, 6, 3, 2]
        conv_pos = [660, 600, 540, 450, 390, 330, 240, 180, 120]
        rm_pos = [540, 480, 420, 390, 330, 270, 210, 180, 120, 60]
        # TODO
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(self.size)
            self.clock = pygame.time.Clock()

        e = pygame.event.get()
        for ev in e:
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                self.pause = True
                break
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_UP:
                self.framerate *= 2.
                self.framerate = max(1, self.framerate)
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_DOWN:
                self.framerate *= 0.5
                self.framerate = max(1, self.framerate)
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                pygame.quit()
                self.screen = None
                self.quit = True
                return

        while self.pause:
            e = pygame.event.get()
            flag = False
            for ev in e:
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                    self.pause = False
                    flag = True
                    break
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_RIGHT:
                    flag = True
                    break
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    pygame.quit()
                    self.screen = None
                    self.quit = True
                    return
            if flag:
                break

        self.clock.tick(self.framerate)
        self.screen.fill(self.WHITE)
        sysfont = pygame.font.SysFont(name='', size=50)

        text = sysfont.render("Rack Master Monitoring Center", True, self.BLACK)
        self.screen.blit(text, (400, 10))
        # define keyboard inputs
        sysfont = pygame.font.SysFont(name='', size=30)

        text = sysfont.render("[Escape] : Quit Visualization", True, self.BLACK)
        self.screen.blit(text, (260, 740))

        text = sysfont.render("[Up Arrow] : Speed x 2", True, self.BLACK)
        self.screen.blit(text, (600, 740))
        text = sysfont.render("[Down Arrow] : Speed  x 0.5", True, self.BLACK)
        self.screen.blit(text, (600, 770))

        text = sysfont.render("[Space] : Pause/Resume", True, self.BLACK)
        self.screen.blit(text, (940, 740))
        text = sysfont.render("[Right Arrow] : 1 Step ", True, self.BLACK)
        self.screen.blit(text, (940, 770))

        sysfont = pygame.font.SysFont(name='', size=24)
        # -------------------------------------- conveyor belts rendering -----------------------------------
        for i, pos in enumerate(conv_pos):
            floor = 2 if (i < 3) else 6 if (i >= 6) else 3
            layer = i % 3 + 1
            # pygame.draw.line(self.screen, self.BLACK, [100, pos], [480, pos], 1)
            pygame.draw.rect(self.screen, (128, 128, 128), [100, pos, 380, 10])
            # pygame.draw.line(self.screen, self.BLACK, [100, pos], [100 + 30 * capacities[i], pos], 3)
            pygame.draw.rect(self.screen, (228, 192, 168), [100, pos, 30 * capacities[i], 15])

            text = sysfont.render("{}F L{}".format(floor, layer), True, self.BLACK)
            self.screen.blit(text, (440, pos - 20))

        waiting_quantities = {}

        for label in self.labels:
            waiting_quantities[label] = self.layers[label].QUEUE_LEN

        for label in self.labels:
            for i in range(waiting_quantities[label]):
                pygame.draw.rect(self.screen, (168, 138, 100), [100 + 30 * i, conv_pos[label - 1] - 30, 30, 30])
                pygame.draw.rect(self.screen, self.BLACK, [100 + 30 * i, conv_pos[label - 1] - 30, 30, 30], 1)

        waiting_destinations = {label: [] for label in self.labels}
        for label in self.labels:
            for lot in self.layers[label].QUEUE:
                waiting_destinations[label].append(lot.destination)

        for label in self.labels:
            for i in range(waiting_quantities[label]):
                text = sysfont.render('{}F'.format(waiting_destinations[label][i]), True, self.BLACK)
                self.screen.blit(text, (105 + 30 * i, conv_pos[label - 1] - 20))
        # ---------------------------------------------------------------------------------------------------

        # --------------------------------------- rack master rendering -------------------------------------

        pygame.draw.rect(self.screen, (128, 128, 128), [36, 50, 8, 640])
        pos = rm_pos[self.rack_pos]
        pygame.draw.rect(self.screen, self.WHITE, [10, pos, 60, 120])
        pygame.draw.rect(self.screen, self.BLACK, [10, pos, 60, 120], 3)
        pygame.draw.line(self.screen, self.BLACK, [10, pos + 60], [69, pos + 60], 3)

        # lower fork rendering
        lower_d, upper_d = self.rack_destination
        if lower_d > 0:
            pygame.draw.rect(self.screen, (168, 138, 100), [25, pos + 88.5, 30, 30])
            pygame.draw.rect(self.screen, self.BLACK, [25, pos + 88.5, 30, 30], 1)
            text = sysfont.render('{}F'.format(lower_d), True, self.BLACK)
            self.screen.blit(text, (30, pos + 100))
        # upper fork rendering
        if upper_d > 0:
            pygame.draw.rect(self.screen, (168, 138, 100), [25, pos + 29, 30, 30])
            pygame.draw.rect(self.screen, self.BLACK, [25, pos + 29, 30, 30], 1)
            text = sysfont.render('{}F'.format(upper_d), True, self.BLACK)
            self.screen.blit(text, (30, pos + 40))
        # ---------------------------------------------------------------------------------------------------
        # ------------------------------------------- operation log -----------------------------------------
        # <1> elapsed time
        h = int(self.t // 3600.)
        m = int(self.t % 3600.) // 60
        s = self.t % 60.

        loc = 80

        loc += 20
        text = sysfont.render("Elapsed Time : {:d}h {}m {:.2f}s".format(h, m, s), True, self.BLACK)
        self.screen.blit(text, (700, loc))
        # <2> carried quantity
        num_total = sum(self.total_amount)
        ratio = 100. * self.num_carried / num_total if num_total > 0 else np.NaN

        loc += 20
        text = sysfont.render("Carried : {} ({:.2f}%)".format(self.num_carried, ratio), True, self.BLACK)
        self.screen.blit(text, (700, loc))

        # <3> total quantity
        loc += 40
        text = sysfont.render("Total (POD): {} ({})".format(num_total,
                                                            self.total_amount[1] + self.total_amount[4]),
                              True,
                              self.BLACK
                              )
        self.screen.blit(text, (700, loc))

        for i, label in enumerate(self.labels):
            loc += 20
            text = sysfont.render("-- {} : {}".format(self.label2str[label], self.total_amount[i]), True, self.BLACK)
            self.screen.blit(text, (700, loc))

        loc += 40
        text = sysfont.render("Load_two : {}".format(self.load_two), True, self.BLACK)
        self.screen.blit(text, (700, loc))

        loc += 20
        text = sysfont.render("Unload_two : {}".format(self.unload_two), True, self.BLACK)
        self.screen.blit(text, (700, loc))

        loc += 40
        text = sysfont.render("Time Statistics", True, self.BLACK)
        self.screen.blit(text, (700, loc))

        loc += 20
        t = np.mean(self.flow_time_log) if len(self.flow_time_log) > 0 else np.nan
        text = sysfont.render("-- Average Waiting Time : {:.2f}s".format(t), True, self.BLACK)
        self.screen.blit(text, (700, loc))

        loc += 20
        t = np.max(self.flow_time_log) if len(self.flow_time_log) > 0 else -np.inf
        text = sysfont.render("-- Max Waiting Time : {:.2f}s".format(t), True, self.BLACK)
        self.screen.blit(text, (700, loc))

        loc += 20
        t = np.mean(self.flow_time_log) if len(self.flow_time_log) > 0 else np.nan
        text = sysfont.render("-- Average Flow Time : {:.2f}s".format(t), True, self.BLACK)
        self.screen.blit(text, (700, loc))

        loc += 20
        t = np.max(self.flow_time_log) if len(self.flow_time_log) > 0 else -np.inf
        text = sysfont.render("-- Max Flow Time : {:.2f}s".format(t), True, self.BLACK)
        self.screen.blit(text, (700, loc))
        # ---------------------------------------------------------------------------------------------------
        # ----------------------------------------- command history -----------------------------------------
        loc += 40
        text = sysfont.render("Operation History".format(t), True, self.BLACK)
        self.screen.blit(text, (700, loc))

        for t, operation in list(self.command_queue):
            loc += 20
            description = operation2str[operation]
            h = int(t // 3600.)
            m = int(t % 3600.) // 60
            s = t % 60.
            text = sysfont.render('({:d}h {}m {:.2f}s) {}'.format(h, m, s, description), True, self.BLACK)
            self.screen.blit(text, (700, loc))
        # ---------------------------------------------------------------------------------------------------
        pygame.display.update()
        return

    def close(self):
        pygame.quit()
        self.screen = None

    def load_lower(self):
        target_label = self.pos2label[self.rack_pos][0]
        assert target_label is not None
        target_conveyor = self.layers[target_label]
        assert not target_conveyor.is_empty
        self.rack.load_lower(target_conveyor.pop())
        return

    def load_upper(self):
        target_label = self.pos2label[self.rack_pos][1]
        assert target_label is not None
        target_conveyor = self.layers[target_label]
        assert not target_conveyor.is_empty
        self.rack.load_upper(target_conveyor.pop())
        return
    ###
    def load_lower_(self):
        target_label = self.pos2label[self.rack_pos][0]
        assert target_label is not None
        target_conveyor = self.layers[target_label]
        assert not target_conveyor.is_empty
        self.rack.load_lower_(target_conveyor.pop())
        return

    def load_upper_(self):
        target_label = self.pos2label[self.rack_pos][1]
        assert target_label is not None
        target_conveyor = self.layers[target_label]
        assert not target_conveyor.is_empty
        self.rack.load_upper_(target_conveyor.pop())
        return
    ###
    @property
    def operation_log(self):
        self.handle_remaining_lots()
        info = {
            'tot' : self.tot,
            'miss': self.miss,
            'missing' : self.missing,
            'carried': self.num_carried,
            'waiting_quantity': self.waiting_quantity,
            'visit_count': self.visit_count,
            'load_two': self.load_two,
            'unload_two': self.unload_two,
            'load_sequential': self.load_sequential,
            'total': self.total_amount,
            'average_waiting_time': np.mean(self.waiting_time_log),
            'max_waiting_time': np.max(self.waiting_time_log),
            'average_flow_time': np.mean(self.flow_time_log),
            'max_flow_time': np.max(self.flow_time_log)
        }
        return info

    def handle_remaining_lots(self):
        for conveyor in self.layers.values():
            for wafer in conveyor.QUEUE:
                self.waiting_time_log.append(self.t - wafer.cmd_time)

    @property
    def waiting_time(self):
        wt = np.zeros(self.num_layers)
        for i, conveyors in enumerate(self.layers.values()):
            wt[i] = 0. if conveyors.is_empty else self.t - conveyors.cmd_time
        return wt

    @property
    def destination(self) -> List[int]:
        return [conveyor.destination for conveyor in self.layers.values()]

    @property
    def rack_destination(self) -> Tuple[int, int]:
        return self.rack.destination

    @property
    def rack_destination_(self) -> Tuple[int, int]:
        return self.rack.destination_

    @property
    def is_pod_loaded(self):
        return self.rack.is_pod_loaded

    @property
    def travel_distance(self):
        # TODO : what if the queue is empty?
        return np.array([self.layers[i].destination - self.label2floor[i] for i in self.layers]) / 4.

    @property
    def elapsed_time(self):
        return self.t

    @property
    def num_lots(self):
        return [(label, conveyor.QUEUE_LEN) for label, conveyor in self.layers.items()]

    @property
    def waiting_quantity(self) -> List[int]:
        return [conveyor.QUEUE_LEN for conveyor in self.layers.values()]

