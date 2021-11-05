import sys

from numpy.lib.function_base import append
packages = [r'C:\\Users\\sahil\\Documents\\Python_environments\\env37', r'C:\\Users\\sahil\\Documents\\Python_environments\\env37\\lib\\site-packages']
for path in packages:
    sys.path.append(path)
import numpy as np
from utils import *
from reward_function import *
from skimage.draw import line, rectangle, rectangle_perimeter, ellipse, polygon
from skimage.feature import blob_dog
from skimage.filters import gaussian
from deepbots.supervisor.controllers.supervisor_emitter_receiver import SupervisorCSV
from pprint import pprint

from controller import Emitter
from controller import Receiver
####Import RL agent 


class RLSupervisor(SupervisorCSV):
    def __init__(self):
        super().__init__()
        #self.observationSpace = 
        #self.actionSpace = 
        self.timestep = 64
        self.robot_nodes = [self.supervisor.getFromDef('bot_red'), self.supervisor.getFromDef('bot_blue')]
        self.bot_red = self.supervisor.getFromDef('bot_red')
        self.bot_blue = self.supervisor.getFromDef('bot_blue')
        self.red_state = [self.bot_red.getPosition()[0], self.bot_red.getPosition()[2], np.pi, 0, 0] # [ x, z, theta, colour, gripper_state]
        self.blue_state = [self.bot_blue.getPosition()[0], self.bot_blue.getPosition()[2], 0, 0, 0] # [ x, z, theta, colour, gripper_state]
        self.occupancy_grid = np.full((80,80), 0.5)
        self._extent = np.array([[6, -4],[-7, -4],[-7, 4],[6, 3]])
        self.episodeCount = 0
        self.episodeLimit = 10000  # Max number of episodes allowed
        self.stepsPerEpisode = 200  # Max number of steps per episode
        self.episodeScore = 0  # Score accumulated during an episode
        self.episodeScoreList = []  # A list to save all the episode scores, used to check if task is solved
        self.emitter.setChannel(Emitter.CHANNEL_BROADCAST)
        self.receiver.setChannel(Receiver.CHANNEL_BROADCAST)
        self.grid_reset = self.supervisor.getFromDef('grid_reset')

        #reward function attributes
        self.ilegal_contacts = {('bot_red', 'bot_blue'):  -20}
        self.block_nodes = []
        for i in range(4):
            block_node = self.supervisor.getFromDef('Block_R' + str(int(i+1)))
            self.block_nodes.append(block_node)
            self.ilegal_contacts[('bot_blue','Block_R' + str(int(i+1)))] = -5
        for i in range(4):
            block_node = self.supervisor.getFromDef('Block_B' + str(int(i+1)))
            self.block_nodes.append(block_node)
            self.ilegal_contacts[('bot_red','Block_B' + str(int(i+1)))] = -5
        self.prev_grid_score = 0
        self.time = 0
        print

    def await_state_data(self):
        state_update = False
        robot_data = []
        while True:
            message=self.handle_receiver() 
            if message == None:
                break
            message2 = []
            for i in range(len(message)):
                if i == 0 or i == 6 or i == 7:
                    message2.append(int(message[i]))
                else:
                    message2.append(float(message[i]))
            robot_data.append(message2)
            state_update = True

        if len(robot_data) > 1:
            if robot_data[0][0] == 1:
                temp = robot_data[0]
                robot_data[0] = robot_data[1]
                robot_data[1] = temp
        return robot_data, state_update

    def update_binary_occupancy_grid(self, robot_data):
        max_range = 1.3
        for i in [0,1]:
            test_position = robot_data[i][1:4]
            psValues = robot_data[i][4:6]
        
            local_coordinate1 = [-psValues[0]*0.707,psValues[0]*0.707]
            local_coordinate2 = [psValues[1]*0.707,psValues[1]*0.707]

            coord1 = np.rint((transform_local_coords(test_position, local_coordinate1)*[100, -100] + np.array([119, 119]))/3)
            coord2 = np.rint((transform_local_coords(test_position, local_coordinate2)*[100, -100] + np.array([119, 119]))/3)

            pos = convert_to_grid_space(robot_data[(i+1)%2][1:3])
            rad = robot_data[(i+1)%2][3]
            grid_robot_state = np.array([pos[0], pos[1], rad])
            robot_extent_global = np.array([[0,0],[0,0],[0,0],[0,0]])
            for i in range(len(self._extent)):
                robot_extent_global[i] = np.rint((transform_local_coords(grid_robot_state, self._extent[i])))
            robot_extent_global = np.transpose(robot_extent_global)
            rr, cc = polygon(robot_extent_global[0], robot_extent_global[1])
            check_coords = np.array([rr, cc]).T

            if np.array([bound(int(coord1[1]),bound(int(coord1[0])))]) not in check_coords and psValues[0] < max_range:
                self.occupancy_grid[bound(int(coord1[1]))][bound(int(coord1[0]))] = update_occupied(self.occupancy_grid, [bound(int(coord1[1])),bound(int(coord1[0]))], p=(0.95 - 0.14*psValues[0]))
            if np.array([bound(int(coord2[1]),bound(int(coord2[0])))]) not in check_coords and psValues[1] < max_range:
                self.occupancy_grid[bound(int(coord2[1]))][bound(int(coord2[0]))] = update_occupied(self.occupancy_grid, [bound(int(coord2[1])),bound(int(coord2[0]))], p=(0.95 - 0.14*psValues[1]))

            current_pos = np.rint((np.array([test_position[0], test_position[1]])*[100, -100] + np.array([119, 119]))/3)

            rr1, cc1, updates1 = update_free(self.occupancy_grid, [bound(int(coord1[1])),bound(int(coord1[0]))], [bound(int(current_pos[1])), bound(int(current_pos[0]))], p=0.4)
            self.occupancy_grid[rr1, cc1] = updates1

            rr2, cc2, updates2 = update_free(self.occupancy_grid, [bound(int(coord2[1])),bound(int(coord2[0]))], [bound(int(current_pos[1])), bound(int(current_pos[0]))], p=0.4)
            self.occupancy_grid[rr2, cc2] = updates2

    def get_observations(self):
        robot_data, update = self.await_state_data()
        if update:
            self.update_binary_occupancy_grid(robot_data)
            for data in robot_data:
                if data[0] == 0:
                    self.red_state = [data[1], data[2], data[3], data[6], data[7]]
                else:
                    self.blue_state = [data[1], data[2], data[3] ,data[6], data[7]]

        return [self.red_state, self.blue_state, self.occupancy_grid]            
        
    def get_reward(self, action):
        self.time += self.get_timestep()/1000
        current_reward, curr_grid_score = step_reward(self.robot_nodes, self.block_nodes, self.ilegal_contacts, self.prev_grid_score, self.time, self.supervisor)
        self.prev_grid_score = curr_grid_score
        return current_reward
    
    def is_done(self):
        complete = done_check(self.time, self.robot_nodes, self.block_nodes)
        return done_check(self.time, self.robot_nodes, self.block_nodes)

    def reset(self):
        #self.supervisor.simulationReset()
        self.supervisor.simulationResetPhysics()
        self.grid_reset.restartController()
        self.red_state = [self.bot_red.getPosition()[0], self.bot_red.getPosition()[2], np.pi, 0, 0] # [ x, z, theta, colour, gripper_state]
        self.blue_state = [self.bot_blue.getPosition()[0], self.bot_blue.getPosition()[2], 0, 0, 0] # [ x, z, theta, colour, gripper_state]
        self.occupancy_grid = np.full((80,80), 0.5)
        self.prev_grid_score = 0
        self.time = 0
        for node in self.robot_nodes:
            node.restartController()
        
        return [self.red_state, self.blue_state, self.occupancy_grid]
        
    
    def get_info(self):
        return None

    def solved(self):
        return None

print("beginning")
supervisor = RLSupervisor()
#pprint(vars(supervisor))

def agent():
    return [5+np.random.uniform(-10,5), 5+np.random.uniform(-10,5), 5+np.random.choice([0,1], p = [0.8, 0.2]), 5+np.random.uniform(-10,5), np.random.uniform(-10,5), np.random.choice([0,1], p = [0.8, 0.2])]


while supervisor.episodeCount < supervisor.episodeLimit:
    visualiser1 = visualiser(1, ["Occupancy Grid"], k=1)
    observation = supervisor.reset()
    supervisor.episodeScore = 0
    while True:
        action = agent()
        newObservation, reward, done, info = supervisor.step(action)
        #print(newObservation)

        if done:
            print(supervisor.episodeScore)
            supervisor.episodeScoreList.append(supervisor.episodeScore)
            break 
            
        supervisor.episodeScore += reward
        observation = newObservation

        visualiser1(supervisor.occupancy_grid)
    
    plt.close()

