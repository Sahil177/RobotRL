"""Evaluator controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Supervisor
from reward_function import *

supervisor = Supervisor()

# get the time step of the current world.
timestep = 64

ilegal_contacts = {('bot_red', 'bot_blue'):  -20}

block_nodes = []
for i in range(4):
    block_node = supervisor.getFromDef('Block_R' + str(int(i+1)))
    block_nodes.append(block_node)
    ilegal_contacts[('bot_blue','Block_R' + str(int(i+1)))] = -5
for i in range(4):
    block_node = supervisor.getFromDef('Block_B' + str(int(i+1)))
    block_nodes.append(block_node)
    ilegal_contacts[('bot_red','Block_B' + str(int(i+1)))] = -5

robot_nodes = [supervisor.getFromDef('bot_red'), supervisor.getFromDef('bot_blue')]


prev_grid_score = 0 
total_reward = 0      
i =0
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while supervisor.step(timestep) != -1:
    i +=1
    time = i*timestep/1000
    current_reward, curr_grid_score = step_reward(robot_nodes, block_nodes, ilegal_contacts, prev_grid_score, time, supervisor)
    total_reward += current_reward
    current_collisions = collisions(robot_nodes, block_nodes, ilegal_contacts, supervisor)
    print(f"reward: {current_reward}, total_reward: {total_reward}, collisions: {current_collisions}")
    prev_grid_score = curr_grid_score
    if done(time, robot_nodes, block_nodes):
        supervisor.simulationSetMode(0)

# Enter here exit cleanup code.
