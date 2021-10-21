"""collision_detector controller."""

from collections import defaultdict
from controller import Robot
from controller import Supervisor
from controller import Node
from controller import ContactPoint

supervisor = Supervisor()

# get the time step of the current world.
timestep = 64

contact_boundary = ['r_gripper_right', 'r_gripper_left' , 'bot_red', 'bot_blue', 'b_gripper_right', 'b_gripper_left']

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

def collisions():
    colliding = []
    for node in robot_nodes:
        contacts = node.getContactPoints(includeDescendants=True)
        contact_locations = [contact.get_point() for contact in contacts]

        contacts_id = [supervisor.getFromId(contact.node_id) for contact in contacts]
        contact_names = [contact.getDef() for contact in contacts_id]
        #print(contacts_name)

        for i in range(len(contact_names)):
            if contact_names[i] in contact_boundary:
                colliding.append((node.getDef(), contact_locations[i]))
    
    for node in block_nodes:
        contacts = node.getContactPoints()
        for contact in contacts:
            if contact.get_point()[1] > 0:
                colliding.append((supervisor.getFromId(contact.node_id).getDef(), contact.get_point()))

    collision_dict = defaultdict(list)
    for obj, point in colliding:
        collision_dict[tuple(point)].append(obj)
    
    collisions = [tuple(val) for key, val in collision_dict.items()]
    ilegal_collisions = [collision for collision in collisions if collision in ilegal_contacts]


    return set(ilegal_collisions)

def done(time):
    done = True
    if time >= 300:
        return done
    for block in block_nodes:
        if block.getDef()[6] == 'R':
            pos = block.getPosition()
            if not 0.8 < pos[0] < 1.2 or not 0.8<pos[2] < 1.2:
                done = False
                return done
        else:
            pos = block.getPosition()
            if not 0.8 < pos[0] < 1.2 or not -1.2<pos[2] < -0.8:
                done = False
                return done
    
    for robot in robot_nodes:
        if robot.getDef()[4] == 'r':
            pos = robot.getPosition()
            if not 0.8 < pos[0] < 1.2 or not 0.8<pos[2] < 1.2:
                done = False
                return done
        else:
            pos = robot.getPosition()
            if not 0.8 < pos[0] < 1.2 or not -1.2<pos[2] < -0.8:
                done = False
                return done
    
    return done

def current_grid_score():
    score = 0
    for block in block_nodes:
        if block.getDef()[6] == 'R':
            pos = block.getPosition()
            if 0.8 < pos[0] < 1.2 and 0.8<pos[2] < 1.2:
                score += 20
        else:
            pos = block.getPosition()
            if  0.8 < pos[0] < 1.2 and -1.2<pos[2] < -0.8:
                score +=20 
    
    if score >= 40:
        for robot in robot_nodes:
            if robot.getDef()[4] == 'r':
                pos = robot.getPosition()
                if 0.8 < pos[0] < 1.2 and 0.8<pos[2] < 1.2:
                    score += 20
            else:
                pos = robot.getPosition()
                if 0.8 < pos[0] < 1.2 and -1.2<pos[2] < -0.8:
                    score += 20
    
    return score

        
score = 0
i =0
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while supervisor.step(timestep) != -1:
    i +=1
    time = i*timestep/1000
    current_collisions = collisions()
    print(f"score: {score}, collisions: {current_collisions}")
    for collision in current_collisions:
        score += ilegal_contacts[collision]
    if done(time):
        final_score = score + current_grid_score()
        print(f"Total score: {final_score}, Time: {time}")
        supervisor.simulationSetMode(0)

# Enter here exit cleanup code.