"""reward function"""

from collections import defaultdict
from controller import Node
from controller import ContactPoint

def collisions(robot_nodes, block_nodes, ilegal_contacts, supervisor):
    contact_boundary = ['r_gripper_right', 'r_gripper_left' , 'bot_red', 'bot_blue', 'b_gripper_right', 'b_gripper_left']
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

def done_check(time, robot_nodes, block_nodes):
    done = True
    if time >= 10:
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

def current_grid_score(block_nodes):
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
    
    return score


def robot_bonus(grid_Score, robot_nodes):
    score =0
    if grid_Score >= 40:
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

def step_reward(robot_nodes, block_nodes, ilegal_contacts, prev_grid_score, time, supervisor):
    penalties = 0
    bonus = 0
    grid_score = 0
    current_collisions = collisions(robot_nodes, block_nodes, ilegal_contacts, supervisor)
    for collision in current_collisions:
        penalties += ilegal_contacts[collision]
    curr_grid_score = current_grid_score(block_nodes)
    if done_check(time, robot_nodes, block_nodes):
        bonus = robot_bonus(curr_grid_score, robot_nodes)
    grid_score = curr_grid_score - prev_grid_score

    return (penalties + grid_score + bonus, curr_grid_score)