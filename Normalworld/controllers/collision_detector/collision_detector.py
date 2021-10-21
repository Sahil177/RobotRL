"""collision_detector controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from collections import defaultdict
from controller import Robot
from controller import Supervisor
from controller import Node
from controller import ContactPoint

# create the Robot instance.
supervisor = Supervisor()

# get the time step of the current world.
timestep = 64

red_node = supervisor.getFromDef('bot_red')
#red_block = supervisor.getFromDef('Block_R1')

contact_boundary = ['r_gripper_right', 'r_gripper_left' , 'bot_red', 'bot_blue', 'b_gripper_right', 'b_gripper_left']


block_nodes = []
for i in range(4):
	block_node = supervisor.getFromDef('Block_R' + str(int(i+1)))
	block_nodes.append(block_node)
for i in range(4):
	block_node = supervisor.getFromDef('Block_B' + str(int(i+1)))
	block_nodes.append(block_node)

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
    
    collisions = [val for key, val in collision_dict.items()]


    return collisions

    




    

    


# Main loop:
# - perform simulation steps until Webots is stopping the controller
while supervisor.step(timestep) != -1:
    
    '''contacts = red_node.getContactPoints(includeDescendants=True)
    print([contact.get_node_id() for contact in contacts])
    print([contact.point for contact in contacts])
    print([contact.get_point() for contact in contacts])
    contacts_id = [supervisor.getFromId(contact.node_id) for contact in contacts]
    contacts_name = [(contact.getDef(), contact.getPosition()) for contact in contacts_id]
    print(contacts_name[0])'''

    print(collisions())

    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)

# Enter here exit cleanup code.