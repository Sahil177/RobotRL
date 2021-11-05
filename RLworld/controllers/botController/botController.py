"""botController controller."""

import sys
packages = [r'C:\\Users\\sahil\\Documents\\Python_environments\\env37', r'C:\\Users\\sahil\\Documents\\Python_environments\\env37\\lib\\site-packages']
for path in packages:
    sys.path.append(path)

import numpy as np
from deepbots.robots.controllers.robot_emitter_receiver_csv import RobotEmitterReceiverCSV

class bot(RobotEmitterReceiverCSV):
    def __init__(self):
        super().__init__()
        self.main_gps = self.robot.getDevice("gps_main")
        self.main_gps.enable(self.get_timestep())
        self.compass  = self.robot.getDevice("compass")
        self.compass.enable(self.get_timestep())
        self.ds_left = self.robot.getDevice("ds_left")
        self.ds_left.enable(self.get_timestep())
        self.ds_right = self.robot.getDevice("ds_right")
        self.ds_right.enable(self.get_timestep())
        self.colour_sensor = self.robot.getDevice("camera")
        self.colour_sensor.enable(self.get_timestep())
        self.emitter = self.robot.getDevice("emitter")
        self.receiver = self.robot.getDevice("receiver")
        self.receiver.enable(self.get_timestep())
        self.wheel1 = self.robot.getDevice("wheel1")
        self.wheel1.setPosition(float('inf'))
        self.wheel1.setVelocity(0.0)
        self.wheel2 = self.robot.getDevice("wheel2")
        self.wheel2.setPosition(float('inf'))
        self.wheel2.setVelocity(0.0)
        self.gripper_left_motor = self.robot.getDevice("Lmotor")
        self.gripper_right_motor = self.robot.getDevice("Rmotor")
        self.gripper_left_motor.setAvailableTorque(0.2)
        self.gripper_right_motor.setAvailableTorque(0.2)
        self.gripper_left_motor.setPosition(0.9)
        self.gripper_right_motor.setPosition(0.9)
        self.grip_state = 0
        self.robot_name = self.robot.getName()
        if self.robot_name == 'red':
            self.robot_id = 0
            self.emitter.setChannel(0)
            self.receiver.setChannel(0)
        else:
            self.robot_id = 1
            self.emitter.setChannel(1)
            self.receiver.setChannel(1)
    
    def grip(self, grip_pos):
        if grip_pos ==1:
            self.gripper_left_motor.setPosition(0.1)
            self.gripper_right_motor.setPosition(0.1)
        else:
            self.gripper_left_motor.setPosition(0.9)
            self.gripper_right_motor.setPosition(0.9)
    
    def create_message(self):
        psValues = [self.ds_right.getValue(), self.ds_left.getValue()]
        rad = np.arctan2(self.compass.getValues()[0], self.compass.getValues()[2])
        global_position = [self.main_gps.getValues()[2],self.main_gps.getValues()[0], rad]

        color_data = self.colour_sensor.getImage()
        red = self.colour_sensor.imageGetRed(color_data, self.colour_sensor.getWidth(), 0,0)
        blue = self.colour_sensor.imageGetBlue(color_data, self.colour_sensor.getWidth(), 0,0)
        green = self.colour_sensor.imageGetGreen(color_data, self.colour_sensor.getWidth(), 0,0)
        gray = self.colour_sensor.imageGetGray(color_data, self.colour_sensor.getWidth(), 0,0)

        if red > 1.5*gray:
            detected_colour = 2
        elif blue > 1.5*gray:
            detected_colour = 1
        else:
            detected_colour = 0

        return [self.robot_id, global_position[0], global_position[1], global_position[2], psValues[0], psValues[1], detected_colour, self.grip_state]


        
    
    def use_message_data(self, message):
        if self.robot_id == 0:
            m = 0
        else:
            m = 3
        
        self.wheel1.setVelocity(float(message[m]))
        self.wheel2.setVelocity(float(message[m+1]))
        self.grip(int(message[m+2]))
        self.grip_state = int(message[m+2])

        

bot_controller = bot()
bot_controller.run()

