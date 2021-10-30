"""botController controller."""

import sys
packages = [r'C:\\Users\\sahil\\Documents\\Python_environments\\env37', r'C:\\Users\\sahil\\Documents\\Python_environments\\env37\\lib\\site-packages']
for path in packages:
    sys.path.append(path)

from deepbots.robots.controllers.robot_emitter_receiver_csv import RobotEmitterReceiverCSV

class bot(RobotEmitterReceiverCSV):
    def __init__(self):
        super().__init__()
        self.main_gps = self.robot.getDevice("gps_main")
        self.main_gps.enable(self.get_timestep())
        self.ds_left = self.robot.getDevice("ds_left")
        self.ds_left.enable(self.get_timestep())
        self.ds_right = self.robot.getDevice("ds_right")
        self.ds_right.enable(self.get_timestep())
        self.colour_sensor = self.robot.getDevice("camera")
        self.emitter = self.robot.getDevice("emitter")
        self.reciever = self.robot.getDevice("reciever")
        self.reciever.enable(self.get_timestep())
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
        self.gripper_left_motor.setPosition(0.1)
        self.gripper_right_motor.setPosition(0.1)
        self.robot_name = self.robot.getName()
        if self.robot_name == 'red':
            self.robot_id = 0
            self.emitter.setChannel(0)
            self.reciever.setChannel(0)
        else:
            self.robot_id = 1
            self.emitter.setChannel(1)
            self.reciever.setChannel(1)
    
    def grip(self, grip_pos):
        if grip_pos ==1:
            self.gripper_left_motor.setPosition(0.1)
            self.gripper_right_motor.setPosition(0.1)
        else:
            self.gripper_left_motor.setPosition(0.9)
            self.gripper_right_motor.setPosition(0.9)
    
    def create_message(self):

        return None
    
    def use_message_data(self, message):

        return None

bot_controller = bot()
bot_controller.run()

