
import sys
packages = [r'C:\\Users\\sahil\\Documents\\Python_environments\\env37', r'C:\\Users\\sahil\\Documents\\Python_environments\\env37\\lib\\site-packages']
for path in packages:
    sys.path.append(path)

from deepbots.supervisor.controllers.supervisor_emitter_receiver import SupervisorCSV
####Import RL agent 


class RLSupervisor(SupervisorCSV):
    def __init__(self):
        super().__init__()
        #self.observationSpace = 
        #self.actionSpace = 
        self.bot_red = self.supervisor.getFromDef('bot_red')
        self.bot_blue = self.supervisor.getFromDef('bot_blue')
        self.messageRecieved = None
        self.episodeLimit = 10000  # Max number of episodes allowed
        self.stepsPerEpisode = 200  # Max number of steps per episode
        self.episodeScore = 0  # Score accumulated during an episode
        self.episodeScoreList = []  # A list to save all the episode scores, used to check if task is solved

        def get_observations(self):
            #TO COMPLETE
            return None
        
        def get_reward(self):
            #TO COMPLETE
            return None
        
        def is_done(self):
            #TO COMPLETE
            return None
        
        def solved(self):
            #TO COMPLETE
            return None

        def reset(self):
            #TO COMPLETE
            return None
        
        def get_info(self):
            return None

        


