import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line

# utiliies for grid updates
def log_odds_ratio(p):
	return np.log(p/(1-p))

def inv_log_odds_ratio(l):
	return 1 - 1/(1+np.exp(l))

def update_occupied(grid, coord, p = 0.6):
	return inv_log_odds_ratio(log_odds_ratio(grid[coord[0]][coord[1]]) + log_odds_ratio(p))

def update_free(grid, coord, position, p=0.4):
	rr, cc = line(position[0], position[1], coord[0], coord[1])
	updates = []
	for i in range(len(rr)-1):
		updates.append(inv_log_odds_ratio(log_odds_ratio(grid[rr[i], cc[i]]) + log_odds_ratio(p)))

	return rr[:-1], cc[:-1], np.array(updates)

### General utility function defs:
def bound(coord, a=0, b=79):
	k = min(coord, b)
	k = max(k, a)
	return k

def transform_local_coords(robot_position, local_coordinate):
	# robot_position = [x, y, theta]
	# local coordinate = [x', y']

	rot_matrix = np.array([[np.cos(robot_position[2]), np.sin(robot_position[2])], [-np.sin(robot_position[2]), np.cos(robot_position[2])]])
	offset_vector = np.array([robot_position[0],robot_position[1]])
	
	return np.matmul(rot_matrix, local_coordinate) + offset_vector

def convert_to_grid_coords(coord):
	return np.flip(np.rint((np.array(coord)*[100, -100] + np.array([119, 119]))/3))

def convert_to_grid_space(coord):
	return np.flip((np.array(coord)*[100, -100] + np.array([119, 119]))/3)

### Visualiser Class:
class visualiser:
	def __init__(self, num_of_displays, titles, k):
		self.num_of_displays = num_of_displays
		self.k = k
		exec('self.fig{} = plt.figure()'.format(k))
		
		#self.fig.suptitle('test title', fontsize=20)
		self.titles = titles
		placeholder = np.full((80,80), 0)
		for i in range(num_of_displays):
			exec('self.ax{}{} = self.fig{}.add_subplot(1, {}, {})'.format(k,i,k, self.num_of_displays, i+1))
			exec('self.display{}{} = self.ax{}{}.imshow(np.full((80,80), 0), vmin=-1, vmax=1, interpolation="None", cmap="RdBu")'.format(k, i, k,i))
			exec('self.ax{}{}.set_title(\"{}\")'.format(k, i, self.titles[i]))
			exec('self.fig{}.canvas.draw()'.format(k))
			exec('self.axbackground{}{} = self.fig{}.canvas.copy_from_bbox(self.ax{}{}.bbox)'.format(k, i, k, k,i))
		plt.show(block = False)

	def __call__(self, *args):
		if len(args) != self.num_of_displays:
			raise ValueError('displays and display number must match')
		else:
			for i in range(self.num_of_displays):
				exec('self.display{}{}.set_data(args[{}])'.format(self.k, i, i))
				exec('self.fig{}.canvas.restore_region(self.axbackground{}{})'.format(self.k,self.k,i))
				exec('self.ax{}{}.draw_artist(self.display{}{})'.format(self.k,i,self.k,i))
				exec('self.fig{}.canvas.blit(self.ax{}{}.bbox)'.format(self.k,self.k,i))
				exec('self.fig{}.canvas.flush_events()'.format(self.k))
	
	def close(self):
		exec('self.fig{}.close()'.format(self.k))
				
