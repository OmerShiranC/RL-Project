import math
import sympy as sp
from sympy import N
import numpy as np
from scipy.optimize import minimize_scalar

# Data Visualization
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

class Settings:
    """Define settings of the environment such as course shape, course characteristics
       possible actions, position, neural-network parameters and training parameters

    Example usage:
        > settings = Settings()
    """
    def __init__(self):
        # Define road segments, this whole sequence is an ellipsis course
        # Starts with straight horizontal line, draw elliptical section 
        # then another straight line and then close with another ellipsis
        self.road_segments = [
            {
                'x': lambda t: t,
                'y': lambda t: 0.000001*t,
                't_start': 0,
                't_end': 10
            },
            {
                'x': lambda t: 10 + 5 * sp.sin(np.pi * (t - 10)),
                'y': lambda t: 5-5 * sp.cos(np.pi * (t - 10)),
                't_start': 10,
                't_end': 11
            },
            {
                'x': lambda t: 21-t,
                'y': lambda t: 8 +2*sp.cos(np.pi * (t - 11)/10),
                't_start': 11,
                't_end': 21
            },
            {
                'x': lambda t: 3 * sp.sin(np.pi * (t - 20)),
                'y': lambda t: 3 - 3 * sp.cos(np.pi * (t - 20)),
                't_start': 21,
                't_end': 22
            }
        ]
        # Define road parameters such as width and length
        self.road_width    = 1
        self.road_length   = 15
        # Closed is a boolean that indicates whether the circuit is closed
        # (I presume)
        self.closed        = False
        # Road resolutions refers to how coarse the grid that constitutes the
        # course is (affects number of states)
        self.road_resolution = 100


        # car settings

        # car initial position
        self.init_car_x = 4
        self.init_car_y = 0
        self.init_car_theta = 0

        self.init_car_speed = .25
        
        #sensors
        self.n_sensors = 7
        self.resolution = .3
        self.max_sensor_range = 3
        
        #actions
        self.action_dim = 7  # Actions for steering
        self.actions = 0.5*np.linspace(-1,1,self.action_dim)
        
        ## NN
        self.state_dim = self.n_sensors  * self.max_sensor_range / self.resolution   # State dimension is the number of sensor
        self.Hidden_layers = [32, 16]
        # Define the gamma and alpha
        self.gamma  = .95
        self.alpha  = .95
                 
        # training
        self.epsilon = 0.05
        self.lr      = 0.001
        self.num_episodes = 100

class RoadEnv:
    """Define Environment of the road. Requires a Settings object to function

    Example usage:
        > settings = Settings()
        > roadenv = RoadEnv(settings)
    """
    def __init__(self, settings):
        # Explain what t is
        self.settings = settings
        self.t = sp.Symbol('t')

    def get_road_limits(self):
        """
        Args:
            self: RoadEnv object
        
        Outputs:
            road_limits: List of three vectorized lambda functions that track the 
                        road limits to the left, right and center respectively
        """
        road_limits = {'left': [], 'right': [], 'center': []}

        for segment in self.settings.road_segments:
            x = segment['x'](self.t)
            y = segment['y'](self.t)

            # Calculate the derivatives of the parametric equations
            dx_dt = sp.diff(x, self.t)
            dy_dt = sp.diff(y, self.t)

            # Calculate the normal vector
            normal_x = -dy_dt / sp.sqrt(dx_dt ** 2 + dy_dt ** 2)
            normal_y =  dx_dt / sp.sqrt(dx_dt ** 2 + dy_dt ** 2)

            # Calculate the road limits
            half_width = self.settings.road_width / 2
            left_x = x + normal_x * half_width
            left_y = y + normal_y * half_width
            right_x = x - normal_x * half_width
            right_y = y - normal_y * half_width

            # Create vectorized lambda functions for the road limits
            left_func   = sp.lambdify(self.t, [left_x, left_y], 'numpy')
            right_func  = sp.lambdify(self.t, [right_x, right_y], 'numpy')
            center_func = sp.lambdify(self.t, [x, y], 'numpy')

            road_limits['left'].append(left_func)
            road_limits['right'].append(right_func)
            road_limits['center'].append(center_func)

        return road_limits

    def distance_road_center(self, x, y):
        """ Tracks distance from a point in the road to the center of the road
        Args:
            self: RoadEnv object
            x: float, position of the point in the road in the x-axis
            y: float, position of the point in the road in the y-axis
        
        Outputs:
            min_distance: float, minimum 
            closest_segment: integer, index of the segment of the road that is closest to the point
            closest_t: sympy Symbol, part of the segment that is closest 
        """
        min_distance = float('inf')
        closest_segment = None
        closest_t = None

        for i, segment in enumerate(self.settings.road_segments):
            road_center_x = sp.lambdify(self.t, sp.sympify(segment['x'](self.t)), 'numpy')
            road_center_y = sp.lambdify(self.t, sp.sympify(segment['y'](self.t)), 'numpy')

            def distance_func(t):
                return (road_center_x(t) - x) ** 2 + (road_center_y(t) - y) ** 2

            result = minimize_scalar(distance_func, bounds=(segment['t_start'], segment['t_end']), method='bounded')
            distance = np.sqrt(result.fun)

            if distance < min_distance:
                min_distance = distance
                closest_segment = i
                closest_t = result.x

        return min_distance, closest_segment, closest_t

    def road_direction_and_terminal(self, x, y):
        """
        Args:
            self: RoadEnv object
            x: float, position of the point in the road in the x-axis
            y: float, position of the point in the road in the y-axis
        
        Outputs:
            distance: float or None, distance from the center
                      evaluates to None if the position is out of bounds
                      
            directions: np.array, array with the possible directions in the road. They are positive by construction
            
            out_of_road: boolean, evaluates to True if out of bounds
            
        """
        distance, segment, t = roadenv.distance_road_center(x, y)


        if distance > self.settings.road_width / 2:
            distance, positions, out_of_road = None, None, True
            return distance, positions, out_of_road

        # Calculate the direction of the road at the closest point
        segment = self.settings.road_segments[segment]
        x_func = sp.lambdify(self.t, segment['x'](self.t), 'numpy')
        y_func = sp.lambdify(self.t, segment['y'](self.t), 'numpy')

        dx_dt = sp.diff(segment['x'](self.t), self.t)
        dy_dt = sp.diff(segment['y'](self.t), self.t)
        
        dx_func = sp.lambdify(self.t, dx_dt, 'numpy')
        dy_func = sp.lambdify(self.t, dy_dt, 'numpy')

        dx = dx_func(t)
        dy = dy_func(t)
        
        direction = np.pi/2-np.arctan2(dx,dy)
        directions = np.where(direction < 0, direction + 2 * np.pi, direction)
        out_of_road = False
        return distance, directions, out_of_road
    
    def reward(self, distance, road_direction, carenv_theta, out_of_road):
        """
        Args:
            self: RoadEnv object
            distance: float, distance from center
            directions:np.array, array with the possible directions in the road. They are positive by construction
            carenv_theta: float, angle          
            out_of_road: boolean, evaluates to True if out of bounds
        Outputs:
            reward: float, reward from the state action pair
            
        """
        reward = -100
        if not out_of_road:
            ang_diff1 = abs(road_direction - carenv_theta)
            ang_diff2 = min(ang_diff1,2*np.pi-ang_diff1)
            ang_diff3 = np.cos(ang_diff2)
            reward = -(self.settings.road_width  - distance)**2 + ang_diff3
        return reward

class CarEnv:
    """Define Environment of the car. Requires a Settings object to function

    Example usage:
        > settings = Settings()
        > carenv = CarEnv(settings)
    """
    def __init__(self, settings, roadenv):
        self.settings = settings
        self.roadenv = roadenv
        self.x = settings.init_car_x
        self.y = settings.init_car_y
        self.theta = settings.init_car_theta
        self.speed = settings.init_car_speed
        self.trajectory = [(self.x, self.y)]
        self.terminal = False

    def car_reset(self):
        self.x = settings.init_car_x
        self.y = settings.init_car_y
        self.theta = settings.init_car_theta
        self.speed = settings.init_car_speed
        self.trajectory = [(self.x, self.y)]
        self.terminal = False

    def move(self, steering_angle):
        """
        Args:
            self: CarEnv object
            steering_angle: float, angle to steer the car
        """
        # Update the car position
        self.x += self.speed * np.cos(self.theta)
        self.y += self.speed * np.sin(self.theta)
        self.theta = (self.theta +steering_angle) % (2 * np.pi)
        self.trajectory.append((self.x, self.y))

    def get_state(self):
        """
        Args:
            self: CarEnv object

        Outputs:
            state: np.array, array with the state of the car
        """
        state = np.zeros(self.settings.n_sensors)
        angles = np.linspace(-np.pi/2, np.pi/2, self.settings.n_sensors)
        for angle in angles:
            x_loc = self.x.copy()
            y_loc = self.y.copy()
            sensor_reading = 0
            while sensor_reading < self.settings.max_sensor_range:
                x_loc += self.settings.resolution*np.cos(self.theta + angle)
                y_loc += self.settings.resolution*np.sin(self.theta + angle)
                dis,_,_ = distance_road_center(self, x_loc, y_loc)
                if dis < self.settings.road_width/2:
                    break
                sensor_reading += self.settings.resolution
            state[angles.index(angle)] = sensor_reading
        return state

    def step(self, action): # go over this function
        """
        Args:
            self: CarEnv object
            action: float, action to take

        Outputs:
            next_state: np.array, array with the state of the car after the action
            reward: float, reward from the state action pair
        """
        self.move(action)
        distance, road_direction, self.terminal = self.roadenv.road_direction_and_terminal(self.x, self.y)
        reward = self.roadenv.reward(distance, road_direction, self.theta, out_of_road)

        if self.terminal:
            state = self.get_state()
        else:
            state = None

        return state, reward




def Visualize(roadenv, carenv, settings):
    """
    Args:
        roadenv: RoadEnv object
        carenv: CarEnv object
        settings: Settings object
        
    """
    road_limits = roadenv.get_road_limits()

    fig, ax = plt.subplots(figsize=(12, 12))

    for i, segment in enumerate(settings.road_segments):
        t = np.linspace(segment['t_start'], segment['t_end'], settings.road_resolution)

        left_x, left_y = road_limits['left'][i](t)
        right_x, right_y = road_limits['right'][i](t)
        center_x, center_y = road_limits['center'][i](t)

        ax.plot(left_x, left_y, '-', color='gold')
        ax.plot(right_x, right_y, '-', color='gold')
        ax.plot(center_x, center_y, 'w--')
        
        # Add direction arrow for each segment
        mid_point = len(t) // 2
        mid_x, mid_y = center_x[mid_point], center_y[mid_point]
        dx = center_x[mid_point + 1] - center_x[mid_point]
        dy = center_y[mid_point + 1] - center_y[mid_point]
        arrow_length = 0.3
        ax.arrow(mid_x, mid_y, arrow_length * dx, arrow_length * dy,\
                 head_width=0.1, head_length=0.1, fc='white', ec='white',\
                 length_includes_head=True, zorder=3)

    ax.plot([], [], '-', color='gold', label='Road edge')
    ax.plot([], [], 'w--', label='Center line')

    ax.set_facecolor('gray')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Piecewise Road Visualization')
    ax.legend()

    # Ensure the aspect ratio is equal
    ax.set_aspect('equal', 'box')

    # Plot the car as an arrow and the trajectory in red
    arrow_length = 0.3
    ax.arrow(carenv.x, carenv.y, arrow_length*np.cos(carenv.theta), arrow_length*np.sin(carenv.theta),
             head_width=0.2, head_length=0.1, fc='r', ec='r', label='Car')
    ax.plot(*zip(*carenv.trajectory), 'r--', label='Trajectory')

    ax.grid(True)
    plt.tight_layout()
    plt.show()


def get_valid_input(a, b):
    """
    Get a valid integer input from the user between a and b
    args:
        param a: lower bound
        param b: upper bound
    return:
        integer between a and b or 'q' to quit
    """
    while True:
        user_input = input(f"Enter an integer between {a} and {b}, or 'q' to quit: ").strip().lower()

        if user_input == 'q':
            return 'q'

        try:
            number = int(user_input)
            if a <= number <= b:
                return number
            else:
                print(f"Please enter a number between {a} and {b}.")
        except ValueError:
            print("Invalid input. Please enter a valid integer or 'q'.")

#
#
# settings = Settings()
# roadenv = RoadEnv(settings)
# carenv = CarEnv(settings)
# # move the car
# for i in range(50):
#     Visualize(roadenv, carenv, settings)
#     upper_bound = 2
#     lower_bound = -2
#     print('')
#     action =  get_and_process_action(f"Step number {i+1}, choose action: ", lower_bound,upper_bound)
#     if action == 'q':
#         break
#     carenv.move(float(action))
#     distance, closest_segment, closest_t = roadenv.distance_road_center(carenv.x, carenv.y)
#     distance, road_direction, out_of_road = roadenv.road_direction_and_terminal(distance, closest_segment, closest_t)
#     reward = roadenv.reward(distance, road_direction, carenv.theta, out_of_road)
#
#     if not  out_of_road:
#         #round the direction to 2 decimal points
#         print(f"   Distance: {distance:.2f}, Direction: {road_direction:.2f}, carenv.theta: {carenv.theta:.2f}, difection diff Direction: {abs(road_direction - carenv.theta):.2f}, Out of road: {out_of_road}")
#         print(f"    reward = {reward} ")
#         print(f" ")
#     if i>2:
#         plt.close('all')
        





    
