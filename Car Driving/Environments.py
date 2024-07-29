from help_functions import get_valid_input

import math
import sympy as sp
from sympy import N
import numpy as np
from scipy.optimize import minimize_scalar


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

        self.max_speed = 2
        self.min_speed = .2
        self.init_car_speed = .5
        
        #sensors
        self.n_sensors = 7
        self.resolution = .3
        self.max_sensor_range = 3
        
        #actions
        self.action_dim = 5  # Actions for steering
        self.actions = 0.5*np.linspace(-1,1,self.action_dim)

        # add acceleration and deceleration
        self.action_dim += 2
        
        ## NN
        self.state_dim = self.n_sensors  + 1   # State dimension is the number of sensor + speed
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
        distance, segment, t = self.distance_road_center(x, y)


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
    
    def reward(self, distance, road_direction, carenv_theta, out_of_road,carenv_speed):
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
            reward = -(self.settings.road_width  - distance)**2 + ang_diff3+ (2*(carenv_speed-3*self.settings.min_speed))**3
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
        self.x = self.settings.init_car_x
        self.y = self.settings.init_car_y + self.settings.road_width / 3 * np.random.uniform(-1, 1)
        self.theta = self.settings.init_car_theta + .3 * np.random.uniform(-1, 1)
        self.speed = self.settings.init_car_speed + .3 * np.random.uniform(-1, 1)
        self.trajectory = [(self.x, self.y)]
        self.terminal = False

    def move(self, action):
        """
        Args:
            self: CarEnv object
            steering_angle: float, angle to steer txhe car
        """
        steering_angle = 0
        print(f'action: {action}, car direction: {self.theta:.2f}, car speed: {self.speed:.2f}')
        if action == self.settings.action_dim - 1: #decelerate
            self.speed = max(self.settings.min_speed, self.speed - 0.1)
            print(f'      decelerate,   new speed: {self.speed :.2f}')
        elif action == self.settings.action_dim - 2: #accelerate
            self.speed = min(self.settings.max_speed, self.speed + 0.1)
            print(f'      accelerate,   new speed: {self.speed :.2f}')
        else:
            steering_angle = self.settings.actions[action]
            self.theta = (self.theta + steering_angle) % (2 * np.pi)
            print(f'      steering_angle   , {steering_angle:.2f}, car new direction  {self.theta:.2f}')

        # Update the car position
        self.x += self.speed * np.cos(self.theta)
        self.y += self.speed * np.sin(self.theta)
        self.trajectory.append((self.x, self.y))

    def get_state(self):
        """
        Args:
            self: CarEnv object

        Outputs:
            state: np.array, array with the state of the car
        """
        state = []
        angles = np.linspace(-np.pi/2, np.pi/2, self.settings.n_sensors)
        for angle in angles:
            x_loc = self.x
            y_loc = self.y
            sensor_reading = 0
            while sensor_reading < self.settings.max_sensor_range:
                x_loc += self.settings.resolution*np.cos(self.theta + angle)
                y_loc += self.settings.resolution*np.sin(self.theta + angle)
                dis,_,_ = self.roadenv.distance_road_center(x_loc, y_loc)
                if dis > self.settings.road_width/2:
                    break
                sensor_reading += self.settings.resolution
            state.append(sensor_reading)
        # Add speed to the state list
        state.append(self.speed)
        return state

    def step(self, action): # go over this function
        """
        Args:
            self: CarEnv object
            action: int, action to take

        Outputs:
            next_state: np.array, array with the state of the car after the action
            reward: float, reward from the state action pair
        """
        self.move(action)
        distance, road_direction, self.terminal = self.roadenv.road_direction_and_terminal(self.x, self.y)
        reward = self.roadenv.reward(distance, road_direction, self.theta, self.terminal,self.speed)

        # if not self.terminal:
        state = self.get_state()
        # else:
        #     state = None

        return state, reward



        





    
