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
    def __init__(self):
        # road settings
        self.road_segments = [
            #Build the road piece by piece
            #First part: x increases linearly and y increases very slowly
            {
                'x': lambda t: t,
                'y': lambda t: 0.000001*t,
                't_start': 0,
                't_end': 5
            },
            #Second part: ellipsis
            {
                'x': lambda t: 5 + 3 * sp.sin(np.pi * (t - 4.9)),
                'y': lambda t: 3+3 * sp.cos(np.pi * (t - 4.9)),
                't_start': 5,
                't_end': 6
            },
            #Third part: x decreases linearly and y decreases very slowly
            {
                'x': lambda t: 5 - (t - 6),
                'y': lambda t: 6-0.000001*t,
                't_start': 6,
                't_end': 11
            },
            #Fourth part: Another ellipsis
            {
                'x': lambda t: 3 * sp.sin(np.pi * (t - 10)),
                'y': lambda t: 3 + 3 * sp.cos(np.pi * (t - 10)),
                't_start': 11,
                't_end': 12
            }
        ]
        #Road characteristics
        self.road_width    = 1
        self.road_length   = 15
        self.closed        = False
        self.road_resulotion = 100

        # car settings

        # car initial position
        self.init_car_x = 4
        self.init_car_y = 0
        self.init_car_theta = 0
        self.init_car_speed = .5

#Define the road environment class
class RoadEnv:
    def __init__(self, settings):
        self.settings = settings
        self.t = sp.Symbol('t')

    def get_road_limits(self):
        road_limits = {'left': [], 'right': [], 'center': []}

        for segment in self.settings.road_segments:
            x = segment['x'](self.t)
            y = segment['y'](self.t)

            # Calculate the derivatives of the parametric equations
            dx_dt = sp.diff(x, self.t)
            dy_dt = sp.diff(y, self.t)

            # Calculate the normal vector
            # Normal vector is necessary because at any point, the limit of the road is perpendicular to the centerline
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

    #Get distance to the center
    def distance_road_center(self, x, y):
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

    #Get direction of the road and whether the car crashes
    def road_direction_and_terminal(self, distance, segment, t):
        if distance > self.settings.road_width / 2:
            return None, None, True

        segment = self.settings.road_segments[segment]
        x = sp.sympify(segment['x'](self.t))
        y = sp.sympify(segment['y'](self.t))

#         # Calculate the direction of the road at the closest point
        dx_dt = sp.diff(x, self.t)
        dy_dt = sp.diff(y, self.t)
        road_direction_x = sp.lambdify(self.t, dx_dt, 'numpy')
        road_direction_y = sp.lambdify(self.t, dy_dt, 'numpy')

        dx = road_direction_x(t)
        dy = road_direction_y(t)

        #dx = self.settings.road_segments[segment]['x'](t+1)-self.settings.road_segments[segment]['x'](t)
        #dy = self.settings.road_segments[segment]['y'](t+1)-self.settings.road_segments[segment]['y'](t)
        
        print(f' dx={dx:.2f},dy={dy:.2f}, {type(dx)}, {type(dy)}')
        direction = np.pi/2-np.arctan2(dx,dy)
        print(f' dx={dx:.2f},dy={dy:.2f}, direction = {direction:.2f}')
        return distance, np.where(direction < 0, direction + 2 * np.pi, direction), False

class CarEnv:
    def __init__(self, settings):
        self.settings = settings
        self.x = settings.init_car_x
        self.y = settings.init_car_y
        self.theta = settings.init_car_theta
        self.speed = settings.init_car_speed
        self.trajectory = [(self.x, self.y)]

    def car_reset(self):
        self.x = settings.init_car_x
        self.y = settings.init_car_y
        self.theta = settings.init_car_theta
        self.speed = settings.init_car_speed
        self.trajectory = [(self.x, self.y)]

    def move(self, steering_angle):
        # Update the car position
        self.x += self.speed * np.cos(self.theta)
        self.y += self.speed * np.sin(self.theta)
        self.theta = (self.theta +steering_angle) % (2 * np.pi)
        self.trajectory.append((self.x, self.y))
        
        


def Visualize(roadenv, carenv, settings):
    road_limits = roadenv.get_road_limits()

    fig, ax = plt.subplots(figsize=(12, 12))

    for i, segment in enumerate(settings.road_segments):
        t = np.linspace(segment['t_start'], segment['t_end'], settings.road_resulotion)

        left_x, left_y = road_limits['left'][i](t)
        right_x, right_y = road_limits['right'][i](t)
        center_x, center_y = road_limits['center'][i](t)

        ax.plot(left_x, left_y, '-', color='gold')
        ax.plot(right_x, right_y, '-', color='gold')
        ax.plot(center_x, center_y, 'w--')

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




settings = Settings()
roadenv = RoadEnv(settings)
carenv = CarEnv(settings)
# move the car
for i in range(50):
    #action = input(f"Step number {i+1}, choose action: ")
    action = np.random.uniform(-.5, .5)
    if action == 'q':
        break
    carenv.move(float(action))
    distance, closest_segment, closest_t = roadenv.distance_road_center(carenv.x, carenv.y)
    distance, direction, out_of_road = roadenv.road_direction_and_terminal(distance, closest_segment, closest_t)
    if not  out_of_road:
        #round the direction to 2 decimal points
        print(f"   Distance: {distance:.2f}, Direction: {direction:.2f}, carenv.theta: {carenv.theta:.2f}, difection diff Direction: {abs(direction - carenv.theta):.2f}, Out of road: {out_of_road}")
        print(f" ")
    if i>2:
        plt.close('all')
        





    Visualize(roadenv, carenv, settings)