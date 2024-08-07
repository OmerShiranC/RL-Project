import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from IPython import display

import numpy as np


def Visualize(roadenv, carenv, settings, trajectories):
    """
    Args:
        roadenv: RoadEnv object
        carenv: CarEnv object
        settings: Settings object

    """
    road_limits = roadenv.get_road_limits()

    fig, ax = plt.subplots(figsize=(12, 7))

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

    if not trajectories:
        # Plot the car as an arrow and the trajectory in red
        arrow_length = 0.3
        ax.arrow(carenv.x, carenv.y, arrow_length*np.cos(carenv.theta), arrow_length*np.sin(carenv.theta),
                 head_width=0.2, head_length=0.1, fc='r', ec='r', label='Car')
        ax.plot(*zip(*carenv.trajectory), 'r--', label='Trajectory')
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    else:
        for trajectory in trajectories:
            x_coords = [coord[0] for coord in trajectory]
            y_coords = [coord[1] for coord in trajectory]
            ax.plot(x_coords, y_coords, 'r--', label='Trajectory')
            ax.grid(True)
            plt.tight_layout()
            plt.ion()  # Turn on interactive mode
            plt.show(block=False)  # Show the plot without blocking execution


def plot_training_progress(all_rewards):
    fig, ax = plt.subplots(figsize=(20, 7))
    line, = ax.plot([], [])


    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Training Progress')

    line.set_xdata(range(len(all_rewards)))
    line.set_ydata(all_rewards)
    ax.relim()
    ax.autoscale_view()
    display.clear_output(wait=True)
    display.display(fig)
    plt.pause(0.1)



fig, (ax_reward, ax_trajectory) = plt.subplots(2, 1, figsize=(10, 8))

def epoch_vis(all_rewards, roadenv, carenv, settings, trajectories, all_speeds):
    """
    Updates the visualization with new data for each epoch.

    Args:
        all_rewards: List of total rewards for each training episode.
        roadenv: RoadEnv object.
        carenv: CarEnv object.
        settings: Settings object.
        trajectories: List of trajectories (episodes) from the training process.
    """
    global ax_reward, ax_trajectory  # Access the global axes

    # Update reward plot
    ax_reward.clear()  # Clear previous data
    ax_reward.plot(range(len(all_rewards)), all_rewards, label="Reward")

    # Create a second y-axis for speed
    ax_speed = ax_reward.twinx()

    # Plot speed curve
    ax_speed.plot(range(len(all_speeds)), all_speeds, color='orange', label="Speed")

    # Set labels and title
    ax_reward.set_xlabel('Episode')
    ax_reward.set_ylabel('Total Reward')
    ax_speed.set_ylabel('Average Speed')
    ax_speed.set_ylim(0, settings.max_speed)
    ax_reward.set_title('Training Progress')
    ax_reward.grid(True)

    # Adjust legend position to avoid overlap
    ax_reward.legend(loc='upper left')
    ax_speed.legend(loc='upper right')

    # Update trajectory plot
    ax_trajectory.clear()
    road_limits = roadenv.get_road_limits()

    for i, segment in enumerate(settings.road_segments):
        t = np.linspace(segment['t_start'], segment['t_end'], settings.road_resolution)

        left_x, left_y = road_limits['left'][i](t)
        right_x, right_y = road_limits['right'][i](t)
        center_x, center_y = road_limits['center'][i](t)

        ax_trajectory.plot(left_x, left_y, '-', color='gold')
        ax_trajectory.plot(right_x, right_y, '-', color='gold')
        ax_trajectory.plot(center_x, center_y, 'w--')


    # Adjust plot formatting
    ax_trajectory.set_facecolor('gray')
    ax_trajectory.set_xlabel('x')
    ax_trajectory.set_ylabel('y')
    ax_trajectory.set_title('Piecewise Road with Trajectory')
    ax_trajectory.set_aspect('equal', 'box')

    # Plot the car and trajectory (if any)
    if not trajectories:
        # Plot car and trajectory (code from Visualize function omitted for brevity)
        pass
    else:
        for trajectory in trajectories:
            x_coords = [coord[0] for coord in trajectory]
            y_coords = [coord[1] for coord in trajectory]
            ax_trajectory.plot(x_coords, y_coords, 'r--', label='Trajectory')

    # Display updated plot
    plt.tight_layout()
    plt.draw()
    #Save
    plt.savefig('TrainingLoop.png')
    plt.pause(0.1)  # Adjust pause time as needed
    
    