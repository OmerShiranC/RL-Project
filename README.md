# Car on Elliptical Road Simulator

This Python environment simulates a car navigating a closed-loop road with elliptical sections. The car receives sensor readings and takes actions (steering and acceleration/deceleration) to stay on the road and potentially maximize rewards. Using reinforcement learning, our car learns from the environment and learns when to steer, accelerate and brake. This is a work in progress.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
  - [Policy Network](#policy-network)
  - [Road Environment](#road-environment)
  - [Car Environment](#car-environment)
  - [Visualization](#visualization)
  - [Main Training Script](#main-training-script)
- [Acknowledgments](#acknowledgments)

## Overview
This project demonstrates the implementation of a reinforcement learning model for navigating a car through a defined road environment using Deep Q-Learning. The project includes three main components: the policy network, the road environment, and the car environment. Additionally, there is a visualization component to visualize the car's trajectory and the road.

## Requirements
To run this project, you will need the following packages:
- torch
- numpy
- sympy
- scipy
- matplotlib

You can install the required packages using the following command:
```bash
pip install -r requirements.txt

![screenshot](TrainingLoop.png)





