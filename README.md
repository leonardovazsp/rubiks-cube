# Rubiks Cube Robot Solver

When an engineer gets bored, he/she builds a Rubiks Cube robot. This is a simple robot that solves a Rubiks Cube. It uses 2x cameras to capture the entirety of the cube at once, and then uses a neural network to determine the cube's state. The solving algorithm is a deep reinforcement learning algorithm that learns to solve the cube by trial and error (just for the fun of it).

## Hardware

![Hardware](assets/rubiks.png)

The robot is built using 2x Raspberry Pi 3, 2x Pi Camera, a 3D printed structure, 6x stepper motors, and a bunch of wires. The robot is controlled by a Python program running on the Raspberry Pi. The camera is used to detect the cube, and the stepper motors are used to rotate the cube. The neural network is implemented using PyTorch.

## Software

The software is written in Python. The neural network is implemented using PyTorch. The robot is controlled by a Python program running on the Raspberry Pi. The neural network is hosted on a remote server. The robot communicates with the server using a REST API.
