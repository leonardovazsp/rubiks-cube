'''
Module to drive the Rubik's Cube.
This module interfaces with raspberry pi GPIO pins to drive the motors.
The motors are driven by A4988 motor driver.
The motors are connected to the raspberry pi GPIO pins as follows:
The pins configuration is stored in the config.json file.
'''

import time
import RPi.GPIO as GPIO
import json

WAIT_TIME = 0.001

# Set the GPIO mode
GPIO.setmode(GPIO.BCM)

# Deactivate warnings
GPIO.setwarnings(False)

# Load config
with open('config.json') as f:
    config = json.load(f)

pins = config['pins']

# Set the GPIO pins
for pin in pins.values():
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, False)

# Define the motor functions
def rotate(pin, steps=50):
    for i in range(steps):
        GPIO.output(pin, True)
        time.sleep(WAIT_TIME)
        GPIO.output(pin, False)
        time.sleep(WAIT_TIME)

def top():
    print('top')
    GPIO.output(pins['direction'], True)
    rotate(pins['top'])

def top_rev():
    GPIO.output(pins['direction'], False)
    rotate(pins['top'])

def right():
    GPIO.output(pins['direction'], True)
    rotate(pins['right'])

def right_rev():
    GPIO.output(pins['direction'], False)
    rotate(pins['right'])

def front():
    GPIO.output(pins['direction'], True)
    rotate(pins['front'])

def front_rev():
    GPIO.output(pins['direction'], False)
    rotate(pins['front'])

def left():
    GPIO.output(pins['direction'], True)
    rotate(pins['left'])

def left_rev():
    GPIO.output(pins['direction'], False)
    rotate(pins['left'])

def back():
    GPIO.output(pins['direction'], True)
    rotate(pins['back'])

def back_rev():
    GPIO.output(pins['direction'], False)
    rotate(pins['back'])

def bottom():
    GPIO.output(pins['direction'], True)
    rotate(pins['bottom'])

def bottom_rev():
    GPIO.output(pins['direction'], False)
    rotate(pins['bottom'])