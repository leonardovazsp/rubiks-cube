# Rubik's Cube Solving System

A complete system for autonomous Rubik’s Cube solving, integrating hardware control, computer vision, deep learning, and reinforcement learning. The system includes camera-based state recognition, precise motor control via Arduino and Raspberry Pi, and intelligent decision-making through trained models and planning algorithms.

![Demo](assets/video.gif)

---

## Features

* **Dual-camera capture** for full cube visibility
* **State recognition** via convolutional or transformer-based models
* **Pose alignment** using a learned classifier for cube reorientation
* **Cube manipulation** through Raspberry Pi-controlled stepper motors
* **End-to-end automation**: from image acquisition to solution execution
* **Deep reinforcement learning** agent trained to solve the cube
* **MCTS-enhanced planning** for fast, interpretable move sequences

---

## System Architecture

* **Hardware Layer**:

  * Raspberry Pi with dual cameras
  * Arduino + A4988 motor drivers
  * Stepper motors for 6 face actuations
  * REST communication between devices

* **Software Layer**:

  * Vision models in PyTorch (CNN & Transformer variants)
  * Reinforcement learning agent (Actor-Critic)
  * Monte Carlo Tree Search planner
  * Pose estimation for cube alignment
  * Dataset generation and API endpoints for remote control

---

## Components

### 🎯 `server/`

* Model training scripts (`trainer.py`, `train.py`)
* State and pose estimators (`model.py`)
* API for dataset labeling and model inference (`api.py`)
* Monte Carlo Tree Search logic (`reinforcement_learning/mcts.py`)
* Actor-Critic policy model with warm-up + exponential LR decay
* Evaluation & tracking via `wandb`

### 🤖 `raspberry-pi/`

* Motor control via serial to Arduino
* Image acquisition with `PiCamera` and OpenCV
* Cube state updates and move execution
* Flask server with ngrok tunnel support
* Pose alignment for multi-orientation data collection

### ⚙️ `arduino/`

* Command listener for motor stepping
* Delay and activation control
* Interface for directional movement across axes

---

## Learning and Decision-Making

This system integrates multiple approaches for decision making:

* **Supervised Learning**: Color and pose recognizers trained on cube images
* **Reinforcement Learning**: A policy trained from scratch to solve the cube using exploration and temporal rewards
* **Monte Carlo Tree Search (MCTS)**: Used on top of the RL policy for sample-efficient pathfinding and evaluation

---

## Setup & Installation

### Raspberry Pi

```bash
cd raspberry-pi
python3 setup.py --server-type master --domain your-domain.ngrok.io --port 8000 --token your-ngrok-token
```

### Arduino

Upload the sketch in `arduino/driver.cpp` using Arduino IDE.

### Server

```bash
cd server
python3 api.py
```

For training:

```bash
python3 test_trainer.py
```

---

## Dataset Generation

Run the following to capture images and associated states:

```bash
python3 generate.py
```

Images and move metadata will be saved in the configured directory.

---

## Example: Solving the Cube

```python
from cube import Cube
cube = Cube()
cube.driver.activate()
cube.driver.set_delay(900)
cube.scramble(20)
cube.solve()
```

---

## Notable Techniques Used

* **Batch-augmented self-supervised labeling**
* **Model-based exploration using MCTS**
* **Learned pose correction for reliable state capture**
* **Cross-platform coordination via serial and HTTP**

---

## License

MIT License.
