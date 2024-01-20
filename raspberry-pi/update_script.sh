#!/bin/bash

PROJECT_DIR="~/projects/rubiks-cube/raspberry-pi"
MAIN_PY_SCRIPT="main.py"

cd $PROJECT_DIR

git pull origin main

pip install -r requirements.txt

systemctl restart rubiks.service