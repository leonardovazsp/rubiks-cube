#!/bin/bash

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$PROJECT_DIR"

git pull origin main
pip install -r requirements.txt
systemctl restart rubiks.service