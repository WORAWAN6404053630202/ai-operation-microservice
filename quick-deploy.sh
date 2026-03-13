#!/bin/bash
# quick-deploy.sh - Quick deploy commands (copy-paste to EC2 terminal)

# Stop old process
pkill -f "uvicorn app:app"
sleep 2

# Update code
cd ~/ai-operation-microservice && git fetch origin && git reset --hard origin/main && git clean -fd

# Check if updated
echo "Latest commit:" && git log -1 --oneline
echo "Checking index.html..." && grep "กรุณารอสักครู่" code/static/index.html && echo "✅ Updated!" || echo "❌ Not updated!"

# Install deps
pip3 install -r requirements.txt

# Start server
export PYTHONPATH="$PWD/code"
cd code && nohup python3 -m uvicorn app:app --host 0.0.0.0 --port 3000 > uvicorn.log 2>&1 &

# Check status
sleep 3 && ps aux | grep uvicorn | grep -v grep && echo "✅ Server running!" || echo "❌ Server not running!"
