#!/bin/bash
# deploy-ec2.sh - Deploy script for EC2

set -e  # Exit on error

echo "🚀 Starting deployment to EC2..."

# 1. Kill existing uvicorn process
echo "📌 Stopping existing uvicorn process..."
pkill -f "uvicorn app:app" || true
sleep 2

# 2. Pull latest code
echo "📥 Pulling latest code from GitHub..."
cd ~/ai-operation-microservice
git fetch origin
git reset --hard origin/main
git clean -fd

# 3. Show latest commits
echo "📝 Latest commits:"
git log -3 --oneline

# 4. Verify critical files changed
echo "🔍 Verifying static files..."
if grep -q "กรุณารอสักครู่" code/static/index.html; then
    echo "✅ index.html updated correctly"
else
    echo "❌ WARNING: index.html might not be updated!"
fi

# 5. Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt --quiet

# 6. Set PYTHONPATH
export PYTHONPATH="$PWD/code"
echo "✅ PYTHONPATH set to: $PYTHONPATH"

# 7. Reingest data (optional - comment out if not needed)
# echo "🔄 Re-ingesting data..."
# cd code
# rm -rf local_chroma_v2
# python3 -m scripts.ingest_local
# cd ..

# 8. Start server
echo "🚀 Starting uvicorn server..."
cd ~/ai-operation-microservice/code
nohup python3 -m uvicorn app:app --host 0.0.0.0 --port 3000 > uvicorn.log 2>&1 &

# 9. Wait for server to start
sleep 3

# 10. Check if server is running
if pgrep -f "uvicorn app:app" > /dev/null; then
    echo "✅ Server started successfully!"
    echo "📊 Server PID: $(pgrep -f 'uvicorn app:app')"
    echo "📝 Logs: tail -f ~/ai-operation-microservice/code/uvicorn.log"
else
    echo "❌ Server failed to start! Check logs:"
    tail -20 uvicorn.log
    exit 1
fi

echo "🎉 Deployment completed!"
echo "🌐 Access: http://43.209.110.234:3000"
