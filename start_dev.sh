#!/bin/bash

# 启动后端
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload &
BACKEND_PID=$!

# 启动前端
cd frontend_streamlit
API_BASE_URL=http://127.0.0.1:8080 streamlit run app.py --server.port 8188 &
FRONTEND_PID=$!
cd ..

# 定义 Ctrl+C 处理函数
cleanup() {
    echo "Stopping backend and frontend..."
    kill $BACKEND_PID $FRONTEND_PID
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "Done!"
    exit 0
}

# 捕获 SIGINT (Ctrl+C) 和 SIGTERM
trap cleanup SIGINT SIGTERM

# 等待两个进程结束
wait $BACKEND_PID $FRONTEND_PID