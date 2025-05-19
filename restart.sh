#!/bin/bash

# 定义端口和日志文件名
PORT=8000
PYTHON_SCRIPT="main.py"
LOG_FILE="nohup.log"
APP_COMMAND="python ${PYTHON_SCRIPT}" # 如果你用python3, 请改成 python3 ${PYTHON_SCRIPT}

echo "正在尝试重启应用..."

# 查找占用指定端口的进程PID
# 使用 lsof 命令。如果 lsof 不可用，可以尝试 netstat 或 ss (如：ss -tulnp | grep ":${PORT}" | awk '{print $7}' | cut -d',' -f1 | sed 's/pid=//')
PID=$(lsof -t -i:${PORT})

# 如果找到了PID，就杀死该进程
if [ -n "$PID" ]; then
  echo "发现进程 (PID: $PID) 正在使用端口 ${PORT}."
  echo "正在终止进程 $PID ..."
  kill $PID
  # 等待一段时间确保进程被杀死
  sleep 2

  # 再次检查进程是否还存在，如果存在则强制杀死
  if ps -p $PID > /dev/null; then
    echo "进程 $PID 未能优雅关闭，尝试强制终止 (kill -9)..."
    kill -9 $PID
    sleep 1
  fi

  if ps -p $PID > /dev/null; then
    echo "错误：无法终止进程 $PID。请手动检查。"
  else
    echo "进程 $PID 已成功终止。"
  fi
else
  echo "未发现占用端口 ${PORT} 的进程。"
fi

# 检查 python 脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: 启动脚本 '$PYTHON_SCRIPT' 未在当前目录找到!"
    exit 1
fi

# 后台启动 python 应用
echo "正在启动 ${APP_COMMAND} ..."
nohup ${APP_COMMAND} > ${LOG_FILE} 2>&1 &

# 获取后台进程的PID
BG_PID=$!

# 短暂等待，让应用有时间启动或失败
sleep 1

# 检查新进程是否成功启动
if ps -p $BG_PID > /dev/null; then
   echo "${PYTHON_SCRIPT} 已成功启动 (PID: $BG_PID)."
   echo "日志输出到: ${LOG_FILE}"
   echo "你可以使用 'tail -f ${LOG_FILE}' 查看实时日志。"
else
   echo "错误: ${PYTHON_SCRIPT} 未能成功启动。请检查 ${LOG_FILE} 获取详细信息。"
fi

echo "重启脚本执行完毕。"
