#!/bin/bash
# Profile Generator 服务管理脚本
# 用于 profile_generator 角色，使用 Llama-3.1-8B-Instruct 模型

# 配置（请根据实际环境修改）
MODEL_PATH="${MODEL_PATH:-/path/to/your/Llama-3.1-8B-Instruct}"
PORT="${PORT:-8001}"
LOG_DIR="${LOG_DIR:-logs}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.3}"
GPU_DEVICES="${GPU_DEVICES:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
SERVICE_NAME="profile_generator"
PYTHON_CMD="${PYTHON_CMD:-python}"

# 查找 vllm 进程
find_vllm_process() {
    pgrep -f "vllm.entrypoints.openai.api_server.*--port ${PORT}"
}

# 检查服务是否运行
check_service() {
    local pid=$(find_vllm_process)
    if [ -n "$pid" ]; then
        echo "服务正在运行 (PID: $pid)"
        return 0
    else
        return 1
    fi
}

# 检查端口是否可用
check_port() {
    curl -s http://localhost:${PORT}/v1/models > /dev/null 2>&1
    return $?
}

# 检查日志中的错误
check_log_for_errors() {
    if [ -f "${LOG_DIR}/vllm-${SERVICE_NAME}-${PORT}.log" ]; then
        local errors=$(tail -20 "${LOG_DIR}/vllm-${SERVICE_NAME}-${PORT}.log" | grep -i "error\|failed\|exception" | head -3)
        if [ -n "$errors" ]; then
            echo ""
            echo "⚠️  检测到错误信息:"
            echo "$errors"
            echo ""
        fi
    fi
}

# 启动服务
start_service() {
    echo "检查服务状态..."

    if check_service; then
        echo "✓ 服务已在运行，无需重复启动"
        local pid=$(find_vllm_process)
        echo "  PID: $pid"
        return 0
    fi

    echo "启动 Profile Generator 服务 (Llama-3.1-8B-Instruct)..."
    echo "模型路径: ${MODEL_PATH}"
    echo "端口: ${PORT}"
    echo "GPU 显存占用上限: ${GPU_MEMORY_UTILIZATION}"

    local tp_size="${TENSOR_PARALLEL_SIZE}"
    if [ -z "$tp_size" ]; then
        if [ -n "$GPU_DEVICES" ]; then
            tp_size=$(echo "$GPU_DEVICES" | awk -F',' '{print NF}')
        else
            tp_size=1
        fi
    fi

    if [ -n "$GPU_DEVICES" ]; then
        echo "GPU 设备: ${GPU_DEVICES}"
    else
        echo "GPU 设备: 未指定 (使用默认设备)"
    fi
    echo "Tensor 并行度: ${tp_size}"

    mkdir -p "$LOG_DIR"
    > ${LOG_DIR}/vllm-${SERVICE_NAME}-${PORT}.log

    echo ""
    echo "正在启动 vllm 服务器..."

    if [ -n "$GPU_DEVICES" ]; then
        CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" \
        nohup ${PYTHON_CMD} -m vllm.entrypoints.openai.api_server \
            --model "${MODEL_PATH}" \
            --port ${PORT} \
            --enforce-eager \
            --max-model-len 4096 \
            --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
            --tensor-parallel-size ${tp_size} \
            > ${LOG_DIR}/vllm-${SERVICE_NAME}-${PORT}.log 2>&1 &
    else
        nohup ${PYTHON_CMD} -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --port ${PORT} \
        --enforce-eager \
        --max-model-len 4096 \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
        --tensor-parallel-size ${tp_size} \
        > ${LOG_DIR}/vllm-${SERVICE_NAME}-${PORT}.log 2>&1 &
    fi

    local pid=$!
    echo "✓ vllm 进程已启动"
    echo "  PID: ${pid}"
    echo "  端口: ${PORT}"
    echo "  日志: ${LOG_DIR}/vllm-${SERVICE_NAME}-${PORT}.log"
    echo ""
    echo "等待服务初始化..."
    echo "提示: vllm 需要加载模型，首次启动可能需要 1-2 分钟"
    echo ""

    local max_wait=120
    local elapsed=0

    while [ $elapsed -lt $max_wait ]; do
        sleep 2
        elapsed=$((elapsed + 2))

        if ! ps -p $pid > /dev/null 2>&1; then
            echo ""
            echo "✗ 进程意外退出"
            check_log_for_errors
            echo "完整日志: ${LOG_DIR}/vllm-${SERVICE_NAME}-${PORT}.log"
            return 1
        fi

        if check_port; then
            echo ""
            echo "✓ 服务启动成功! (用时 ${elapsed} 秒)"
            echo ""
            echo "可用的模型:"
            curl -s http://localhost:${PORT}/v1/models | ${PYTHON_CMD} -m json.tool 2>/dev/null || echo "模型信息获取失败"
            return 0
        fi

        if [ $((elapsed % 10)) -eq 0 ]; then
            echo "[$elapsed/${max_wait}s] 等待中..."
            if [ -f "${LOG_DIR}/vllm-${SERVICE_NAME}-${PORT}.log" ]; then
                local last_line=$(tail -1 "${LOG_DIR}/vllm-${SERVICE_NAME}-${PORT}.log")
                if [ -n "$last_line" ]; then
                    echo "  最新日志: $last_line"
                fi
            fi
        else
            echo -n "."
        fi
    done

    echo ""
    echo "✗ 服务启动超时 (等待了 ${max_wait} 秒)"
    echo ""
    check_log_for_errors
    echo "请查看完整日志: ${LOG_DIR}/vllm-${SERVICE_NAME}-${PORT}.log"
    return 1
}

# 停止服务
stop_service() {
    echo "停止 Profile Generator 服务..."

    local pid=$(find_vllm_process)

    if [ -n "$pid" ]; then
        echo "找到进程 PID: $pid"
        kill $pid
        echo "已发送停止信号，等待进程退出..."

        for i in {1..10}; do
            sleep 1
            if ! ps -p $pid > /dev/null 2>&1; then
                echo "✓ 服务已停止"
                return 0
            fi
            echo -n "."
        done

        echo ""
        echo "温和终止失败，强制终止..."
        kill -9 $pid
        echo "✓ 服务已强制停止"
    else
        echo "没有找到运行中的服务"
        pkill -9 -f "vllm.entrypoints.openai.api_server.*--port ${PORT}"
    fi
}

# 重启服务
restart_service() {
    echo "重启 Profile Generator 服务..."
    stop_service
    echo ""
    sleep 2
    start_service
}

# 查看服务状态
status_service() {
    echo "=== Profile Generator 服务状态 ==="
    echo ""

    local pid=$(find_vllm_process)
    if [ -n "$pid" ]; then
        echo "进程状态: ✓ 运行中"
        echo "PID: $pid"
        echo ""
        echo "进程详情:"
        ps -p $pid -o pid,ppid,%cpu,%mem,etime,cmd --no-headers | sed 's/^/  /'
    else
        echo "进程状态: ✗ 未运行"
    fi

    echo ""

    echo "端口检查 (${PORT}):"
    if check_port; then
        echo "✓ 端口可访问"
        echo ""
        echo "模型信息:"
        curl -s http://localhost:${PORT}/v1/models | ${PYTHON_CMD} -m json.tool 2>/dev/null || echo "无法获取模型信息"
    else
        echo "✗ 端口不可访问"

        if [ -n "$pid" ]; then
            echo ""
            echo "提示: 进程正在运行但端口不可访问，可能正在初始化模型"
        fi
    fi
}

# 查看日志
view_logs() {
    if [ -f "${LOG_DIR}/vllm-${SERVICE_NAME}-${PORT}.log" ]; then
        echo "实时查看日志 (按 Ctrl+C 退出):"
        echo "================================"
        tail -f "${LOG_DIR}/vllm-${SERVICE_NAME}-${PORT}.log"
    else
        echo "日志文件不存在: ${LOG_DIR}/vllm-${SERVICE_NAME}-${PORT}.log"
    fi
}

# 主函数
case "$1" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        status_service
        ;;
    logs)
        view_logs
        ;;
    check)
        if check_port; then
            echo "✓ Profile Generator 服务正常运行"
            exit 0
        else
            echo "✗ Profile Generator 服务不可用"
            exit 1
        fi
        ;;
    *)
        echo "用法: $0 {start|stop|restart|status|logs|check}"
        echo ""
        echo "环境变量配置:"
        echo "  MODEL_PATH            - 模型路径"
        echo "  PORT                  - 服务端口 (默认: 8000)"
        echo "  GPU_DEVICES           - GPU设备ID"
        echo "  PYTHON_CMD            - Python命令"
        exit 1
        ;;
esac
