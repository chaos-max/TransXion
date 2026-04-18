#!/bin/bash
# 交易数据生成脚本控制工具

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_SERVICE="${SCRIPT_DIR}/manage_llm_service.sh"

# 检查 LLM 服务
check_llm() {
    if [ -x "$LLM_SERVICE" ]; then
        $LLM_SERVICE check > /dev/null 2>&1
        return $?
    else
        echo "✗ 找不到 LLM 服务管理脚本: $LLM_SERVICE"
        return 1
    fi
}

case "$1" in
    start)
        echo "准备启动交易数据生成..."
        # LLM 服务检查由 main_txn_async.py 根据配置自动处理：
        #   - 纯 API 模式（deepseek/openai）：无需本地服务，直接启动
        #   - 本地 vLLM 模式：Python 会检测对应端点并提示手动启动
        # 如需手动管理本地 vLLM，使用: bash run_txn.sh llm start

        echo ""
        echo "启动交易数据生成..."
        cd "$SCRIPT_DIR"
        nohup python main_txn_async.py > transaction_run.log 2>&1 &
        echo "进程已启动，PID: $!"
        echo "查看日志: tail -f $SCRIPT_DIR/transaction_run.log"
        ;;
    stop)
        echo "停止交易数据生成..."
        pkill -f "python main_txn_async.py"
        echo "进程已停止"
        ;;
    status)
        echo "检查进程状态..."
        ps aux | grep "python main_txn_async.py" | grep -v grep
        ;;
    log)
        tail -f "$SCRIPT_DIR/transaction_run.log"
        ;;
    llm)
        shift
        if [ -x "$LLM_SERVICE" ]; then
            $LLM_SERVICE "$@"
        else
            echo "✗ 找不到 LLM 服务管理脚本"
            exit 1
        fi
        ;;
    *)
        echo "用法: $0 {start|stop|status|log|llm}"
        echo ""
        echo "主要命令:"
        echo "  start  - 启动进程（LLM 服务检查由 Python 根据配置自动处理）"
        echo "  stop   - 停止进程"
        echo "  status - 查看状态"
        echo "  log    - 查看日志"
        echo ""
        echo "LLM 服务管理:"
        echo "  llm start   - 启动 LLM 服务"
        echo "  llm stop    - 停止 LLM 服务"
        echo "  llm restart - 重启 LLM 服务"
        echo "  llm status  - 查看 LLM 服务状态"
        echo "  llm logs    - 查看 LLM 服务日志"
        exit 1
        ;;
esac
