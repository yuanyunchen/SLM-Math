#!/bin/bash

# 简单测试脚本，验证并行启动

echo "Testing parallel execution..."

# 启动三个后台任务，每个任务打印开始和结束时间
(
    export CUDA_VISIBLE_DEVICES=0
    echo "[GPU 0] Started at $(date '+%H:%M:%S')"
    sleep 5
    echo "[GPU 0] Finished at $(date '+%H:%M:%S')"
) > /tmp/gpu0_test.log 2>&1 &
PID0=$!

(
    export CUDA_VISIBLE_DEVICES=1
    echo "[GPU 1] Started at $(date '+%H:%M:%S')"
    sleep 5
    echo "[GPU 1] Finished at $(date '+%H:%M:%S')"
) > /tmp/gpu1_test.log 2>&1 &
PID1=$!

(
    export CUDA_VISIBLE_DEVICES=2
    echo "[GPU 2] Started at $(date '+%H:%M:%S')"
    sleep 5
    echo "[GPU 2] Finished at $(date '+%H:%M:%S')"
) > /tmp/gpu2_test.log 2>&1 &
PID2=$!

echo "All tasks started:"
echo "  GPU 0: PID $PID0"
echo "  GPU 1: PID $PID1"
echo "  GPU 2: PID $PID2"
echo ""
echo "Waiting for completion..."
wait

echo ""
echo "Results:"
cat /tmp/gpu0_test.log
cat /tmp/gpu1_test.log
cat /tmp/gpu2_test.log

