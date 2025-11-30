# Multi-GPU Agent Evaluation Guide

## Quick Start

### 一键启动所有评测
```bash
cd /root/autodl-tmp/SLM-Math
./scripts/run_all_agents_8gpu.sh
```

### 一键停止所有评测
```bash
./scripts/stop_all_agents.sh
```

### 监控进度
```bash
# 方法1: 查看所有agent sessions概览
./scripts/list_agent_sessions.sh

# 方法2: 进入特定agent的tmux session查看实时输出
tmux attach -t agent_with_python_tools    # 查看agent_with_python_tools
tmux attach -t majority_vote               # 查看majority_vote
tmux attach -t solver_checker_chat         # 查看solver_checker_chat
# ... 等等

# 方法3: 列出所有tmux sessions
tmux ls

# 方法4: 快速检查评测结果（不进入tmux）
./scripts/check_agent_status.sh
```

---

## 评测配置

### 模型和数据集
- **模型**: Qwen2.5-Math-1.5B
- **数据集**: GSM8K (500条) + MATH (500条)
- **轮次名**: full_eval

### 超参数
- **Max Iterations**: 3 (多轮agent)
- **Num Runs**: 3 (majority_vote)
- **Max Subproblems**: 3 (plan_and_reflection)

### Agent-GPU分配
```
GPU 0: agent_with_python_tools
GPU 1: majority_vote
GPU 2: plan_and_reflection
GPU 3: solver_checker_chat
GPU 4: solver_checker_stateless
GPU 5: solver_checker_summarizer
GPU 6: solver_checker_summarizer_chat
GPU 7: solver_checker_with_tools
```

### 运行顺序
每个GPU上的agent会按顺序运行：
1. GSM8K (500条)
2. MATH (500条)

---

## Tmux使用指南

### Session架构
**每个agent有独立的tmux session，以agent名称命名**

```bash
# 列出所有agent sessions
tmux ls

# 进入特定agent session
tmux attach -t agent_with_python_tools
tmux attach -t majority_vote
tmux attach -t plan_and_reflection
tmux attach -t solver_checker_chat
# ... 等等
```

### 基本操作
| 快捷键 | 功能 |
|--------|------|
| `Ctrl+b` 然后 `d` | 退出当前session（任务继续运行） |
| `Ctrl+b` 然后 `[` | 进入滚动模式查看历史输出（q退出） |
| `Ctrl+b` 然后 `?` | 显示所有快捷键帮助 |

### Session命名规则
```
agent_with_python_tools       -> GPU 0
majority_vote                 -> GPU 1
plan_and_reflection           -> GPU 2
solver_checker_chat           -> GPU 3
solver_checker_stateless      -> GPU 4
solver_checker_summarizer     -> GPU 5
solver_checker_summarizer_chat -> GPU 6
solver_checker_with_tools     -> GPU 7
```

---

## 结果查看

### 结果目录结构
```
results/
├── full_eval_Qwen2.5-Math-1.5B_gsm8k_500_1124/
│   ├── logging.log       # 完整日志
│   ├── metrics.csv       # 性能指标表
│   ├── summary.txt       # 人类可读总结
│   └── answer.json       # 详细预测结果
└── full_eval_Qwen2.5-Math-1.5B_math_500_1124/
    └── ...
```

### 快速查看汇总
```bash
# 查看所有agent的最新结果
./scripts/check_agent_status.sh

# 查看特定agent的结果
cat results/full_eval_*agent_with_python_tools*/summary.txt
```

---

## 常见问题

### Q1: 如何停止所有评测？
```bash
# 方法1: 停止特定agent
tmux attach -t agent_with_python_tools
# 然后按 Ctrl+C

# 方法2: 直接杀掉所有eval_agent进程
pkill -f eval_agent

# 方法3: 杀掉所有agent sessions
tmux kill-session -t agent_with_python_tools
tmux kill-session -t majority_vote
# ... 或者用循环
for session in agent_with_python_tools majority_vote plan_and_reflection solver_checker_chat solver_checker_stateless solver_checker_summarizer solver_checker_summarizer_chat solver_checker_with_tools; do
    tmux kill-session -t $session 2>/dev/null
done
```

### Q2: 如何重新运行某个agent？
```bash
# 进入tmux，找到对应GPU窗口，重新运行命令
# 或者直接运行：
export CUDA_VISIBLE_DEVICES=0  # 指定GPU
python -m evaluation.eval_agent \
    --model "Qwen2.5-Math-1.5B" \
    --agent "agent_with_python_tools" \
    --round "full_eval" \
    --dataset "gsm8k" \
    --count 500 \
    --max_iterations 3 \
    --detailed "false" \
    --save_interval 50
```

### Q3: GPU内存不足怎么办？
```bash
# 检查GPU状态
nvidia-smi

# 如果某个GPU内存占用过高，可以：
# 1. 停止该GPU上的任务 (Ctrl+C)
# 2. 清理缓存
python -c "import torch; torch.cuda.empty_cache()"
# 3. 重新启动任务
```

### Q4: 如何查看实时准确率？
```bash
# 进入对应GPU窗口，日志会显示每50条的统计
# 或者查看 logging.log
tail -f results/full_eval_*/logging.log
```

---

## 预计运行时间

基于Qwen2.5-Math-1.5B模型和8个GPU并行：
- **GSM8K 500条**: 约30-60分钟/agent
- **MATH 500条**: 约30-60分钟/agent
- **总时间**: 约1-2小时（所有agent并行完成）

---

## 注意事项

1. **确保conda环境已激活**: 脚本会自动激活 `slm_math` 环境
2. **检查磁盘空间**: 16个结果目录，每个约100-500MB
3. **监控GPU温度**: 长时间运行时注意散热
4. **不要关闭终端**: 即使退出tmux，进程也会继续运行
5. **定期检查状态**: 使用 `check_agent_status.sh` 监控进度

---

## 技术支持

如有问题，检查：
1. 日志文件: `results/full_eval_*/logging.log`
2. GPU状态: `nvidia-smi`
3. 进程状态: `ps aux | grep eval_agent`
4. Tmux会话: `tmux ls`

