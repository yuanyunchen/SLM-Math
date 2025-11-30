# 8-GPU Agent评测 - 快速上手

## 🚀 一键启动
```bash
cd /root/autodl-tmp/SLM-Math
./scripts/run_all_agents_8gpu.sh
```

## 📊 监控进度
```bash
# 查看所有sessions概览
./scripts/list_agent_sessions.sh

# 进入特定agent查看实时输出
tmux attach -t agent_with_python_tools
# 按 Ctrl+b 然后 d 退出

# 查看所有tmux sessions
tmux ls
```

## ⏸️ 停止评测
```bash
./scripts/stop_all_agents.sh
```

## 🔍 查看结果
```bash
./scripts/check_agent_status.sh
```

---

## Session列表
每个agent有独立的tmux session：

| GPU | Agent Session名 | 进入命令 |
|-----|----------------|---------|
| 0 | agent_with_python_tools | `tmux attach -t agent_with_python_tools` |
| 1 | majority_vote | `tmux attach -t majority_vote` |
| 2 | plan_and_reflection | `tmux attach -t plan_and_reflection` |
| 3 | solver_checker_chat | `tmux attach -t solver_checker_chat` |
| 4 | solver_checker_stateless | `tmux attach -t solver_checker_stateless` |
| 5 | solver_checker_summarizer | `tmux attach -t solver_checker_summarizer` |
| 6 | solver_checker_summarizer_chat | `tmux attach -t solver_checker_summarizer_chat` |
| 7 | solver_checker_with_tools | `tmux attach -t solver_checker_with_tools` |
| 0* | solver_checker (base) | `tmux attach -t solver_checker` |

**注**: GPU 0* 表示与 agent_with_python_tools 共享GPU 0  
**注**: `solver_checker` 是 base/stateless 版本

---

## 常用命令
```bash
# 启动主要8个agents
./scripts/run_all_agents_8gpu.sh

# 启动solver_checker_base (单独)
./scripts/run_solver_checker_base.sh

# 查看sessions
./scripts/list_agent_sessions.sh

# 查看结果
./scripts/check_agent_status.sh

# 停止所有
./scripts/stop_all_agents.sh

# 进入session
tmux attach -t <agent_name>

# 退出session (不停止任务)
Ctrl+b 然后 d
```

---

详细文档: `scripts/MULTI_GPU_EVAL_README.md`

