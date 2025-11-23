# Multi-Agent Workflow

从 `dev/multiagent` 分支提取的多智能体评估系统（Solver-Checker 迭代工作流）。

## 📁 文件说明

```
agent/
├── README.md                       # 本文件
├── run_multi_agent_eval.sh        # ⭐ 运行脚本
├── analyze_results.py              # ⭐ 结果分析工具
│
└── 以下是 multiagent 版本的核心模块（对应主项目的修改版）:
    ├── eval_pipeline_multiagent.py     # evaluation/eval_pipeline.py
    ├── prompt_utils_multiagent.py      # utils/prompt_utils.py
    └── inference_multiagent.py         # models/inference.py
```

## 🚀 快速开始

### 1. 运行评估

```bash
cd agent
./run_multi_agent_eval.sh
```

### 2. 修改配置（可选）

编辑 `run_multi_agent_eval.sh`:

```bash
MODEL="Qwen2.5-Math-1.5B"      # 模型名称
DATASET="gsm8k"                # 数据集 (gsm8k/math)
COUNT=20                       # 样本数量 (0=全部)
```

### 3. 分析结果

```bash
python analyze_results.py
```

## 💡 Multi-Agent 工作流

```
问题
 ↓
Solver 生成答案
 ↓
Checker 验证 → 判断: CORRECT / INCORRECT / UNCLEAR
 ↓
如果 CORRECT: 完成 ✓
如果不正确: 提供反馈 → Solver 重试 (最多5次)
```

## 📊 分析报告 - 4类案例

运行 `python analyze_results.py` 后会生成 CSV 报告，包含：

| 类型 | 说明 | 意义 |
|------|------|------|
| **Type 1: Improved** | 第一次错误 → 后来正确 | ✅ 系统有效 |
| **Type 2: Degraded** | 第一次正确 → 后来错误 | ⚠️ 需要改进 |
| **Type 3: First Try** | 一次成功 | 🎯 效率高 |
| **Type 4: Unnecessary** | 正确但 Checker 未识别 | 🔍 可优化 |

## 🔧 关键参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `MODEL` | Solver 模型 | Qwen2.5-Math-1.5B |
| `CHECKER_MODEL` | Checker 模型（可选） | Qwen2.5-Math-1.5B-Instruct |
| `DATASET` | 数据集 | gsm8k, math |
| `COUNT` | 样本数 | 20 (0=全部) |
| `MODE` | 必须为 multi_agent | multi_agent |

## 📈 输出文件

### 评估结果
```
results/<ROUND>_<MODEL>_<DATASET>_<COUNT>_<MMDD>/
├── log/*.log          # 详细日志（每次迭代对话）
├── metrics.csv        # 准确率等指标
├── summary.txt        # 摘要
└── answer.json        # 详细答案
```

### 分析报告
```
summary/<dataset>_<model>_<count>problems_<timestamp>_analysis.csv
```

## 🆚 与 Main 分支的区别

| 特性 | Main 分支 | Multiagent 分支 |
|------|-----------|-----------------|
| 评估模式 | standard, thinking | **+ multi_agent** |
| 迭代机制 | 无 | **Solver-Checker 循环** |
| 分析工具 | 基础 | **4类案例自动分析** |

## 💻 使用示例

### 示例 1: 基础评估

```bash
./run_multi_agent_eval.sh
```

### 示例 2: 自定义参数

```bash
# 编辑脚本
nano run_multi_agent_eval.sh

# 修改:
MODEL="Qwen3-1.7B"
COUNT=100
DATASET="math"

# 运行
./run_multi_agent_eval.sh
```

### 示例 3: 使用不同的 Checker

```bash
# 在 run_multi_agent_eval.sh 中取消注释:
CHECKER_MODEL="Qwen2.5-Math-1.5B-Instruct"
```

## 📝 核心文件说明

### Python 模块

| 文件 | 来源 | 主要修改 |
|------|------|---------|
| `eval_pipeline_multiagent.py` | evaluation/eval_pipeline.py | 添加 multi_agent 模式 |
| `prompt_utils_multiagent.py` | utils/prompt_utils.py | 添加 Solver/Checker prompt |
| `inference_multiagent.py` | models/inference.py | 优化推理参数 |
| `analyze_results.py` | 新增 | 4类案例分析工具 |

### 关键函数（prompt_utils_multiagent.py）

```python
format_prompt_solver(question, checker_feedback=None)    # Solver prompt
format_prompt_checker(question, solver_response)         # Checker prompt
parse_checker_verdict(checker_response)                  # 提取判断
parse_checker_tip(checker_response)                      # 提取反馈
```

## 🐛 常见问题

### Q: 如何运行？
```bash
./run_multi_agent_eval.sh
```

### Q: 结果在哪里？
- 评估: `../results/<最新目录>/`
- 分析: `../summary/*.csv`

### Q: 如何分析？
```bash
python analyze_results.py
```

### Q: Checker 总是返回 UNCLEAR？
优化 `prompt_utils_multiagent.py` 中的 `format_prompt_checker()` 函数。

## 🎯 优化建议

根据分析报告：

1. **Improved Cases 多** → 系统有效，继续使用
2. **Degraded Cases 多** → 优化 Checker prompt
3. **Unnecessary Iterations 多** → 优化 Checker 识别能力
4. **First Try Rate 低** → 优化 Solver prompt

## ✅ 检查清单

运行前:
- [ ] 模型文件在 `../pretrained_models/`
- [ ] 数据集在 `../data/`
- [ ] 已配置 `run_multi_agent_eval.sh`

运行后:
- [ ] 查看 `../results/<dir>/summary.txt`
- [ ] 运行 `python analyze_results.py`
- [ ] 查看 4类案例统计

---

**快速开始**: `./run_multi_agent_eval.sh`  
**来源**: `dev/multiagent` 分支
