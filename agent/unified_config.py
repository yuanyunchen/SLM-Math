"""
Unified configuration for all agents
统一的agent配置，确保第一轮solver输出可以公平比较

核心原则：
1. FIRST_ROUND_SOLVER_CONFIG: 第一次solver生成必须使用此配置（贪婪解码）
   - 保证所有agent的first_try_accuracy可比较
   - do_sample=False 使用贪婪解码，确保确定性输出
   
2. SUBSEQUENT_ROUND_CONFIG: 后续轮次可以使用不同配置
   - 允许一定随机性以探索不同解法
"""

# ============================================================================
# FIRST ROUND SOLVER CONFIG (统一配置 - 贪婪解码)
# ============================================================================
# 所有agent的第一轮solver必须使用这些参数
# 确保first_try_accuracy在不同agent间可比较
# 注意: do_sample=False 时使用贪婪解码，temperature/top_p 不影响输出
FIRST_ROUND_SOLVER_CONFIG = {
    'temperature': 0.7,         # 贪婪解码时不生效
    'do_sample': False,         # 贪婪解码（核心参数）
    'top_p': 0.95,              # 贪婪解码时不生效
    'max_new_tokens': 2048,     # 足够长的输出
    'repetition_penalty': 1.15  # 适度的重复惩罚
}

# ============================================================================
# SUBSEQUENT ROUND CONFIG (后续轮次配置 - 允许随机性)
# ============================================================================
# 后续轮次可以使用随机性来探索不同解法
SUBSEQUENT_ROUND_CONFIG = {
    'temperature': 0.7,         # 中等随机性
    'do_sample': True,          # 使用采样
    'top_p': 0.9,               # nucleus sampling
    'top_k': 50,                # top-k sampling
    'max_new_tokens': 2048,     # 足够长的输出
    'repetition_penalty': 1.15  # 适度的重复惩罚
}

# ============================================================================
# MAJORITY VOTE CONFIG
# ============================================================================
# majority_vote特殊配置：
# - 第一个run使用FIRST_ROUND_SOLVER_CONFIG（确定性）
# - 后续runs使用不同seed的随机配置
MAJORITY_VOTE_FIRST_RUN_CONFIG = FIRST_ROUND_SOLVER_CONFIG.copy()

MAJORITY_VOTE_OTHER_RUNS_CONFIG = {
    'temperature': 0.7,         # 需要随机性来获得不同答案
    'do_sample': True,
    'top_p': 0.95,
    'top_k': 50,
    'max_new_tokens': 2048,
    'repetition_penalty': 1.2
}

# ============================================================================
# CHECKER CONFIG
# ============================================================================
# Checker通常需要更确定性的输出
CHECKER_CONFIG = {
    'temperature': 0.0,         # 确定性验证
    'do_sample': False,
    'top_p': 1.0,
    'top_k': 1,
    'max_new_tokens': 512,      # checker输出较短
    'repetition_penalty': 1.1
}


def get_first_round_config():
    """获取第一轮solver配置（确定性）"""
    return FIRST_ROUND_SOLVER_CONFIG.copy()


def get_subsequent_round_config():
    """获取后续轮次配置（允许随机性）"""
    return SUBSEQUENT_ROUND_CONFIG.copy()


def get_checker_config():
    """获取checker配置"""
    return CHECKER_CONFIG.copy()


# 说明文档
"""
使用指南：

1. solver_checker系列agent:
   - 第1轮solver: 使用 FIRST_ROUND_SOLVER_CONFIG
   - 第2+轮solver: 使用 SUBSEQUENT_ROUND_CONFIG
   - checker: 使用 CHECKER_CONFIG

2. majority_vote:
   - 第1个run: 使用 MAJORITY_VOTE_FIRST_RUN_CONFIG（确定性）
   - 第2+个run: 使用 MAJORITY_VOTE_OTHER_RUNS_CONFIG（带seed的随机）

3. plan_and_reflection:
   - 第1次integrate/直接解题: 使用 FIRST_ROUND_SOLVER_CONFIG
   - 后续迭代: 使用 SUBSEQUENT_ROUND_CONFIG

这样保证：
- first_try_accuracy 在所有agent间可比较（都是确定性的第一次生成）
- improved_rate/degraded_rate 的baseline一致
"""
