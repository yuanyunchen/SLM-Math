"""
Standardized Generation Configuration
All agents and evaluation scripts should use these parameters for fair comparison.
"""

# ============================================================================
# GENERATION PARAMETERS (Standardized across all agents and eval.py)
# ============================================================================

# Maximum tokens to generate
MAX_NEW_TOKENS = 2048

# Temperature for sampling (0.0 = greedy, higher = more random)
# 0.7 is a good balance for math reasoning
TEMPERATURE = 0.7

# Whether to use sampling (vs greedy decoding)
DO_SAMPLE = True

# Nucleus sampling: only consider tokens with cumulative probability <= top_p
TOP_P = 0.95

# Penalty for repeating tokens (1.0 = no penalty, higher = more penalty)
REPETITION_PENALTY = 1.15

# ============================================================================
# AGENT-SPECIFIC OVERRIDES (if needed for specific roles)
# ============================================================================

# Checker-specific parameters (for verification, can be more conservative)
CHECKER_MAX_TOKENS = 512
CHECKER_TEMPERATURE = 0.3
CHECKER_TOP_P = 0.9
CHECKER_REPETITION_PENALTY = 1.2

# Summary-specific parameters (for condensing information)
SUMMARY_MAX_TOKENS = 256
SUMMARY_TEMPERATURE = 0.5
SUMMARY_TOP_P = 0.9
SUMMARY_REPETITION_PENALTY = 1.1

# ============================================================================
# COMPATIBILITY NOTE
# ============================================================================
# For backward compatibility with existing code that uses old parameter names:
MAX_TOKEN = MAX_NEW_TOKENS  # Legacy alias


