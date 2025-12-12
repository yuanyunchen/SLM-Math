"""
Unified configuration for all agents
Ensures first-round solver outputs are comparable across agents.

Core principles:
1. FIRST_ROUND_SOLVER_CONFIG: the first solver generation must use this config (greedy decoding)
   - Keeps first_try_accuracy comparable across agents.
   - do_sample=False enforces deterministic greedy decoding.
2. SUBSEQUENT_ROUND_CONFIG: later rounds may use different settings
   - Allows randomness to explore alternative solutions.
"""

# ============================================================================
# FIRST ROUND SOLVER CONFIG (unified - greedy)
# ============================================================================
# All agents must use these params for the first solver turn.
# Ensures first_try_accuracy is comparable across agents.
# Note: with do_sample=False, greedy decoding is used; temperature/top_p do not matter.
FIRST_ROUND_SOLVER_CONFIG = {
    'temperature': 0.7,         # Ignored during greedy decoding
    'do_sample': False,         # Greedy decoding (key setting)
    'top_p': 0.95,              # Ignored during greedy decoding
    'max_new_tokens': 2048,     # Allow long outputs
    'repetition_penalty': 1.15  # Mild repetition penalty
}

# ============================================================================
# SUBSEQUENT ROUND CONFIG (later rounds - allow randomness)
# ============================================================================
# Later rounds can use randomness to explore alternatives
SUBSEQUENT_ROUND_CONFIG = {
    'temperature': 0.7,         # Moderate randomness
    'do_sample': True,          # Enable sampling
    'top_p': 0.9,               # nucleus sampling
    'top_k': 50,                # top-k sampling
    'max_new_tokens': 2048,     # Allow long outputs
    'repetition_penalty': 1.15  # Mild repetition penalty
}

# ============================================================================
# MAJORITY VOTE CONFIG
# ============================================================================
# Majority vote specific config:
# - First run uses FIRST_ROUND_SOLVER_CONFIG (deterministic).
# - Later runs use randomized configs with different seeds.
MAJORITY_VOTE_FIRST_RUN_CONFIG = FIRST_ROUND_SOLVER_CONFIG.copy()

MAJORITY_VOTE_OTHER_RUNS_CONFIG = {
    'temperature': 0.7,         # Randomness to diversify answers
    'do_sample': True,
    'top_p': 0.95,
    'top_k': 50,
    'max_new_tokens': 2048,
    'repetition_penalty': 1.2
}

# ============================================================================
# CHECKER CONFIG
# ============================================================================
# Checkers typically need more deterministic output.
CHECKER_CONFIG = {
    'temperature': 0.0,         # Deterministic verification
    'do_sample': False,
    'top_p': 1.0,
    'top_k': 1,
    'max_new_tokens': 512,      # Shorter checker outputs
    'repetition_penalty': 1.1
}


def get_first_round_config():
    """Get deterministic first-round solver config."""
    return FIRST_ROUND_SOLVER_CONFIG.copy()


def get_subsequent_round_config():
    """Get later-round config (allows randomness)."""
    return SUBSEQUENT_ROUND_CONFIG.copy()


def get_checker_config():
    """Get checker config."""
    return CHECKER_CONFIG.copy()


# Documentation
"""
Usage guide:

1. solver_checker family:
   - Solver round 1: use FIRST_ROUND_SOLVER_CONFIG.
   - Solver rounds 2+: use SUBSEQUENT_ROUND_CONFIG.
   - Checker: use CHECKER_CONFIG.

2. majority_vote:
   - Run 1: MAJORITY_VOTE_FIRST_RUN_CONFIG (deterministic).
   - Runs 2+: MAJORITY_VOTE_OTHER_RUNS_CONFIG (seeded randomness).

3. plan_and_reflection:
   - First integrate/direct solve: FIRST_ROUND_SOLVER_CONFIG.
   - Later iterations: SUBSEQUENT_ROUND_CONFIG.

This ensures:
- first_try_accuracy is comparable across agents (deterministic first generation).
- Baselines for improved_rate/degraded_rate stay consistent.
"""
