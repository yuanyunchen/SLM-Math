"""
Training Scripts Index
Maps report sections to corresponding training scripts

This file documents which Python scripts correspond to which training methods
mentioned in the final report (finalreport_version3.tex).
"""

TRAINING_SCRIPTS_MAPPING = {
    # =========================================================================
    # Main Training Methods (Section 3, Table 2 in report)
    # =========================================================================
    
    "SFT (LoRA rank=16)": {
        "script": "train_sft_lora.py",
        "shell": "scripts/train_sft_lora.sh",
        "report_section": "Section 3.3, Line 157-158",
        "config": {
            "learning_rate": "1e-4",
            "lora_rank": 16,
            "epochs": 2,
            "batch_size": "128 (effective: 16 x 8)",
        },
        "results": {
            "GSM8K": "80.0%",
            "MATH-500": "67.2%"
        }
    },
    
    "SFT (Full)": {
        "script": "train_sft_full.py",
        "shell": "scripts/train_sft_full.sh",
        "report_section": "Section 3.3, Line 157-158",
        "config": {
            "learning_rate": "5e-5",
            "epochs": 2,
            "batch_size": "128 (effective: 16 x 8)",
        },
        "results": {
            "GSM8K": "81.6%",
            "MATH-500": "67.0%"
        }
    },
    
    "GRPO (RL)": {
        "script": "train_rl_grpo.py",
        "shell": "scripts/train_rl_grpo.sh",
        "report_section": "Section 3.4, Line 162-186",
        "config": {
            "learning_rate": "5e-6",
            "kl_coefficient": 0.05,
            "batch_size": "64 (effective: 16 x 4)",
            "num_return_sequences": 2,
            "epochs": 1,
        },
        "results": {
            "GSM8K": "82.4%",
            "MATH-500": "68.2%"
        }
    },
    
    # =========================================================================
    # Specialized Training (for agent components)
    # =========================================================================
    
    "Solver Training (with code)": {
        "script": "train_sft_solver.py",
        "shell": "scripts/train_sft_solver.sh",
        "report_section": "Section 5.2 (Emergent Code Generation)",
        "data": "data/sft_solver_training/run_1209_1011/",
        "description": "Trains solver to generate step-by-step reasoning with Python code"
    },
    
    "Verifier Training": {
        "script": "train_sft_verifier.py",
        "shell": "scripts/train_sft_verifier.sh",
        "report_section": "Section 3.5 (Solver-Verifier)",
        "data": "data/sft_verifier_training/",
        "description": "Trains verifier to check solutions and provide verdicts"
    },
    
    "Solver-Verifier RL": {
        "script": "train_rl_solver_verifier.py",
        "shell": "scripts/train_rl_solver_verifier_current_v2.sh",
        "report_section": "Table 2, Line 337 (Solver-Verifier SFT+RL)",
        "description": "Joint RL training for solver-verifier system",
        "results": {
            "GSM8K": "86.8%",
            "MATH-500": "68.8%"
        }
    },
    
    "Multi-Agent RL": {
        "script": "train_rl_solver_verifier_multi.py",
        "shell": "scripts/train_rl_solver_verifier_multi.sh",
        "report_section": "Section 3.5 (Agentic Workflows)",
        "description": "Multi-turn interaction training for solver-verifier"
    },
    
    # =========================================================================
    # Supporting/Alternative Methods
    # =========================================================================
    
    "Agent SFT": {
        "script": "train_agent_sft.py",
        "shell": "scripts/train_agent_sft.sh",
        "description": "General agent training with SFT"
    },
    
    "Code Feedback (SFT+RL)": {
        "script": "train_rl_code_feedback.py",
        "shell": "scripts/train_rl_code_feedback.sh",
        "alt_script": "train_rl_agent_with_code_feedback.py",
        "alt_shell": "scripts/train_rl_agent_with_code_feedback.sh",
        "report_section": "Table 2, Line 342 (Code Feedback SFT+RL)",
        "description": "Two-step generation with code execution feedback",
        "config": {
            "learning_rate": "5e-6",
            "kl_coefficient": 0.05,
            "epochs": 1,
            "batch_size": "16 (effective: 2 x 8)",
        },
        "results": {
            "GSM8K": "84.6%",
            "MATH-500": "67.8%"
        }
    },
    
    "Plain SFT": {
        "script": "train_sft_plain.py",
        "shell": "scripts/train_sft_plain.sh",
        "description": "Plain answer training without CoT"
    },
    
    "Distillation": {
        "script": "train_distill.py",
        "shell": "scripts/train_distill.sh",
        "description": "Knowledge distillation from larger models"
    },
    
    # =========================================================================
    # Base/Generic Scripts
    # =========================================================================
    
    "SFT Baseline": {
        "script": "train_sft_baseline.py",
        "shell": "scripts/train_sft_baseline_lora.sh",
        "description": "Generic SFT trainer (supports both LoRA and full)"
    },
    
    "RL Base": {
        "script": "train_rl_base.py",
        "shell": "scripts/train_rl_baseline.sh",
        "description": "Generic GRPO RL trainer"
    },
    
    "RL (Legacy)": {
        "script": "train_RL.py",
        "shell": "scripts/train_rl.sh",
        "description": "Legacy RL training script"
    },
}


# Quick lookup by report table
REPORT_TABLE_2_MAPPING = {
    "Base Model": None,  # No training
    "SFT (LoRA)": "train_sft_lora.py",
    "SFT (Full)": "train_sft_full.py",
    "GRPO": "train_rl_grpo.py",
    "Solver--Verifier (Base)": None,  # Inference only
    "Solver--Verifier (SFT Solver)": "train_sft_solver.py",
    "Solver--Verifier (SFT Verifier)": "train_sft_verifier.py",
    "Solver--Verifier (SFT Both)": ["train_sft_solver.py", "train_sft_verifier.py"],
    "Solver--Verifier (SFT+RL)": "train_rl_solver_verifier.py",
    "Code Feedback (Base)": None,  # Inference only
    "Code Feedback (SFT)": "train_sft_lora.py",  # Uses standard SFT
    "Code Feedback (SFT+RL)": "train_rl_agent_with_code_feedback.py",
}


def print_training_index():
    """Print formatted training scripts index"""
    print("=" * 80)
    print("TRAINING SCRIPTS INDEX")
    print("=" * 80)
    print()
    
    for method_name, info in TRAINING_SCRIPTS_MAPPING.items():
        print(f"Method: {method_name}")
        print(f"  Script: models/{info['script']}")
        if 'shell' in info:
            print(f"  Shell:  {info['shell']}")
        if 'report_section' in info:
            print(f"  Report: {info['report_section']}")
        if 'results' in info:
            print(f"  Results: GSM8K {info['results']['GSM8K']}, MATH {info['results']['MATH-500']}")
        if 'description' in info:
            print(f"  Desc:   {info['description']}")
        print()


if __name__ == "__main__":
    print_training_index()

