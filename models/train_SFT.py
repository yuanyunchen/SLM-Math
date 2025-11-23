"""
Supervised Fine-Tuning (SFT) training script for Qwen2.5-Math models.
Trains models to generate Chain-of-Thought reasoning in DeepSeek-R1 style format.

Usage:
    python -m models.train_SFT --config configs/sft_config.yaml
    python -m models.train_SFT --config configs/sft_config.yaml --model_name Qwen2.5-Math-1.5B
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import yaml
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, prepare_model_for_kbit_training

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.train_utils import (
    CoTDataConfig,
    prepare_dataset,
    tokenize_function,
    compute_data_statistics,
)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


class LossLoggingCallback(TrainerCallback):
    """Callback to log training and evaluation losses to a file."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.loss_file = output_dir / "loss_history.csv"
        self.initialized = False
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize loss logging file."""
        if not self.initialized:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.loss_file, 'w') as f:
                f.write("step,epoch,train_loss,eval_loss,learning_rate\n")
            self.initialized = True
            logger.info(f"Loss logging initialized: {self.loss_file}")
    
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs: Dict[str, float], **kwargs):
        """Log losses after each logging step."""
        if not self.initialized:
            return
        
        step = state.global_step
        epoch = state.epoch if state.epoch is not None else 0
        train_loss = logs.get('loss', '')
        eval_loss = logs.get('eval_loss', '')
        lr = logs.get('learning_rate', '')
        
        # Write to file
        with open(self.loss_file, 'a') as f:
            f.write(f"{step},{epoch:.4f},{train_loss},{eval_loss},{lr}\n")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def get_optimal_device_and_dtype():
    """Determine optimal device and dtype for training.
    
    Returns:
        Tuple of (device_map, torch_dtype, device_name)
    """
    if torch.cuda.is_available():
        # CUDA available - prefer bfloat16 if supported (Ampere+)
        if torch.cuda.is_bf16_supported():
            return "auto", torch.bfloat16, "CUDA (BF16)"
        else:
            return "auto", torch.float16, "CUDA (FP16)"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon MPS - use float32 for stability
        return None, torch.float32, "MPS (Apple Silicon)"
    else:
        # CPU fallback
        return None, torch.float32, "CPU"


def setup_model_and_tokenizer(
    model_config: Dict,
    lora_config: Dict,
    qlora_config: Dict = None,
    use_lora: bool = True,
    use_qlora: bool = False
):
    """Load model and tokenizer, optionally with LoRA or QLoRA.
    Supports CUDA, MPS (Apple Silicon), and CPU.
    
    Args:
        model_config: Model configuration dictionary
        lora_config: LoRA configuration dictionary
        qlora_config: QLoRA configuration dictionary
        use_lora: Whether to use LoRA
        use_qlora: Whether to use QLoRA (4-bit quantization + LoRA)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model_path = Path(project_root) / model_config['path']
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    logger.info(f"  Model name: {model_config['name']}")
    logger.info(f"  LoRA enabled: {use_lora}")
    logger.info(f"  QLoRA enabled: {use_qlora}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=model_config.get('trust_remote_code', True),
        use_fast=True,
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    
    # Add special tokens for thinking format if not present
    special_tokens = {'additional_special_tokens': ['<think>', '</think>']}
    num_added = tokenizer.add_special_tokens(special_tokens)
    if num_added > 0:
        logger.info(f"Added {num_added} special tokens: <think>, </think>")
    
    # Setup QLoRA quantization config if enabled
    quantization_config = None
    if use_qlora and qlora_config and qlora_config.get('enabled', False):
        if not torch.cuda.is_available():
            logger.warning("QLoRA requires CUDA. Falling back to regular LoRA on current device.")
            use_qlora = False
        else:
            logger.info("Setting up QLoRA (4-bit quantization):")
            logger.info(f"  Bits: {qlora_config.get('bits', 4)}")
            logger.info(f"  Quant type: {qlora_config.get('quant_type', 'nf4')}")
            logger.info(f"  Double quant: {qlora_config.get('use_double_quant', True)}")
            logger.info(f"  Compute dtype: {qlora_config.get('compute_dtype', 'bfloat16')}")
            
            # Map compute dtype string to torch dtype
            compute_dtype_map = {
                'float16': torch.float16,
                'bfloat16': torch.bfloat16,
                'float32': torch.float32,
            }
            compute_dtype = compute_dtype_map.get(
                qlora_config.get('compute_dtype', 'bfloat16'), 
                torch.bfloat16
            )
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=(qlora_config.get('bits', 4) == 4),
                load_in_8bit=(qlora_config.get('bits', 4) == 8),
                bnb_4bit_quant_type=qlora_config.get('quant_type', 'nf4'),
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=qlora_config.get('use_double_quant', True),
            )
    
    # Determine optimal device and dtype
    device_map, torch_dtype, device_name = get_optimal_device_and_dtype()
    
    # Override for QLoRA (must use CUDA with device_map="auto")
    if use_qlora and quantization_config:
        device_map = "auto"
        device_name = "CUDA (QLoRA)"
    
    logger.info(f"  Device: {device_name}")
    logger.info(f"  Device map: {device_map if device_map else 'None (manual placement)'}")
    if not use_qlora:
        logger.info(f"  Torch dtype: {torch_dtype}")
    
    # Load model with device-specific settings
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=model_config.get('trust_remote_code', True),
            torch_dtype=torch_dtype if not use_qlora else None,
            device_map=device_map,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        )
        
        # If using MPS or CPU without device_map, manually place model
        if device_map is None and not use_qlora:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                model = model.to("mps")
                logger.info("  Model moved to MPS device")
            else:
                model = model.to("cpu")
                logger.info("  Model on CPU")
    
    except Exception as e:
        logger.warning(f"  Failed to load with optimal settings: {e}")
        if use_qlora:
            logger.error("  QLoRA loading failed. Please check CUDA and bitsandbytes installation.")
            raise
        logger.warning("  Falling back to CPU with float32...")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=model_config.get('trust_remote_code', True),
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model = model.to("cpu")
        logger.info("  Model loaded on CPU")
    
    # Resize token embeddings if we added special tokens
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized model embeddings to {len(tokenizer)}")
    
    # Apply LoRA or QLoRA if enabled
    if use_lora or use_qlora:
        # Prepare model for k-bit training if using QLoRA
        if use_qlora:
            logger.info("Preparing model for QLoRA (k-bit training)...")
            model = prepare_model_for_kbit_training(model)
        
        logger.info("Applying LoRA configuration:")
        logger.info(f"  Rank (r): {lora_config['r']}")
        logger.info(f"  Alpha: {lora_config['alpha']}")
        logger.info(f"  Dropout: {lora_config['dropout']}")
        logger.info(f"  Target modules: {lora_config['target_modules']}")
        
        peft_config = LoraConfig(
            r=lora_config['r'],
            lora_alpha=lora_config['alpha'],
            lora_dropout=lora_config['dropout'],
            target_modules=lora_config['target_modules'],
            bias=lora_config.get('bias', 'none'),
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        if use_qlora:
            logger.info("✓ QLoRA setup complete (4-bit quantized model + LoRA)")
    else:
        logger.info("Training with full fine-tuning (no LoRA)")
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable params: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")
    
    return model, tokenizer


def create_training_arguments(
    config: Dict,
    output_dir: Path
) -> TrainingArguments:
    """Create TrainingArguments from configuration.
    
    Args:
        config: Full configuration dictionary
        output_dir: Output directory for checkpoints
    
    Returns:
        TrainingArguments instance
    """
    train_config = config['training']
    opt_config = config['optimizer']
    precision_config = config['precision']
    memory_config = config['memory']
    checkpoint_config = config['checkpoint']
    logging_config = config['logging']
    eval_config = config['evaluation']
    misc_config = config['misc']
    
    # Create logging directory
    log_dir = Path(train_config.get('logging_dir', 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    args = TrainingArguments(
        # Output
        output_dir=str(output_dir),
        
        # Training hyperparameters
        num_train_epochs=train_config['num_train_epochs'],
        max_steps=train_config.get('max_steps', -1),
        per_device_train_batch_size=train_config['per_device_train_batch_size'],
        per_device_eval_batch_size=eval_config.get('per_device_eval_batch_size', train_config['per_device_eval_batch_size']),
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        
        # Learning rate
        learning_rate=train_config['learning_rate'],
        lr_scheduler_type=train_config['lr_scheduler_type'],
        warmup_ratio=train_config['warmup_ratio'],
        
        # Optimizer
        optim=opt_config['type'],
        adam_beta1=opt_config.get('adam_beta1', 0.9),
        adam_beta2=opt_config.get('adam_beta2', 0.999),
        adam_epsilon=opt_config.get('adam_epsilon', 1e-8),
        weight_decay=train_config['weight_decay'],
        max_grad_norm=train_config['max_grad_norm'],
        
        # Precision
        fp16=precision_config['fp16'],
        bf16=precision_config['bf16'],
        fp16_full_eval=precision_config.get('fp16_full_eval', False),
        bf16_full_eval=precision_config.get('bf16_full_eval', False),
        
        # Memory
        gradient_checkpointing=memory_config['gradient_checkpointing'],
        gradient_checkpointing_kwargs=memory_config.get('gradient_checkpointing_kwargs', None),
        
        # Checkpointing
        save_strategy=checkpoint_config['save_strategy'],
        save_steps=checkpoint_config.get('save_steps', 500),
        save_total_limit=checkpoint_config.get('save_total_limit', 3),
        load_best_model_at_end=checkpoint_config.get('load_best_model_at_end', True),
        metric_for_best_model=checkpoint_config.get('metric_for_best_model', 'eval_loss'),
        greater_is_better=checkpoint_config.get('greater_is_better', False),
        
        # Logging
        logging_dir=str(log_dir),
        logging_strategy=logging_config['logging_strategy'],
        logging_steps=logging_config['logging_steps'],
        report_to=logging_config.get('report_to', ['tensorboard']),
        logging_first_step=True,
        log_level=logging_config.get('log_level', 'info'),
        log_on_each_node=logging_config.get('log_on_each_node', True),
        
        # Evaluation
        eval_strategy=eval_config.get('eval_strategy', eval_config.get('evaluation_strategy', 'steps')),
        eval_steps=eval_config.get('eval_steps', 500),
        eval_accumulation_steps=eval_config.get('eval_accumulation_steps', None),
        
        # Miscellaneous
        seed=misc_config.get('seed', 42),
        data_seed=misc_config.get('data_seed', 42),
        use_cpu=misc_config.get('use_cpu', misc_config.get('no_cuda', False)),
        dataloader_num_workers=misc_config.get('dataloader_num_workers', 4),
        dataloader_pin_memory=misc_config.get('dataloader_pin_memory', True),
        remove_unused_columns=misc_config.get('remove_unused_columns', True),
        label_smoothing_factor=misc_config.get('label_smoothing_factor', 0.0),
        group_by_length=misc_config.get('group_by_length', False),
        length_column_name=misc_config.get('length_column_name', 'length'),
        resume_from_checkpoint=misc_config.get('resume_from_checkpoint', None),
        ignore_data_skip=misc_config.get('ignore_data_skip', False),
        disable_tqdm=misc_config.get('disable_tqdm', False),
        
        # Hub
        push_to_hub=misc_config.get('push_to_hub', False),
        hub_model_id=misc_config.get('hub_model_id', None),
        hub_token=misc_config.get('hub_token', None),
    )
    
    return args


def save_training_config(config: Dict, output_dir: Path):
    """Save training configuration to output directory.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory
    """
    config_file = output_dir / "training_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved training configuration to {config_file}")


def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train Qwen2.5-Math with SFT on CoT data")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/sft_config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        help='Override model name from config'
    )
    parser.add_argument(
        '--data_files',
        type=str,
        nargs='+',
        default=None,
        help='Override training data files from config'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Override output directory from config'
    )
    parser.add_argument(
        '--no_lora',
        action='store_true',
        help='Disable LoRA (use full fine-tuning)'
    )
    parser.add_argument(
        '--qlora',
        action='store_true',
        help='Enable QLoRA (4-bit quantization + LoRA) - requires CUDA'
    )
    args = parser.parse_args()
    
    # Load configuration
    logger.info("="*80)
    logger.info("SFT Training for Qwen2.5-Math - Chain-of-Thought Reasoning")
    logger.info("="*80)
    
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.model_name:
        config['model']['name'] = args.model_name
        logger.info(f"Override model name: {args.model_name}")
    
    if args.data_files:
        config['data']['train_files'] = args.data_files
        logger.info(f"Override data files: {args.data_files}")
    
    # Handle LoRA/QLoRA options
    use_qlora = args.qlora or (config.get('qlora', {}).get('enabled', False) and not args.no_lora)
    use_lora = (config['lora']['enabled'] or use_qlora) and not args.no_lora
    
    if use_qlora:
        logger.info("QLoRA mode enabled")
        if not torch.cuda.is_available():
            logger.warning("QLoRA requires CUDA, but CUDA is not available!")
            logger.warning("Falling back to regular LoRA")
            use_qlora = False
    
    # Setup output directory
    run_name = config['training'].get('run_name', 'sft_run')
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_base = Path(config['training']['output_dir'])
    output_dir = output_base / f"{run_name}_{timestamp}"
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Save configuration
    save_training_config(config, output_dir)
    
    # Load model and tokenizer
    logger.info("\n" + "="*80)
    logger.info("Loading Model and Tokenizer")
    logger.info("="*80)
    model, tokenizer = setup_model_and_tokenizer(
        config['model'],
        config['lora'],
        qlora_config=config.get('qlora', {}),
        use_lora=use_lora,
        use_qlora=use_qlora
    )
    
    # Prepare datasets
    logger.info("\n" + "="*80)
    logger.info("Preparing Datasets")
    logger.info("="*80)
    
    data_config = CoTDataConfig(
        max_seq_length=config['data']['max_seq_length'],
        train_split_ratio=config['data']['train_split_ratio'],
    )
    
    dataset_dict = prepare_dataset(
        data_files=config['data']['train_files'],
        config=data_config,
        dataset_name=config['data']['dataset_name']
    )
    
    logger.info(f"Dataset splits:")
    logger.info(f"  Train: {len(dataset_dict['train'])} samples")
    logger.info(f"  Validation: {len(dataset_dict['validation'])} samples")
    
    # Compute statistics
    compute_data_statistics(dataset_dict['train'], tokenizer)
    
    # Tokenize datasets
    logger.info("\nTokenizing datasets...")
    tokenized_datasets = dataset_dict.map(
        lambda examples: tokenize_function(
            examples,
            tokenizer,
            max_seq_length=config['data']['max_seq_length']
        ),
        batched=True,
        num_proc=config['data'].get('num_workers', 4),
        remove_columns=dataset_dict['train'].column_names,
        desc="Tokenizing"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Create training arguments
    logger.info("\n" + "="*80)
    logger.info("Training Configuration")
    logger.info("="*80)
    
    training_args = create_training_arguments(config, output_dir)
    
    # Calculate effective batch size
    effective_batch_size = (
        training_args.per_device_train_batch_size *
        training_args.gradient_accumulation_steps *
        max(1, training_args.n_gpu)
    )
    
    logger.info(f"Effective batch size: {effective_batch_size}")
    logger.info(f"  per_device_batch_size: {training_args.per_device_train_batch_size}")
    logger.info(f"  gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
    logger.info(f"  num_gpus: {max(1, training_args.n_gpu)}")
    
    # Estimate training steps
    num_train_samples = len(tokenized_datasets['train'])
    steps_per_epoch = num_train_samples // effective_batch_size
    total_steps = steps_per_epoch * training_args.num_train_epochs
    
    logger.info(f"\nTraining estimates:")
    logger.info(f"  Samples per epoch: {num_train_samples}")
    logger.info(f"  Steps per epoch: {steps_per_epoch}")
    logger.info(f"  Total epochs: {training_args.num_train_epochs}")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Save every: {training_args.save_steps} steps")
    logger.info(f"  Eval every: {training_args.eval_steps} steps")
    
    # Create callbacks
    callbacks = [
        LossLoggingCallback(output_dir)
    ]
    
    # Create Trainer
    logger.info("\n" + "="*80)
    logger.info("Initializing Trainer")
    logger.info("="*80)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        processing_class=tokenizer,  # transformers 5.0: tokenizer -> processing_class
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # Start training
    logger.info("\n" + "="*80)
    logger.info("Starting Training")
    logger.info("="*80)
    logger.info(f"Training will start now...")
    logger.info(f"Monitor with: tensorboard --logdir {training_args.logging_dir}")
    logger.info("="*80 + "\n")
    
    try:
        train_result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        
        # Save final model
        logger.info("\n" + "="*80)
        logger.info("Training Completed Successfully!")
        logger.info("="*80)
        
        final_model_path = output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        logger.info(f"Saved final model to: {final_model_path}")
        
        # Save training metrics
        metrics = train_result.metrics
        metrics_file = output_dir / "training_metrics.json"
        trainer.save_metrics("train", metrics)
        logger.info(f"Saved training metrics to: {metrics_file}")
        
        # Log final metrics
        logger.info("\nFinal Training Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
        
        # Save training state
        trainer.save_state()
        
        logger.info("\n" + "="*80)
        logger.info("All outputs saved to:")
        logger.info(f"  {output_dir}")
        logger.info("="*80)
        
    except KeyboardInterrupt:
        logger.warning("\n" + "="*80)
        logger.warning("Training interrupted by user!")
        logger.warning("="*80)
        logger.info("Saving checkpoint...")
        trainer.save_model(str(output_dir / "interrupted_checkpoint"))
        logger.info(f"Checkpoint saved to: {output_dir / 'interrupted_checkpoint'}")
        sys.exit(1)
    
    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("Training failed with error:")
        logger.error("="*80)
        logger.error(str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

