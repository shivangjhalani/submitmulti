import argparse
import json
import logging
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.distributed as dist
import torch.utils.checkpoint as checkpoint_module
from transformers import AutoModelForCausalLM, TrainingArguments
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
from multicoco.config import MultiCoCoConfig, TrainingMode
from multicoco.constants import COCONUT_SPECIAL_TOKENS, DEFAULT_BATCH_SIZE, DEFAULT_EVAL_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_LOG_DIR, DEFAULT_MODEL_NAME, DEFAULT_NUM_EPOCHS, DEFAULT_OUTPUT_DIR, IMAGE_TOKEN, TEST_DATASET_LIMIT
from multicoco.data import SupervisedDataset, collate_fn
from multicoco.exceptions import ConfigurationError, DataLoadingError, EvaluationError, ModelInitializationError
from multicoco.latent_wrapper import LatentWrapper
from multicoco.model import MultiCoCo
from multicoco.trainer import CoCoTrainer
logger = logging.getLogger(__name__)

class MultiCoCoRunner:

    def __init__(self, config: MultiCoCoConfig):
        self.config = config
        self.model: Optional[MultiCoCo] = None
        self.trainer: Optional[CoCoTrainer] = None
        self.train_dataset: Optional[SupervisedDataset] = None
        self.eval_dataset: Optional[SupervisedDataset] = None
        self.wandb_run: Optional[Any] = None
        self.run_log_dir: Optional[str] = None
        self._initialize()
        mode_type = 'training' if config.training.mode != TrainingMode.EVAL_ONLY else 'evaluation'
        logger.info(f'MultiCoCoRunner initialized for {mode_type}')

    def _initialize(self) -> None:
        if self.config.training.seed is not None:
            self._set_random_seeds(self.config.training.seed)
        self._setup_logging()
        self._setup_cuda()
        self._setup_wandb()

    def _set_random_seeds(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f'Set random seed to {seed} with deterministic operations enabled')

    def _setup_cuda(self) -> None:
        if torch.cuda.is_available():
            if not getattr(torch.backends.cudnn, 'deterministic', False):
                torch.backends.cudnn.benchmark = True
                logger.info(f'CUDA available with {torch.cuda.device_count()} devices (performance optimized)')
            else:
                logger.info(f'CUDA available with {torch.cuda.device_count()} devices (deterministic mode)')
        else:
            logger.warning('CUDA not available, using CPU')

    def _setup_logging(self) -> None:
        local_rank = int(os.environ.get('LOCAL_RANK', -1))
        if local_rank > 0:
            logging.getLogger().setLevel(logging.CRITICAL)
            return
        log_cfg = self.config.logging
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        run_name = log_cfg.run_name or 'run'
        self.run_log_dir = os.path.join(log_cfg.log_dir, f'{run_name}_{timestamp}')
        os.makedirs(self.run_log_dir, exist_ok=True)
        root_logger = logging.getLogger()
        try:
            log_level = getattr(logging, log_cfg.log_level.upper())
        except AttributeError:
            log_level = logging.INFO
            logger.warning(f'Invalid log level "{log_cfg.log_level}", falling back to INFO')
        root_logger.setLevel(log_level)
        if root_logger.hasHandlers():
            root_logger.handlers.clear()
        if log_cfg.console_output:
            console_handler = logging.StreamHandler()
            if log_cfg.verbose:
                console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            else:
                console_formatter = logging.Formatter('%(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        if log_cfg.log_to_file:
            run_log_path = os.path.join(self.run_log_dir, 'run.log')
            handler = logging.FileHandler(run_log_path)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)
        self._setup_evaluation_logger()
        logger.info(f'Logging initialized. Output saved to: {self.run_log_dir}')

    def _setup_evaluation_logger(self) -> None:
        eval_logger = logging.getLogger('evaluation_details')
        eval_logger.setLevel(logging.INFO)
        eval_logger.propagate = False
        if eval_logger.hasHandlers():
            eval_logger.handlers.clear()
        json_formatter = logging.Formatter('%(message)s')
        if self.config.training.mode == TrainingMode.EVAL_ONLY:
            eval_log_path = os.path.join(self.run_log_dir, 'evaluation.log')
            # OPTIMIZATION: Add buffering to reduce file I/O overhead
            eval_handler = logging.FileHandler(eval_log_path, mode='a', encoding='utf-8')
            eval_handler.setFormatter(json_formatter)
            eval_logger.addHandler(eval_handler)
        self.eval_logger = eval_logger
        self.json_formatter = json_formatter

    def _setup_wandb(self) -> None:
        if not self.config.logging.use_wandb:
            return
        local_rank = int(os.environ.get('LOCAL_RANK', -1))
        if local_rank not in [-1, 0]:
            return
        try:
            import wandb
            from dataclasses import asdict
            run_name = self.config.logging.run_name or self.config.training.name or f'run_{random.randint(1000, 999999)}'
            project = self.config.logging.project or 'multicoco'
            tags = []
            if self.config.training.mode:
                tags.append(str(self.config.training.mode.value))
            if self.config.coconut.enabled:
                tags.append('coconut')
                tags.append(f'c_thought_{self.config.coconut.c_thought}')
                tags.append(f'max_stage_{self.config.coconut.max_latent_stage}')
            else:
                tags.append('cot')
            self.wandb_run = wandb.init(project=project, name=run_name, tags=tags, reinit=True)
            cfg_dict = asdict(self.config)
            self.wandb_run.config.update(cfg_dict, allow_val_change=True)
            self.generation_table = wandb.Table(columns=['epoch', 'question', 'generated_text', 'ground_truth', 'prediction', 'correct'])
            logger.info(f'Initialized wandb run: project={project}, name={run_name}, tags={tags}')
        except ImportError:
            logger.warning('wandb not found; skipping integration')
            self.config.logging.use_wandb = False

    def initialize_model(self) -> None:
        try:
            model_config = self.config.model
            coconut_config = self.config.coconut
            training_mode = self.config.training.mode
            special_tokens = self._get_special_tokens(coconut_config, training_mode)
            base_model_source, checkpoint_path = self._get_model_source()
            self.model = MultiCoCo(model_id=base_model_source, config_id=model_config.config_id, tokenizer_id=model_config.tokenizer_id, image_processor_id=model_config.image_processor_id, special_tokens=special_tokens, torch_dtype=model_config.torch_dtype, trust_remote_code=model_config.trust_remote_code, low_cpu_mem_usage=model_config.low_cpu_mem_usage)
            self._finalize_model_setup(checkpoint_path, special_tokens, coconut_config, training_mode)
        except Exception as e:
            raise ModelInitializationError(f'Model initialization failed: {e}') from e

    def _finalize_model_setup(self, checkpoint_path: Optional[str], special_tokens: list, coconut_config, training_mode) -> None:
        if checkpoint_path:
            self._load_checkpoint_weights(checkpoint_path)
        if self._has_latent_tokens(special_tokens):
            self._initialize_latent_token_embeddings()
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Moving model to GPU: {torch.cuda.get_device_name()}")
            self.model = self.model.to(device)
        else:
            logger.info("CUDA not available, keeping model on CPU")
        
        if self._needs_latent_wrapper(coconut_config, training_mode):
            self.model = LatentWrapper(self.model, self.model.tokenizer)
        self._log_model_info(checkpoint_path, training_mode, coconut_config)

    def _get_special_tokens(self, coconut_config, training_mode) -> list:
        # For training modes, use coconut config
        if training_mode != TrainingMode.EVAL_ONLY:
            if coconut_config.enabled or training_mode == TrainingMode.COCONUT_TRAIN:
                special_tokens = list(set(self.config.model.get_special_tokens(coconut_config)) | set(COCONUT_SPECIAL_TOKENS))
                logger.info(f'Adding latent special tokens: {special_tokens}')
            else:
                special_tokens = self.config.model.get_special_tokens(coconut_config)
                logger.info('CoT training phase - no latent tokens added')
        else:
            # For evaluation-only mode, add latent tokens only for coconut evaluation
            base_tokens = self.config.model.get_special_tokens(coconut_config)
            if self.config.evaluation.coconut:
                special_tokens = list(set(base_tokens) | set(COCONUT_SPECIAL_TOKENS))
                logger.info(f'CoCoNut evaluation - adding latent special tokens: {special_tokens}')
            else:
                special_tokens = base_tokens
                logger.info(f'{self.config.evaluation.get_eval_type().upper()} evaluation - no latent tokens added')
        return special_tokens

    def _get_model_source(self) -> tuple[str, Optional[str]]:
        model_config = self.config.model
        if model_config.load_model_path:
            logger.info(f'Loading from checkpoint: {model_config.load_model_path}')
            return (model_config.model_name, model_config.load_model_path)
        else:
            logger.info(f'Loading base model: {model_config.model_name}')
            return (model_config.model_name, None)

    def _has_latent_tokens(self, special_tokens: list) -> bool:
        return any((tok in special_tokens for tok in COCONUT_SPECIAL_TOKENS))

    def _needs_latent_wrapper(self, coconut_config, training_mode) -> bool:
        # For training modes, use the standard logic
        if training_mode != TrainingMode.EVAL_ONLY:
            return coconut_config.enabled or training_mode == TrainingMode.COCONUT_TRAIN
        
        # For evaluation-only mode, only use LatentWrapper if we're doing coconut evaluation
        return self.config.evaluation.coconut

    def _log_model_info(self, checkpoint_path: Optional[str], training_mode, coconut_config) -> None:
        source_info = f'checkpoint: {checkpoint_path}' if checkpoint_path else f'base model: {self.config.model.model_name}'
        logger.info(f'Model initialized from {source_info}')
        logger.info(f'Dtype: {self.config.model.torch_dtype}, BF16: {self.config.training.bf16}, FP16: {self.config.training.fp16}')
        logger.info(f'Mode: {training_mode}, CoCoNut: {coconut_config.enabled}')

    def _load_checkpoint_weights(self, checkpoint_path: str) -> None:
        if self.model is None:
            raise ModelInitializationError('Model must be initialized first')
        if not os.path.exists(checkpoint_path):
            raise ModelInitializationError(f'Checkpoint path does not exist: {checkpoint_path}')
        try:
            tokenizer_info_path = os.path.join(checkpoint_path, 'tokenizer_info.json')
            checkpoint_tokenizer_info = None
            if os.path.exists(tokenizer_info_path):
                with open(tokenizer_info_path, 'r') as f:
                    checkpoint_tokenizer_info = json.load(f)
                logger.info(f"Found tokenizer info: checkpoint vocab_size={checkpoint_tokenizer_info['vocab_size']}, current vocab_size={len(self.model.tokenizer)}")
            model_files = ['model.safetensors', 'pytorch_model.bin']
            model_file = None
            for f in model_files:
                full_path = os.path.join(checkpoint_path, f)
                if os.path.exists(full_path):
                    model_file = full_path
                    break
            if not model_file:
                raise ModelInitializationError(f'No model file found in {checkpoint_path}. Expected one of: {model_files}')
            if model_file.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(model_file)
                logger.info(f'Loading checkpoint from safetensors: {model_file}')
            else:
                state_dict = torch.load(model_file, map_location='cpu')
                logger.info(f'Loading checkpoint from pytorch: {model_file}')
            current_vocab_size = len(self.model.tokenizer)
            checkpoint_vocab_size = None
            for key in state_dict.keys():
                if 'embed_tokens.weight' in key or 'lm_head.weight' in key:
                    checkpoint_vocab_size = state_dict[key].shape[0]
                    break
            if checkpoint_vocab_size and checkpoint_vocab_size != current_vocab_size:
                logger.warning(f'Vocabulary size mismatch: checkpoint={checkpoint_vocab_size}, current={current_vocab_size}')
                logger.info('Handling vocabulary size mismatch by resizing embeddings...')
                if hasattr(self.model.model, 'language_model'):
                    embed_layer = self.model.model.language_model.model.embed_tokens
                    lm_head = self.model.model.language_model.lm_head
                else:
                    embed_layer = self.model.model.get_input_embeddings()
                    lm_head = self.model.model.get_output_embeddings()
                if hasattr(self.model.model, 'language_model'):
                    self.model.model.language_model.resize_token_embeddings(checkpoint_vocab_size)
                else:
                    self.model.model.resize_token_embeddings(checkpoint_vocab_size)
                logger.info(f'Temporarily resized model embeddings to {checkpoint_vocab_size} to match checkpoint')
            target_model = self.model
            missing_keys, unexpected_keys = target_model.load_state_dict(state_dict, strict=False)
            if checkpoint_vocab_size and checkpoint_vocab_size != current_vocab_size:
                if hasattr(self.model.model, 'language_model'):
                    self.model.model.language_model.resize_token_embeddings(current_vocab_size)
                else:
                    self.model.model.resize_token_embeddings(current_vocab_size)
                logger.info(f'Resized model embeddings back to current vocab size: {current_vocab_size}')
                logger.info('New token embeddings will be randomly initialized')
            if checkpoint_vocab_size and checkpoint_vocab_size != current_vocab_size:
                missing_keys = [k for k in missing_keys if 'embed_tokens' not in k and 'lm_head' not in k]
                unexpected_keys = [k for k in unexpected_keys if 'embed_tokens' not in k and 'lm_head' not in k]
            if missing_keys:
                logger.warning(f'Missing keys when loading checkpoint: {missing_keys[:5]}...' if len(missing_keys) > 5 else missing_keys)
            if unexpected_keys:
                logger.warning(f'Unexpected keys when loading checkpoint: {unexpected_keys[:5]}...' if len(unexpected_keys) > 5 else unexpected_keys)
            logger.info(f'Successfully loaded model weights from {model_file}')
        except Exception as e:
            raise ModelInitializationError(f'Failed to load checkpoint weights: {e}') from e

    def _initialize_latent_token_embeddings(self) -> None:
        if self.model is None:
            raise ModelInitializationError('Model must be initialized first')
        try:
            embed_layer = self.model.get_input_embeddings()
            with torch.no_grad():
                eos_token_id = self.model.tokenizer.eos_token_id
                eos_embedding = embed_layer.weight[eos_token_id].clone()
                image_token_id = self.model.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
                if image_token_id is None or image_token_id >= embed_layer.weight.size(0):
                    multimodal_embedding = eos_embedding
                else:
                    image_embedding = embed_layer.weight[image_token_id].clone()
                    multimodal_embedding = (eos_embedding + image_embedding) / 2.0
                for token in COCONUT_SPECIAL_TOKENS:
                    token_id = self.model.tokenizer.convert_tokens_to_ids(token)
                    if token_id is not None and token_id < embed_layer.weight.size(0):
                        embed_layer.weight[token_id] = multimodal_embedding.clone()
        except Exception as e:
            raise ModelInitializationError(f'Failed to initialize latent token embeddings: {e}') from e

    def setup_datasets(self) -> None:
        try:
            data_config = self.config.data
            test_limit = data_config.limit_for_testing
            if isinstance(test_limit, bool):
                test_limit = TEST_DATASET_LIMIT if test_limit else None
            if self.config.training.mode != TrainingMode.EVAL_ONLY and data_config.train_data_path:
                self.train_dataset = SupervisedDataset(data_path=data_config.train_data_path, data_dir=data_config.data_dir, test_limit=test_limit)
                logger.info(f'Training dataset: {len(self.train_dataset)} samples')
            if data_config.eval_data_path:
                self.eval_dataset = SupervisedDataset(data_path=data_config.eval_data_path, data_dir=data_config.data_dir, test_limit=test_limit)
                
                # Apply evaluation-specific preprocessing for coconut mode
                if self.config.evaluation.coconut and self.config.evaluation.eval_latent_tokens is not None:
                    from multicoco.data import create_progressive_latent_dataset
                    logger.info(f'Preprocessing evaluation dataset for CoCoNut evaluation with {self.config.evaluation.eval_latent_tokens} latent tokens')
                    
                    # Convert to base format for preprocessing
                    base_data = []
                    for i in range(len(self.eval_dataset)):
                        item = self.eval_dataset.data[i]
                        base_data.append(item)
                    
                    # Apply latent preprocessing - use max stage to get the specified number of latent tokens
                    processed_data = create_progressive_latent_dataset(
                        scheduled_stage=self.config.evaluation.eval_latent_tokens,  # Stage determines number of latent tokens
                        base_dataset=base_data,
                        c_thought=0,  # Not used for evaluation
                        max_latent_stage=self.config.evaluation.eval_latent_tokens,
                        uniform_prob=0.0,  # Deterministic for evaluation
                        pad_latent_to_max=False,
                        no_cot=True  # Skip CoT steps, just add latent tokens
                    )
                    
                    # Update the dataset with processed data
                    self.eval_dataset.data = processed_data
                    logger.info(f'Applied CoCoNut preprocessing to evaluation dataset: {len(processed_data)} samples with latent tokens')
                
                logger.info(f'Evaluation dataset: {len(self.eval_dataset)} samples')
        except Exception as e:
            raise DataLoadingError(f'Dataset loading failed: {e}') from e

    def create_trainer(self) -> None:
        if self.model is None:
            raise ModelInitializationError('Model must be initialized first')
        try:
            training_args = self._create_training_arguments()
            
            # Get tokenizer and image_processor - handle LatentWrapper case
            tokenizer = getattr(self.model, 'tokenizer', None)
            image_processor = getattr(self.model, 'image_processor', None)
            
            # If wrapped by LatentWrapper, get from base_model
            if hasattr(self.model, 'base_model'):
                tokenizer = tokenizer or getattr(self.model.base_model, 'tokenizer', None) 
                image_processor = image_processor or getattr(self.model.base_model, 'image_processor', None)
            
            self.trainer = CoCoTrainer(
                model=self.model, 
                args=training_args, 
                train_dataset=self.train_dataset, 
                eval_dataset=self.eval_dataset, 
                data_collator=lambda batch: collate_fn(batch, tokenizer, image_processor), 
                runner=self
            )
            if self.config.coconut.enabled:
                self._set_coconut_trainer_params()
            setattr(self.trainer.args, 'log_per_sample', self.config.evaluation.log_per_sample)
            logger.info('Trainer created successfully')
        except Exception as e:
            raise ModelInitializationError(f'Trainer creation failed: {e}') from e

    def _set_coconut_trainer_params(self) -> None:
        if self.trainer is None:
            return
        coconut_cfg = self.config.coconut
        attrs = ['c_thought', 'max_latent_stage', 'epochs_per_stage', 'uniform_prob', 'pad_latent_to_max', 'reset_optimizer']
        for attr in attrs:
            setattr(self.trainer.args, attr, getattr(coconut_cfg, attr))

    def _create_training_arguments(self) -> TrainingArguments:
        training_config = self.config.training
        eval_strategy = getattr(training_config, 'eval_strategy', 'epoch')
        if hasattr(training_config, 'skip_eval_during_training') and training_config.skip_eval_during_training:
            eval_strategy = 'no'
            logger.warning(
                "Conflicting configuration detected: "
                "skip_eval_during_training=True but load_best_model_at_end=True. "
                "Cannot load best model without evaluations. Consider setting load_best_model_at_end=False."
            )
        common_args = {'output_dir': training_config.output_dir, 'num_train_epochs': training_config.num_epochs, 'per_device_train_batch_size': training_config.batch_size, 'per_device_eval_batch_size': training_config.eval_batch_size, 'gradient_accumulation_steps': training_config.gradient_accumulation_steps, 'eval_accumulation_steps': training_config.eval_accumulation_steps, 'learning_rate': training_config.learning_rate, 'warmup_steps': training_config.warmup_steps, 'logging_steps': training_config.logging_steps, 'save_steps': training_config.save_steps, 'eval_steps': training_config.eval_steps, 'save_strategy': training_config.save_strategy, 'eval_strategy': eval_strategy, 'save_total_limit': training_config.save_total_limit, 'load_best_model_at_end': training_config.load_best_model_at_end, 'metric_for_best_model': training_config.metric_for_best_model, 'greater_is_better': training_config.greater_is_better, 'bf16': training_config.bf16, 'fp16': training_config.fp16, 'remove_unused_columns': training_config.remove_unused_columns, 'dataloader_pin_memory': training_config.dataloader_pin_memory, 'dataloader_num_workers': training_config.dataloader_num_workers, 'gradient_checkpointing': training_config.gradient_checkpointing, 'gradient_checkpointing_kwargs': training_config.gradient_checkpointing_kwargs, 'weight_decay': training_config.weight_decay, 'max_grad_norm': training_config.max_grad_norm, 'lr_scheduler_type': training_config.lr_scheduler_type, 'seed': training_config.seed, 'data_seed': training_config.data_seed, 'report_to': self.config.get_wandb_report_to()}
        return TrainingArguments(**common_args)

    def run_training(self) -> None:
        if self.trainer is None:
            raise ModelInitializationError('Trainer must be initialized first')
        if self.train_dataset is None or len(self.train_dataset) == 0:
            raise DataLoadingError('Training dataset is empty or not loaded')
        logger.info('Starting training...')
        self._log_model_config_to_wandb()
        resume_from_checkpoint = self.config.training.resume_from_checkpoint if self.config.training.resume_from_checkpoint else None
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        if hasattr(self.trainer, '_log_performance_summary'):
            self.trainer._log_performance_summary()

    def run_evaluation(self) -> Dict[str, float]:
        if self.trainer is None or self.eval_dataset is None or len(self.eval_dataset) == 0:
            raise EvaluationError('Evaluation dataset is empty or not initialized')
        logger.info('Starting evaluation...')
        # Use evaluate() which respects the log_per_sample setting from args
        metrics = self.trainer.evaluate()
        self._log_evaluation_results(metrics)
        return metrics

    def run(self) -> Dict[str, float]:
        try:
            self.initialize_model()
            self.setup_datasets()
            mode = self.config.training.mode
            if mode == TrainingMode.EVAL_ONLY:
                return self._run_eval_only()
            elif mode == TrainingMode.COT_TRAIN:
                return self._run_training_mode()
            elif mode == TrainingMode.COCONUT_TRAIN:
                return self._run_coconut_mode()
            else:
                raise ConfigurationError(f'Invalid training mode: {mode}')
        except (ConfigurationError, ModelInitializationError, DataLoadingError, EvaluationError) as e:
            logger.error(f'Pipeline failed: {e}')
            raise
        finally:
            self.cleanup()

    def _run_eval_only(self) -> Dict[str, float]:
        logger.info('Starting evaluation only...')
        self.create_trainer()
        return self.run_evaluation()

    def _run_training_mode(self) -> Dict[str, float]:
        logger.info('Starting CoT training...')
        self.create_trainer()
        self.run_training()
        return self._run_final_evaluation()

    def _run_coconut_mode(self) -> Dict[str, float]:
        logger.info('Starting CoCoNut multi-stage training...')
        self.create_trainer()
        if hasattr(self, 'wandb_run') and self.wandb_run is not None:
            coconut_config = {'coconut/max_latent_stage': self.config.coconut.max_latent_stage, 'coconut/epochs_per_stage': self.config.coconut.epochs_per_stage, 'coconut/c_thought': self.config.coconut.c_thought, 'coconut/total_stages': self.config.coconut.max_latent_stage + 1, 'coconut/uniform_prob': self.config.coconut.uniform_prob}
            self.wandb_run.log(coconut_config)
            logger.info(f'Logged CoCoNut configuration to wandb: {coconut_config}')
        resume_from_checkpoint = self.config.training.resume_from_checkpoint if self.config.training.resume_from_checkpoint else None
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logger.info('Running final CoCoNut evaluation...')
        return self.trainer.perform_evaluation(log_per_sample=True)

    def _run_final_evaluation(self) -> Dict[str, float]:
        logger.info('Running final evaluation...')
        return self.run_evaluation()

    def _log_evaluation_results(self, metrics: Dict[str, float]) -> None:
        if self.trainer and self.trainer.is_world_process_zero():
            logger.info('\n' + '=' * 50)
            logger.info('EVALUATION SUMMARY')
            logger.info('=' * 50)
            for key, value in metrics.items():
                logger.info(f'  {key}: {value:.4f}')
            logger.info('=' * 50)
            if hasattr(self, 'wandb_run') and self.wandb_run is not None:
                wandb_metrics = {}
                for key, value in metrics.items():
                    wandb_metrics[f'eval/{key}'] = value
                self.wandb_run.log(wandb_metrics)
                logger.info(f'Logged evaluation metrics to wandb: {wandb_metrics}')

    def _log_training_metrics(self, epoch: int, step: int, loss: float, learning_rate: float=None) -> None:
        if hasattr(self, 'wandb_run') and self.wandb_run is not None:
            metrics = {'train/epoch': epoch, 'train/step': step, 'train/loss': loss}
            if learning_rate is not None:
                metrics['train/learning_rate'] = learning_rate
            self.wandb_run.log(metrics)

    def _log_coconut_stage_metrics(self, stage: int, epoch: int, stage_progress: float) -> None:
        if hasattr(self, 'wandb_run') and self.wandb_run is not None:
            stage_metrics = {'coconut/current_stage': stage, 'coconut/stage_epoch': epoch, 'coconut/stage_progress': stage_progress, 'coconut/latent_replacement_ratio': stage / max(1, self.config.coconut.max_latent_stage)}
            self.wandb_run.log(stage_metrics)

    def _log_generation_samples(self, questions: List[str], generated_texts: List[str], ground_truth: List[str], predictions: List[str], max_samples: int=10) -> None:
        if not hasattr(self, 'wandb_run') or self.wandb_run is None:
            return
        try:
            import wandb
            columns = ['Question', 'Generated Text', 'Ground Truth', 'Prediction', 'Correct']
            data = []
            for i in range(min(len(questions), max_samples)):
                is_correct = predictions[i] == ground_truth[i] if i < len(predictions) and i < len(ground_truth) else False
                data.append([questions[i][:200] + '...' if len(questions[i]) > 200 else questions[i], generated_texts[i][:500] + '...' if len(generated_texts[i]) > 500 else generated_texts[i], ground_truth[i] if i < len(ground_truth) else 'N/A', predictions[i] if i < len(predictions) else 'N/A', '✓' if is_correct else '✗'])
            generation_table = wandb.Table(columns=columns, data=data)
            self.wandb_run.log({'eval/generation_samples': generation_table})
            logger.info(f'Logged {len(data)} generation samples to wandb')
        except Exception as e:
            logger.warning(f'Failed to log generation samples to wandb: {e}')

    def _log_model_config_to_wandb(self) -> None:
        if not hasattr(self, 'wandb_run') or self.wandb_run is None:
            return
        try:
            from dataclasses import asdict
            config_dict = asdict(self.config)
            self.wandb_run.config.update(config_dict, allow_val_change=True)
            if hasattr(self.model, 'config'):
                model_config = {'model_type': getattr(self.model.config, 'model_type', 'unknown'), 'hidden_size': getattr(self.model.config, 'hidden_size', 'unknown'), 'num_attention_heads': getattr(self.model.config, 'num_attention_heads', 'unknown'), 'num_hidden_layers': getattr(self.model.config, 'num_hidden_layers', 'unknown')}
                self.wandb_run.config.update({'model_details': model_config}, allow_val_change=True)
            logger.info('Logged comprehensive configuration to wandb')
        except Exception as e:
            logger.warning(f'Failed to log model config to wandb: {e}')

    def setup_epoch_evaluation_logger(self, epoch: int) -> None:
        if self.config.training.mode == TrainingMode.EVAL_ONLY:
            return
        eval_logger = logging.getLogger('evaluation_details')
        if eval_logger.hasHandlers():
            eval_logger.handlers.clear()
        eval_log_path = os.path.join(self.run_log_dir, f'evaluation_epoch_{epoch + 1}.log')
        eval_handler = logging.FileHandler(eval_log_path)
        eval_handler.setFormatter(self.json_formatter)
        eval_logger.addHandler(eval_handler)

    def cleanup(self) -> None:
        if hasattr(self, 'wandb_run') and self.wandb_run is not None:
            try:
                import wandb
                logger.info('Finishing wandb run...')
                self.wandb_run.finish()
            except ImportError:
                pass
        logger.info('MultiCoCo runner cleanup complete')

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='MultiCoCo: Two-phase training for multimodal models.')
    parser.add_argument('config_path', type=str, help='Path to YAML configuration file')
    parser.add_argument('--eval-only', action='store_true', help='Run evaluation only (skip training)')
    parser.add_argument('--output-dir', type=str, help='Override output directory')
    parser.add_argument('--model-name', type=str, help='Override model name')
    return parser

def apply_cli_overrides(config: MultiCoCoConfig, args: argparse.Namespace) -> MultiCoCoConfig:
    if args.eval_only:
        config.training.mode = TrainingMode.EVAL_ONLY
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.model_name:
        config.model.model_name = args.model_name
    return config

def _load_config(config_path: str) -> MultiCoCoConfig:
    base_cfg_path = os.path.join(os.path.dirname(config_path), 'base.yaml')
    return MultiCoCoConfig.load_with_base(config_path=config_path, base_config_path=base_cfg_path)

def main() -> None:
    try:
        parser = create_parser()
        args = parser.parse_args()
        config = _load_config(args.config_path)
        config = apply_cli_overrides(config, args)
        if config.training.mode == TrainingMode.COT_TRAIN:
            config.evaluation.cot = True
            config.evaluation.vanilla = False
        runner = MultiCoCoRunner(config)
        metrics = runner.run()
        print('\n' + '=' * 50)
        print('FINAL RESULTS')
        print('=' * 50)
        for key, value in metrics.items():
            print(f'{key}: {value}')
        print('=' * 50)
    except KeyboardInterrupt:
        print('\nInterrupted by user')
        sys.exit(1)
    except Exception as e:
        print(f'Error: {e}')
        sys.exit(1)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
if __name__ == '__main__':
    main()