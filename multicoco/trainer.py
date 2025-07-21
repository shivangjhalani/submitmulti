import gc
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Trainer
from transformers.trainer_pt_utils import find_batch_size
from transformers.trainer_utils import TrainOutput
from .answer_extraction import extract_answer_choice
from .constants import DEFAULT_MAX_NEW_TOKENS
from .exceptions import EvaluationError
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logger = logging.getLogger(__name__)

class CoCoTrainer(Trainer):

    def __init__(self, *args, runner=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_train_steps = 0
        self.runner = runner
        logger.info('CoCoTrainer initialized.')

    def train(self, resume_from_checkpoint: Optional[Union[str, bool]]=None, **kwargs) -> TrainOutput:
        is_coconut_mode = hasattr(self.args, 'epochs_per_stage') and hasattr(self.args, 'max_latent_stage')
        if is_coconut_mode:
            return self._train_with_coconut_stages(resume_from_checkpoint, **kwargs)
        else:
            return self._train_standard(resume_from_checkpoint, **kwargs)

    def _train_standard(self, resume_from_checkpoint: Optional[Union[str, bool]]=None, **kwargs) -> TrainOutput:
        self._setup_epoch_training()
        start_epoch, checkpoint_path = self._handle_checkpoint_resumption(resume_from_checkpoint)
        train_dataloader = self.get_train_dataloader()
        steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        total_steps = steps_per_epoch * int(self.args.num_train_epochs)
        self._log_training_setup(steps_per_epoch, total_steps)
        model = self._wrap_model(self.model_wrapped)
        self.create_optimizer_and_scheduler(num_training_steps=total_steps)
        if checkpoint_path:
            self._load_optimizer_scheduler_states(checkpoint_path)
        for epoch in range(start_epoch, int(self.args.num_train_epochs)):
            self._train_single_epoch(model, train_dataloader, epoch, steps_per_epoch)
            gc.collect()
            torch.cuda.empty_cache()
        logger.info('Training completed!')
        return TrainOutput(global_step=self.total_train_steps, training_loss=0.0, metrics={})

    def _train_with_coconut_stages(self, resume_from_checkpoint: Optional[Union[str, bool]]=None, **kwargs) -> TrainOutput:
        logger.info('Starting CoCoNut multi-stage training with stage transitions')
        self._setup_epoch_training()
        start_epoch, checkpoint_path = self._handle_checkpoint_resumption(resume_from_checkpoint)
        self._last_stage = -1
        
        # If resuming from checkpoint, calculate the current stage based on the start epoch
        if start_epoch > 0:
            current_stage = min(start_epoch // self.args.epochs_per_stage, self.args.max_latent_stage)
            self._last_stage = current_stage - 1  # Will trigger stage update on first epoch
            logger.info(f'Resuming CoCoNut training from epoch {start_epoch + 1}, stage {current_stage}')
        
        train_dataloader = self.get_train_dataloader()
        steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        total_steps = steps_per_epoch * int(self.args.num_train_epochs)
        self._log_training_setup(steps_per_epoch, total_steps)
        model = self._wrap_model(self.model_wrapped)
        self.create_optimizer_and_scheduler(num_training_steps=total_steps)
        if checkpoint_path:
            self._load_optimizer_scheduler_states(checkpoint_path)
        for epoch in range(start_epoch, int(self.args.num_train_epochs)):
            current_stage = min(epoch // self.args.epochs_per_stage, self.args.max_latent_stage)
            if current_stage != self._last_stage:
                self._update_for_stage(current_stage)
                self._last_stage = current_stage
                train_dataloader = self.get_train_dataloader()
            stage_epoch = epoch % self.args.epochs_per_stage
            stage_progress = (stage_epoch + 1) / self.args.epochs_per_stage
            logger.info(f'Epoch {epoch + 1}/{int(self.args.num_train_epochs)} - CoCoNut Stage {current_stage}/{self.args.max_latent_stage} (Stage Epoch {stage_epoch + 1}/{self.args.epochs_per_stage})')
            self._log_coconut_stage_metrics(current_stage, stage_epoch, stage_progress)
            self._train_single_epoch(model, train_dataloader, epoch, steps_per_epoch)
            gc.collect()
            torch.cuda.empty_cache()
        logger.info('CoCoNut multi-stage training completed!')
        return TrainOutput(global_step=self.total_train_steps, training_loss=0.0, metrics={})

    def _log_training_setup(self, steps_per_epoch: int, total_steps: int) -> None:
        logger.info('Starting epoch-based training:')
        logger.info(f'  Steps per epoch: {steps_per_epoch}')
        logger.info(f'  Total epochs: {int(self.args.num_train_epochs)}')
        logger.info(f'  Total steps: {total_steps}')

    def _train_single_epoch(self, model: torch.nn.Module, train_dataloader: DataLoader, epoch: int, steps_per_epoch: int) -> None:
        epoch_start_time = time.time()
        logger.info(f'\nStarting Epoch {epoch + 1}/{int(self.args.num_train_epochs)}')
        self._train_one_epoch(model, train_dataloader, epoch, steps_per_epoch)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        # Check if evaluation should run based on eval_strategy
        should_evaluate = self._should_evaluate_after_epoch(epoch)
        
        if should_evaluate:
            if self.runner and hasattr(self.runner, 'setup_epoch_evaluation_logger'):
                self.runner.setup_epoch_evaluation_logger(epoch)
                logger.info(f'Running evaluation after epoch {epoch + 1}...')
                logger.debug(f'Epoch evaluation logger configured for epoch {epoch + 1}')
            else:
                logger.warning('No runner or setup_epoch_evaluation_logger method available. Per-sample evaluation logs may not be written.')
            eval_metrics = self.evaluate()
        else:
            logger.info(f'Skipping evaluation after epoch {epoch + 1} due to eval_strategy setting')
            eval_metrics = {}
        checkpoint_dir = self._save_checkpoint_with_metrics(epoch, eval_metrics)
        epoch_time = time.time() - epoch_start_time
        self._log_epoch_summary(epoch, eval_metrics, checkpoint_dir, epoch_time)

    def _handle_checkpoint_resumption(self, resume_from_checkpoint: Optional[Union[str, bool]]) -> Tuple[int, Optional[str]]:
        start_epoch = 0
        checkpoint_path = None
        if resume_from_checkpoint:
            if resume_from_checkpoint is True:
                checkpoint_path = self._get_last_epoch_checkpoint(self.args.output_dir)
            else:
                checkpoint_path = str(resume_from_checkpoint)
            if checkpoint_path and os.path.exists(checkpoint_path):
                logger.info(f'Resuming training from checkpoint: {checkpoint_path}')
                start_epoch = self._load_model_and_training_state(checkpoint_path)
            else:
                logger.warning('`resume_from_checkpoint` is set but no checkpoint found. Starting from scratch.')
                checkpoint_path = None
        return (start_epoch, checkpoint_path)

    def _get_last_epoch_checkpoint(self, output_dir: str) -> Optional[str]:
        if not os.path.exists(output_dir):
            logger.warning(f'Output directory does not exist: {output_dir}')
            return None
        epoch_dirs = [d for d in os.listdir(output_dir) if d.startswith('epoch-')]
        if not epoch_dirs:
            logger.warning(f'No epoch directories found in: {output_dir}')
            return None
        epoch_nums = [int(d.split('-')[1]) for d in epoch_dirs if d.split('-')[1].isdigit()]
        if not epoch_nums:
            logger.warning(f'No valid epoch numbers found in: {output_dir}')
            return None
        latest_epoch = max(epoch_nums)
        checkpoint_path = os.path.join(output_dir, f'epoch-{latest_epoch}')
        logger.info(f'Found latest checkpoint: {checkpoint_path}')
        return checkpoint_path

    def _load_model_and_training_state(self, checkpoint_path: str) -> int:
        try:
            if not os.path.exists(checkpoint_path):
                logger.error(f'Checkpoint directory does not exist: {checkpoint_path}')
                return 0
            model_files = ['pytorch_model.bin', 'model.safetensors', 'config.json']
            has_model_file = any((os.path.exists(os.path.join(checkpoint_path, f)) for f in model_files))
            if not has_model_file:
                logger.error(f'No model files found in checkpoint directory: {checkpoint_path}')
                return 0
            epoch_num = int(os.path.basename(checkpoint_path).split('-')[1])
            logger.info(f'Loading model and training state from epoch {epoch_num}: {checkpoint_path}')
            
            # Handle tokenizer info for vocabulary size mismatch
            tokenizer_info_path = os.path.join(checkpoint_path, 'tokenizer_info.json')
            checkpoint_tokenizer_info = None
            if os.path.exists(tokenizer_info_path):
                with open(tokenizer_info_path, 'r') as f:
                    checkpoint_tokenizer_info = json.load(f)
                logger.info(f"Found tokenizer info: checkpoint vocab_size={checkpoint_tokenizer_info['vocab_size']}")
            
            model_file = None
            for f in model_files:
                full_path = os.path.join(checkpoint_path, f)
                if os.path.exists(full_path) and f in ['pytorch_model.bin', 'model.safetensors']:
                    model_file = full_path
                    break
            if model_file:
                if model_file.endswith('.safetensors'):
                    from safetensors.torch import load_file
                    state_dict = load_file(model_file)
                else:
                    state_dict = torch.load(model_file, map_location='cpu')
                
                # Check for vocabulary size mismatch and handle it
                current_vocab_size = len(self.tokenizer)
                checkpoint_vocab_size = None
                
                # Detect checkpoint vocab size from state dict
                for key in state_dict.keys():
                    if 'embed_tokens.weight' in key or 'lm_head.weight' in key:
                        checkpoint_vocab_size = state_dict[key].shape[0]
                        break
                
                if checkpoint_vocab_size and checkpoint_vocab_size != current_vocab_size:
                    logger.warning(f'Vocabulary size mismatch: checkpoint={checkpoint_vocab_size}, current={current_vocab_size}')
                    logger.info('This is expected when loading CoT checkpoint for CoCoNut training (latent tokens added)')
                    
                    # Temporarily resize model to match checkpoint for loading
                    target_model = self.model
                    if hasattr(target_model, 'base_model'):  # LatentWrapper case
                        underlying_model = target_model.base_model
                    else:
                        underlying_model = target_model
                    
                    # Resize to checkpoint size temporarily
                    if hasattr(underlying_model, 'model') and hasattr(underlying_model.model, 'language_model'):
                        underlying_model.model.language_model.resize_token_embeddings(checkpoint_vocab_size)
                    elif hasattr(underlying_model, 'model'):
                        underlying_model.model.resize_token_embeddings(checkpoint_vocab_size)
                    else:
                        underlying_model.resize_token_embeddings(checkpoint_vocab_size)
                    
                    logger.info(f'Temporarily resized model embeddings to {checkpoint_vocab_size} to match checkpoint')
                
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                
                # Resize back to current vocabulary size if there was a mismatch
                if checkpoint_vocab_size and checkpoint_vocab_size != current_vocab_size:
                    target_model = self.model
                    if hasattr(target_model, 'base_model'):  # LatentWrapper case
                        underlying_model = target_model.base_model
                    else:
                        underlying_model = target_model
                    
                    # Resize back to current size
                    if hasattr(underlying_model, 'model') and hasattr(underlying_model.model, 'language_model'):
                        underlying_model.model.language_model.resize_token_embeddings(current_vocab_size)
                    elif hasattr(underlying_model, 'model'):
                        underlying_model.model.resize_token_embeddings(current_vocab_size)
                    else:
                        underlying_model.resize_token_embeddings(current_vocab_size)
                    
                    logger.info(f'Resized model embeddings back to current vocab size: {current_vocab_size}')
                    logger.info('New latent token embeddings will be randomly initialized')
                    
                    # Filter out embedding-related keys from warnings since we expect them to be missing
                    missing_keys = [k for k in missing_keys if 'embed_tokens' not in k and 'lm_head' not in k]
                    unexpected_keys = [k for k in unexpected_keys if 'embed_tokens' not in k and 'lm_head' not in k]
                
                if missing_keys:
                    logger.warning(f'Missing keys when loading checkpoint: {missing_keys[:5]}...' if len(missing_keys) > 5 else missing_keys)
                if unexpected_keys:
                    logger.warning(f'Unexpected keys when loading checkpoint: {unexpected_keys[:5]}...' if len(unexpected_keys) > 5 else unexpected_keys)
                logger.info(f'Successfully loaded model weights from {model_file}')
            else:
                logger.error(f'No valid model file found in {checkpoint_path}')
                return 0
            
            state_file = os.path.join(checkpoint_path, 'training_state.pt')
            if os.path.exists(state_file):
                try:
                    training_state = torch.load(state_file, map_location='cpu')
                    self.state.global_step = training_state.get('global_step', 0)
                    self.total_train_steps = training_state.get('total_train_steps', 0)
                    logger.info(f'Restored training state: global_step={self.state.global_step}, total_train_steps={self.total_train_steps}')
                except Exception as e:
                    logger.warning(f'Failed to load training state: {e}')
            next_epoch = epoch_num
            logger.info(f'Model and training state loaded successfully. Next training epoch: {next_epoch} (0-indexed)')
            return next_epoch
        except ValueError as e:
            logger.error(f'Invalid checkpoint path format {checkpoint_path}: {e}')
            return 0
        except Exception as e:
            logger.error(f'Failed to load model and training state {checkpoint_path}: {e}')
            logger.error(f'Exception type: {type(e).__name__}')
            return 0

    def _load_optimizer_scheduler_states(self, checkpoint_path: str) -> None:
        try:
            optimizer_file = os.path.join(checkpoint_path, 'optimizer.pt')
            if os.path.exists(optimizer_file) and hasattr(self, 'optimizer') and (self.optimizer is not None):
                try:
                    optimizer_state = torch.load(optimizer_file, map_location='cpu')
                    self.optimizer.load_state_dict(optimizer_state)
                    logger.info('Successfully loaded optimizer state')
                except Exception as e:
                    logger.warning(f'Failed to load optimizer state: {e}')
            scheduler_file = os.path.join(checkpoint_path, 'scheduler.pt')
            if os.path.exists(scheduler_file) and hasattr(self, 'lr_scheduler') and (self.lr_scheduler is not None):
                try:
                    scheduler_state = torch.load(scheduler_file, map_location='cpu')
                    self.lr_scheduler.load_state_dict(scheduler_state)
                    logger.info('Successfully loaded scheduler state')
                except Exception as e:
                    logger.warning(f'Failed to load scheduler state: {e}')
        except Exception as e:
            logger.warning(f'Failed to load optimizer/scheduler states from {checkpoint_path}: {e}')

    def _setup_epoch_training(self) -> None:
        self.state.global_step = 0
        self.state.epoch = 0
        self.state.total_flos = 0
        logger.info('Training state initialized for epoch-based training')

    def _train_one_epoch(self, model: torch.nn.Module, train_dataloader: DataLoader, epoch: int, steps_per_epoch: int) -> None:
        model.train()
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}', total=len(train_dataloader), disable=not self.is_world_process_zero())
        epoch_loss = 0.0
        step_count = 0
        tr_loss = torch.tensor(0.0).to(model.device)
        for step, inputs in enumerate(pbar):
            loss = self.training_step(model, inputs)
            if loss is not None:
                tr_loss += loss.detach()
                epoch_loss += loss.item()
                step_count += 1
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                if dist.is_initialized():
                    dist.barrier()
                if hasattr(self.args, 'max_grad_norm') and self.args.max_grad_norm > 0:
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1.0 / 2)
                    self._last_grad_norm = total_norm
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                model.zero_grad()
                self.state.global_step += 1
                self.total_train_steps += 1
                avg_loss = tr_loss.item() / self.args.gradient_accumulation_steps
                current_lr = self.get_lr()
                logger.debug(f"Current LR: {current_lr} (full float: {self.optimizer.param_groups[0]['lr']})")
                pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{current_lr:.6f}'})
                tr_loss = torch.tensor(0.0).to(model.device)
                self._log_training_step(loss, step, epoch)
        pbar.close()
        if step_count > 0:
            final_avg_loss = epoch_loss / step_count
            logger.info(f'Epoch {epoch + 1} training complete. Average loss: {final_avg_loss:.4f}')
            if 'wandb' in self.args.report_to:
                try:
                    import wandb
                    if wandb.run:
                        wandb.log({'train/epoch_avg_loss': final_avg_loss, 'train/epoch': epoch + 1, 'train/steps_per_epoch': step_count})
                except ImportError:
                    pass

    def _log_training_step(self, loss: torch.Tensor, step: int, epoch: int=None) -> None:
        if (step + 1) % self.args.gradient_accumulation_steps == 0 and 'wandb' in self.args.report_to:
            try:
                import wandb
                if wandb.run:
                    log_dict = {'train/batch_loss': loss.item(), 'train/step': self.total_train_steps, 'train/global_step': self.state.global_step, 'train/learning_rate': self.get_lr()}
                    if epoch is not None:
                        log_dict['train/epoch'] = epoch + 1
                    elif hasattr(self.state, 'epoch') and self.state.epoch is not None:
                        log_dict['train/epoch'] = self.state.epoch
                    if hasattr(self, '_last_grad_norm'):
                        log_dict['train/grad_norm'] = self._last_grad_norm
                    log_dict['train/gradient_accumulation_steps'] = self.args.gradient_accumulation_steps
                    wandb.log(log_dict)
            except ImportError:
                pass

    def _log_epoch_summary(self, epoch: int, eval_metrics: Dict[str, float], checkpoint_dir: str, epoch_time: float) -> None:
        summary = [f'\nEPOCH {epoch + 1} SUMMARY', f'Checkpoint: {checkpoint_dir}', f'Epoch time: {epoch_time:.2f}s']
        if eval_metrics:
            summary.append('Evaluation metrics:')
            summary.extend([f'  {k}: {v:.4f}' for k, v in eval_metrics.items()])
        for line in summary:
            logger.info(line)
        if 'wandb' in self.args.report_to:
            try:
                import wandb
                if wandb.run:
                    epoch_summary = {'epoch/number': epoch + 1, 'epoch/time_seconds': epoch_time, 'epoch/checkpoint_dir': checkpoint_dir}
                    if eval_metrics:
                        for key, value in eval_metrics.items():
                            epoch_summary[f'epoch/{key}'] = value
                    wandb.log(epoch_summary)
            except ImportError:
                pass

    def _log_validation_loss(self, val_loss: float, epoch: int) -> None:
        if 'wandb' in self.args.report_to:
            try:
                import wandb
                if wandb.run:
                    wandb.log({'eval/loss': val_loss, 'eval/epoch': epoch + 1})
                    logger.info(f'Validation loss: {val_loss:.4f}')
            except ImportError:
                pass

    def get_lr(self) -> float:
        if self.optimizer is None:
            return 0.0
        return self.optimizer.param_groups[0]['lr']

    def _save_checkpoint_with_metrics(self, epoch: int, metrics: Dict[str, float]) -> str:
        checkpoint_dir = os.path.join(self.args.output_dir, f'epoch-{epoch + 1}')
        self.save_model(checkpoint_dir)
        if self.is_world_process_zero():
            if hasattr(self.model, 'tokenizer'):
                tokenizer_info = {'vocab_size': len(self.model.tokenizer), 'special_tokens': self.model.tokenizer.get_added_vocab(), 'pad_token_id': self.model.tokenizer.pad_token_id, 'eos_token_id': self.model.tokenizer.eos_token_id}
                tokenizer_info_path = os.path.join(checkpoint_dir, 'tokenizer_info.json')
                with open(tokenizer_info_path, 'w') as f:
                    json.dump(tokenizer_info, f, indent=4)
                logger.debug(f'Saved tokenizer info to {tokenizer_info_path}')
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                optimizer_path = os.path.join(checkpoint_dir, 'optimizer.pt')
                torch.save(self.optimizer.state_dict(), optimizer_path)
                logger.debug(f'Saved optimizer state to {optimizer_path}')
            if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                scheduler_path = os.path.join(checkpoint_dir, 'scheduler.pt')
                torch.save(self.lr_scheduler.state_dict(), scheduler_path)
                logger.debug(f'Saved scheduler state to {scheduler_path}')
            training_state = {'epoch': epoch + 1, 'global_step': self.state.global_step, 'total_train_steps': self.total_train_steps}
            state_path = os.path.join(checkpoint_dir, 'training_state.pt')
            torch.save(training_state, state_path)
            logger.debug(f'Saved training state to {state_path}')
            metrics_path = os.path.join(checkpoint_dir, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f'Checkpoint saved with metrics: {checkpoint_dir}')
        return checkpoint_dir

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval') -> Dict[str, float]:
        log_per_sample = getattr(self.args, 'log_per_sample', False)
        return self.perform_evaluation(eval_dataset, metric_key_prefix, log_per_sample=log_per_sample)

    def perform_evaluation(self, eval_dataset=None, metric_key_prefix='eval', log_per_sample=False) -> Dict[str, float]:
        eval_dataset = eval_dataset or self.eval_dataset
        if eval_dataset is None:
            raise EvaluationError('No evaluation dataset provided')
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        all_preds, all_labels, all_questions, all_gen_texts, all_gen_tokens, all_ext_ans = ([], [], [], [], [], [])
        all_latencies = []  # Track latencies
        max_new_tokens = getattr(self.args, 'eval_max_new_tokens', DEFAULT_MAX_NEW_TOKENS)
        progress_bar = tqdm(eval_dataloader, desc='Evaluating', total=len(eval_dataloader), disable=not self.is_world_process_zero())
        with torch.no_grad():
            for batch in progress_bar:
                preds, gen_texts, gen_tokens, latencies = self._generate_batch_predictions_with_details(batch, max_new_tokens)
                all_preds.extend(preds)
                all_labels.extend(batch.get('answers', []))
                all_questions.extend(batch.get('questions', []))
                all_gen_texts.extend(gen_texts)
                all_gen_tokens.extend(gen_tokens)
                all_ext_ans.extend(preds)
                all_latencies.extend(latencies)  # Collect latencies
        progress_bar.close()
        gathered = self._gather_evaluation_results(all_preds, all_labels, all_questions, all_gen_texts, all_gen_tokens, all_ext_ans)
        all_preds, all_labels, all_questions, all_gen_texts, all_gen_tokens, all_ext_ans = gathered
        # Note: latencies are not gathered across processes to keep implementation simple
        if self.is_world_process_zero():
            metrics = self._compute_evaluation_metrics(all_preds, all_labels, metric_key_prefix)
            
            # Add latency metrics if enabled and available
            log_latency = getattr(self.runner.config.evaluation, 'log_latency', True) if hasattr(self, 'runner') and self.runner else True
            if log_latency and all_latencies:
                avg_latency = sum(all_latencies) / len(all_latencies)
                min_latency = min(all_latencies)
                max_latency = max(all_latencies)
                total_eval_time = sum(all_latencies)
                
                latency_metrics = {
                    f'{metric_key_prefix}/avg_latency_sec': avg_latency,
                    f'{metric_key_prefix}/min_latency_sec': min_latency,
                    f'{metric_key_prefix}/max_latency_sec': max_latency,
                    f'{metric_key_prefix}/total_eval_time_sec': total_eval_time
                }
                metrics.update(latency_metrics)
                logger.info(f'Latency Metrics: Avg={avg_latency:.4f}s, Min={min_latency:.4f}s, Max={max_latency:.4f}s, Total={total_eval_time:.2f}s')
            
            logger.info(f'{metric_key_prefix.upper()} METRICS: {metrics}')
            if 'wandb' in self.args.report_to:
                try:
                    import wandb
                    if wandb.run:
                        wandb_metrics = {}
                        for key, value in metrics.items():
                            wandb_metrics[f'{metric_key_prefix}/{key}'] = value
                        total_samples = len(all_preds)
                        correct_predictions = sum((1 for pred, label in zip(all_preds, all_labels) if pred == label))
                        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
                        wandb_metrics[f'{metric_key_prefix}/acc'] = accuracy
                        wandb_metrics[f'{metric_key_prefix}/total_samples'] = total_samples
                        wandb_metrics[f'{metric_key_prefix}/correct_predictions'] = correct_predictions
                        if all_gen_texts:
                            cot_exact_matches = 0
                            for gen_text, label in zip(all_gen_texts, all_labels):
                                if label.lower().strip() in gen_text.lower():
                                    cot_exact_matches += 1
                            cot_em_rate = cot_exact_matches / total_samples if total_samples > 0 else 0.0
                            wandb_metrics[f'{metric_key_prefix}/cot_em'] = cot_em_rate
                        wandb.log(wandb_metrics)
                        logger.info(f'Logged comprehensive evaluation metrics to wandb: {wandb_metrics}')
                        if log_per_sample and len(all_questions) > 0:
                            self._log_evaluation_samples_to_wandb(all_questions[:10], all_gen_texts[:10], all_labels[:10], all_preds[:10], metric_key_prefix)
                except ImportError:
                    logger.warning('wandb not available for logging evaluation metrics')
            if log_per_sample:
                correctness = np.array(all_preds) == np.array(all_labels)
                logger.info(f'Logging {len(all_questions)} per-sample evaluation details to file...')
                self._log_per_sample_details(all_questions, all_labels, all_gen_texts, all_ext_ans, all_gen_tokens, correctness, all_latencies)
                logger.debug(f'Completed logging per-sample evaluation details')
            return metrics
        return {}

    def _log_evaluation_samples_to_wandb(self, questions: List[str], generated_texts: List[str], labels: List[str], predictions: List[str], metric_prefix: str) -> None:
        try:
            import wandb
            if wandb.run:
                columns = ['Question', 'Generated Text', 'Ground Truth', 'Prediction', 'Correct']
                data = []
                for i in range(len(questions)):
                    is_correct = predictions[i] == labels[i] if i < len(predictions) and i < len(labels) else False
                    data.append([questions[i][:300] + '...' if len(questions[i]) > 300 else questions[i], generated_texts[i][:500] + '...' if len(generated_texts[i]) > 500 else generated_texts[i], labels[i] if i < len(labels) else 'N/A', predictions[i] if i < len(predictions) else 'N/A', '✓' if is_correct else '✗'])
                eval_table = wandb.Table(columns=columns, data=data)
                wandb.log({f'{metric_prefix}/sample_generations': eval_table})
        except Exception as e:
            logger.warning(f'Failed to log evaluation samples to wandb: {e}')

    def _gather_evaluation_results(self, predictions: List[str], labels: List[str], questions: List[str], generated_texts: List[str], generated_tokens: List[int], extracted_answers: List[str]) -> Tuple[List[str], List[str], List[str], List[str], List[int], List[str]]:
        if dist.is_initialized() and dist.get_world_size() > 1:
            local_results = list(zip(predictions, labels, questions, generated_texts, generated_tokens, extracted_answers))
            gathered_results = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_results, local_results)
            all_results = [item for sublist in gathered_results for item in sublist]
            all_predictions, all_labels, all_questions, all_generated_texts, all_generated_tokens, all_extracted = zip(*all_results)
            return (list(all_predictions), list(all_labels), list(all_questions), list(all_generated_texts), list(all_generated_tokens), list(all_extracted))
        return (predictions, labels, questions, generated_texts, generated_tokens, extracted_answers)

    def _log_per_sample_details(self, questions, labels, generated_texts, extracted, generated_tokens, correctness, latencies=None):
        eval_logger = logging.getLogger('evaluation_details')
        if not eval_logger.hasHandlers():
            logger.warning('evaluation_details logger has no handlers configured. Per-sample logs may not be written to file.')
            return
        
        # OPTIMIZATION: Batch JSON serialization and logging instead of individual calls
        # This significantly reduces I/O overhead from N individual writes to 1 batch write
        batch_details = []
        for i in range(len(questions)):
            details = {
                'question': questions[i], 
                'ground_truth': labels[i], 
                'generated_answer': generated_texts[i], 
                'extracted_answer': extracted[i], 
                'generated_tokens': generated_tokens[i], 
                'correct': bool(correctness[i])
            }
            # Add latency if available
            if latencies and i < len(latencies):
                details['latency_sec'] = latencies[i]
            batch_details.append(details)
        
        # Batch serialize and log all samples at once
        import json
        try:
            # Single JSON serialization for all samples
            batch_json = '\n'.join(json.dumps(details) for details in batch_details)
            # Single logging call instead of N individual calls
            eval_logger.info(batch_json)
        except Exception as e:
            logger.warning(f'Failed to batch log evaluation details: {e}')
            # Fallback to individual logging if batch fails
            for details in batch_details:
                try:
                    eval_logger.info(json.dumps(details))
                except Exception as inner_e:
                    logger.warning(f'Failed to log individual sample: {inner_e}')

    def _generate_batch_predictions_with_details(self, batch: Dict[str, Any], max_new_tokens: int) -> Tuple[List[str], List[str], List[int], List[float]]:
        import time
        
        device_batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch_size = find_batch_size(batch)
        pixel_values = device_batch.get('pixel_values')
        input_ids = device_batch.get('input_ids')
        if input_ids is None or batch_size == 0:
            return ([''] * batch_size, [''] * batch_size, [0] * batch_size, [0.0] * batch_size)
        generation_config = self._get_generation_config(max_new_tokens)
        batch_predictions, batch_generated_texts, batch_generated_tokens, batch_latencies = ([], [], [], [])
        
        # Check if we have a LatentWrapper (CoCoNut mode)
        from .latent_wrapper import LatentWrapper
        is_latent_wrapper = isinstance(self.model, LatentWrapper)
        
        # Get the underlying model for chat interface
        underlying_model = self.model.model if is_latent_wrapper else self.model.model
        
        if hasattr(underlying_model, 'chat') and pixel_values is not None:
            # Use chat interface - this will handle latent injection if needed
            for i in range(batch_size):
                start_time = time.time()
                try:
                    sample_pixel_values = pixel_values[i:i + 1] if pixel_values is not None else None
                    question = device_batch['questions'][i] if 'questions' in device_batch else ''
                    
                    if is_latent_wrapper:
                        # Use LatentWrapper's chat method which handles latent injection
                        response = self.model.chat(tokenizer=self.tokenizer, pixel_values=sample_pixel_values.to(dtype=next(self.model.parameters()).dtype), question=question, generation_config=generation_config)
                    else:
                        # Use underlying model's chat method for vanilla/CoT modes
                        # Debug: Check tensor shapes before calling chat
                        logger.debug(f"Sample {i}: pixel_values shape = {sample_pixel_values.shape if sample_pixel_values is not None else 'None'}")
                        logger.debug(f"Sample {i}: question = {question[:50]}...")
                        
                        response = underlying_model.chat(tokenizer=self.tokenizer, pixel_values=sample_pixel_values.to(dtype=next(self.model.parameters()).dtype), question=question, generation_config=generation_config)
                    
                    batch_predictions.append(extract_answer_choice(response))
                    batch_generated_texts.append(response)
                    response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
                    batch_generated_tokens.append(len(response_tokens))
                    
                except Exception as e:
                    logger.error(f"Error processing sample {i}: {e}")
                    logger.error(f"Pixel values shape: {pixel_values.shape if pixel_values is not None else 'None'}")
                    logger.error(f"Input IDs shape: {input_ids.shape if input_ids is not None else 'None'}")
                    logger.error(f"Batch size: {batch_size}")
                    raise e
                finally:
                    latency = time.time() - start_time
                    batch_latencies.append(latency)
        else:
            # Use generate interface - this will also handle latent injection if needed
            start_time = time.time()
            if is_latent_wrapper:
                # Use LatentWrapper's generate method
                generated_ids = self.model.generate(pixel_values=pixel_values, input_ids=input_ids, attention_mask=device_batch.get('attention_mask'), pad_token_id=self.tokenizer.eos_token_id, **generation_config)
            else:
                # Use base model's generate method
                generated_ids = self.model.generate(pixel_values=pixel_values, input_ids=input_ids, attention_mask=device_batch.get('attention_mask'), pad_token_id=self.tokenizer.eos_token_id, **generation_config)
            
            batch_latency = time.time() - start_time
            
            input_length = input_ids.shape[1]
            for i in range(batch_size):
                gen_part = generated_ids[i, input_length:]
                full_text = self.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                gen_text = self.tokenizer.decode(gen_part, skip_special_tokens=True)
                batch_predictions.append(extract_answer_choice(gen_text))
                batch_generated_texts.append(full_text)
                batch_generated_tokens.append(len(gen_part.tolist()))
                batch_latencies.append(batch_latency / batch_size)  # Average per sample
        return (batch_predictions, batch_generated_texts, batch_generated_tokens, batch_latencies)

    def _compute_evaluation_metrics(self, predictions: List[str], labels: List[str], prefix: str) -> Dict[str, float]:
        if not predictions or not labels:
            return {f'{prefix}_accuracy': 0.0}
        correct = sum((1 for pred, label in zip(predictions, labels) if pred.lower().strip() == label.lower().strip()))
        accuracy = correct / len(labels) if labels else 0.0
        return {f'{prefix}_accuracy': accuracy, f'{prefix}_num_samples': len(labels), f'{prefix}_correct': correct}

    @property
    def tokenizer(self):
        if hasattr(self.model, 'tokenizer'):
            return self.model.tokenizer
        elif hasattr(self.model, 'module') and hasattr(self.model.module, 'tokenizer'):
            return self.model.module.tokenizer
        raise AttributeError('Tokenizer not found in model')

    def _log_training_data_sample(self, batch: Dict, epoch: int, step: int) -> None:
        if not (step == 0 and epoch == 0):
            return
        if 'wandb' in self.args.report_to:
            try:
                import wandb
                if wandb.run and self.is_world_process_zero():
                    input_ids = batch.get('input_ids', [])
                    labels = batch.get('labels', [])
                    if hasattr(self, 'tokenizer') and len(input_ids) > 0:
                        columns = ['step', 'sample_id', 'token_id', 'label_id', 'token_text']
                        data = []
                        max_samples = min(2, len(input_ids))
                        for sample_idx in range(max_samples):
                            sample_input_ids = input_ids[sample_idx]
                            sample_labels = labels[sample_idx] if sample_idx < len(labels) else None
                            max_tokens = min(50, len(sample_input_ids))
                            for token_idx in range(max_tokens):
                                token_id = sample_input_ids[token_idx].item() if hasattr(sample_input_ids[token_idx], 'item') else sample_input_ids[token_idx]
                                label_id = sample_labels[token_idx].item() if sample_labels is not None and hasattr(sample_labels[token_idx], 'item') else -100
                                token_text = self.tokenizer.decode([token_id]) if hasattr(self, 'tokenizer') else f'token_{token_id}'
                                data.append([self.total_train_steps, sample_idx, token_id, label_id, token_text.replace('\n', '\\n')])
                        if data:
                            training_data_table = wandb.Table(columns=columns, data=data)
                            wandb.log({'train/data_samples': training_data_table})
            except Exception as e:
                logger.warning(f'Failed to log training data samples: {e}')

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if hasattr(self, 'state') and self.state.global_step == 0:
            self._log_training_data_sample(inputs, self.state.epoch or 0, 0)
        return super().training_step(model, inputs)

    def _track_best_performance(self, metrics: Dict[str, float], epoch: int, checkpoint_dir: str) -> bool:
        acc_key = None
        for key in ['accuracy', 'acc', 'eval_accuracy']:
            if key in metrics:
                acc_key = key
                break
        if acc_key is None:
            logger.warning('No accuracy metric found for best model tracking')
            return False
        current_acc = metrics[acc_key]
        if not hasattr(self, 'best_accuracy'):
            self.best_accuracy = 0.0
            self.best_epoch = -1
            self.best_checkpoint = None
        is_best = current_acc > self.best_accuracy
        if is_best:
            self.best_accuracy = current_acc
            self.best_epoch = epoch
            self.best_checkpoint = checkpoint_dir
            logger.info(f'🎉 New best accuracy: {current_acc:.4f} at epoch {epoch + 1}')
            if 'wandb' in self.args.report_to:
                try:
                    import wandb
                    if wandb.run:
                        best_metrics = {'best/accuracy': self.best_accuracy, 'best/epoch': self.best_epoch + 1, 'best/checkpoint': checkpoint_dir}
                        for key, value in metrics.items():
                            best_metrics[f'best/{key}'] = value
                        wandb.log(best_metrics)
                        logger.info(f'Updated best model metrics in wandb: {best_metrics}')
                except ImportError:
                    pass
        else:
            logger.info(f'Current accuracy: {current_acc:.4f}, Best: {self.best_accuracy:.4f} (epoch {self.best_epoch + 1})')
        return is_best

    def _log_performance_summary(self) -> None:
        if hasattr(self, 'best_accuracy') and 'wandb' in self.args.report_to:
            try:
                import wandb
                if wandb.run and self.is_world_process_zero():
                    summary_metrics = {'summary/best_accuracy': self.best_accuracy, 'summary/best_epoch': self.best_epoch + 1, 'summary/total_train_steps': self.total_train_steps}
                    if hasattr(self, 'best_checkpoint'):
                        summary_metrics['summary/best_checkpoint'] = self.best_checkpoint
                    wandb.log(summary_metrics)
                    logger.info(f'Logged training summary to wandb: {summary_metrics}')
                    wandb.finish()
            except ImportError:
                pass

    def _update_for_stage(self, stage: int) -> None:
        logger.info(f'Transitioning to CoCoNut stage {stage}')
        if hasattr(self.train_dataset, 'apply_progressive_curriculum'):
            if hasattr(self.train_dataset, 'data') and len(self.train_dataset.data) > 0:
                sample_before = self.train_dataset.data[0] if len(self.train_dataset.data) > 0 else None
                logger.info(f"Dataset sample before curriculum update (stage {stage}): steps={(sample_before.get('steps', 'N/A') if sample_before else 'No data')}")
            self.train_dataset.apply_progressive_curriculum(scheduled_stage=stage, c_thought=self.args.c_thought, max_latent_stage=self.args.max_latent_stage, uniform_prob=self.args.uniform_prob, pad_latent_to_max=self.args.pad_latent_to_max, no_cot=False)
            if hasattr(self.train_dataset, 'data') and len(self.train_dataset.data) > 0:
                sample_after = self.train_dataset.data[0] if len(self.train_dataset.data) > 0 else None
                logger.info(f"Dataset sample after curriculum update (stage {stage}): steps={(sample_after.get('steps', 'N/A') if sample_after else 'No data')}")
            logger.info(f'Applied progressive curriculum for stage {stage} - Dataset size: {len(self.train_dataset)}')
        else:
            logger.warning('Training dataset does not support progressive curriculum')
        if hasattr(self, '_last_train_dataloader'):
            del self._last_train_dataloader
        logger.info('Dataloader will be refreshed for updated curriculum')
        if hasattr(self.args, 'reset_optimizer') and self.args.reset_optimizer:
            logger.info('Resetting optimizer and scheduler for new stage')
            # Calculate remaining training steps for the new stage
            train_dataloader = self.get_train_dataloader()
            steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            current_epoch = getattr(self.state, 'epoch', None) or 0
            remaining_epochs = int(self.args.num_train_epochs) - current_epoch
            remaining_steps = steps_per_epoch * max(1, remaining_epochs)
            
            # Recreate both optimizer and scheduler for the new stage
            self.create_optimizer_and_scheduler(remaining_steps)
            logger.info(f'Reset optimizer and scheduler with {remaining_steps} remaining training steps')
        if 'wandb' in self.args.report_to:
            try:
                import wandb
                if wandb.run:
                    stage_transition = {'coconut/stage_transition': stage, 'coconut/latent_tokens_count': stage * self.args.c_thought, 'coconut/stage_timestamp': time.time(), 'coconut/dataset_size_after_update': len(self.train_dataset) if hasattr(self, 'train_dataset') else 0}
                    wandb.log(stage_transition)
                    logger.info(f'Logged stage transition to wandb: stage {stage}')
            except ImportError:
                pass

    def _log_coconut_stage_metrics(self, current_stage: int, stage_epoch: int, stage_progress: float) -> None:
        if 'wandb' in self.args.report_to:
            try:
                import wandb
                if wandb.run:
                    # Calculate latent/explicit token ratio for this stage
                    latent_token_count = current_stage * getattr(self.args, 'c_thought', 0)
                    total_possible_latent = self.args.max_latent_stage * getattr(self.args, 'c_thought', 0)
                    latent_explicit_ratio = latent_token_count / max(1, total_possible_latent)
                    
                    stage_metrics = {
                        'coconut/current_stage': current_stage, 
                        'coconut/stage_epoch': stage_epoch, 
                        'coconut/stage_progress': stage_progress, 
                        'coconut/latent_replacement_ratio': current_stage / max(1, self.args.max_latent_stage),
                        'coconut/latent_token_count': latent_token_count,
                        'coconut/latent_explicit_ratio': latent_explicit_ratio
                    }
                    wandb.log(stage_metrics)
            except ImportError:
                pass

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.create_optimizer()
        self.lr_scheduler = self.create_scheduler(num_training_steps, self.optimizer)
        logger.info(f'Optimizer created with initial LR: {self.get_lr()}')
        logger.info(f'Scheduler: {type(self.lr_scheduler).__name__}, num_training_steps={num_training_steps}')
        if hasattr(self.args, 'warmup_steps'):
            logger.info(f'Warmup steps: {self.args.warmup_steps}')
        else:
            logger.info('No warmup steps configured')
        if hasattr(self.args, 'max_grad_norm'):
            logger.info(f'Gradient clipping enabled with max_grad_norm: {self.args.max_grad_norm}')
        else:
            logger.info('No gradient clipping configured')

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer=None):
        if optimizer is None:
            optimizer = self.optimizer
        warmup_steps = getattr(self.args, 'warmup_steps', 0)
        scheduler_type = getattr(self.args, 'lr_scheduler_type', 'linear')
        if scheduler_type.lower() == 'cosine':
            try:
                from transformers.optimization import get_cosine_schedule_with_warmup
                scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
                logger.info(f'Created cosine scheduler with warmup_steps={warmup_steps}, total_steps={num_training_steps}')
                return scheduler
            except ImportError:
                logger.warning('Could not import get_cosine_schedule_with_warmup, falling back to default scheduler')
                return super().create_scheduler(num_training_steps, optimizer)
        else:
            logger.info(f'Using default linear scheduler with warmup_steps={warmup_steps}, total_steps={num_training_steps}')
            return super().create_scheduler(num_training_steps, optimizer)

    def _get_generation_config(self, max_new_tokens: int) -> Dict[str, Any]:
        generation_config = {'max_new_tokens': max_new_tokens}
        if self.runner and hasattr(self.runner, 'config') and hasattr(self.runner.config, 'generation'):
            config_dict = self.runner.config.generation
            if isinstance(config_dict, dict):
                supported_params = ['do_sample', 'num_beams', 'temperature', 'top_p', 'top_k']
                for param in supported_params:
                    if param in config_dict:
                        generation_config[param] = config_dict[param]
        else:
            generation_config.update({'do_sample': False, 'num_beams': 1, 'temperature': 1.0, 'top_p': 1.0})
        logger.debug(f'Using generation config: {generation_config}')
        return generation_config

    def _should_evaluate_after_epoch(self, epoch: int) -> bool:
        """
        Check if evaluation should run after the given epoch based on eval_strategy.
        """
        eval_strategy = getattr(self.args, 'eval_strategy', 'epoch')
        
        if eval_strategy == 'no':
            # Never evaluate during training
            return False
        elif eval_strategy == 'epoch':
            # Evaluate after every epoch (default behavior)
            return True
        elif eval_strategy == 'steps':
            # For steps-based evaluation, we still evaluate at epoch end
            # unless specifically configured otherwise
            return True
        else:
            # Default to evaluating if strategy is unknown
            logger.warning(f'Unknown eval_strategy: {eval_strategy}, defaulting to evaluation')
            return True