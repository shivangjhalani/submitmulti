import os
import random
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from .constants import COCONUT_SPECIAL_TOKENS, DEFAULT_BATCH_SIZE, DEFAULT_C_THOUGHT, DEFAULT_EVAL_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_MAX_LATENT_STAGE, DEFAULT_MODEL_NAME, DEFAULT_NUM_EPOCHS, DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)

class TrainingMode(str, Enum):
    EVAL_ONLY = 'eval_only'
    COT_TRAIN = 'cot_train'
    COCONUT_TRAIN = 'coconut_train'

@dataclass
class EvaluationConfig:
    vanilla: bool = True
    cot: bool = False
    coconut: bool = False
    eval_latent_tokens: Optional[int] = None
    log_per_sample: bool = False
    detailed_logging: bool = False
    log_latency: bool = True  # Track wall clock latency during evaluation

    def get_eval_type(self) -> str:
        if self.coconut:
            return 'coconut'
        elif self.cot:
            return 'cot'
        return 'vanilla'

@dataclass
class CoCoNutConfig:
    enabled: bool = False
    c_thought: int = DEFAULT_C_THOUGHT
    max_latent_stage: int = DEFAULT_MAX_LATENT_STAGE
    epochs_per_stage: int = 1
    special_tokens: List[str] = field(default_factory=lambda: COCONUT_SPECIAL_TOKENS.copy())
    uniform_prob: float = 0.0
    pad_latent_to_max: bool = False
    reset_optimizer: bool = True

@dataclass
class DataConfig:
    data_dir: str = ''
    train_data_path: Optional[str] = None
    eval_data_path: Optional[str] = None
    limit_for_testing: Union[bool, int] = False

    def __post_init__(self):
        if self.data_dir:
            self.data_dir = os.path.abspath(self.data_dir)
        if self.train_data_path:
            self.train_data_path = os.path.abspath(self.train_data_path)
        if self.eval_data_path:
            self.eval_data_path = os.path.abspath(self.eval_data_path)

@dataclass
class ModelConfig:
    model_name: str = DEFAULT_MODEL_NAME
    config_id: Optional[str] = None
    tokenizer_id: Optional[str] = None
    image_processor_id: Optional[str] = None
    torch_dtype: str = 'bfloat16'
    trust_remote_code: bool = True
    low_cpu_mem_usage: bool = True
    load_model_path: Optional[str] = None
    torch_compile: bool = False
    use_flash_attention_2: bool = False

    def __post_init__(self):
        """IMPROVEMENT: Add multimodal-specific validation."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Validate multimodal configuration
        if self._is_multimodal_model():
            if not self.image_processor_id:
                logger.warning(f"Multimodal model {self.model_name} detected but no image_processor_id specified. Using model_name as fallback.")
                self.image_processor_id = self.model_name
            
            # InternVL-specific validation
            if 'internvl' in self.model_name.lower():
                if not self.trust_remote_code:
                    raise ValueError("InternVL models require trust_remote_code=True")
                
                # Recommend optimal settings
                if self.torch_dtype not in ['bfloat16', 'float16']:
                    logger.warning(f"For InternVL models, torch_dtype='{self.torch_dtype}' may not be optimal. Consider 'bfloat16' or 'float16'")
        
        # Validate torch_dtype
        valid_dtypes = ['auto', 'float16', 'bfloat16', 'float32']
        if self.torch_dtype not in valid_dtypes:
            raise ValueError(f"torch_dtype must be one of {valid_dtypes}, got {self.torch_dtype}")
    
    def _is_multimodal_model(self) -> bool:
        """Check if the model is multimodal based on name patterns."""
        multimodal_patterns = ['internvl', 'llava', 'qwen-vl', 'blip', 'flamingo', 'clip']
        return any(pattern in self.model_name.lower() for pattern in multimodal_patterns)
    
    def get_model_type(self) -> str:
        """Get the model type for logging purposes."""
        if self._is_multimodal_model():
            return 'multimodal'
        return 'text-only'

    def get_special_tokens(self, coconut_config: CoCoNutConfig) -> List[str]:
        return []

@dataclass
class TrainingConfig:
    output_dir: str = DEFAULT_OUTPUT_DIR
    num_epochs: int = DEFAULT_NUM_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE
    gradient_accumulation_steps: int = 1
    eval_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: Dict[str, Any] = field(default_factory=lambda: {'use_reentrant': False})
    learning_rate: float = DEFAULT_LEARNING_RATE
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = 'linear'
    logging_steps: int = 10
    save_steps: int = 1000
    eval_steps: int = 1000
    eval_strategy: str = 'epoch'
    save_strategy: str = 'epoch'
    skip_eval_during_training: bool = False
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = 'accuracy'
    greater_is_better: bool = False
    bf16: bool = True
    fp16: bool = False
    remove_unused_columns: bool = False
    resume_from_checkpoint: bool = False
    dataloader_pin_memory: bool = False
    dataloader_num_workers: int = 4
    weight_decay: float = 0.01
    seed: Optional[int] = None
    data_seed: Optional[int] = None
    mode: TrainingMode = TrainingMode.COT_TRAIN
    name: Optional[str] = None
    max_checkpoints_to_keep: int = 3
    keep_best_checkpoints: bool = True
    use_run_name_in_output_dir: bool = True

    def __post_init__(self):
        if self.seed is None:
            self.seed = random.randint(0, 2 ** 32 - 1)
        if self.data_seed is None:
            self.data_seed = self.seed
        if self.use_run_name_in_output_dir and self.name:
            base_dir = os.path.dirname(self.output_dir) or 'checkpoints'
            dir_name = os.path.basename(self.output_dir)
            self.output_dir = os.path.join(base_dir, f'{dir_name}_{self.name}')
        os.makedirs(self.output_dir, exist_ok=True)
        if self.mode == TrainingMode.EVAL_ONLY:
            self.load_best_model_at_end = False

@dataclass
class LoggingConfig:
    log_dir: str = 'logs'
    log_level: str = 'INFO'
    use_wandb: bool = True
    log_to_file: bool = True
    console_output: bool = True
    verbose: bool = False
    run_name: Optional[str] = None
    project: str = 'multicoco'

    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)

@dataclass
class MultiCoCoConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    coconut: CoCoNutConfig = field(default_factory=CoCoNutConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    generation: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        self._validate_training()
        self._validate_coconut()
        self._validate_data_requirements()
        self._validate_file_existence()
        self._validate_generation_config()
        self._validate_logging()
        self._validate_multimodal_config()

    def _validate_multimodal_config(self) -> None:
        """Validate multimodal-specific configuration."""
        # Check if model appears to be multimodal
        model_name = self.model.model_name.lower()
        is_likely_multimodal = any(keyword in model_name for keyword in ['intern', 'llava', 'blip', 'flamingo', 'clip'])
        
        if is_likely_multimodal:
            # For multimodal models, ensure we have image processor configuration
            if not self.model.image_processor_id and not any('vision' in str(self.model.__dict__).lower() for _ in [None]):
                logger.info(f'Model {self.model.model_name} appears multimodal but no explicit image_processor_id set. Will use model default.')
            
            # Warn if using very large batch sizes with images (potential OOM)
            if self.training.batch_size > 4:
                logger.warning(f'Large batch size ({self.training.batch_size}) with multimodal model may cause OOM. Consider reducing batch_size.')
                

    def _validate_training(self) -> None:
        if self.training.learning_rate <= 0:
            raise ValueError('learning_rate must be positive')
        if self.training.batch_size <= 0:
            raise ValueError('batch_size must be positive')
        if self.training.num_epochs <= 0:
            raise ValueError('num_epochs must be positive')
        if self.training.bf16 and self.training.fp16:
            raise ValueError('Cannot enable both bf16 and fp16 simultaneously')
        valid_save_strategies = ['epoch', 'steps', 'no']
        if self.training.save_strategy not in valid_save_strategies:
            raise ValueError(f'save_strategy must be one of {valid_save_strategies}, got {self.training.save_strategy}')
        valid_eval_strategies = ['epoch', 'steps', 'no']
        if self.training.eval_strategy not in valid_eval_strategies:
            raise ValueError(f'eval_strategy must be one of {valid_eval_strategies}, got {self.training.eval_strategy}')
        valid_schedulers = ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']
        if self.training.lr_scheduler_type not in valid_schedulers:
            raise ValueError(f'lr_scheduler_type must be one of {valid_schedulers}, got {self.training.lr_scheduler_type}')
        if self.training.load_best_model_at_end and (not self.training.metric_for_best_model):
            raise ValueError('metric_for_best_model must be specified when load_best_model_at_end is True')

    def _validate_coconut(self) -> None:
        if self.coconut.c_thought < 0:
            raise ValueError('c_thought must be non-negative')
        if self.coconut.max_latent_stage < 0:
            raise ValueError('max_latent_stage must be non-negative')
        if self.coconut.epochs_per_stage < 0:
            raise ValueError('epochs_per_stage must be non-negative')
        if not 0.0 <= self.coconut.uniform_prob <= 1.0:
            raise ValueError('uniform_prob must be between 0.0 and 1.0')

    def _validate_data_requirements(self) -> None:
        is_training = self.training.mode != TrainingMode.EVAL_ONLY
        if is_training and (not self.data.train_data_path):
            raise ValueError('Training data path required for training modes')
        if self.training.mode == TrainingMode.EVAL_ONLY and (not self.data.eval_data_path):
            raise ValueError('Evaluation data path required for eval_only mode')
        
        # Only validate coconut compatibility for training modes, not evaluation-only
        if is_training and self.coconut.enabled and (not any([self.evaluation.coconut, self.evaluation.cot])):
            raise ValueError('CoCoNut training enabled but no compatible evaluation configured')

    def _validate_file_existence(self) -> None:
        if self.data.train_data_path and (not os.path.exists(self.data.train_data_path)):
            raise FileNotFoundError(f'Training data not found: {self.data.train_data_path}')
        if self.data.eval_data_path and (not os.path.exists(self.data.eval_data_path)):
            raise FileNotFoundError(f'Evaluation data not found: {self.data.eval_data_path}')

    def _validate_logging(self) -> None:
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.logging.log_level.upper() not in valid_log_levels:
            raise ValueError(f'log_level must be one of {valid_log_levels}, got {self.logging.log_level}')
        if not self.logging.project or not isinstance(self.logging.project, str):
            raise ValueError('project name must be a non-empty string')

    def _validate_generation_config(self) -> None:
        if not isinstance(self.generation, dict):
            raise ValueError('generation config must be a dictionary')
        if 'max_new_tokens' in self.generation:
            if not isinstance(self.generation['max_new_tokens'], int) or self.generation['max_new_tokens'] <= 0:
                raise ValueError('max_new_tokens must be a positive integer')
        if 'temperature' in self.generation:
            if not isinstance(self.generation['temperature'], (int, float)) or self.generation['temperature'] <= 0:
                raise ValueError('temperature must be a positive number')
        if 'top_p' in self.generation:
            if not isinstance(self.generation['top_p'], (int, float)) or not 0 < self.generation['top_p'] <= 1:
                raise ValueError('top_p must be a number between 0 and 1')
        if 'top_k' in self.generation:
            if not isinstance(self.generation['top_k'], int) or self.generation['top_k'] < 0:
                raise ValueError('top_k must be a non-negative integer')
        if 'num_beams' in self.generation:
            if not isinstance(self.generation['num_beams'], int) or self.generation['num_beams'] <= 0:
                raise ValueError('num_beams must be a positive integer')

    @classmethod
    def load_with_base(cls, config_path: str, base_config_path: str='args/base.yaml') -> 'MultiCoCoConfig':
        import yaml
        base_dict = cls._load_yaml_file(base_config_path) if os.path.exists(base_config_path) else {}
        config_dict = cls._load_yaml_file(config_path)
        merged_dict = cls._merge_configs(base_dict, config_dict)
        return cls.from_dict(merged_dict)

    @staticmethod
    def _load_yaml_file(file_path: str) -> Dict[str, Any]:
        import yaml
        with open(file_path, 'r') as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _merge_configs(base_dict: Dict[str, Any], config_dict: Dict[str, Any]) -> Dict[str, Any]:
        merged_dict = {**base_dict, **config_dict}
        for key in ['eval_config', 'coconut', 'generation']:
            if key in base_dict and key in config_dict and isinstance(base_dict[key], dict) and isinstance(config_dict[key], dict):
                merged_dict[key] = {**base_dict[key], **config_dict[key]}
        return merged_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MultiCoCoConfig':
        torch_dtype = cls._determine_torch_dtype(config_dict)
        config_builders = {'model': lambda: cls._build_model_config(config_dict, torch_dtype), 'training': lambda: cls._build_training_config(config_dict), 'data': lambda: cls._build_data_config(config_dict), 'evaluation': lambda: cls._build_evaluation_config(config_dict), 'coconut': lambda: cls._build_coconut_config(config_dict)}
        configs = {name: builder() for name, builder in config_builders.items()}
        configs['logging'] = cls._build_logging_config(config_dict, configs['training'])
        configs['generation'] = config_dict.get('generation', {})
        return cls(**configs)

    @staticmethod
    def _determine_torch_dtype(config_dict: Dict[str, Any]) -> str:
        if config_dict.get('bf16', True):
            return 'bfloat16'
        elif config_dict.get('fp16', False):
            return 'float16'
        return 'float32'

    @staticmethod
    def _build_model_config(config_dict: Dict[str, Any], torch_dtype: str) -> ModelConfig:
        return ModelConfig(model_name=config_dict.get('model_name', DEFAULT_MODEL_NAME), torch_dtype=torch_dtype, config_id=config_dict.get('config_id'), tokenizer_id=config_dict.get('tokenizer_id'), image_processor_id=config_dict.get('image_processor_id'), trust_remote_code=config_dict.get('trust_remote_code', True), low_cpu_mem_usage=config_dict.get('low_cpu_mem_usage', True), load_model_path=config_dict.get('load_model_path'), torch_compile=config_dict.get('torch_compile', False), use_flash_attention_2=config_dict.get('use_flash_attention_2', False))

    @staticmethod
    def _build_training_config(config_dict: Dict[str, Any]) -> TrainingConfig:
        name = config_dict.get('name') or config_dict.get('run_name')
        
        # Handle potential config conflicts
        skip_eval = config_dict.get('skip_eval_during_training', False)
        load_best = config_dict.get('load_best_model_at_end', True)
        
        # Auto-correct conflicting settings
        if skip_eval and load_best:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "Auto-correcting conflicting config: skip_eval_during_training=True but "
                "load_best_model_at_end=True. Setting load_best_model_at_end=False."
            )
            load_best = False
        
        return TrainingConfig(output_dir=config_dict.get('output_dir', DEFAULT_OUTPUT_DIR), num_epochs=config_dict.get('num_epochs', DEFAULT_NUM_EPOCHS), batch_size=config_dict.get('batch_size', DEFAULT_BATCH_SIZE), eval_batch_size=config_dict.get('eval_batch_size', DEFAULT_EVAL_BATCH_SIZE), learning_rate=float(config_dict.get('learning_rate', DEFAULT_LEARNING_RATE)), gradient_accumulation_steps=config_dict.get('gradient_accumulation_steps', 1), eval_accumulation_steps=config_dict.get('eval_accumulation_steps', 1), resume_from_checkpoint=config_dict.get('resume_from_checkpoint', False), mode=TrainingMode(config_dict.get('mode', 'cot_train')), bf16=config_dict.get('bf16', True), fp16=config_dict.get('fp16', False), gradient_checkpointing=config_dict.get('gradient_checkpointing', True), gradient_checkpointing_kwargs=config_dict.get('gradient_checkpointing_kwargs', {'use_reentrant': False}), warmup_steps=config_dict.get('warmup_steps', 500), max_grad_norm=config_dict.get('max_grad_norm', 1.0), lr_scheduler_type=config_dict.get('lr_scheduler_type', 'linear'), logging_steps=config_dict.get('logging_steps', 10), save_steps=config_dict.get('save_steps', 1000), eval_steps=config_dict.get('eval_steps', 1000), eval_strategy=config_dict.get('eval_strategy', 'epoch'), save_strategy=config_dict.get('save_strategy', 'epoch'), skip_eval_during_training=skip_eval, save_total_limit=config_dict.get('save_total_limit', 2), max_checkpoints_to_keep=config_dict.get('max_checkpoints_to_keep', 3), keep_best_checkpoints=config_dict.get('keep_best_checkpoints', True), use_run_name_in_output_dir=config_dict.get('use_run_name_in_output_dir', True), load_best_model_at_end=load_best, metric_for_best_model=config_dict.get('metric_for_best_model', 'accuracy'), greater_is_better=config_dict.get('greater_is_better', False), dataloader_num_workers=config_dict.get('dataloader_num_workers', 4), weight_decay=config_dict.get('weight_decay', 0.01), seed=config_dict.get('seed'), data_seed=config_dict.get('data_seed'), name=name)

    @staticmethod
    def _build_data_config(config_dict: Dict[str, Any]) -> DataConfig:
        eval_data_path = config_dict.get('eval_data_path') or config_dict.get('val_data_path')
        return DataConfig(data_dir=config_dict.get('data_dir', ''), train_data_path=config_dict.get('train_data_path'), eval_data_path=eval_data_path, limit_for_testing=config_dict.get('limit_for_testing', False))

    @staticmethod
    def _build_evaluation_config(config_dict: Dict[str, Any]) -> EvaluationConfig:
        eval_config_dict = config_dict.get('eval_config', {})
        return EvaluationConfig(vanilla=eval_config_dict.get('vanilla', True), coconut=eval_config_dict.get('coconut', False), cot=eval_config_dict.get('cot', False), eval_latent_tokens=eval_config_dict.get('eval_latent_tokens'), log_per_sample=eval_config_dict.get('log_per_sample', False), detailed_logging=eval_config_dict.get('detailed_logging', False), log_latency=eval_config_dict.get('log_latency', True))

    @staticmethod
    def _build_coconut_config(config_dict: Dict[str, Any]) -> CoCoNutConfig:
        coconut_dict = config_dict.get('coconut', {})
        if isinstance(coconut_dict, bool):
            coconut_enabled = coconut_dict
            coconut_dict = {}
        else:
            coconut_enabled = coconut_dict.get('enabled', config_dict.get('coconut', False))

        def get_coconut_value(key: str, default: Any) -> Any:
            return coconut_dict.get(key, config_dict.get(key, default))
        return CoCoNutConfig(enabled=coconut_enabled, c_thought=get_coconut_value('c_thought', DEFAULT_C_THOUGHT), max_latent_stage=get_coconut_value('max_latent_stage', DEFAULT_MAX_LATENT_STAGE), epochs_per_stage=get_coconut_value('epochs_per_stage', 1), uniform_prob=get_coconut_value('uniform_prob', 0.0), pad_latent_to_max=get_coconut_value('pad_latent_to_max', False), reset_optimizer=get_coconut_value('reset_optimizer', True))

    @staticmethod
    def _build_logging_config(config_dict: Dict[str, Any], training_config: TrainingConfig) -> LoggingConfig:
        logging_dict = config_dict.get('logging', {})
        return LoggingConfig(log_dir=logging_dict.get('log_dir', 'logs'), log_level=logging_dict.get('log_level', 'INFO'), use_wandb=logging_dict.get('use_wandb', True), log_to_file=logging_dict.get('log_to_file', True), console_output=logging_dict.get('console_output', True), verbose=logging_dict.get('verbose', False), run_name=training_config.name or logging_dict.get('run_name'), project=logging_dict.get('project', 'multicoco'))

    def get_wandb_report_to(self) -> List[str]:
        return ['wandb'] if self.logging.use_wandb else []