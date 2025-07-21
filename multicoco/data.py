import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from PIL import Image
from torch.utils.data import Dataset
from .constants import DEFAULT_MAX_LENGTH, END_LATENT_TOKEN, FALLBACK_IMAGE_SIZE, IMAGE_TOKEN, LATENT_TOKEN, LOSS_IGNORE_INDEX, START_LATENT_TOKEN
from .exceptions import DataLoadingError, DatasetError, ImageProcessingError
logger = logging.getLogger(__name__)

class SupervisedDataset(Dataset):

    def __init__(self, data_path: str, data_dir: str, test_limit: Optional[int]=None) -> None:
        super().__init__()
        self._validate_paths(data_path, data_dir)
        self.data = self._load_data(data_path, test_limit)
        self.data_dir = data_dir
        self._original_data = self.data.copy()
        logger.info(f'Loaded {len(self.data)} samples from {data_path}')

    def _validate_paths(self, data_path: str, data_dir: str) -> None:
        for path, name in [(data_path, 'Data file'), (data_dir, 'Data directory')]:
            if not os.path.exists(path):
                raise DataLoadingError(f'{name} not found: {path}')

    def _load_data(self, data_path: str, test_limit: Optional[int]) -> List[Dict]:
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise DataLoadingError(f'Failed to load data from {data_path}: {e}') from e
        if test_limit is not None:
            data = data[:test_limit]
            logger.info(f'Limited dataset to {test_limit} samples for testing')
        return data

    def __len__(self) -> int:
        return len(self.data)

    def apply_progressive_curriculum(self, scheduled_stage: int, c_thought: int, max_latent_stage: int, uniform_prob: float=0.0, pad_latent_to_max: bool=False, no_cot: bool=False) -> None:
        logger.info(f'Applying progressive curriculum for stage {scheduled_stage}')
        self.data = create_progressive_latent_dataset(scheduled_stage=scheduled_stage, base_dataset=self._original_data, c_thought=c_thought, max_latent_stage=max_latent_stage, uniform_prob=uniform_prob, pad_latent_to_max=pad_latent_to_max, no_cot=no_cot)
        logger.info(f'Dataset updated with {len(self.data)} curriculum samples')

    def __getitem__(self, index: int) -> Dict[str, Union[Image.Image, str]]:
        if index >= len(self.data):
            raise DatasetError(f'Index {index} out of range for dataset of size {len(self.data)}')
        item = self.data[index]
        self._validate_item(item, index)
        image = self._load_image(item['image'])
        question = item['question']
        answer = item.get('answer', item.get('direct_answer', ''))
        result = {'image': image, 'question': question, 'answer': answer}
        if (steps := item.get('steps')):
            result['steps'] = steps
        if (reasoning := item.get('reasoning')):
            result['reasoning'] = reasoning
        return result

    def _validate_item(self, item: Dict, index: int) -> None:
        required_fields = ['image', 'question']
        missing_fields = [field for field in required_fields if field not in item]
        if missing_fields:
            raise DatasetError(f'Sample {index} missing fields: {missing_fields}')

    def _load_image(self, image_file: str) -> Image.Image:
        if os.path.isabs(image_file):
            image_path = image_file
        else:
            image_path = os.path.join(self.data_dir, image_file)
        try:
            if not os.path.exists(image_path):
                logger.warning(f'Image file not found: {image_path}')
                return Image.new('RGB', (FALLBACK_IMAGE_SIZE, FALLBACK_IMAGE_SIZE), color=(0, 0, 0))
            image = Image.open(image_path)
            return image.convert('RGB')
        except (OSError, IOError) as e:
            logger.warning(f'Failed to load image {image_path}: {e}')
            return Image.new('RGB', (FALLBACK_IMAGE_SIZE, FALLBACK_IMAGE_SIZE), color=(0, 0, 0))
        except Exception as e:
            logger.warning(f'Unexpected error loading image {image_path}: {e}')
            return Image.new('RGB', (FALLBACK_IMAGE_SIZE, FALLBACK_IMAGE_SIZE), color=(0, 0, 0))

def collate_fn(batch: List[Dict[str, Any]], tokenizer: Any, image_processor: Any) -> Dict[str, torch.Tensor]:
    if not batch:
        raise DatasetError('Empty batch provided to collate function')
    try:
        images = [item['image'] for item in batch]
        questions = [item['question'] for item in batch]
        answers = [item['answer'] for item in batch]
        pixel_values = _process_images(images, image_processor)
        
        # Get the model to determine the correct number of image tokens
        # This is a workaround - ideally the model should be passed to collate_fn
        # For now, we'll use the standard num_image_token
        from .constants import IMG_CONTEXT_TOKEN
        
        full_texts, prompts = _create_chat_formatted_texts(batch, questions, answers, tokenizer)
        full_encodings = tokenizer(full_texts, padding=True, truncation=True, max_length=DEFAULT_MAX_LENGTH, return_tensors='pt', add_special_tokens=True)
        labels = _create_training_labels(full_encodings['input_ids'], prompts, tokenizer)
        return {'pixel_values': pixel_values, 'input_ids': full_encodings['input_ids'], 'attention_mask': full_encodings['attention_mask'], 'labels': labels, 'questions': questions, 'answers': answers}
    except Exception as e:
        raise DatasetError(f'Failed to collate batch: {e}') from e

def _create_chat_formatted_texts(batch: List[Dict[str, Any]], questions: List[str], answers: List[str], tokenizer: Any) -> Tuple[List[str], List[str]]:
    """
    Create chat formatted texts ensuring proper image-latent-text ordering.
    Fix: Ensure latent reasoning happens after image context is established.
    Updated: Use proper multimodal format with correct number of IMG_CONTEXT tokens.
    """
    from .constants import IMG_CONTEXT_TOKEN
    from .image_tokens import get_tokenizer_image_token_count
    
    full_texts = []
    prompts = []
    
    # FIX: Get the actual number of image tokens instead of hardcoding 256
    # This prevents the severe image token count mismatch that causes assertion failures
    # or silent truncation of visual information
    num_image_tokens = get_tokenizer_image_token_count(tokenizer, fallback=256)
    
    logger.debug(f"Using {num_image_tokens} IMG_CONTEXT tokens for multimodal formatting")
    
    # Create the proper multimodal format: <img><IMG_CONTEXT>×N</img>
    img_context = IMG_CONTEXT_TOKEN * num_image_tokens
    multimodal_image_token = f'<img>{img_context}</img>'
    
    for i, (question, answer) in enumerate(zip(questions, answers)):
        assistant_part = _build_assistant_response(batch[i], answer)
        # Use proper multimodal format instead of just IMAGE_TOKEN
        prompt = f'<|im_start|>user\n{multimodal_image_token}\n{question}<|im_end|><|im_start|>assistant\n'
        full_text = f'{prompt}{assistant_part}'
        full_texts.append(full_text)
        prompts.append(prompt)
    return (full_texts, prompts)

def _build_assistant_response(item: Dict[str, Any], answer: str) -> str:
    if (reasoning_text := item.get('reasoning', '')):
        return f'{reasoning_text} The answer is {answer}'
    elif (reasoning_steps := item.get('steps', [])):
        reasoning_combined = ' '.join(reasoning_steps)
        return f'{reasoning_combined} The answer is {answer}'
    return answer

def _process_images(images: List[Image.Image], image_processor: Any) -> torch.Tensor:
    try:
        processed = image_processor(images, return_tensors='pt')
        return processed['pixel_values']
    except Exception as e:
        raise ImageProcessingError(f'Error during image processing: {e}') from e

def _create_training_labels(input_ids: torch.Tensor, prompts: List[str], tokenizer: Any) -> torch.Tensor:
    labels = input_ids.clone()
    for i, prompt in enumerate(prompts):
        prompt_tokens = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids[0]
        prompt_length = len(prompt_tokens)
        labels[i, :prompt_length] = LOSS_IGNORE_INDEX
    return labels

def create_progressive_latent_dataset(scheduled_stage: int, base_dataset: List[Dict], c_thought: int, max_latent_stage: int, uniform_prob: float=0.0, pad_latent_to_max: bool=False, no_cot: bool=False) -> List[Dict]:
    """
    IMPROVEMENT: Enhanced progressive dataset creation with image-aware logic and randomness.
    """
    logger.info(f'Creating progressive latent dataset for stage {scheduled_stage}')
    processed_samples = []
    
    for sample in base_dataset:
        steps = _parse_reasoning_steps(sample.get('steps', []))
        
        # IMPROVEMENT: Add randomness to curriculum based on content type
        if random.random() < uniform_prob:
            # Random stage selection with bias towards visual reasoning steps
            stage_to_train = _select_random_stage_with_bias(steps, max_latent_stage, sample)
        else:
            stage_to_train = scheduled_stage
            
        n_skip_steps, n_latent_tokens = _calculate_curriculum_params(stage_to_train, max_latent_stage, steps, pad_latent_to_max, no_cot)
        
        # IMPROVEMENT: Apply image-aware latent token adjustment
        total_latent_tokens = _adjust_latent_tokens_for_multimodal(n_latent_tokens * c_thought, sample, steps)
        
        reasoning_text = _build_reasoning_text(total_latent_tokens, steps, n_skip_steps)
        processed_sample = {**sample, 'reasoning': reasoning_text, 'stage': stage_to_train, 'n_latent_tokens': total_latent_tokens, 'n_skip_steps': n_skip_steps}
        processed_samples.append(processed_sample)
    return processed_samples

def _select_random_stage_with_bias(steps: List[str], max_latent_stage: int, sample: Dict) -> int:
    """
    IMPROVEMENT: Select random stage with bias towards image-related reasoning.
    """
    import random  # Import at function start to avoid scoping issues
    
    # Check if steps contain visual reasoning keywords
    visual_keywords = ['see', 'look', 'image', 'picture', 'visual', 'color', 'shape', 'object', 'appears', 'shows']
    
    # Count visual reasoning steps
    visual_steps = []
    for i, step in enumerate(steps):
        if any(keyword in step.lower() for keyword in visual_keywords):
            visual_steps.append(i)
    
    # If we have visual reasoning steps, bias towards later stages that can handle more visual complexity
    if visual_steps:
        # Bias towards higher stages for visual reasoning
        weights = [1.0] * (len(steps) + 1)
        for i in range(len(visual_steps), len(steps) + 1):
            weights[i] *= 2.0  # Double weight for stages that include visual reasoning
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Weighted random selection
        return random.choices(range(len(steps) + 1), weights=weights)[0]
    else:
        # Standard uniform selection for non-visual samples
        return random.choice(range(len(steps) + 1))

def _adjust_latent_tokens_for_multimodal(base_latent_tokens: int, sample: Dict, steps: List[str]) -> int:
    """
    IMPROVEMENT: Adjust latent token count based on multimodal complexity.
    """
    if base_latent_tokens == 0:
        return 0
    
    # Check for visual complexity indicators
    question = sample.get('question', '').lower()
    visual_complexity_indicators = [
        'describe', 'detail', 'analyze', 'compare', 'identify', 'count', 
        'relationship', 'interaction', 'scene', 'complex'
    ]
    
    complexity_bonus = 0
    
    # Add bonus tokens for visually complex questions
    for indicator in visual_complexity_indicators:
        if indicator in question:
            complexity_bonus += 1
    
    # Add bonus for multi-step visual reasoning
    visual_reasoning_steps = sum(1 for step in steps if any(
        word in step.lower() for word in ['see', 'look', 'observe', 'notice', 'visual']
    ))
    
    if visual_reasoning_steps > 2:
        complexity_bonus += min(2, visual_reasoning_steps - 2)  # Cap at 2 extra tokens
    
    # Apply complexity adjustment (max 50% increase)
    max_bonus = max(1, base_latent_tokens // 2)
    final_bonus = min(complexity_bonus, max_bonus)
    
    return base_latent_tokens + final_bonus

def _parse_reasoning_steps(steps: Union[List[str], str]) -> List[str]:
    if isinstance(steps, str):
        return [step.strip() for step in steps.split('\n') if step.strip()]
    return steps

def _calculate_curriculum_params(stage_to_train: int, max_latent_stage: int, steps: List[str], pad_latent_to_max: bool, no_cot: bool) -> Tuple[int, int]:
    if no_cot:
        # For pure evaluation without CoT, use no latent tokens but don't skip all steps
        return (len(steps), 0)  # Skip all explicit steps, use no latent tokens
    if stage_to_train > max_latent_stage:
        n_skip_steps = 10000
        n_latent_tokens = max_latent_stage if pad_latent_to_max else min(len(steps), max_latent_stage)
    else:
        n_skip_steps = stage_to_train
        n_latent_tokens = stage_to_train
    return (n_skip_steps, n_latent_tokens)

def _build_reasoning_text(total_latent_tokens: int, steps: List[str], n_skip_steps: int) -> str:
    """
    Build reasoning text with latent tokens, ensuring proper positioning relative to images.
    Fix: Place latent tokens after image understanding, not before reasoning about images.
    """
    reasoning_parts = []
    
    # Add latent tokens for reasoning (place them after any image context)
    if total_latent_tokens > 0:
        latent_block = ' '.join([LATENT_TOKEN] * total_latent_tokens)
        reasoning_parts.append(f'{START_LATENT_TOKEN} {latent_block} {END_LATENT_TOKEN}')
    
    # Add remaining explicit reasoning steps
    if (remaining_steps := steps[n_skip_steps:]):
        reasoning_parts.append(' '.join(remaining_steps))
    
    return ' '.join(reasoning_parts).strip()