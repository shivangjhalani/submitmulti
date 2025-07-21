#!/usr/bin/env python3
"""
Utility functions for determining the correct number of image tokens for different models.
This fixes the image token count mismatch issue.
"""

import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


def get_model_image_token_count(model: Any, fallback: int = 256) -> int:
    """
    Determine the correct number of image tokens for a given model.
    
    This function tries multiple methods to get the actual number of image tokens
    that the model produces, which is critical for proper multimodal processing.
    
    Args:
        model: The model object (could be wrapped in LatentWrapper)
        fallback: Fallback value if no method works
        
    Returns:
        The correct number of image tokens for this model
    """
    # Unwrap LatentWrapper if needed
    base_model = getattr(model, 'base_model', model)
    
    # Method 1: Check config.num_image_token
    if hasattr(base_model, 'config'):
        config = base_model.config
        if hasattr(config, 'num_image_token'):
            num_tokens = config.num_image_token
            logger.debug(f"Found num_image_token in config: {num_tokens}")
            return num_tokens
    
    # Method 2: Check model.num_image_token attribute
    if hasattr(base_model, 'num_image_token'):
        num_tokens = base_model.num_image_token
        logger.debug(f"Found num_image_token as model attribute: {num_tokens}")
        return num_tokens
    
    # Method 3: Calculate from vision config
    if hasattr(base_model, 'config'):
        config = base_model.config
        
        # Check for vision_config
        vision_config = getattr(config, 'vision_config', None)
        if vision_config:
            # Calculate based on image size and patch size
            image_size = getattr(vision_config, 'image_size', None)
            patch_size = getattr(vision_config, 'patch_size', None)
            
            if image_size and patch_size:
                # Calculate number of patches
                if isinstance(image_size, (list, tuple)):
                    image_size = image_size[0]  # Assume square
                if isinstance(patch_size, (list, tuple)):
                    patch_size = patch_size[0]  # Assume square
                
                patches_per_side = image_size // patch_size
                num_tokens = patches_per_side * patches_per_side
                
                # Apply downsample ratio if available
                downsample_ratio = getattr(config, 'downsample_ratio', None)
                if downsample_ratio and downsample_ratio != 1.0:
                    num_tokens = int(num_tokens * downsample_ratio)
                
                logger.debug(f"Calculated from vision config: {num_tokens} tokens "
                           f"(image_size={image_size}, patch_size={patch_size}, downsample={downsample_ratio})")
                return num_tokens
    
    # Method 4: InternVL-specific calculation
    if hasattr(base_model, 'config'):
        config = base_model.config
        model_name = getattr(config, '_name_or_path', '') or getattr(config, 'model_type', '')
        
        if 'internvl' in model_name.lower():
            # InternVL3-1B typically uses 448x448 images with 14x14 patches
            # and a specific downsample ratio
            if hasattr(config, 'downsample_ratio'):
                # Standard calculation for InternVL
                base_patches = 32 * 32  # 1024 base patches
                downsample_factor = config.downsample_ratio
                num_tokens = int(base_patches * downsample_factor)
                logger.debug(f"InternVL calculation: {num_tokens} tokens (downsample_ratio={downsample_factor})")
                return num_tokens
            else:
                # Known InternVL3-1B defaults
                num_tokens = 256  # This should be updated based on actual model inspection
                logger.debug(f"InternVL fallback: {num_tokens} tokens")
                return num_tokens
    
    # Method 5: Try to extract from model's extract_feature output
    try:
        if hasattr(base_model, 'extract_feature'):
            # Create a dummy image tensor to test
            import torch
            dummy_pixel_values = torch.randn(1, 3, 448, 448)  # Standard input size
            
            with torch.no_grad():
                dummy_features = base_model.extract_feature(dummy_pixel_values)
                if dummy_features is not None and hasattr(dummy_features, 'shape'):
                    num_tokens = dummy_features.shape[1]  # Sequence length dimension
                    logger.debug(f"Extracted from extract_feature: {num_tokens} tokens")
                    return num_tokens
    except Exception as e:
        logger.debug(f"Could not extract token count from extract_feature: {e}")
    
    # Method 6: Check tokenizer if available
    # This is less reliable but might work for some models
    
    logger.warning(f"Could not determine image token count for model. Using fallback: {fallback}")
    return fallback


def get_tokenizer_image_token_count(tokenizer: Any, fallback: int = 256) -> int:
    """
    Try to get image token count from tokenizer attributes.
    
    Args:
        tokenizer: The tokenizer object
        fallback: Fallback value if no method works
        
    Returns:
        The number of image tokens
    """
    # Method 1: Check tokenizer.model.num_image_token
    if hasattr(tokenizer, 'model') and hasattr(tokenizer.model, 'num_image_token'):
        num_tokens = tokenizer.model.num_image_token
        logger.debug(f"Found num_image_token in tokenizer.model: {num_tokens}")
        return num_tokens
    
    # Method 2: Check tokenizer.num_image_token
    if hasattr(tokenizer, 'num_image_token'):
        num_tokens = tokenizer.num_image_token
        logger.debug(f"Found num_image_token in tokenizer: {num_tokens}")
        return num_tokens
    
    logger.debug(f"Could not determine image token count from tokenizer. Using fallback: {fallback}")
    return fallback


def validate_image_token_count(prompt: str, image_embeds: Optional[Any], expected_count: int) -> bool:
    """
    Validate that the prompt has the correct number of IMG_CONTEXT tokens
    to match the image embeddings.
    
    Args:
        prompt: The text prompt containing IMG_CONTEXT tokens
        image_embeds: The image embeddings tensor
        expected_count: Expected number of IMG_CONTEXT tokens
        
    Returns:
        True if counts match, False otherwise
    """
    prompt_count = prompt.count('<IMG_CONTEXT>')
    
    if image_embeds is not None:
        embed_count = image_embeds.shape[1] if len(image_embeds.shape) > 1 else image_embeds.shape[0]
        
        if prompt_count != embed_count:
            logger.error(f"Image token count mismatch: prompt has {prompt_count} <IMG_CONTEXT> tokens, "
                        f"but image_embeds has {embed_count} tokens")
            return False
        
        if prompt_count != expected_count:
            logger.warning(f"Unexpected token count: expected {expected_count}, "
                          f"got {prompt_count} in prompt and {embed_count} in embeddings")
    
    return prompt_count == expected_count
