#!/usr/bin/env python3
"""
Enhanced Test Utilities Module for CoCoNut Algorithm Testing

This module provides comprehensive testing utilities for the CoCoNut algorithm implementation,
including realistic mock models, intelligent tokenizer integration, and common test functions.

Key features:
- EnhancedMockModel: Sophisticated mock model with realistic behavior
- SmartMockTokenizer: Intelligent tokenizer with real/mock fallback capability  
- Enhanced mock components with proper structure and error handling
- Utility functions for common test operations and validations
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Constants for test configuration
DEFAULT_VOCAB_SIZE = 200000  # Realistic vocabulary size
DEFAULT_HIDDEN_SIZE = 4096   # InternVL3-1B hidden size
DEFAULT_NUM_LAYERS = 24      # InternVL3-1B layer count
DEFAULT_NUM_HEADS = 32       # Standard attention head count
DEFAULT_HEAD_DIM = 128       # Standard head dimension

# Special tokens for testing
TEST_SPECIAL_TOKENS = [
    '<|start_latent|>',
    '<|latent|>',
    '<|end_latent|>',
    '<IMG_CONTEXT>',
    '<|im_start|>',
    '<|im_end|>',
    '<pad>',
    '<eos>',
    '<unk>'
]

# Common test vocabulary for fallback scenarios
TEST_VOCAB = {
    # Special tokens
    '<|start_latent|>': 151672,
    '<|latent|>': 151673,
    '<|end_latent|>': 151674,
    '<IMG_CONTEXT>': 151667,
    '<|im_start|>': 151644,
    '<|im_end|>': 151645,
    '<pad>': 151643,
    '<eos>': 151643,
    '<unk>': 151999,
    
    # Common words for testing
    'hello': 15339,
    'world': 23040,
    'test': 1985,
    'the': 279,
    'answer': 4320,
    'is': 374,
    'question': 3488,
    'image': 2217,
    'a': 264,
    'an': 459,
    'this': 420,
    'that': 430,
    'what': 1602,
    'where': 1405,
    'when': 994,
    'how': 1268,
    'why': 3249,
    'can': 649,
    'will': 690,
    'should': 1288,
    'would': 1053,
    'could': 1436,
    'yes': 9820,
    'no': 912,
    'true': 2575,
    'false': 905,
    'good': 1695,
    'bad': 1958,
    'right': 1314,
    'wrong': 5076,
    'correct': 4495,
    'incorrect': 15465,
    'A': 32,
    'B': 33,
    'C': 34,
    'D': 35,
    '0': 15,
    '1': 16,
    '2': 17,
    '3': 18,
    '4': 19,
    '5': 20,
    '6': 21,
    '7': 22,
    '8': 23,
    '9': 24,
    'first': 1176,
    'second': 2132,
    'third': 4948,
    'fourth': 11999,
    'one': 832,
    'two': 1403,
    'three': 2380,
    'four': 3116,
    'five': 4236,
    'six': 4161,
    'seven': 8223,
    'eight': 8223,
    'nine': 11324,
    'ten': 5935
}


class MockEmbedding(nn.Module):
    """Enhanced mock embedding layer with realistic structure and initialization"""
    
    def __init__(self, vocab_size: int = DEFAULT_VOCAB_SIZE, embed_dim: int = DEFAULT_HIDDEN_SIZE):
        super().__init__()
        self.num_embeddings = vocab_size
        self.embedding_dim = embed_dim
        
        # Initialize weights with proper scaling similar to real models
        self.weight = nn.Parameter(torch.randn(vocab_size, embed_dim) * 0.02)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass with proper bounds checking"""
        if input_ids.max() >= self.num_embeddings:
            logger.warning(f"Input ID {input_ids.max()} exceeds vocab size {self.num_embeddings}")
            # Clamp to valid range instead of failing
            input_ids = torch.clamp(input_ids, 0, self.num_embeddings - 1)
        
        return torch.nn.functional.embedding(input_ids, self.weight)
    
    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.forward(input_ids)


class SmartMockTokenizer:
    """Intelligent tokenizer that uses real AutoTokenizer when possible, falls back to enhanced mock"""
    
    def __init__(self, model_name: str = 'microsoft/DialoGPT-medium', 
                 fallback_vocab_size: int = DEFAULT_VOCAB_SIZE):
        self.model_name = model_name
        self.is_real_tokenizer = False
        self.vocab_size = fallback_vocab_size
        
        # Try to load real tokenizer first
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.vocab_size = len(self.tokenizer.get_vocab())
            self.is_real_tokenizer = True
            
            # Add special tokens if not present
            self._add_special_tokens()
            
            logger.info(f"✓ Loaded real tokenizer: {model_name} (vocab_size: {self.vocab_size})")
            
        except Exception as e:
            logger.warning(f"Failed to load real tokenizer {model_name}: {e}")
            logger.info("Falling back to enhanced mock tokenizer")
            self._create_mock_tokenizer()
    
    def _add_special_tokens(self):
        """Add special tokens to real tokenizer if needed"""
        if not self.is_real_tokenizer:
            return
            
        existing_tokens = set(self.tokenizer.get_vocab().keys())
        new_tokens = [token for token in TEST_SPECIAL_TOKENS if token not in existing_tokens]
        
        if new_tokens:
            self.tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
            self.vocab_size = len(self.tokenizer.get_vocab())
            logger.info(f"Added {len(new_tokens)} special tokens: {new_tokens}")
    
    def _create_mock_tokenizer(self):
        """Create enhanced mock tokenizer with realistic vocabulary"""
        self.is_real_tokenizer = False
        
        # Create comprehensive vocabulary
        self.vocab = TEST_VOCAB.copy()
        
        # Add more tokens to reach target vocab size
        for i in range(len(self.vocab), self.vocab_size):
            self.vocab[f'token_{i}'] = i
            
        # Set special token IDs
        self.unk_token_id = self.vocab.get('<unk>', self.vocab_size - 1)
        self.pad_token_id = self.vocab.get('<pad>', 151643)
        self.eos_token_id = self.vocab.get('<eos>', 151643)
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.eos_token = '<eos>'
        
        # Create reverse vocabulary for decoding
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def convert_tokens_to_ids(self, token: Union[str, List[str]]) -> Union[int, List[int]]:
        """Convert tokens to IDs with proper handling"""
        if isinstance(token, str):
            if self.is_real_tokenizer:
                return self.tokenizer.convert_tokens_to_ids(token)
            else:
                return self.vocab.get(token, self.unk_token_id)
        else:
            # Handle list of tokens
            return [self.convert_tokens_to_ids(t) for t in token]
    
    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """Convert IDs to tokens"""
        if isinstance(ids, int):
            if self.is_real_tokenizer:
                return self.tokenizer.convert_ids_to_tokens(ids)
            else:
                return self.inverse_vocab.get(ids, self.unk_token)
        else:
            return [self.convert_ids_to_tokens(id_) for id_ in ids]
    
    def encode(self, text: str, add_special_tokens: bool = False, 
               return_tensors: Optional[str] = None) -> Union[List[int], torch.Tensor]:
        """Encode text to token IDs"""
        if self.is_real_tokenizer:
            return self.tokenizer.encode(text, add_special_tokens=add_special_tokens, 
                                       return_tensors=return_tensors)
        else:
            # Simple word-based tokenization for mock
            tokens = text.split()
            ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
            
            if return_tensors == "pt":
                return torch.tensor([ids])
            return ids
    
    def decode(self, ids: Union[torch.Tensor, List[int]], 
               skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
            
        if self.is_real_tokenizer:
            return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        else:
            tokens = [self.inverse_vocab.get(id_, f'<unk_{id_}>') for id_ in ids]
            return ' '.join(tokens)
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary dictionary"""
        if self.is_real_tokenizer:
            return self.tokenizer.get_vocab()
        else:
            return self.vocab.copy()
    
    def add_special_tokens(self, special_tokens_dict: Dict[str, Union[str, List[str]]]) -> int:
        """Add special tokens to vocabulary"""
        if self.is_real_tokenizer:
            num_added = self.tokenizer.add_special_tokens(special_tokens_dict)
            self.vocab_size = len(self.tokenizer.get_vocab())
            return num_added
        else:
            # Add to mock vocab
            num_added = 0
            for key, tokens in special_tokens_dict.items():
                if isinstance(tokens, str):
                    tokens = [tokens]
                for token in tokens:
                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)
                        num_added += 1
            
            # Update reverse vocab
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
            self.vocab_size = len(self.vocab)
            return num_added


class EnhancedMockModel(nn.Module):
    """Enhanced mock model that realistically simulates InternVL3-1B behavior"""
    
    def __init__(self, vocab_size: int = DEFAULT_VOCAB_SIZE, 
                 hidden_size: int = DEFAULT_HIDDEN_SIZE,
                 num_layers: int = DEFAULT_NUM_LAYERS,
                 num_heads: int = DEFAULT_NUM_HEADS):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Create nested structure similar to InternVL3-1B
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.embed_tokens = MockEmbedding(vocab_size, hidden_size)
        
        # Vision model for multimodal testing
        self.vision_model = nn.Module()
        
        # Set required attributes
        self.dtype = torch.float32
        self.img_context_token_id = TEST_VOCAB.get('<IMG_CONTEXT>', 151667)
        self._device = torch.device('cpu')
        
        # Mock transformer layers for realistic KV cache
        self.transformer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=16,  # Reduced for efficiency
                batch_first=True,
                activation='gelu'
            )
            for _ in range(min(num_layers, 6))  # Limit for efficiency
        ])
        
        logger.info(f"✓ Created EnhancedMockModel: vocab={vocab_size}, hidden={hidden_size}, layers={num_layers}")
    
    def get_input_embeddings(self) -> MockEmbedding:
        """Get input embedding layer"""
        return self.language_model.model.embed_tokens
    
    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        """Resize token embeddings to match tokenizer"""
        old_embeddings = self.language_model.model.embed_tokens
        new_embeddings = MockEmbedding(new_num_tokens, self.hidden_size)
        
        # Copy old weights where possible
        num_tokens_to_copy = min(old_embeddings.num_embeddings, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy] = old_embeddings.weight.data[:num_tokens_to_copy]
        
        self.language_model.model.embed_tokens = new_embeddings
        self.vocab_size = new_num_tokens
        
        logger.info(f"✓ Resized embeddings from {old_embeddings.num_embeddings} to {new_num_tokens}")
    
    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Mock vision feature extraction"""
        batch_size = pixel_values.shape[0]
        # Return realistic image features (256 tokens per image)
        return torch.randn(batch_size, 256, self.hidden_size) * 0.02
    
    def forward(self, input_ids: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                output_hidden_states: bool = False,
                use_cache: bool = False,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                **kwargs) -> 'MockOutputs':
        """Enhanced forward pass with realistic behavior"""
        
        # Handle inputs
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        elif inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        
        batch_size, seq_len, hidden_size = inputs_embeds.shape
        
        # Generate realistic hidden states with some variation
        hidden_states_list = [inputs_embeds] if output_hidden_states else []
        current_hidden = inputs_embeds
        
        # Simulate layer processing
        for i in range(min(self.num_layers, 6)):  # Limit for efficiency
            # Add some realistic transformation
            current_hidden = current_hidden + torch.randn_like(current_hidden) * 0.01
            if output_hidden_states:
                hidden_states_list.append(current_hidden)
        
        final_hidden = current_hidden
        
        # Generate realistic logits
        logits = torch.randn(batch_size, seq_len, self.vocab_size) * 0.1
        
        # Generate realistic KV cache if requested
        kv_cache = None
        if use_cache:
            kv_cache = []
            for _ in range(self.num_layers):
                key = torch.randn(batch_size, self.num_heads, seq_len, self.head_dim) * 0.02
                value = torch.randn(batch_size, self.num_heads, seq_len, self.head_dim) * 0.02
                kv_cache.append((key, value))
        
        return MockOutputs(
            logits=logits,
            hidden_states=hidden_states_list if output_hidden_states else None,
            past_key_values=kv_cache,
            last_hidden_state=final_hidden
        )
    
    def generate(self, input_ids: Optional[torch.Tensor] = None,
                 inputs_embeds: Optional[torch.Tensor] = None,
                 attention_mask: Optional[torch.Tensor] = None,
                 pixel_values: Optional[torch.Tensor] = None,
                 max_new_tokens: int = 10,
                 **kwargs) -> torch.Tensor:
        """Mock generation method"""
        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_len, _ = inputs_embeds.shape
            # Generate mock input_ids for continuation
            input_ids = torch.randint(1, min(1000, self.vocab_size), (batch_size, seq_len))
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        
        # Generate new tokens (simple random generation for testing)
        new_tokens = torch.randint(1, min(1000, self.vocab_size), (batch_size, max_new_tokens))
        
        return torch.cat([input_ids, new_tokens], dim=1)
    
    def chat(self, tokenizer, pixel_values: Optional[torch.Tensor] = None,
             question: str = "", generation_config=None, **kwargs) -> str:
        """Mock chat method for compatibility"""
        return f"Mock response to: {question}"
    
    def parameters(self):
        """Mock parameters method for device/dtype detection"""
        param = torch.tensor([1.0], device=self._device, dtype=self.dtype)
        yield param
    
    @property
    def device(self) -> torch.device:
        """Get model device"""
        return self._device


class MockOutputs:
    """Mock model outputs structure"""
    
    def __init__(self, logits: torch.Tensor, 
                 hidden_states: Optional[List[torch.Tensor]] = None,
                 past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                 last_hidden_state: Optional[torch.Tensor] = None,
                 loss: Optional[torch.Tensor] = None):
        self.logits = logits
        self.hidden_states = hidden_states
        self.past_key_values = past_key_values
        self.last_hidden_state = last_hidden_state
        self.loss = loss


# Utility Functions

def create_test_input_ids(tokenizer: SmartMockTokenizer, 
                         text_tokens: List[str],
                         latent_spans: Optional[List[Tuple[int, int]]] = None,
                         include_image_tokens: bool = False) -> torch.Tensor:
    """Create test input IDs with optional latent spans and image tokens"""
    
    input_tokens = []
    
    # Add image tokens if requested
    if include_image_tokens:
        img_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        input_tokens.extend([img_token_id, img_token_id])  # Two image tokens
    
    # Add text tokens
    token_ids = [tokenizer.convert_tokens_to_ids(token) for token in text_tokens]
    
    # Insert latent spans if specified
    if latent_spans:
        start_id = tokenizer.convert_tokens_to_ids('<|start_latent|>')
        latent_id = tokenizer.convert_tokens_to_ids('<|latent|>')
        end_id = tokenizer.convert_tokens_to_ids('<|end_latent|>')
        
        for span_start, span_end in latent_spans:
            # Insert latent span at specified position
            span_tokens = [start_id] + [latent_id] * (span_end - span_start - 2) + [end_id]
            token_ids[span_start:span_start] = span_tokens
    
    input_tokens.extend(token_ids)
    return torch.tensor([input_tokens])


def validate_model_outputs(outputs: MockOutputs, 
                          expected_batch_size: int,
                          expected_seq_len: int,
                          expected_vocab_size: int) -> bool:
    """Validate model outputs have correct shapes and types"""
    try:
        # Check logits
        if outputs.logits.shape != (expected_batch_size, expected_seq_len, expected_vocab_size):
            logger.error(f"Logits shape mismatch: {outputs.logits.shape} vs expected {(expected_batch_size, expected_seq_len, expected_vocab_size)}")
            return False
        
        # Check hidden states if present
        if outputs.hidden_states is not None:
            for i, hidden in enumerate(outputs.hidden_states):
                if hidden.shape[:2] != (expected_batch_size, expected_seq_len):
                    logger.error(f"Hidden state {i} shape mismatch: {hidden.shape}")
                    return False
        
        # Check KV cache if present
        if outputs.past_key_values is not None:
            for i, (key, value) in enumerate(outputs.past_key_values):
                if key.shape[0] != expected_batch_size or value.shape[0] != expected_batch_size:
                    logger.error(f"KV cache {i} batch size mismatch")
                    return False
        
        logger.debug("✓ Model outputs validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Model outputs validation failed: {e}")
        return False


def create_test_environment(model_name: str = 'microsoft/DialoGPT-medium',
                           vocab_size: int = DEFAULT_VOCAB_SIZE,
                           hidden_size: int = DEFAULT_HIDDEN_SIZE) -> Tuple[EnhancedMockModel, SmartMockTokenizer]:
    """Create a complete test environment with enhanced mock model and tokenizer"""
    
    # Create smart tokenizer
    tokenizer = SmartMockTokenizer(model_name, vocab_size)
    
    # Create enhanced model with matching vocabulary
    model = EnhancedMockModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size
    )
    
    # Resize model embeddings if tokenizer was extended
    if tokenizer.vocab_size != vocab_size:
        model.resize_token_embeddings(tokenizer.vocab_size)
    
    logger.info(f"✓ Created test environment with vocab_size={tokenizer.vocab_size}")
    return model, tokenizer


def run_basic_model_test(model: EnhancedMockModel, tokenizer: SmartMockTokenizer) -> bool:
    """Run basic tests to verify model and tokenizer work correctly"""
    try:
        # Test basic forward pass
        test_tokens = ['hello', 'world', 'test']
        input_ids = create_test_input_ids(tokenizer, test_tokens)
        
        outputs = model.forward(input_ids=input_ids, output_hidden_states=True, use_cache=True)
        
        # Validate outputs
        if not validate_model_outputs(outputs, 1, input_ids.shape[1], model.vocab_size):
            return False
        
        # Test generation
        generated = model.generate(input_ids=input_ids, max_new_tokens=5)
        if generated.shape[1] <= input_ids.shape[1]:
            logger.error("Generation failed to produce new tokens")
            return False
        
        # Test tokenizer encoding/decoding
        text = "hello world test"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        logger.info(f"✓ Basic tests passed - Original: '{text}', Decoded: '{decoded}'")
        return True
        
    except Exception as e:
        logger.error(f"Basic model test failed: {e}")
        return False


# Module-level test when imported
if __name__ == "__main__":
    # Quick self-test when run directly
    print("🧪 Running test_utils self-tests...")
    
    model, tokenizer = create_test_environment()
    if run_basic_model_test(model, tokenizer):
        print("✅ All test_utils self-tests passed!")
    else:
        print("❌ test_utils self-tests failed!")
