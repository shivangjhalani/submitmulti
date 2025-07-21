```python
import torch
import torch.distributed
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel

model = AutoModel.from_pretrained("OpenGVLab/InternVL3-1B-Pretrained", trust_remote_code=True, torch_dtype=torch.bfloat16, use_flash_attn=True, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B-Pretrained", trust_remote_code=True, use_fast=False)
```

```python
print(type(model))
print(dir(model)) # This will list all attributes and methods
```
<class 'transformers_modules.OpenGVLab.InternVL3-1B-Pretrained.d3292416b2ecc894c0f4009a6dae424fbf164249.modeling_internvl_chat.InternVLChatModel'>
['T_destination', '__annotations__', '__call__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattr__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_apply', '_assisted_decoding', '_auto_class', '_autoset_attn_implementation', '_backward_compatibility_gradient_checkpointing', '_backward_hooks', '_backward_pre_hooks', '_beam_search', '_beam_search_has_unfinished_sequences', '_buffers', '_cache_dependant_input_preparation', '_cache_dependant_input_preparation_exporting', '_call_impl', '_check_and_enable_flash_attn_2', '_check_and_enable_flash_attn_3', '_check_and_enable_flex_attn', '_check_and_enable_sdpa', '_checkpoint_conversion_mapping', '_compiled_call_impl', '_constrained_beam_search', '_contrastive_search', '_convert_head_mask_to_5d', '_copy_lm_head_original_to_resized', '_create_repo', '_device_mesh', '_dispatch_accelerate_model', '_dola_decoding', '_expand_inputs_for_generation', '_fix_state_dict_key_on_load', '_fix_state_dict_key_on_save', '_fix_state_dict_keys_on_save', '_flatten_beam_dim', '_forward_hooks', '_forward_hooks_always_called', '_forward_hooks_with_kwargs', '_forward_pre_hooks', '_forward_pre_hooks_with_kwargs', '_from_config', '_gather_beams', '_get_backward_hooks', '_get_backward_pre_hooks', '_get_cache', '_get_candidate_generator', '_get_files_timestamps', '_get_initial_cache_position', '_get_key_renaming_mapping', '_get_layer_device_map_for_cache_init', '_get_logits_processor', '_get_name', '_get_no_split_modules', '_get_resized_embeddings', '_get_resized_lm_head', '_get_running_beams_for_next_iteration', '_get_stopping_criteria', '_get_top_k_continuations', '_group_beam_search', '_has_unfinished_sequences', '_hf_peft_config_loaded', '_hook_rss_memory_post_forward', '_hook_rss_memory_pre_forward', '_init_added_embeddings_weights_with_mean', '_init_added_lm_head_bias_with_mean', '_init_added_lm_head_weights_with_mean', '_init_weights', '_initialize_missing_keys', '_initialize_weights', '_is_full_backward_hook', '_is_hf_initialized', '_is_stateful', '_keep_in_fp32_modules', '_keep_in_fp32_modules', '_keep_in_fp32_modules_strict', '_keep_in_fp32_modules_strict', '_keys_to_ignore_on_load_missing', '_keys_to_ignore_on_load_unexpected', '_keys_to_ignore_on_save', '_load_from_flax', '_load_from_state_dict', '_load_from_tf', '_load_pretrained_model', '_load_state_dict_post_hooks', '_load_state_dict_pre_hooks', '_maybe_initialize_input_ids_for_generation', '_maybe_warn_non_full_backward_hook', '_merge_criteria_processor_list', '_modules', '_move_missing_keys_from_meta_to_cpu', '_named_members', '_no_split_modules', '_no_split_modules', '_non_persistent_buffers_set', '_parameters', '_pp_plan', '_prefill_chunking', '_prepare_attention_mask_for_generation', '_prepare_cache_for_generation', '_prepare_decoder_input_ids_for_generation', '_prepare_encoder_decoder_kwargs_for_generation', '_prepare_generated_length', '_prepare_generation_config', '_prepare_model_inputs', '_prepare_special_tokens', '_register_load_state_dict_pre_hook', '_register_state_dict_hook', '_reorder_cache', '_replicate_for_data_parallel', '_resize_token_embeddings', '_sample', '_save_to_state_dict', '_set_default_torch_dtype', '_set_gradient_checkpointing', '_skip_keys_device_placement', '_slow_forward', '_state_dict_hooks', '_state_dict_pre_hooks', '_supports_attention_backend', '_supports_cache_class', '_supports_default_dynamic_cache', '_supports_flash_attn_2', '_supports_flash_attn_3', '_supports_flex_attn', '_supports_logits_to_keep', '_supports_quantized_cache', '_supports_sdpa', '_supports_static_cache', '_temporary_reorder_cache', '_tie_encoder_decoder_weights', '_tie_or_clone_weights', '_tied_weights_keys', '_tp_plan', '_tp_size', '_tp_size', '_unflatten_beam_dim', '_update_finished_beams', '_update_model_kwargs_for_generation', '_upload_modified_files', '_valid_auto_compile_criteria', '_validate_assistant', '_validate_generated_length', '_validate_model_kwargs', '_version', '_wrapped_call_impl', 'active_adapter', 'active_adapters', 'add_adapter', 'add_memory_hooks', 'add_model_tags', 'add_module', 'apply', 'base_model', 'base_model_prefix', 'batch_chat', 'bfloat16', 'buffers', 'call_super_init', 'can_generate', 'chat', 'children', 'compile', 'compute_transition_scores', 'config', 'config_class', 'conv_template', 'cpu', 'create_extended_attention_mask_for_decoder', 'cuda', 'delete_adapter', 'dequantize', 'device', 'disable_adapters', 'disable_input_require_grads', 'double', 'downsample_ratio', 'dtype', 'dummy_inputs', 'dump_patches', 'enable_adapters', 'enable_input_require_grads', 'estimate_tokens', 'eval', 'extra_repr', 'extract_feature', 'float', 'floating_point_ops', 'forward', 'framework', 'from_pretrained', 'generate', 'generate_batch', 'generation_config', 'get_adapter_state_dict', 'get_buffer', 'get_compiled_call', 'get_extended_attention_mask', 'get_extra_state', 'get_head_mask', 'get_init_context', 'get_input_embeddings', 'get_memory_footprint', 'get_output_embeddings', 'get_parameter', 'get_parameter_or_buffer', 'get_position_embeddings', 'get_submodule', 'gradient_checkpointing_disable', 'gradient_checkpointing_enable', 'half', 'heal_tokens', 'img_context_token_id', 'init_continuous_batching', 'init_weights', 'initialize_weights', 'invert_attention_mask', 'ipu', 'is_backend_compatible', 'is_gradient_checkpointing', 'is_parallelizable', 'language_model', 'lm_head', 'load_adapter', 'load_custom_generate', 'load_state_dict', 'loss_function', 'loss_type', 'main_input_name', 'mlp1', 'model_tags', 'modules', 'mtia', 'name_or_path', 'named_buffers', 'named_children', 'named_modules', 'named_parameters', 'num_image_token', 'num_parameters', 'parameters', 'patch_size', 'pixel_shuffle', 'post_init', 'prepare_inputs_for_generation', 'prune_heads', 'ps_version', 'push_to_hub', 'register_backward_hook', 'register_buffer', 'register_for_auto_class', 'register_forward_hook', 'register_forward_pre_hook', 'register_full_backward_hook', 'register_full_backward_pre_hook', 'register_load_state_dict_post_hook', 'register_load_state_dict_pre_hook', 'register_module', 'register_parameter', 'register_state_dict_post_hook', 'register_state_dict_pre_hook', 'requires_grad_', 'reset_memory_hooks_state', 'resize_position_embeddings', 'resize_token_embeddings', 'retrieve_modules_from_names', 'reverse_bettertransformer', 'save_pretrained', 'select_layer', 'set_adapter', 'set_extra_state', 'set_input_embeddings', 'set_submodule', 'share_memory', 'smart_apply', 'state_dict', 'supports_gradient_checkpointing', 'supports_pp_plan', 'supports_tp_plan', 'system_message', 'template', 'tie_weights', 'to', 'to_bettertransformer', 'to_empty', 'tp_size', 'train', 'training', 'type', 'vision_model', 'warn_if_padding_and_no_attention_mask', 'warnings_issued', 'xpu', 'zero_grad']




```python
print(model)
```
InternVLChatModel(
  (vision_model): InternVisionModel(
    (embeddings): InternVisionEmbeddings(
      (patch_embedding): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14))
    )
    (encoder): InternVisionEncoder(
      (layers): ModuleList(
        (0-23): 24 x InternVisionEncoderLayer(
          (attn): InternAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): InternMLP(
            (act): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (drop_path1): Identity()
          (drop_path2): Identity()
        )
      )
    )
  )
  (language_model): Qwen2ForCausalLM(
    (model): Qwen2Model(
      (embed_tokens): Embedding(151674, 896)
      (layers): ModuleList(
        (0-23): 24 x Qwen2DecoderLayer(
          (self_attn): Qwen2Attention(
            (q_proj): Linear(in_features=896, out_features=896, bias=True)
            (k_proj): Linear(in_features=896, out_features=128, bias=True)
            (v_proj): Linear(in_features=896, out_features=128, bias=True)
            (o_proj): Linear(in_features=896, out_features=896, bias=False)
          )
          (mlp): Qwen2MLP(
            (gate_proj): Linear(in_features=896, out_features=4864, bias=False)
            (up_proj): Linear(in_features=896, out_features=4864, bias=False)
            (down_proj): Linear(in_features=4864, out_features=896, bias=False)
            (act_fn): SiLU()
          )
          (input_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
          (post_attention_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
        )
      )
      (norm): Qwen2RMSNorm((896,), eps=1e-06)
      (rotary_emb): Qwen2RotaryEmbedding()
    )
    (lm_head): Linear(in_features=896, out_features=151674, bias=False)
  )
  (mlp1): Sequential(
    (0): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
    (1): Linear(in_features=4096, out_features=896, bias=True)
    (2): GELU(approximate='none')
    (3): Linear(in_features=896, out_features=896, bias=True)
  )
)



```python
print(model.config)
```

InternVLChatConfig {
  "architectures": [
    "InternVLChatModel"
  ],
  "auto_map": {
    "AutoConfig": "configuration_internvl_chat.InternVLChatConfig",
    "AutoModel": "modeling_internvl_chat.InternVLChatModel",
    "AutoModelForCausalLM": "modeling_internvl_chat.InternVLChatModel"
  },
  "downsample_ratio": 0.5,
  "dynamic_image_size": true,
  "force_image_size": 448,
  "image_fold": null,
  "llm_config": {
    "_name_or_path": "./pretrained/Qwen2.5-32B-Instruct",
    "architectures": [
      "Qwen2ForCausalLM"
    ],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151643,
    "hidden_act": "silu",
    "hidden_size": 896,
    "initializer_range": 0.02,
    "intermediate_size": 4864,
    "layer_types": [
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention"
    ],
    "max_position_embeddings": 32768,
    "max_window_layers": 70,
    "model_type": "qwen2",
    "moe_config": null,
    "num_attention_heads": 14,
    "num_hidden_layers": 24,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
      "factor": 2.0,
      "rope_type": "dynamic",
      "type": "dynamic"
    },
    "rope_theta": 1000000.0,
    "sliding_window": null,
    "torch_dtype": "bfloat16",
    "use_bfloat16": true,
    "use_cache": true,
    "use_sliding_window": false,
    "vocab_size": 151674
  },
  "max_dynamic_patch": 12,
  "min_dynamic_patch": 1,
  "model_type": "internvl_chat",
  "output_attentions": false,
  "pad2square": false,
  "ps_version": "v2",
  "select_layer": -1,
  "template": "internvl2_5",
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": null,
  "use_backbone_lora": 0,
  "use_llm_lora": 0,
  "use_thumbnail": true,
  "vision_config": {
    "_name_or_path": "OpenGVLab/InternViT-6B-448px-V1-5",
    "architectures": [
      "InternVisionModel"
    ],
    "attention_dropout": 0.0,
    "auto_map": {
      "AutoConfig": "configuration_intern_vit.InternVisionConfig",
      "AutoModel": "modeling_intern_vit.InternVisionModel"
    },
    "capacity_factor": 1.2,
    "drop_path_rate": 0.0,
    "dropout": 0.0,
    "eval_capacity_factor": 1.4,
    "hidden_act": "gelu",
    "hidden_size": 1024,
    "image_size": 448,
    "initializer_factor": 0.1,
    "initializer_range": 1e-10,
    "intermediate_size": 4096,
    "laux_allreduce": "all_nodes",
    "layer_norm_eps": 1e-06,
    "model_type": "intern_vit_6b",
    "moe_coeff_ratio": 0.5,
    "moe_intermediate_size": 768,
    "moe_output_scale": 4.0,
    "noisy_gate_policy": "RSample_before",
    "norm_type": "layer_norm",
    "num_attention_heads": 16,
    "num_channels": 3,
    "num_experts": 8,
    "num_hidden_layers": 24,
    "num_routed_experts": 4,
    "num_shared_experts": 4,
    "patch_size": 14,
    "qk_normalization": false,
    "qkv_bias": true,
    "shared_expert_intermediate_size": 3072,
    "torch_dtype": "bfloat16",
    "use_bfloat16": true,
    "use_flash_attn": false,
    "use_moe": false,
    "use_residual": true,
    "use_rts": false,
    "use_weighted_residual": false
  }
}



```python
print(tokenizer)
```

Qwen2Tokenizer(name_or_path='OpenGVLab/InternVL3-1B-Pretrained', vocab_size=151643, model_max_length=1000000, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'eos_token': '<|im_end|>', 'pad_token': '<|endoftext|>', 'additional_special_tokens': ['<|im_start|>', '<|im_end|>', '<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', '<|box_end|>', '<|quad_start|>', '<|quad_end|>', '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>', '<|image_pad|>', '<|video_pad|>']}, clean_up_tokenization_spaces=False, added_tokens_decoder={
	151643: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151644: AddedToken("<|im_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151645: AddedToken("<|im_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151646: AddedToken("<|object_ref_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151647: AddedToken("<|object_ref_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151648: AddedToken("<|box_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151649: AddedToken("<|box_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151650: AddedToken("<|quad_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151651: AddedToken("<|quad_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151652: AddedToken("<|vision_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151653: AddedToken("<|vision_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151654: AddedToken("<|vision_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151655: AddedToken("<|image_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151656: AddedToken("<|video_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151657: AddedToken("<tool_call>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151658: AddedToken("</tool_call>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151659: AddedToken("<|fim_prefix|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151660: AddedToken("<|fim_middle|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151661: AddedToken("<|fim_suffix|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151662: AddedToken("<|fim_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151663: AddedToken("<|repo_name|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151664: AddedToken("<|file_sep|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151665: AddedToken("<img>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151666: AddedToken("</img>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151667: AddedToken("<IMG_CONTEXT>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151668: AddedToken("<quad>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151669: AddedToken("</quad>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151670: AddedToken("<ref>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151671: AddedToken("</ref>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151672: AddedToken("<box>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151673: AddedToken("</box>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
}
)



```python
# Inspect model attributes for image input format information
print("Model attributes potentially related to image input:")
for attr_name in dir(model.config.vision_config):
    if "image" in attr_name or "patch" in attr_name or "size" in attr_name:
        try:
            print(f"{attr_name}: {getattr(model.config.vision_config, attr_name)}")
        except:
            pass
```

Model attributes potentially related to image input:
__sizeof__: <built-in method __sizeof__ of InternVisionConfig object at 0x7f58e1906e90>
chunk_size_feed_forward: 0
cross_attention_hidden_size: None
encoder_no_repeat_ngram_size: 0
hidden_size: 1024
image_size: 448
intermediate_size: 4096
moe_intermediate_size: 768
no_repeat_ngram_size: 0
patch_size: 14
shared_expert_intermediate_size: 3072



```python
import torch

# Create dummy inputs based on the model configuration
image_size = model.config.vision_config.image_size
patch_size = model.config.vision_config.patch_size
hidden_size = model.config.vision_config.hidden_size
llm_hidden_size = model.config.llm_config.hidden_size
max_position_embeddings = model.config.llm_config.max_position_embeddings
vocab_size = model.config.llm_config.vocab_size

# Dummy image input (batch_size, num_channels, height, width)
# Cast to bfloat16 to match model's expected type
dummy_image_input = torch.randn(1, 3, image_size, image_size).to(torch.bfloat16)
print(f"Dummy image input shape: {dummy_image_input.shape}")
print(f"Dummy image input dtype: {dummy_image_input.dtype}")


# Dummy text input (batch_size, sequence_length)
dummy_text_input = torch.randint(0, vocab_size, (1, 10))
print(f"Dummy text input shape: {dummy_text_input.shape}")

# Dummy attention mask (batch_size, sequence_length)
dummy_attention_mask = torch.ones(1, 10)
print(f"Dummy attention mask shape: {dummy_attention_mask.shape}")

# You can also create dummy inputs for other potential arguments like token_type_ids if needed,
# but these are the most common for this type of model.

# To see intermediate tensor shapes, you would need to forward the dummy inputs through the model
# and potentially add hooks or inspect the model's forward method. This can be complex and depends
# on the specific model architecture.

# Example of forwarding through the vision model to see the output shape
with torch.no_grad():
    vision_output = model.vision_model(dummy_image_input)
    print(f"Vision model output shape (last hidden state): {vision_output.last_hidden_state.shape}")

# Example of forwarding through the language model to see the output shape
# Note: This requires matching the dimensions and types correctly based on the model's forward method
# and the output of the vision model if you want to combine them.
# This is a simplified example and might need adjustments based on the exact model implementation.
with torch.no_grad():
    # In a multimodal model, the text input might be combined with the vision output
    # Let's create a dummy combined input shape based on potential concatenation
    # This is a simplification - the actual combination depends on the model's forward method
    dummy_combined_input = torch.randn(1, vision_output.last_hidden_state.shape[1] + dummy_text_input.shape[1], llm_hidden_size)
    print(f"Dummy combined input shape (example): {dummy_combined_input.shape}")

    # To get the language model output, you'd typically pass the combined input and attention mask
    # The exact method call depends on the model.
    # For demonstration, let's assume a method like 'generate' or 'forward' that takes inputs
    # print(f"Language model output shape: ...") # This would require calling the model's forward method with correctly formatted inputs
```

Dummy image input shape: torch.Size([1, 3, 448, 448])
Dummy image input dtype: torch.bfloat16
Dummy text input shape: torch.Size([1, 10])
Dummy attention mask shape: torch.Size([1, 10])
Vision model output shape (last hidden state): torch.Size([1, 1025, 1024])
Dummy combined input shape (example): torch.Size([1, 1035, 896])




```python
print(model.forward)
print(help(model.forward))
```

<bound method InternVLChatModel.forward of InternVLChatModel(
  (vision_model): InternVisionModel(
    (embeddings): InternVisionEmbeddings(
      (patch_embedding): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14))
    )
    (encoder): InternVisionEncoder(
      (layers): ModuleList(
        (0-23): 24 x InternVisionEncoderLayer(
          (attn): InternAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): InternMLP(
            (act): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (drop_path1): Identity()
          (drop_path2): Identity()
        )
      )
    )
  )
  (language_model): Qwen2ForCausalLM(
    (model): Qwen2Model(
      (embed_tokens): Embedding(151674, 896)
      (layers): ModuleList(
        (0-23): 24 x Qwen2DecoderLayer(
          (self_attn): Qwen2Attention(
            (q_proj): Linear(in_features=896, out_features=896, bias=True)
            (k_proj): Linear(in_features=896, out_features=128, bias=True)
            (v_proj): Linear(in_features=896, out_features=128, bias=True)
            (o_proj): Linear(in_features=896, out_features=896, bias=False)
          )
          (mlp): Qwen2MLP(
            (gate_proj): Linear(in_features=896, out_features=4864, bias=False)
            (up_proj): Linear(in_features=896, out_features=4864, bias=False)
            (down_proj): Linear(in_features=4864, out_features=896, bias=False)
            (act_fn): SiLU()
          )
          (input_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
          (post_attention_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
        )
      )
      (norm): Qwen2RMSNorm((896,), eps=1e-06)
      (rotary_emb): Qwen2RotaryEmbedding()
    )
    (lm_head): Linear(in_features=896, out_features=151674, bias=False)
  )
  (mlp1): Sequential(
    (0): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
    (1): Linear(in_features=4096, out_features=896, bias=True)
    (2): GELU(approximate='none')
    (3): Linear(in_features=896, out_features=896, bias=True)
  )
)>
Help on method forward in module transformers_modules.OpenGVLab.InternVL3-1B-Pretrained.d3292416b2ecc894c0f4009a6dae424fbf164249.modeling_internvl_chat:

forward(pixel_values: torch.FloatTensor, input_ids: torch.LongTensor = None, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, image_flags: Optional[torch.LongTensor] = None, past_key_values: Optional[List[torch.FloatTensor]] = None, labels: Optional[torch.LongTensor] = None, use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, transformers.modeling_outputs.CausalLMOutputWithPast] method of transformers_modules.OpenGVLab.InternVL3-1B-Pretrained.d3292416b2ecc894c0f4009a6dae424fbf164249.modeling_internvl_chat.InternVLChatModel instance
    Define the computation performed at every call.
    
    Should be overridden by all subclasses.
    
    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        registered hooks while the latter silently ignores them.

None
