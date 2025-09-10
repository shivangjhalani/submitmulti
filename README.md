```
Install toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

Configure runtime for Docker and restart
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

# MultiCoCo: Multimodal Chain-of-Continuous-Thought

MultiCoCo is a comprehensive training framework that extends the original [CoCoNut (Chain-of-Continuous-Thought)](https://arxiv.org/abs/2412.06769) methodology to multimodal models, specifically InternVL3-1B for Visual Question Answering.

## Overview

MultiCoCo implements the progressive curriculum learning approach from the original CoCoNut paper, enabling multimodal models to reason in continuous latent space through a multi-stage training process.

### Key Features

- **Progressive Curriculum Learning**: Stage-by-stage replacement of reasoning steps with latent tokens
- **Multimodal Support**: Extended CoCoNut methodology for vision-language models (InternVL3-1B)
- **Two-Phase Training**: Separate CoT training and CoCoNut multi-stage training commands
- **Distributed Training**: Full DDP support for multi-GPU training
- **Wandb Integration**: Comprehensive experiment tracking and logging
- **Evaluation Suite**: Support for A-OKVQA dataset evaluation
- **Flexible Generation**: Configurable generation parameters optimized for each evaluation type

## Training Methodology

MultiCoCo follows the original CoCoNut methodology with two distinct phases:

### Phase 1: Chain-of-Thought (CoT) Training
```bash
torchrun --nnodes 1 --nproc_per_node 1 run.py args/aokvqa_cot.yaml
```

This phase trains the model with full reasoning chains:
```
Question: What is this animal?
Reasoning: I can see this is a four-legged animal with stripes. The black and white stripe pattern is characteristic of zebras.
Answer: Zebra
```

### Phase 2: CoCoNut Multi-Stage Training
```bash
torchrun --nnodes 1 --nproc_per_node 1 run.py args/aokvqa_coconut.yaml
```

This phase implements progressive curriculum learning:

- **Stage 1**: Replace 1st reasoning step with latent tokens
  ```
  Question: What is this animal?
  <latent> The black and white stripe pattern is characteristic of zebras.
  Answer: Zebra
  ```

- **Stage 2**: Replace 2nd reasoning step with additional latent tokens
  ```
  Question: What is this animal?
  <latent> <latent> Answer: Zebra
  ```

- **Stages 3-6**: Progressive replacement until reaching pure latent reasoning

## Generation Configuration

MultiCoCo uses different generation parameters optimized for each evaluation type:

### Chain-of-Thought (CoT) Evaluation
```yaml
generation:
  max_new_tokens: 256    # Allow longer reasoning sequences
  do_sample: true        # Enable creative generation
  temperature: 1.0       # High creativity for detailed reasoning
  top_p: 0.9            # Nucleus sampling for coherent text
  num_beams: 1          # Single beam for efficiency
```

### CoCoNut Latent Reasoning Evaluation
```yaml
generation:
  max_new_tokens: 256    # Allow sufficient answer generation
  do_sample: true        # Enable sampling for diversity
  temperature: 0.8       # Balanced creativity for latent reasoning
  top_p: 0.9            # Nucleus sampling
  num_beams: 1          # Single beam for efficiency
```

### Vanilla Direct Answering Evaluation
```yaml
generation:
  max_new_tokens: 128    # Shorter answers for direct response
  do_sample: true        # Enable some variability
  temperature: 0.6       # Conservative creativity
  top_p: 0.8            # Slightly more focused sampling
  num_beams: 1          # Single beam for efficiency
```

## Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.36+
- A-OKVQA dataset
- GPU with sufficient memory (recommended: 24GB+ VRAM)

### Installation
```bash
git clone https://github.com/your-repo/multicoco.git
cd multicoco
pip install -r requirements.txt
```

### Data Preparation
Place your A-OKVQA dataset files in the `data/` directory:
```
data/
├── aokvqa_train.json
├── aokvqa_validation.json
└── aokvqa_test.json
```

### Step 1: CoT Training
```bash
torchrun --nnodes 1 --nproc_per_node 1 run.py args/aokvqa_cot.yaml
```

**Purpose**: Train the model to generate explicit chain-of-thought reasoning.

**Expected Outcome**: Model learns to provide step-by-step reasoning before answering.

### Step 2: CoCoNut Multi-Stage Training
```bash
torchrun --nnodes 1 --nproc_per_node 1 run.py args/aokvqa_coconut.yaml
```

**Purpose**: Progressive curriculum learning that gradually replaces reasoning steps with latent tokens.

**Configuration**: Update `load_model_path` in the config to point to the best CoT checkpoint.

**Training Process**:
- Automatically calculates training stages based on `epochs_per_stage`
- Applies progressive curriculum to dataset for each stage
- Optionally resets optimizer between stages
- Saves checkpoints after each epoch

### Step 3: Evaluation
```bash
torchrun --nnodes 1 --nproc_per_node 1 run.py args/aokvqa_coconut_eval.yaml
```

**Purpose**: Evaluate the trained CoCoNut model on the validation set.

## Configuration Files

MultiCoCo uses a clean configuration inheritance system to reduce redundancy. All configurations inherit common settings from `args/base.yaml`, and specific configs only define what differs.

### Base Configuration (`args/base.yaml`)
Contains common settings shared across all configurations:
- Project and model settings
- Data paths and logging configuration
- Default training parameters
- Checkpoint management settings
- Default generation parameters

### CoT Training (`args/aokvqa_cot.yaml`)
```yaml
# Inherits from base.yaml with overrides
mode: "cot_train"
name: "aokvqa-cot-stage0"
output_dir: "checkpoints/aokvqa_cot"
num_epochs: 10
batch_size: 16
eval_batch_size: 64

# Enhanced generation for reasoning
generation:
  max_new_tokens: 256
  do_sample: true
  temperature: 1.0
  top_p: 0.9
  num_beams: 1

eval_config:
  cot: true
  coconut: false
```

### CoCoNut Training (`args/aokvqa_coconut.yaml`)
```yaml
# Inherits from base.yaml with overrides
mode: "coconut_train"
name: "aokvqa-coconut-multistage"
output_dir: "checkpoints/aokvqa_coconut"
num_epochs: 50
load_model_path: "checkpoints/aokvqa_cot"

# Balanced generation for latent reasoning
generation:
  max_new_tokens: 256
  do_sample: true
  temperature: 0.8
  top_p: 0.9
  num_beams: 1

coconut:
  enabled: true
  c_thought: 1
  epochs_per_stage: 5
  max_latent_stage: 6

eval_config:
  cot: false
  coconut: true
```

### CoT Evaluation (`args/aokvqa_cot_eval.yaml`)
```yaml
# Inherits from base.yaml with overrides
mode: "eval_only"
name: "aokvqa-cot-eval"
load_model_path: "checkpoints/aokvqa_cot"

generation:
  max_new_tokens: 256
  do_sample: true
  temperature: 1.0
  top_p: 0.9
  num_beams: 1

eval_config:
  cot: true
  coconut: false
```

### CoCoNut Evaluation (`args/aokvqa_coconut_eval.yaml`)
```yaml
# Inherits from base.yaml with overrides
mode: "eval_only"
name: "aokvqa-coconut-evaluation"
load_model_path: "checkpoints/aokvqa_coconut"

generation:
  max_new_tokens: 256
  do_sample: true
  temperature: 0.8
  top_p: 0.9
  num_beams: 1

coconut:
  enabled: true
  c_thought: 1
  max_latent_stage: 6

eval_config:
  cot: false
  coconut: true
  eval_latent_tokens: 6
```

### Vanilla Evaluation (`args/aokvqa_vanilla_eval.yaml`)
```yaml
# Inherits from base.yaml with overrides
mode: "eval_only"
name: "aokvqa-vanilla-eval"

generation:
  max_new_tokens: 128
  do_sample: true
  temperature: 0.6
  top_p: 0.8
  num_beams: 1

eval_config:
  cot: false
  coconut: false
  vanilla: true
```

## Checkpoint Management

MultiCoCo includes sophisticated checkpoint management features:

### Configuration Options
```yaml
# In base.yaml or specific configs
max_checkpoints_to_keep: 10        # Maximum number of checkpoints to keep
keep_best_checkpoints: true       # Keep best checkpoints based on eval accuracy
use_run_name_in_output_dir: true  # Include run name in checkpoint directory path
```

### Features
- **Smart Cleanup**: Automatically removes worst-performing checkpoints when limit is exceeded
- **Best Model Retention**: Always keeps the best-performing checkpoints based on evaluation accuracy
- **Run-Specific Directories**: Optional run name inclusion in output paths for better organization
- **Distributed Training Support**: Handles checkpoint management across multiple GPUs

## Advanced Features

### Progressive Curriculum Learning
The CoCoNut training phase implements sophisticated curriculum learning:
- **Stage-wise Training**: Each stage trains for a specified number of epochs
- **Dynamic Data Modification**: Dataset is dynamically modified for each stage
- **Flexible Token Replacement**: Configurable number of latent tokens per reasoning step
- **Curriculum Mixing**: Optional mixing of data from different stages for regularization

### Evaluation Modes
MultiCoCo supports three evaluation modes:
1. **Vanilla**: Direct question answering without explicit reasoning
2. **CoT**: Chain-of-thought evaluation with explicit reasoning steps
3. **CoCoNut**: Latent reasoning evaluation using trained latent tokens

### Logging and Monitoring
- **Detailed Evaluation Logs**: Per-sample logging with generated text and extracted answers
- **Wandb Integration**: Comprehensive experiment tracking and visualization
- **Distributed Logging**: Proper log aggregation across multiple processes
- **Checkpoint Logging**: Detailed logging of checkpoint creation and cleanup

## Model Architecture

MultiCoCo extends InternVL3-1B with:
- **Latent Token Integration**: Special tokens for latent reasoning
- **Progressive Training**: Multi-stage curriculum learning
- **Flexible Evaluation**: Support for multiple reasoning modes

## Performance Tips

1. **Memory Optimization**: Use gradient checkpointing and bf16 precision
2. **Batch Size Tuning**: Adjust batch size based on available GPU memory
3. **Generation Parameters**: Tune temperature and top_p for desired creativity level
4. **Checkpoint Management**: Use appropriate checkpoint limits to manage disk space

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size or enable gradient checkpointing
2. **No Chain of Thought**: Check generation parameters (temperature, do_sample)
3. **Checkpoint Loading**: Verify model path and checkpoint compatibility
4. **Generation Quality**: Adjust temperature and sampling parameters

## Citation

If you use MultiCoCo in your research, please cite:

```bibtex
@article{multicoco2024,
  title={MultiCoCo: Multimodal Chain-of-Continuous-Thought},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## License

[Insert your license information here]

## Acknowledgments

This work builds upon the original CoCoNut methodology and the InternVL series of models. We thank the authors of these works for their contributions to the field.
