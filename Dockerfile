# 1. Base Image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pyyaml tqdm safetensors

# 4. Copy the rest of the application
COPY . .

# 5. Set the default command to download checkpoint, preprocess, evaluate, and display logs
CMD ["sh", "-c", "python scripts/download_checkpoint.py && python preprocessing/aokvqa.py && torchrun --nnodes 1 --nproc_per_node 1 run.py args/aokvqa_cot_eval.yaml && cat $(ls -td logs/aokvqa-cot-eval_* | head -n 1)/evaluation.log"]
