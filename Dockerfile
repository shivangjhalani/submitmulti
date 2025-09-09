# 1. Base Image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pyyaml

# 4. Copy the rest of the application
COPY . .

# 5. Create checkpoints directory and download model checkpoint
RUN mkdir -p checkpoints && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ThefirstM/checkpoints', repo_type='dataset', local_dir='checkpoints', allow_patterns=['aokvqa_cot_aokvqa-cot-stage0/epoch-8/**'])"

# 6. Run preprocessing
RUN python preprocessing/aokvqa.py

# 7. Set the default command to run evaluation and display logs
CMD ["sh", "-c", "torchrun --nnodes 1 --nproc_per_node 1 run.py args/aokvqa_cot_eval.yaml && cat $(ls -td logs/aokvqa-cot-eval_* | head -n 1)/evaluation.log"]
