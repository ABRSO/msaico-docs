# vLLM with NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition

Wednesday, December 24th 2025 7:26 PM

This document details the installation steps and configurations taken to get vLLM working with NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition cards. Each LLM requires specific versions of vLLM, libraries, GPU driver and CUDA.

> **GGUF models with vLLM**

> vLLM has experimental GGUF support, and cannot support multi-part gguf files.[^1] If it is essential to use a multi-part gguf file, then they can be merged into a single file with the help of gguf-split feature of llama.cpp.[^2]

## **openai/gpt-oss-120b**

```shell
# Create virtual environment
uv venv .gpt120b --python 3.12 --seed  
source .gpt120b/bin/activate

# Install vLLM  
UV_HTTP_TIMEOUT=1000 uv pip install --pre vllm --extra-index-url https://wheels.vllm.ai/gpt-oss/ --extra-index-url https://download.pytorch.org/whl/nightly/cu128 --index-strategy unsafe-best-match

# Serve gpt-oss-120b model
/home/sadmin/.gpt120b/bin/vllm serve openai/gpt-oss-120b --port 11434 --max_model_len 2048 --gpu-memory-utilization 0.8
```

- If you run into `openai_harmony.HarmonyError: error downloading or loading vocab file: failed to download or load vocab` error, this can be worked around by downloading the tiktoken encoding files in advance and setting the `TIKTOKEN_ENCODINGS_BASE` environment variable. This is caused by a bug in openai_harmony code.

```shell
mkdir -p tiktoken_encodings
wget -O tiktoken_encodings/o200k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
wget -O tiktoken_encodings/cl100k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
export TIKTOKEN_ENCODINGS_BASE=${PWD}/tiktoken_encodings
```

**Environment variables for running this model**:

```shell
CUDA_HOME=/usr/local/cuda-12.8
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:%(LD_LIBRARY_PATH)s
TIKTOKEN_ENCODINGS_BASE=/home/sadmin/tiktoken_encodings
```

**systemctl service file for running gpt-oss-120b**:

```shell
[Unit]
Description=vLLM serve GPT-OSS 120b
After=network.target
Wants=network-online.target

[Service]
# Run as the non-root user who owns the venv/workspace
User=sadmin
Group=sadmin

# Make sure essential CUDA envs are visible to the service
Environment=CUDA_HOME=/usr/local/cuda-12.8
Environment=LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:%(LD_LIBRARY_PATH)s
Environment=TIKTOKEN_ENCODINGS_BASE=/home/sadmin/tiktoken_encodings

# ExecStart: call the venv vllm binary directly (no wrapper script)
ExecStart=/home/sadmin/.gpt120b/bin/vllm serve openai/gpt-oss-120b --port 11434 --max_model_len 2048 --gpu-memory-utilization 0.8

# Restart behavior: resilient and restarts after a crash or system reboot
Restart=on-failure
RestartSec=5
# Run the service after boot (also restart on systemd daemon reload)
StartLimitIntervalSec=60
StartLimitBurst=5

StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

## **RedHatAI/Llama-4-Maverick-17B-128E-Instruct-NVFP4**

```shell
# Create virtual environment  
uv venv .llama4-maverick --python 3.12 --seed    
source .llama4-maverick/bin/activate

# Install vLLM    
UV_HTTP_TIMEOUT=1000 uv pip install vllm --extra-index-url https://download.pytorch.org/whl/nightly/cu128 --index-strategy unsafe-best-match

# Run model
/home/sadmin/.llama4-maverick/bin/vllm serve RedHatAI/Llama-4-Maverick-17B-128E-Instruct-NVFP4 --port 11434 --max_model_len 2048 --gpu-memory-utilization 0.7 --cpu-offload-gb 200

# unsloth GGUF model
/home/sadmin/llama.cpp/build/bin/llama-server -m /home/sadmin/.cache/huggingface/hub/llama-4-Maverick-17B-128E-Instruct-Q4/Llama-4-Maverick-17B-128E-Instruct-UD-Q4_K_XL-00001-of-00005.gguf -c 2048 --port 11434 --host 0.0.0.0 --parallel 50
```

> While this model is offloaded to CPU with vLLM, the performance is not up to mark and the inference generation is not coherent, check the below screenshots for reference. With llama.cpp this is not a problem.

**systemctl service file for running RedHatAI/Llama-4-Maverick-17B-128E-Instruct-NVFP4 using vLLM**:

```shell
[Unit]
Description=vLLM serve Llama-4-Maverick
After=network.target
Wants=network-online.target

[Service]
# Run as the non-root user who owns the venv/workspace
User=sadmin
Group=sadmin

# Make sure essential CUDA envs are visible to the service
Environment=CUDA_HOME=/usr/local/cuda-12.8
Environment=LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:%(LD_LIBRARY_PATH)s

# ExecStart: call the venv vllm binary directly (no wrapper script)
ExecStart=/home/sadmin/.llama4-maverick/bin/vllm serve RedHatAI/Llama-4-Maverick-17B-128E-Instruct-NVFP4 --port 11434 --max_model_len 2048 --gpu-memory-utilization 0.7 --cpu-offload-gb 200

# Restart behavior: resilient and restarts after a crash or system reboot
Restart=on-failure
RestartSec=5
# Run the service after boot (also restart on systemd daemon reload)
StartLimitIntervalSec=60
StartLimitBurst=5

StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**systemctl service file for running unsloth/Llama-4-Maverick-17B-128E-Instruct-UD-Q4_K_XL using llama.cpp**:

```shell
[Unit]
Description=vLLM serve Llama-4-Maverick
After=network.target
Wants=network-online.target

[Service]
# Run as the non-root user who owns the venv/workspace
User=sadmin
Group=sadmin

# Make sure essential CUDA envs are visible to the service
Environment=CUDA_HOME=/usr/local/cuda-12.8
Environment=LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:%(LD_LIBRARY_PATH)s

# ExecStart: call the venv vllm binary directly (no wrapper script)
ExecStart=/home/sadmin/llama.cpp/build/bin/llama-server -m /home/sadmin/.cache/huggingface/hub/llama-4-Maverick-17B-128E-Instruct-Q4/Llama-4-Maverick-17B-128E-Instruct-UD-Q4_K_XL-00001-of-00005.gguf -c 2048 --port 11434 --host 0.0.0.0 --parallel 50

# Restart behavior: resilient and restarts after a crash or system reboot
Restart=on-failure
RestartSec=5
# Run the service after boot (also restart on systemd daemon reload)
StartLimitIntervalSec=60
StartLimitBurst=5

StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

## **meta-llama/Llama-3.3-70B-Instruct**

```shell
# Create virtual environment  
uv venv .llama3.3-70b --python 3.12 --seed    
source .llama3.3-70b/bin/activate

# Install vLLM    
UV_HTTP_TIMEOUT=1000 uv pip install vllm --extra-index-url https://download.pytorch.org/whl/nightly/cu128 --index-strategy unsafe-best-match

# Run model
/home/sadmin/.llama3.3-70b/bin/vllm serve meta-llama/Llama-3.3-70B-Instruct --port 11434 --max_model_len 8192 --gpu-memory-utilization 0.8 --tensor-parallel-size 4
```

**systemd service file for running meta-llama/Llama-3.3-70B-Instruct using vLLM**:

```shell
[Unit]
Description=vLLM serve llama-3.3-70B
After=network.target
Wants=network-online.target

[Service]
# Run as the non-root user who owns the venv/workspace
User=sadmin
Group=sadmin

# Make sure essential CUDA envs are visible to the service
Environment=CUDA_HOME=/usr/local/cuda-12.8
Environment=LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:%(LD_LIBRARY_PATH)s
# vLLM / torch runtime hints
#Environment=VLLM_ENGINE_ITERATION_TIMEOUT_S=120
#Environment=VLLM_TORCH_COMPILE=0
#Environment=NCCL_HEARTBEAT_TIMEOUT=0
#Environment=NCCL_ENABLE_MONITORING=0

# ExecStart: call the venv vllm binary directly (no wrapper script)
ExecStart=/home/sadmin/.llama3.3-70b/bin/vllm serve meta-llama/Llama-3.3-70B-Instruct --port 11434 --max_model_len 8192 --gpu-memory-utilization 0.8 --tensor-parallel-size 2

# Restart behavior: resilient and restarts after a crash or system reboot
Restart=on-failure
RestartSec=5
# Run the service after boot (also restart on systemd daemon reload)
StartLimitIntervalSec=60
StartLimitBurst=5

StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

## intfloat/e5-large-v2

```shell
# Create a virtual environment
uv venv .e5-large-v2 --python 3.12 --seed    
source .e5-large-v2/bin/activate

# Install vLLM    
UV_HTTP_TIMEOUT=1000 uv pip install vllm --extra-index-url https://download.pytorch.org/whl/nightly/cu128 --index-strategy unsafe-best-match

# Run model
CUDA_VISIBLE_DEVICES=1 /home/sadmin/.e5-large-v2/bin/vllm serve intfloat/e5-large-v2 --port 8080 --gpu-memory-utilization 0.15
```

Refer to Online Serving and Pooling API in vLLM docs for embedding model serving[^3]

**systemd service file for running intfloat/e5-large-v2 using vLLM**:

```shell
[Unit]
Description=vLLM serve e5-large-v2
After=network.target
Wants=network-online.target

[Service]
# Run as the non-root user who owns the venv/workspace
User=sadmin
Group=sadmin

# Make sure essential CUDA envs are visible to the service
Environment=CUDA_HOME=/usr/local/cuda-12.8
Environment=LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:%(LD_LIBRARY_PATH)s
Environment=CUDA_VISIBLE_DEVICES=1

# ExecStart: call the venv vllm binary directly (no wrapper script)
ExecStart=/home/sadmin/.e5-large-v2/bin/vllm serve intfloat/e5-large-v2 --port 8080 --gpu-memory-utilization 0.15

# Restart behavior: resilient and restarts after a crash or system reboot
Restart=on-failure
RestartSec=5
# Run the service after boot (also restart on systemd daemon reload)
StartLimitIntervalSec=60
StartLimitBurst=5

StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**cURL request**:

```shell
curl -X 'POST' \
  'http://10.141.1.242:8080/pooling' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "intfloat/e5-large-v2",
  "input": ["Hello, there!"],
  "task": "embed"
}'
```

## References

[^1]: [GGUF - vLLM](https://docs.vllm.ai/en/latest/features/quantization/gguf/)

[^2]: [gguf-split: split and merge gguf per batch of tensors by phymbert · Pull Request #6135 · ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/pull/6135)

[^3]: [Online Pooling serving](https://github.com/vllm-project/vllm/blob/main/docs/models/pooling_models.md#online-serving)
