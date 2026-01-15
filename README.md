# [AAAI 2026 Oral] Bridging the Copyright Gap: Do Large Vision-Language Models Recognize and Respect Copyrighted Content?

[![AAAI 2026 (Oral)](https://img.shields.io/badge/AAAI%202026-Oral-blue)](https://aaai.org/conference/aaai/aaai-26/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.21871-b31b1b.svg)](https://arxiv.org/abs/2512.21871)

This is the official implementation of **"Bridging the Copyright Gap: Do Large Vision-Language Models Recognize and Respect Copyrighted Content?"** (AAAI 2026 Oral).

## 📑 Table of Contents

- [Overview](#-overview)
- [News](#-news)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Citation](#-citation)

## 📌 Overview

This repository provides a comprehensive framework for evaluating and enhancing copyright compliance in Large Vision-Language Models (LVLMs). It includes:

- **Benchmark Dataset**: A collection of 50,000 multimodal query-content pairs spanning 4 content categories
- **Evaluation Framework**: Comprehensive metrics and tools for assessing copyright compliance
- **CopyGuard**: A tool-augmented defense framework designed to enhance copyright compliance in LVLMs

The benchmark dataset contains **50,000 query-content pairs** organized across 4 categories:

| Category | Content Files | Notices | Task Types | Queries per Task | Total Pairs |
|----------|--------------|---------|------------|------------------|-------------|
| Books | 100 | 5 | 4 | 10 | 20,000 |
| Code | 50 | 5 | 4 | 10 | 10,000 |
| Lyrics | 50 | 5 | 4 | 10 | 10,000 |
| News | 50 | 5 | 4 | 10 | 10,000 |

**Calculation**: 250 material sources × 5 forms of copyright notice × 4 task types × 10 queries per task = 50,000


## 📢 News

- **[December 2025]** 🎉 Our paper "Bridging the Copyright Gap: Do Large Vision-Language Models Recognize and Respect Copyrighted Content?" has been accepted to **AAAI 2026 (Oral Presentation)**!


## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/copyright-compliance.git
cd copyright-compliance/Code
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

For API-based models:
```bash
pip install openai litellm
```

### 3. Download Dataset

Download the image datasets from [Google Drive](https://drive.google.com/drive/folders/1YTkxqmDBFvMpVO3HJLeNcLX7_MN4umX_?usp=sharing) and place them in the `dataset/` directory. The required files are:

- `book_copyright_images.zip`
- `code_copyright_images.zip`
- `lyrics_copyright_images.zip`
- `news_copyright_images.zip`

After downloading, extract all zip files:
```bash
cd dataset
python extract_images.py
cd ..
```

**⚠️ Fair Use and Copyright Notice**: This repository contains copyrighted materials intended **solely for research and educational purposes**. The dataset and code are provided exclusively for academic research and education, copyright compliance evaluation and analysis, non-commercial scientific purposes, and criticism, comment, and scholarship.

Users are responsible for ensuring that their use of this code and dataset complies with all applicable copyright laws in their jurisdiction. If you are a copyright holder and have concerns regarding the inclusion of your work, please contact us.

### 4. Configuration

Create your configuration file by copying the example:
```bash
cp config/config.example.yaml config/config.yaml
```

Then, edit `config/config.yaml` and update it with your settings:
```yaml
api_keys:
  openai_api_key: "your-key"
  serper_api_key: "your-key"

models:
  model_paths:
    qwen2_5_vl_7b: "/path/to/model"
```


## 🚀 Quick Start

This section provides a step-by-step guide to evaluate models with and without the CopyGuard defense mechanism.

### Supported Models

The framework supports the following LVLMs:

**Local Models:**
- **Qwen2.5-VL** (7B, 3B): State-of-the-art multimodal model
  - Install: `pip install qwen-vl-utils`
- **DeepSeek-VL** (7B): High-performance vision-language model
  - Install: `git clone https://github.com/deepseek-ai/DeepSeek-VL.git && cd DeepSeek-VL && pip install -e .`
- **LLaVA** (1.5-7B, 1.5-13B, Next): Popular open-source VLM
  - Install: `git clone https://github.com/haotian-liu/LLaVA.git && export PYTHONPATH=/path/to/LLaVA:$PYTHONPATH`
- **GLM-4V** (9B): Bilingual vision-language model
- **InstructBLIP** (Vicuna-7B): Instruction-tuned BLIP
- **Janus-Pro** (7B): Unified vision understanding and generation model

**API-Based Models:**
- **GPT-4o, GPT-4o-mini**: OpenAI's multimodal models
- **Claude 3** (Opus, Sonnet, Haiku): Anthropic's vision models
- **Gemini Pro**: Google's multimodal model

### Step 1: Generate Baseline Responses (Without Defense)

Generate baseline model responses without any defense mechanism:

```bash
python -m models.generate_responses \
    --model-type qwen \
    --model-path /path/to/qwen2.5-vl-7b \
    --dataset dataset/book_copyright.json \
    --output results/book_baseline.json \
    --image-mode 0 \
    --notice-mode 0
```

### Step 2: Evaluate Baseline Responses

Evaluate the generated baseline responses:

```bash
python -m evaluation.evaluator \
    --input results/book_baseline.json \
    --output results/book_baseline_eval.json \
    --csv results/book_baseline_metrics.csv
```

### Step 3: Generate Responses With CopyGuard Defense

Generate model responses with the CopyGuard defense mechanism enabled:

```bash
python -m models.generate_responses_with_defense \
    --model-type qwen \
    --model-path /path/to/qwen2.5-vl-7b \
    --dataset dataset/book_copyright.json \
    --output results/book_defense.json \
    --image-mode 0 \
    --notice-mode 0
```

**Parameter Explanation**:

The benchmark evaluates 5 forms of copyright notices by combining `--image-mode` and `--notice-mode` parameters:

| Copyright Notice Form | `--image-mode` | `--notice-mode` | Description |
|----------------------|----------------|-----------------|-------------|
| 1. No notice (baseline) | `0` | `0` | Plain text image with no copyright notice |
| 2. Generic text notice | `0` | `1` | Plain text image with "All rights reserved" in query |
| 3. Original text notice | `0` | `2` | Plain text image with original copyright text in query |
| 4. Generic image notice | `1` | `0` | Image with "All rights reserved" embedded, no text notice |
| 5. Original image notice | `2` | `0` | Image with original copyright text embedded, no text notice |

- `--image-mode`: Specifies the image presentation mode. Options: `0` = plain text image (no copyright notice), `1` = image with generic copyright notice ("All rights reserved"), `2` = image with original copyright notice. This parameter evaluates how the modality of copyright notices (embedded in image vs. presented as text) affects model compliance.
- `--notice-mode`: Specifies the copyright notice format in the query. Options: `0` = no notice, `1` = generic notice ("All rights reserved"), `2` = original notice (content-specific copyright text). This parameter evaluates how different types of copyright notices impact model behavior, as discussed in the paper.

### Step 4: Evaluate Defense-Enabled Responses

Evaluate the responses generated with CopyGuard defense:

```bash
python -m evaluation.evaluator \
    --input results/book_defense.json \
    --output results/book_defense_eval.json \
    --csv results/book_defense_metrics.csv
```

## 📚 Advanced Usage

### Using Different Models

#### Local Models

**Qwen2.5-VL:**
```bash
python -m models.generate_responses \
    --model-type qwen \
    --model-path /path/to/Qwen2.5-VL-7B-Instruct \
    --dataset dataset/book_copyright.json \
    --output results/qwen_results.json
```

**DeepSeek-VL:**
```bash
python -m models.generate_responses \
    --model-type deepseek \
    --model-path /path/to/deepseek-vl-7b-chat \
    --dataset dataset/book_copyright.json \
    --output results/deepseek_results.json
```

**LLaVA:**
```bash
python -m models.generate_responses \
    --model-type llava \
    --model-path /path/to/llava-v1.5-7b \
    --llava-repo-path /path/to/LLaVA \
    --dataset dataset/book_copyright.json \
    --output results/llava_results.json
```

**GLM-4V:**
```bash
python -m models.generate_responses \
    --model-type glm \
    --model-path /path/to/glm-4v-9b \
    --dataset dataset/book_copyright.json \
    --output results/glm_results.json
```

#### API-Based Models

**GPT-4o:**
```bash
export OPENAI_API_KEY="your-api-key"
python -m models.generate_responses \
    --model-type gpt-4o \
    --model-path gpt-4o \
    --dataset dataset/book_copyright.json \
    --output results/gpt4o_results.json
```

**Claude:**
```bash
export ANTHROPIC_API_KEY="your-api-key"
python -m models.generate_responses \
    --model-type claude \
    --model-path claude-3-7-sonnet-20250219 \
    --api-base https://api.anthropic.com/v1 \
    --dataset dataset/book_copyright.json \
    --output results/claude_results.json
```

### CopyGuard Defense Options

The CopyGuard defense framework can be customized with various options:

```bash
python -m models.generate_responses_with_defense \
    --model-type qwen \
    --model-path /path/to/qwen2.5-vl-7b \
    --dataset dataset/book_copyright.json \
    --output results/book_defense.json \
    --enable-ocr \              # Enable OCR-based copyright detection
    --enable-verifier \         # Enable copyright status verification
    --enable-risk-analyzer \    # Enable query risk analysis
    --enable-reminder \         # Enable copyright reminders
    --block-risky              # Block risky queries (vs. warning only)
```

**Defense Components:**
- **Notice Identifier**: Uses PaddleOCR to detect copyright notices in images
- **Status Verifier**: Uses Serper API + DeepSeek-R1 to verify copyright status
- **Risk Analyzer**: Analyzes queries for potential copyright infringement
- **Status Reminder**: Provides copyright guidance to the model

### Batch Processing

Process all 5 copyright notice configurations:

```bash
#!/bin/bash
MODEL_TYPE="qwen"
MODEL_PATH="/path/to/qwen2.5-vl-7b"
DATASET="dataset/book_copyright.json"

for image_mode in 0 1 2; do
    for notice_mode in 0 1 2; do
        if [ $image_mode -eq 0 ] || [ $notice_mode -eq 0 ]; then
            echo "Processing image_mode=$image_mode, notice_mode=$notice_mode"
            python -m models.generate_responses \
                --model-type $MODEL_TYPE \
                --model-path $MODEL_PATH \
                --dataset $DATASET \
                --output results/book_${image_mode}_${notice_mode}.json \
                --image-mode $image_mode \
                --notice-mode $notice_mode
        fi
    done
done
```

### Dataset Structure

The dataset JSON should follow this structure:

```json
[
    {
        "txt_file": "dataset/book_copyright/1Q84.txt",
        "img_file_0": "dataset/book_copyright_images/0/1Q84_sample_1.png",
        "img_file_1": "dataset/book_copyright_images/1/1Q84_sample_1.png",
        "img_file_2": "dataset/book_copyright_images/2/1Q84_sample_1.png",
        "text": "Full text content from the book...",
        "copyright_text": "Copyright © 2009 by Haruki Murakami. All rights reserved."
    }
]
```

Each entry contains:
- `txt_file`: Path to the text file
- `img_file_0/1/2`: Paths to images with different copyright notice modes
- `text`: The actual text content
- `copyright_text`: Original copyright notice text

## 📄 Citation

If you find this work useful for your research, please cite our [paper](https://arxiv.org/abs/2512.21871):

```bibtex
@inproceedings{xu2026bridging,
  title={Bridging the Copyright Gap: Do Large Vision-Language Models Recognize and Respect Copyrighted Content?},
  author={Xu, Naen and Zhang, Jinghuai and Li, Changjiang and An, Hengyu and Zhou, Chunyi and Wang, Jun and Xu, Boyu and Li, Yuyuan and Du, Tianyu and Ji, Shouling},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```
