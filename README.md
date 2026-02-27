# Noise-Aware In-Context Learning (NAICL)

## Hallucination Evaluation and Mitigation for Large Audio-Language Models



------

## Overview

This repository provides:

- A hallucination benchmark for audio captioning (Clotho-1K)
- A hallucination evaluation pipeline
- A Noise-aware In-Context Learning (NIC) mitigation method

You can use this repository in two ways:

1. **Evaluate your own audio-language model**
2. **Apply our NIC method for hallucination mitigation**

------

# Dataset Preparation

Before running any experiments, you must download the **Clotho development dataset**.

Download from:

https://zenodo.org/records/3490684

After downloading, place the `development` split audio files into:

```
data/clotho/
```

The directory structure should look like:

```
data/
 └── clotho/
      ├── audio_1.wav
      ├── audio_2.wav
      ├── ...
```


## Step 1 — Install Requirements

Install all required dependencies:

```
pip install -r requirements.txt
```

------

# Part I — Evaluate Your Model

If you only want to evaluate your model on the Clotho-1K benchmark, follow the three steps below.

## Step 2 — Run Inference

Modify the corresponding `run_inference_*` script to load your model.

You have **two options**:

- **Local deployment**
- **API-based deployment**

Both options are provided with example scripts.

------

### Option 1 — Local Deployment

If you want to run the model locally, you must first deploy the model on your machine.

Install Hugging Face dependencies:

```
pip install transformers accelerate
```

Then:

- Download the model from Hugging Face
- Set the local model path in the inference script
- For different model you may need modify the script.

Run:

```
python run_inference_local.py
```

------

### Option 2 — API-Based Deployment

If you prefer API-based inference:

- Deploy your model using an OpenAI-compatible API format
   (e.g., OpenAI API or vLLM with OpenAI-compatible endpoints)
- Configure your API key and endpoint in the script

Run:

```
python run_inference_api.py
```


## Step 3 — Run Evaluation and Calculate Metrics

First run hallucination detection:

```
python evaluation.py
```

Then compute final metrics:

```
python calculate.py
```

This will produce:

- Hallucination Rate (HR)
- Hallucination type distribution
- Keyword frequency statistics (Event / Definite / Acoustic)

------

# Part II — Apply the NAICL Method

If you would like to reproduce our **Noise-aware In-Context Learning (NAICL)** method, additional preparation is required.

------

## Step 2 — Download BEATs Official Weights

NIC relies on BEATs as the acoustic encoder for noise retrieval.

Download BEATs from the official repository:

https://github.com/microsoft/unilm/tree/master/beats

Download the official checkpoint (e.g.,
 `BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt`) and place it in:

```
checkpoints/
```

Make sure the checkpoint path is correctly set in the NIC inference script.

------

## Step 3 — Run NIC Inference

Use the NIC-enabled inference script:

```
python run_inference_nic.py
```

## Step 4 — Evaluate NIC Results

After inference, run:

```
python evaluation.py
python calculate.py
```

------

# Notes

- NIC is a training-free, inference-time mitigation method.
- All evaluation metrics follow the definitions described in the associated paper.
