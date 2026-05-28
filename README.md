# History-Focused LLM Fine-Tuning

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red?style=for-the-badge&logo=pytorch)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-green?style=for-the-badge)
![BitsAndBytes](https://img.shields.io/badge/BitsAndBytes-8bit_Quantization-orange?style=for-the-badge)
![Mistral](https://img.shields.io/badge/Mistral-7B_Instruct-purple?style=for-the-badge)
![Kaggle](https://img.shields.io/badge/Kaggle-GPU_Notebook-blue?style=for-the-badge&logo=kaggle)
![Datasets](https://img.shields.io/badge/Datasets-SQuAD_+_WikiQA-teal?style=for-the-badge)
![Trainer](https://img.shields.io/badge/HF-SFTTrainer-yellowgreen?style=for-the-badge)
![Evaluation](https://img.shields.io/badge/Evaluation-ROUGE_Score-darkgreen?style=for-the-badge)
![Deployment](https://img.shields.io/badge/Deployment-HuggingFace_Hub-gold?style=for-the-badge&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

<img width="724" height="402" alt="image" src="https://github.com/user-attachments/assets/08f33267-7c7d-4056-8bdd-4fb465ad8789" />


## Project Overview

This project implements Parameter-Efficient Fine-Tuning (PEFT) via Low-Rank Adaptation (LoRA) to specialize a Mistral-7B-Instruct-v0.2 base model for historical question-answering tasks. By consolidating and restructuring the text fields of two prominent benchmark datasets—SQuAD (contextual question-answering) and WikiQA (open-domain questions)—the pipeline unifies complex data schemas into Mistral’s native instruction format (`<s>[INST]...[/INST]`). Using 8-bit quantized optimization via bitsandbytes and the Hugging Face SFTTrainer, the model was trained for 100 steps on a Kaggle GPU environment, effectively dropping the initial training loss from a high of 2.44 down to a stabilized convergence around 1.80.

The engineering lifecycle concludes with an end-to-end evaluation and model deployment workflow. Instead of altering the massive 14 GB base model directly, the training process produced a highly efficient, modular 26 MB set of adapter weights (`adapter_model.safetensors`). Performance validation was handled through a side-by-side behavioral benchmark tracking string similarity (ROUGE scores) against the frozen base model. The finalized, lightweight adapter layers were then successfully packaged and deployed to the Hugging Face Hub, allowing any downstream application to immediately dynamically snap the history-tuned behaviors right onto the official Mistral base architecture for zero-context inference.
