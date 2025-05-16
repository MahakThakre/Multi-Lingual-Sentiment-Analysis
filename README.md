# 🧠 Multilingual Sentiment Analysis Project

## 🚀 Overview
This project is developed as part of the **Multilingual Sentiment Analysis Competition** hosted on Kaggle. The goal is to fine-tune the **LLaMA 3.1-8B-Instruct** model to classify text sentiment (Positive or Negative) across **13 Indian languages** using **LoRA-based parameter-efficient fine-tuning**, all within **Kaggle Notebook constraints**.

## 🏁 Competition Timeline
- **Start:** February 14, 2025  
- **End:** February 18, 2025  
- **Evaluation Metric:** F1 Score  
- **Restrictions:**  
  - Must use **LLaMA 3.1-8B-Instruct**  
  - Training only on **provided dataset**  
  - **No external data or local training allowed**  
  - **Kaggle Notebooks only**

- ### 🔧 Technologies Used and Their Roles

| Technology                | Purpose / Where It Was Used                                                                 |
|--------------------------|---------------------------------------------------------------------------------------------|
| **[LLaMA 3.1-8B-Instruct](https://huggingface.co/meta-llama)** | Main instruction-tuned language model used for generating answers based on the input prompt. |
| **[Unsloth](https://github.com/unslothai/unsloth)**              | Efficient fine-tuning of the LLaMA 3.1-8B model using QLoRA, reducing training time and memory usage. |
| **[LoRA](https://arxiv.org/abs/2106.09685)** (Low-Rank Adaptation) | Applied to fine-tune large language models efficiently by freezing base weights and adapting only low-rank matrices. |
| **[Hugging Face Transformers](https://huggingface.co/transformers)** | Used to load and run the LLaMA model, manage tokenizer, and integrate Unsloth for training and inference. |
| **[Datasets](https://huggingface.co/docs/datasets)** | Used to load, preprocess, and manage dataset splits (train/validation) for model training. |
| **[Pandas](https://pandas.pydata.org/)**               | Employed for data manipulation, EDA (exploratory data analysis), and converting CSVs to required formats. |
| **[Scikit-learn](https://scikit-learn.org/)**         | Used for computing evaluation metrics like F1 score and for additional preprocessing tools. |
| **[Kaggle Notebook](https://www.kaggle.com/code)**      | Entire development, experimentation, and final submissions were done using Kaggle’s GPU-powered notebooks. |
  
## 🧾 Dataset
- **Train File:** `train.csv`  
- **Test File:** `test.csv`  
- **Languages Covered:**
  - Assamese, Bodo, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu, Urdu

## 🧪 Methodology

### ✅ Fine-Tuning Strategy
- **Model Loading:** Loaded the base LLaMA 3.1-8B-Instruct model with 4-bit quantization.
- **LoRA Config:**
  - Rank (`r`): 16
  - LoRA Alpha: 16
  - Target Modules: QKV, MLP
  - Dropout: 0
  - Gradient Checkpointing: Enabled via `"unsloth"` for memory efficiency.
- **Prompt Template:** Custom Alpaca-style instruction format per sample.

### 🔄 Data Handling
- Labeled sentences mapped as `1` for Positive and `0` for Negative.
- Mapped language codes to full names for clarity in prompts.
- Used 80/20 train-validation split.

## 🧠 Training Configuration
- **Sequence Length:** 4096
- **Batch Size:** Auto-managed by Unsloth for optimal GPU usage
- **Evaluation Strategy:** F1 Score on validation set
- **Trainer:** `SFTTrainer` from TRL for supervised fine-tuning

## 📊 Evaluation
- **Metric:** F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
- **Grading Policy:**
  - For `F1 Score ≤ 0.5`: `Score = 2 * F1 * 50`
  - For `F1 Score > 0.5`: `Score = ((F1 - 0.5) / (MaxF1 - 0.5)) * 50 + 50`

## 📂 Directory Structure
```
.
├── train.csv
├── test.csv
├── llama_finetune_sentiment.ipynb  # Main notebook
├── README.md
```

## 📄 License
Released under the Apache 2.0 License.
