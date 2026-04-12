# Intent Detection in Banking using LLaMA-3 (Fine-Tuning with Unsloth)

This project fine-tunes a large language model (`unsloth/llama-3-8b-bnb-4bit`) on the **BANKING77** dataset to accurately classify customer intents across 77 distinct banking query categories. It uses **Unsloth** and **LoRA** (Low-Rank Adaptation) to achieve high-performance fine-tuning within the memory constraints of a free Google Colab T4 GPU.

## Video Demo

You can watch the video demonstration of this project here: [Video Demo (Google Drive)](https://drive.google.com/drive/folders/1TrqQV7YLZe3FAH97z7ozKB_UcT0dQbW8?usp=sharing)

## Project Structure

```text
├── configs/
│   ├── train.yaml            # Hyperparameters for training and LoRA config
│   └── inference.yaml        # Configuration for loading the model during inference
├── sample_data/              # Generated from preprocessing script
│   ├── train.csv             # Sub-sampled training data
│   └── test.csv              # Sub-sampled test data
├── scripts/
│   ├── preprocess_data.py    # Downloads and splits the dataset
│   ├── train.py              # Main training script using SFTTrainer
│   ├── inference.py          # Standalone inference class to predict a single intent
│   ├── train.ipynb               # The main Google Colab notebook to run the entire pipeline
│   └── evaluate.py           # Full evaluation script on test.csv
├── train.sh
├── inference.sh
├── requirements.txt          # PIP dependencies for local setup (if applicable)
└── README.md                 # Project documentation
```

## How to Run (Google Colab - Recommended)

Because Unsloth is deeply integrated with Linux and CUDA, **running this project entirely on Google Colab (with a T4 GPU)** is highly recommended. You can run all steps sequentially using the `train.ipynb` notebook provided in this repository.

1. **Open Google Colab:** Upload `train.ipynb` or open it directly, and ensure your Runtime is set to `T4 GPU`.
2. **Install Dependencies:**
   ```python
   !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   !pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
   !pip install pyyaml pandas datasets
   ```
3. **Clone the Repository:**
   ```bash
   !git clone https://github.com/CPTMinh/ANLP-Project02-FINE-TUNING-INTENT-DETECTION-MODEL-WITH-BANKING-DATASET.git
   %cd ANLP-Project02-FINE-TUNING-INTENT-DETECTION-MODEL-WITH-BANKING-DATASET
   ```
4. **Run Data Preprocessing:**
   ```bash
   !python scripts/preprocess_data.py
   ```
5. **Mount Google Drive (Optional but Recommended):**
   Run the code cells in the notebook to mount your drive and modify `configs/train.yaml` automatically so checkpoints save safely to your Google Drive.
6. **Run Training:**
   ```bash
   !python scripts/train.py
   ```
7. **Run Inference & Evaluation:**
   After training finishes, execute the evaluation scripts to test the model on the generated `test.csv`:
   ```bash
   !python scripts/inference.py
   !python scripts/evaluate.py
   ```

## Inference Usage

As required by the grading instructions, `scripts/inference.py` contains a standalone `IntentClassification` class with an `__init__` and `__call__` method.

Example:
```python
from scripts.inference import IntentClassification

# Initialize with the configuration file
classifier = IntentClassification(model_path="configs/inference.yaml")

# Make a prediction
predicted_label = classifier("I lost my credit card yesterday, how do I freeze my account?")
print(f"Predicted Intent ID: {predicted_label}")
```