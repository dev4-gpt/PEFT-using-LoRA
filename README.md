Parameter-Efficient Fine-Tuning of RoBERTa with LoRA
This repository contains the code and documentation for the project "Finetuning with LORA," completed as part of the Deep Learning course at New York University. The project focuses on efficiently fine-tuning a roberta-base model for a multi-class text classification task on the AG News dataset using Parameter-Efficient Fine-Tuning (PEFT), specifically the LoRA (Low-Rank Adaptation) technique.

Link to Full Project Report (PDF) | Link to Jupyter Notebook (ipynb)

Project Overview
Transformer-based models like RoBERTa have set the standard for NLP tasks but are computationally expensive to fully fine-tune. This project demonstrates a resource-efficient approach by leveraging LoRA to adapt a large pre-trained model to a downstream task with minimal trainable parameters.

The core objective was to achieve high classification accuracy while keeping the number of trainable parameters under 1 million. Our final model successfully met this constraint, showcasing the power and efficiency of PEFT methods.

Key Achievements:
High-Accuracy Fine-Tuning: Achieved a 90.04% evaluation accuracy on the AG News classification task.

Extreme Parameter Efficiency: Fine-tuned the model by training only 907,012 parameters, which constitutes just 0.72% of the total 125 million parameters in the roberta-base model.

Efficient Methodology: Implemented a robust training pipeline using the Hugging Face ecosystem, including transformers, peft, and datasets.

In-depth Analysis: The project includes a detailed analysis of the training process, hyperparameter choices, and the impact of the LoRA configuration on model performance.

Methodology
1. Dataset and Preprocessing
Dataset: AG News, a collection of over 120,000 news articles categorized into four classes: "World," "Sports," "Business," and "Sci/Tech."

Preprocessing: Text was tokenized using the RobertaTokenizer. The dataset was then split into a 90% training set and a 10% evaluation set.

2. Model Architecture
Base Model: roberta-base, a powerful pre-trained transformer model.

PEFT Technique: LoRA was applied to the model. The core idea is to freeze the pre-trained model weights and inject trainable low-rank matrices into the attention layers.

LoRA Configuration:

r (Rank): 8

lora_alpha (Scaling Factor): 16

target_modules: Applied to the query and value projections in the self-attention layers.

bias: Set to 'lora_only'.

3. Training Strategy
Optimizer: AdamW with a learning rate of 1e-4 and weight decay of 0.01.

Scheduler: Linear learning rate scheduler with a warmup ratio of 0.1.

Batch Size: 32 for training, 64 for evaluation.

Epochs: 6

Environment: Trained on a GPU with mixed-precision (FP16) enabled to accelerate training.

Results
The model's performance was tracked throughout the training process. The evaluation accuracy steadily improved, demonstrating stable convergence.


Figure: Training/Evaluation Loss and Evaluation Accuracy plotted against training steps.

Metric

Value

Trainable Parameters

907,012

Total Parameters

125,537,288

Percentage Trainable

0.7225%

Final Evaluation Accuracy

90.04%

How to Run
Clone the repository:

git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

Set up the environment:
Install the required Python libraries.

pip install transformers datasets evaluate accelerate peft trl bitsandbytes

Run the Jupyter Notebook:
Open and run the Roberta_PEFT.ipynb notebook in an environment like Google Colab (with GPU) or a local Jupyter instance with the necessary dependencies installed.
