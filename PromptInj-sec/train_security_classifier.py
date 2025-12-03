import random
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    DebertaV2Tokenizer,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from scipy.special import expit # Sigmoid function for probability conversion

# --- 1. Configuration ---
MODEL_NAME = "microsoft/deberta-v3-base"
DATA_FILE = "training_data.jsonl"  # Balanced dataset from dataset_generator.py
NUM_LABELS = 3 # L0_INJECTION, L1_HARMFUL_GOAL, L2_SAFE

# Label IDs based on the generator script
LABEL_MAP = {
    0: "L0_INJECTION",
    1: "L1_HARMFUL_GOAL",
    2: "L2_SAFE",
}

# --- 2. Data Loading and Preprocessing ---

def load_and_prepare_data(file_path: str):
    """Loads the JSONL file, converts it to a Hugging Face Dataset, and splits it."""
    print(f"Loading data from {file_path}...")
    
    # Load JSONL file into a Pandas DataFrame
    df = pd.read_json(file_path, lines=True)

    # Convert the list of labels into separate columns for training purposes
    labels_df = pd.DataFrame(df['labels'].to_list(), columns=LABEL_MAP.values())
    df = pd.concat([df['text'], labels_df], axis=1)

    # Split data 80/10/10 for train/validation/test
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Convert Pandas DataFrames to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))
    
    print(f"Train/Validation/Test split complete: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset


def tokenize_function(examples, tokenizer):
    """Tokenizes the text and formats the labels for multi-label training."""
    # Tokenize the text with explicit max_length to control memory usage
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

    # Prepare the labels. Trainer expects a 'labels' key in the tokenized output.
    labels = [examples[label_name] for label_name in LABEL_MAP.values()]
    
    # Transpose the list of lists: (num_classes, batch_size) -> (batch_size, num_classes)
    tokenized["labels"] = list(zip(*labels)) 
    
    # Convert tuples of labels back to lists before returning
    tokenized["labels"] = [list(l) for l in tokenized["labels"]]

    return tokenized

# --- 3. Custom Metric Computation for Multi-Label ---

def compute_metrics(eval_pred):
    """
    Computes standard metrics (Accuracy, AUC, and F1-score) for multi-label classification.
    """
    # predictions is a tuple (logits, labels)
    logits, labels = eval_pred
    
    # Convert logits (raw outputs) to probabilities using sigmoid
    probabilities = expit(logits)
    
    # Convert probabilities to binary predictions (0 or 1) using a threshold (e.g., 0.5)
    predictions = (probabilities >= 0.5).astype(int)

    # Flatten labels and predictions for macro/micro metrics
    y_true = labels.flatten()
    y_pred = predictions.flatten()
    
    # Calculate metrics, focusing on the required Confidentiality F1-score
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    micro_f1 = f1_score(labels, predictions, average='micro', zero_division=0)
    accuracy = accuracy_score(labels, predictions)

    # Calculate AUC (Area Under the Curve)
    try:
        auc = roc_auc_score(labels, probabilities, average='macro')
    except ValueError:
        # AUC is not well-defined if only one class is present in the batch
        auc = 0.0

    return {
        'accuracy': accuracy,
        'f1_macro': macro_f1,
        'f1_micro': micro_f1,
        'auc_macro': auc,
    }


# --- 4. Main Training Function ---

def fine_tune_deberta(train_dataset, val_dataset):
    """Initializes and runs the fine-tuning process for DeBERTa."""
    
    # Initialize Tokenizer and Model (using slow tokenizer to avoid conversion issues with DeBERTa v3)
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
    
    # Load DeBERTa for Sequence Classification with the correct number of labels
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    
    # Apply tokenization to the datasets
    tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_val = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    # Drop original columns to keep only 'input_ids', 'attention_mask', and 'labels'
    tokenized_train = tokenized_train.remove_columns(list(LABEL_MAP.values()) + ["text"])
    tokenized_val = tokenized_val.remove_columns(list(LABEL_MAP.values()) + ["text"])

    # Define Training Arguments (CPU training for DeBERTa v3 base)
    training_args = TrainingArguments(
        output_dir="./deberta_pi_classifier",
        num_train_epochs=3,                     # Number of epochs to train
        per_device_train_batch_size=8,          # Batch size for CPU
        per_device_eval_batch_size=8,           # Batch size for evaluation
        gradient_accumulation_steps=2,          # Effective batch = 16
        warmup_steps=200,                       # Warmup steps
        weight_decay=0.01,                      # Strength of weight decay
        logging_dir='./logs',                   # Directory for storing logs
        logging_steps=50,
        eval_strategy="epoch",                  # Evaluate at the end of each epoch
        save_strategy="epoch",                  # Save checkpoint at the end of each epoch
        load_best_model_at_end=True,            # Load the best model found during training
        metric_for_best_model="f1_macro",       # Use F1-macro to select the best model
        fp16=False,                             # CPU doesn't use fp16
        use_cpu=True,                           # Force CPU training (avoids MPS memory issues)
        dataloader_pin_memory=False,            # Not needed for CPU
    )

    # Define a custom Trainer for Multi-Label Loss (BCEWithLogitsLoss)
    class MultiLabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            # Convert labels to float (required for BCEWithLogitsLoss)
            labels = labels.float()
            
            # Use Binary Cross Entropy with Logits for Multi-Label Classification
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1, self.model.config.num_labels))
            
            return (loss, outputs) if return_outputs else loss

    # Initialize the Trainer
    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    print("\n--- Starting Model Training (Stage 1 Detection) ---")
    trainer.train()
    print("--- Training Complete ---")

    # Evaluate on the validation set
    print("\n--- Validation Results ---")
    validation_metrics = trainer.evaluate(tokenized_val)
    print(validation_metrics)
    
    # Save the final model
    final_model_path = "./final_deberta_classifier"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\n✅ Final model and tokenizer saved to: {final_model_path}")

# --- 5. Execution ---
if __name__ == "__main__":
    
    # Ensure a consistent random state for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    try:
        # Load, prepare, and split data
        train_ds, val_ds, test_ds = load_and_prepare_data(DATA_FILE)
        
        # Run the fine-tuning process
        fine_tune_deberta(train_ds, val_ds)
        
        # Added explicit note about the file name for confirmation.
        FILE_NAME = "train_security_classifier.py"
        print(f"\n--- Script Confirmation ---")
        print(f"This script, '{FILE_NAME}', has completed its fine-tuning process.")
        print("To complete the evaluation, run the trainer.evaluate() command on the 'test_ds' using the saved model.")
        
    except FileNotFoundError:
        print(f"\nERROR: The data file '{DATA_FILE}' was not found.")
        print("Please ensure you run 'generate_security_dataset.py' successfully first to create the training data.")