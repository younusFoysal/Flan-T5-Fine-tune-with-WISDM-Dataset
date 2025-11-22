"""
FLAN-T5 Fine-tuning for WISDM Activity Recognition Dataset
Following DataCamp tutorial: https://www.datacamp.com/tutorial/flan-t5-tutorial

This script fine-tunes FLAN-T5 on the WISDM accelerometer dataset by converting
time-series sensor data into text format for sequence-to-text generation.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import torch
from tqdm import tqdm


# ==================== STEP 1: Load WISDM Data ====================
def load_wisdm_data(file_path):
    """
    Load WISDM raw accelerometer data from text file.
    
    Args:
        file_path (str): Path to WISDM_ar_v1.1_raw.txt
    
    Returns:
        pd.DataFrame: DataFrame with columns [user, activity, timestamp, x, y, z]
    """
    print("Loading WISDM data...")
    
    # Read the file line by line and parse
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            # Remove trailing semicolon and whitespace
            line = line.strip().rstrip(';')
            if not line:
                continue
            
            try:
                parts = line.split(',')
                if len(parts) == 6:
                    user = int(parts[0])
                    activity = parts[1]
                    timestamp = int(parts[2])
                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5])
                    data.append([user, activity, timestamp, x, y, z])
            except:
                continue
    
    df = pd.DataFrame(data, columns=['user', 'activity', 'timestamp', 'x', 'y', 'z'])
    print(f"Loaded {len(df)} sensor readings")
    print(f"Activities: {df['activity'].unique()}")
    print(f"Activity distribution:\n{df['activity'].value_counts()}")
    
    return df


# ==================== STEP 2: Create Sliding Windows ====================
def create_sliding_windows(df, window_size=80, step_size=40):
    """
    Create sliding windows from time-series data for each user and activity.
    
    Args:
        df (pd.DataFrame): Input dataframe with sensor readings
        window_size (int): Number of samples per window (default: 80 ≈ 4 seconds at 20Hz)
        step_size (int): Step size for sliding window (default: 40, 50% overlap)
    
    Returns:
        list: List of (window_data, activity_label) tuples
    """
    print(f"\nCreating sliding windows (size={window_size}, step={step_size})...")
    
    windows = []
    
    # Group by user and activity to maintain continuity
    for (user, activity), group in tqdm(df.groupby(['user', 'activity'])):
        # Sort by timestamp
        group = group.sort_values('timestamp')
        
        # Extract sensor values
        sensor_data = group[['x', 'y', 'z']].values
        
        # Create windows
        for i in range(0, len(sensor_data) - window_size + 1, step_size):
            window = sensor_data[i:i + window_size]
            if len(window) == window_size:
                windows.append((window, activity))
    
    print(f"Created {len(windows)} windows")
    return windows


# ==================== STEP 3: Convert Windows to Text ====================
def window_to_text(window):
    """
    Convert a numeric sensor window to text format for T5 input.
    
    Args:
        window (np.ndarray): Array of shape (window_size, 3) with x, y, z values
    
    Returns:
        str: Text representation of the window
    """
    # Strategy: Convert window to text by describing sensor readings
    # Format: "accelerometer data: x1 y1 z1, x2 y2 z2, ..."
    # To keep it manageable, we'll use rounded values
    
    readings = []
    for i, (x, y, z) in enumerate(window):
        # Round to 2 decimal places to reduce token count
        readings.append(f"{x:.2f} {y:.2f} {z:.2f}")
    
    # Join with commas, limit to prevent too long sequences
    # Sample every 4th reading to keep input manageable (20 readings instead of 80)
    sampled_readings = [readings[i] for i in range(0, len(readings), 4)]
    
    text = "classify activity from accelerometer: " + ", ".join(sampled_readings)
    return text


def create_text_dataset(windows):
    """
    Convert windows to text dataset format for T5.
    
    Args:
        windows (list): List of (window_data, activity_label) tuples
    
    Returns:
        pd.DataFrame: DataFrame with 'input_text' and 'target_text' columns
    """
    print("\nConverting windows to text format...")
    
    data = []
    for window, activity in tqdm(windows):
        input_text = window_to_text(window)
        target_text = activity
        data.append({'input_text': input_text, 'target_text': target_text})
    
    df = pd.DataFrame(data)
    print(f"Created {len(df)} text examples")
    print(f"\nSample input: {df['input_text'].iloc[0][:200]}...")
    print(f"Sample target: {df['target_text'].iloc[0]}")
    
    return df


# ==================== STEP 4: Prepare Dataset for T5 ====================
def prepare_dataset(text_df, test_size=0.2, val_size=0.1):
    """
    Split data and create HuggingFace Dataset objects.
    
    Args:
        text_df (pd.DataFrame): DataFrame with input_text and target_text
        test_size (float): Proportion for test set
        val_size (float): Proportion for validation set (from training set)
    
    Returns:
        DatasetDict: Dictionary with train, validation, and test datasets
    """
    print("\nSplitting dataset...")
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        text_df, test_size=test_size, random_state=42, stratify=text_df['target_text']
    )
    
    # Second split: train vs val
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, random_state=42, stratify=train_val_df['target_text']
    )
    
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    # Create HuggingFace datasets
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train_df, preserve_index=False),
        'validation': Dataset.from_pandas(val_df, preserve_index=False),
        'test': Dataset.from_pandas(test_df, preserve_index=False)
    })
    
    return dataset_dict


# ==================== STEP 5: Tokenization ====================
def preprocess_function(examples, tokenizer, max_input_length=512, max_target_length=10):
    """
    Tokenize input and target texts.
    
    Args:
        examples: Batch of examples from dataset
        tokenizer: T5 tokenizer
        max_input_length (int): Max length for input tokens
        max_target_length (int): Max length for target tokens
    
    Returns:
        dict: Tokenized inputs and labels
    """
    # Tokenize inputs
    model_inputs = tokenizer(
        examples['input_text'],
        max_length=max_input_length,
        truncation=True,
        padding=False  # Will pad dynamically in data collator
    )
    
    # Tokenize targets
    labels = tokenizer(
        examples['target_text'],
        max_length=max_target_length,
        truncation=True,
        padding=False
    )
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


def tokenize_dataset(dataset_dict, tokenizer):
    """
    Apply tokenization to all splits in the dataset.
    
    Args:
        dataset_dict (DatasetDict): Dataset with train/val/test splits
        tokenizer: T5 tokenizer
    
    Returns:
        DatasetDict: Tokenized dataset
    """
    print("\nTokenizing dataset...")
    
    tokenized_datasets = dataset_dict.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset_dict['train'].column_names
    )
    
    print("Tokenization complete!")
    return tokenized_datasets


# ==================== STEP 6: Fine-tune FLAN-T5 ====================
def train_model(tokenized_datasets, model, tokenizer, output_dir='./results', num_epochs=3):
    """
    Fine-tune FLAN-T5 model using HuggingFace Trainer.
    
    Args:
        tokenized_datasets (DatasetDict): Tokenized train/val/test data
        model: FLAN-T5 model
        tokenizer: T5 tokenizer
        output_dir (str): Directory to save model checkpoints
        num_epochs (int): Number of training epochs
    
    Returns:
        Seq2SeqTrainer: Trained model trainer
    """
    print("\nSetting up training...")
    
    # Data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none"  # Disable wandb/tensorboard for simplicity
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    print("Starting training...")
    trainer.train()
    
    print("\nTraining complete!")
    return trainer


# ==================== STEP 7: Evaluate Model ====================
def evaluate_model(trainer, tokenized_datasets, tokenizer, model):
    """
    Evaluate the fine-tuned model on test set.
    
    Args:
        trainer (Seq2SeqTrainer): Trained model trainer
        tokenized_datasets (DatasetDict): Dataset with test split
        tokenizer: T5 tokenizer
        model: Fine-tuned model
    
    Returns:
        dict: Evaluation metrics
    """
    print("\nEvaluating on test set...")
    
    # Evaluate
    metrics = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
    print(f"Test Loss: {metrics['eval_loss']:.4f}")
    
    return metrics


def test_predictions(model, tokenizer, test_inputs, num_samples=5):
    """
    Generate predictions for sample inputs.
    
    Args:
        model: Fine-tuned model
        tokenizer: T5 tokenizer
        test_inputs (list): List of test input texts
        num_samples (int): Number of samples to test
    
    Returns:
        list: Generated predictions
    """
    print("\n" + "="*60)
    print("Testing predictions on sample inputs...")
    print("="*60)
    
    model.eval()
    device = next(model.parameters()).device
    
    predictions = []
    for i, input_text in enumerate(test_inputs[:num_samples]):
        # Tokenize input
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(device)
        
        # Generate prediction
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=10,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode prediction
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(prediction)
        
        print(f"\nSample {i+1}:")
        print(f"Input: {input_text[:150]}...")
        print(f"Predicted Activity: {prediction}")
    
    print("="*60)
    return predictions


# ==================== MAIN PIPELINE ====================
def main():
    """
    Main pipeline to fine-tune FLAN-T5 on WISDM dataset.
    """
    print("="*60)
    print("FLAN-T5 Fine-tuning for WISDM Activity Recognition")
    print("="*60)
    
    # Configuration
    WISDM_FILE = "WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"
    MODEL_NAME = "google/flan-t5-small"  # Using small version for faster training
    OUTPUT_DIR = "./flan-t5-wisdm"
    WINDOW_SIZE = 80  # ~4 seconds at 20Hz
    STEP_SIZE = 40    # 50% overlap
    NUM_EPOCHS = 3
    
    # Check if WISDM data exists
    if not os.path.exists(WISDM_FILE):
        print(f"Error: WISDM data file not found at {WISDM_FILE}")
        print("Please ensure WISDM_ar_latest.tar.gz is extracted.")
        return
    
    # Step 1: Load WISDM data
    df = load_wisdm_data(WISDM_FILE)
    
    # Optionally limit data for faster training (remove for full dataset)
    # df = df.groupby('activity').head(10000)
    
    # Step 2: Create sliding windows
    windows = create_sliding_windows(df, window_size=WINDOW_SIZE, step_size=STEP_SIZE)
    
    # Step 3: Convert to text dataset
    text_df = create_text_dataset(windows)
    
    # Limit dataset size for demonstration (remove for full training)
    # text_df = text_df.groupby('target_text').head(500)
    
    # Step 4: Prepare train/val/test splits
    dataset_dict = prepare_dataset(text_df)
    
    # Step 5: Load model and tokenizer
    print(f"\nLoading FLAN-T5 model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Step 6: Tokenize dataset
    tokenized_datasets = tokenize_dataset(dataset_dict, tokenizer)
    
    # Step 7: Train model
    trainer = train_model(
        tokenized_datasets,
        model,
        tokenizer,
        output_dir=OUTPUT_DIR,
        num_epochs=NUM_EPOCHS
    )
    
    # Step 8: Evaluate model
    metrics = evaluate_model(trainer, tokenized_datasets, tokenizer, model)
    
    # Step 9: Test predictions on samples
    test_inputs = [dataset_dict['test'][i]['input_text'] for i in range(min(5, len(dataset_dict['test'])))]
    test_targets = [dataset_dict['test'][i]['target_text'] for i in range(min(5, len(dataset_dict['test'])))]
    
    predictions = test_predictions(model, tokenizer, test_inputs)
    
    # Print accuracy on test samples
    print("\n" + "="*60)
    print("Sample Predictions vs Ground Truth:")
    print("="*60)
    for i, (pred, target) in enumerate(zip(predictions, test_targets)):
        print(f"Sample {i+1}: Predicted='{pred}' | Actual='{target}' | Match={pred==target}")
    
    # Save final model
    print(f"\nSaving final model to {OUTPUT_DIR}/final_model...")
    trainer.save_model(f"{OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"Model saved to: {OUTPUT_DIR}/final_model")
    print("You can now use this model for activity recognition from accelerometer data.")


if __name__ == "__main__":
    main()
