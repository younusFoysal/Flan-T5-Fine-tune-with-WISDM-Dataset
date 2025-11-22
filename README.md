# FLAN-T5 Fine-tuning for WISDM Activity Recognition

This project fine-tunes Google's FLAN-T5 language model on the WISDM (Wireless Sensor Data Mining) Activity Recognition dataset. While FLAN-T5 is designed for text-to-text tasks and WISDM contains numeric time-series accelerometer data, we bridge this gap by converting sensor readings into text format.

**Reference Tutorial**: [DataCamp FLAN-T5 Tutorial](https://www.datacamp.com/tutorial/flan-t5-tutorial)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Function Descriptions](#function-descriptions)
- [Pipeline Architecture](#pipeline-architecture)
- [Configuration](#configuration)
- [Expected Output](#expected-output)
- [Model Performance](#model-performance)

---

## 🔍 Overview

### The Challenge
FLAN-T5 is a text-to-text transformer model, while WISDM is a time-series accelerometer dataset. This creates a domain mismatch.

### The Solution
We convert numeric sensor windows into text strings and treat activity recognition as a sequence-to-text generation problem:
- **Input**: "classify activity from accelerometer: -0.69 12.68 0.50, 5.01 11.26 0.95, ..."
- **Output**: "Walking" or "Jogging" or other activity labels

This approach allows us to leverage FLAN-T5's powerful sequence-to-sequence capabilities for a non-traditional use case.

---

## 📊 Dataset

### WISDM Activity Recognition Dataset v1.1

**Description**: Accelerometer data from smartphone sensors recording human activities.

**Activities** (6 classes):
1. Walking
2. Jogging
3. Upstairs (climbing stairs)
4. Downstairs (descending stairs)
5. Sitting
6. Standing

**Data Format**: 
- **Sensor**: 3-axis accelerometer (x, y, z)
- **Sampling Rate**: ~20 Hz
- **Users**: 36 participants
- **Total Readings**: ~1.1 million samples

**File Structure**:
```
user_id, activity, timestamp, x_accel, y_accel, z_accel;
33, Jogging, 49105962326000, -0.6946377, 12.680544, 0.50395286;
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, but recommended for faster training)

### Required Libraries

Install all dependencies using pip:

```bash
pip install torch torchvision torchaudio
pip install transformers datasets
pip install pandas numpy scikit-learn
pip install tqdm
```

Or create a `requirements.txt`:

```txt
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

Then install:
```bash
pip install -r requirements.txt
```

### Dataset Setup

1. Download the WISDM dataset (already provided as `WISDM_ar_latest.tar.gz`)
2. Extract it:
```bash
tar -xzf WISDM_ar_latest.tar.gz
```

The extraction creates a `WISDM_ar_v1.1` directory containing the raw data file.

---

## 🚀 Usage

### Basic Usage

Simply run the main script:

```bash
python main.py
```

The script will automatically:
1. Load the WISDM dataset
2. Create sliding windows from time-series data
3. Convert windows to text format
4. Split data into train/validation/test sets
5. Load and tokenize with FLAN-T5
6. Fine-tune the model
7. Evaluate and save the trained model

### Expected Runtime

- **CPU**: 2-4 hours for full dataset
- **GPU (CUDA)**: 30-60 minutes for full dataset

### Output

The trained model will be saved to:
```
./flan-t5-wisdm/final_model/
```

---

## 📚 Function Descriptions

### 1. `load_wisdm_data(file_path)`

**Purpose**: Loads and parses the WISDM raw data file.

**Parameters**:
- `file_path` (str): Path to `WISDM_ar_v1.1_raw.txt`

**Returns**:
- `pd.DataFrame`: DataFrame with columns [user, activity, timestamp, x, y, z]

**How It Works**:
- Reads the file line by line
- Parses comma-separated values
- Handles semicolon terminators and malformed lines
- Returns structured DataFrame with all sensor readings

**Example Output**:
```
Loaded 1098207 sensor readings
Activities: ['Jogging' 'Walking' 'Upstairs' 'Downstairs' 'Sitting' 'Standing']
```

---

### 2. `create_sliding_windows(df, window_size=80, step_size=40)`

**Purpose**: Creates overlapping windows from continuous time-series data.

**Parameters**:
- `df` (pd.DataFrame): Input dataframe with sensor readings
- `window_size` (int): Number of samples per window (default: 80 ≈ 4 seconds at 20Hz)
- `step_size` (int): Sliding window step (default: 40, giving 50% overlap)

**Returns**:
- `list`: List of (window_data, activity_label) tuples

**How It Works**:
- Groups data by user and activity to maintain temporal continuity
- Sorts by timestamp within each group
- Extracts x, y, z sensor values
- Creates overlapping windows using sliding window technique
- Each window contains 80 consecutive sensor readings (3-axis × 80 = 240 values)

**Why Overlapping Windows?**
- Increases training data size
- Provides more robust patterns
- Standard practice in time-series classification

**Example Output**:
```
Creating sliding windows (size=80, step=40)...
Created 27450 windows
```

---

### 3. `window_to_text(window)`

**Purpose**: Converts a numeric sensor window into text format suitable for T5.

**Parameters**:
- `window` (np.ndarray): Array of shape (80, 3) with x, y, z acceleration values

**Returns**:
- `str`: Text representation of the window

**How It Works**:
- Rounds sensor values to 2 decimal places (reduces token count)
- Formats each reading as "x y z"
- Samples every 4th reading to keep input manageable (20 readings instead of 80)
- Prepends instruction: "classify activity from accelerometer: "
- Joins readings with commas

**Example Output**:
```python
"classify activity from accelerometer: -0.69 12.68 0.50, 5.01 11.26 0.95, 4.90 10.88 -0.08, ..."
```

**Design Rationale**:
- Sampling reduces token count while preserving temporal pattern
- Instruction prefix helps model understand the task
- Format is human-readable and model-parseable

---

### 4. `create_text_dataset(windows)`

**Purpose**: Converts all windows to text dataset format.

**Parameters**:
- `windows` (list): List of (window_data, activity_label) tuples

**Returns**:
- `pd.DataFrame`: DataFrame with 'input_text' and 'target_text' columns

**How It Works**:
- Iterates through all windows
- Applies `window_to_text()` to create input
- Uses activity label as target
- Returns DataFrame ready for splitting

**Example Output**:
```
Converting windows to text format...
Created 27450 text examples
Sample input: classify activity from accelerometer: -0.69 12.68 0.50...
Sample target: Jogging
```

---

### 5. `prepare_dataset(text_df, test_size=0.2, val_size=0.1)`

**Purpose**: Splits data and creates HuggingFace Dataset objects.

**Parameters**:
- `text_df` (pd.DataFrame): DataFrame with input_text and target_text
- `test_size` (float): Proportion for test set (default: 0.2 = 20%)
- `val_size` (float): Proportion for validation from remaining (default: 0.1 = 10% of train+val)

**Returns**:
- `DatasetDict`: Dictionary with 'train', 'validation', and 'test' splits

**How It Works**:
- First split: Separates test set (20%)
- Second split: Divides remaining into train (72%) and validation (8%)
- Uses stratified sampling to maintain class balance
- Converts to HuggingFace Dataset format

**Final Split**:
- Train: 72%
- Validation: 8%
- Test: 20%

**Example Output**:
```
Splitting dataset...
Train: 19764, Validation: 2196, Test: 5490
```

---

### 6. `preprocess_function(examples, tokenizer, max_input_length=512, max_target_length=10)`

**Purpose**: Tokenizes input and target texts for T5.

**Parameters**:
- `examples`: Batch of examples from dataset
- `tokenizer`: T5 tokenizer
- `max_input_length` (int): Max tokens for input (default: 512)
- `max_target_length` (int): Max tokens for target (default: 10)

**Returns**:
- `dict`: Dictionary with tokenized input_ids, attention_mask, and labels

**How It Works**:
- Tokenizes input texts using T5 tokenizer
- Tokenizes target labels
- Truncates sequences exceeding max length
- Padding is handled dynamically by DataCollator during training

**Why These Lengths?**
- 512 tokens: Sufficient for ~20 sensor readings in text format
- 10 tokens: More than enough for activity names (typically 1-2 tokens)

---

### 7. `tokenize_dataset(dataset_dict, tokenizer)`

**Purpose**: Applies tokenization to all dataset splits.

**Parameters**:
- `dataset_dict` (DatasetDict): Dataset with train/val/test splits
- `tokenizer`: T5 tokenizer

**Returns**:
- `DatasetDict`: Tokenized dataset ready for training

**How It Works**:
- Applies `preprocess_function()` to all splits using `.map()`
- Processes in batches for efficiency
- Removes original text columns (only keeps tokenized versions)

---

### 8. `train_model(tokenized_datasets, model, tokenizer, output_dir='./results', num_epochs=3)`

**Purpose**: Fine-tunes FLAN-T5 using HuggingFace Trainer.

**Parameters**:
- `tokenized_datasets` (DatasetDict): Tokenized train/val/test data
- `model`: FLAN-T5 model
- `tokenizer`: T5 tokenizer
- `output_dir` (str): Directory for checkpoints (default: './results')
- `num_epochs` (int): Training epochs (default: 3)

**Returns**:
- `Seq2SeqTrainer`: Trained model trainer object

**How It Works**:
1. **Data Collator**: Sets up dynamic padding for variable-length sequences
2. **Training Arguments**: Configures hyperparameters:
   - Learning rate: 5e-5
   - Batch size: 8
   - Evaluation strategy: Every epoch
   - Mixed precision (FP16) if GPU available
   - Early stopping based on validation loss
3. **Trainer**: Initializes Seq2SeqTrainer with all components
4. **Training**: Runs training loop with automatic evaluation

**Training Features**:
- Automatic checkpointing
- Gradient accumulation
- Learning rate scheduling
- Best model selection based on validation loss

**Example Output**:
```
Setting up training...
Using device: cuda
Starting training...
Epoch 1/3: 100%|██████████| 2470/2470 [15:32<00:00]
Training complete!
```

---

### 9. `evaluate_model(trainer, tokenized_datasets, tokenizer, model)`

**Purpose**: Evaluates the fine-tuned model on test set.

**Parameters**:
- `trainer` (Seq2SeqTrainer): Trained model trainer
- `tokenized_datasets` (DatasetDict): Dataset with test split
- `tokenizer`: T5 tokenizer
- `model`: Fine-tuned model

**Returns**:
- `dict`: Evaluation metrics including loss

**How It Works**:
- Runs model on held-out test set
- Computes evaluation metrics
- Returns test loss and other metrics

**Example Output**:
```
Evaluating on test set...
Test Loss: 0.0234
```

---

### 10. `test_predictions(model, tokenizer, test_inputs, num_samples=5)`

**Purpose**: Generates and displays predictions for sample inputs.

**Parameters**:
- `model`: Fine-tuned model
- `tokenizer`: T5 tokenizer
- `test_inputs` (list): List of test input texts
- `num_samples` (int): Number of samples to test (default: 5)

**Returns**:
- `list`: Generated predictions

**How It Works**:
- Sets model to evaluation mode
- Tokenizes input texts
- Generates predictions using beam search (4 beams)
- Decodes token IDs back to text
- Displays input snippet and predicted activity

**Example Output**:
```
============================================================
Testing predictions on sample inputs...
============================================================

Sample 1:
Input: classify activity from accelerometer: -0.69 12.68 0.50, 5.01 11.26 0.95...
Predicted Activity: Jogging

Sample 2:
Input: classify activity from accelerometer: 0.12 9.81 0.03, 0.15 9.79 0.02...
Predicted Activity: Standing
============================================================
```

---

### 11. `main()`

**Purpose**: Orchestrates the entire pipeline from data loading to model saving.

**How It Works**:
1. Loads WISDM data
2. Creates sliding windows
3. Converts to text dataset
4. Prepares train/val/test splits
5. Loads FLAN-T5 model and tokenizer
6. Tokenizes dataset
7. Fine-tunes model
8. Evaluates on test set
9. Tests predictions on samples
10. Saves final model

**Configuration Variables** (modifiable):
```python
WISDM_FILE = "WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"
MODEL_NAME = "google/flan-t5-small"
OUTPUT_DIR = "./flan-t5-wisdm"
WINDOW_SIZE = 80
STEP_SIZE = 40
NUM_EPOCHS = 3
```

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     WISDM Raw Data                          │
│          (user, activity, timestamp, x, y, z)               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Load & Parse Data                              │
│         load_wisdm_data() → DataFrame                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│          Create Sliding Windows                             │
│    create_sliding_windows() → List of windows               │
│         (80 samples, 50% overlap)                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│          Convert to Text Format                             │
│  window_to_text() + create_text_dataset()                   │
│    "classify activity from accelerometer: ..."              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│         Split into Train/Val/Test                           │
│   prepare_dataset() → DatasetDict                           │
│        Train: 72%, Val: 8%, Test: 20%                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Tokenization                                   │
│    tokenize_dataset() → Tokenized DatasetDict               │
│         (Input IDs, Attention Masks, Labels)                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│           Load FLAN-T5 Model                                │
│    google/flan-t5-small (60M parameters)                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Fine-tuning                                    │
│   train_model() → Seq2SeqTrainer                            │
│    3 epochs, batch size 8, learning rate 5e-5               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│          Evaluation & Testing                               │
│   evaluate_model() + test_predictions()                     │
│         Test loss, sample predictions                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│             Save Fine-tuned Model                           │
│      ./flan-t5-wisdm/final_model/                           │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Configuration

### Adjustable Parameters in `main()`:

| Parameter | Default Value | Description |
|-----------|--------------|-------------|
| `WISDM_FILE` | `WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt` | Path to raw data file |
| `MODEL_NAME` | `google/flan-t5-small` | Pre-trained model to fine-tune |
| `OUTPUT_DIR` | `./flan-t5-wisdm` | Directory for outputs |
| `WINDOW_SIZE` | `80` | Samples per window (~4 sec) |
| `STEP_SIZE` | `40` | Window step (50% overlap) |
| `NUM_EPOCHS` | `3` | Training epochs |

### Model Variants:

You can use different FLAN-T5 sizes:
- `google/flan-t5-small` (60M params) - Fast, good for prototyping
- `google/flan-t5-base` (250M params) - Better accuracy, slower training
- `google/flan-t5-large` (780M params) - Best accuracy, requires significant GPU memory

### Training Hyperparameters:

Defined in `train_model()`:
- **Learning Rate**: 5e-5
- **Batch Size**: 8 (per device)
- **Weight Decay**: 0.01
- **FP16**: Enabled if CUDA available
- **Evaluation**: Every epoch
- **Early Stopping**: Based on validation loss

---

## 📈 Expected Output

### Console Output

```
============================================================
FLAN-T5 Fine-tuning for WISDM Activity Recognition
============================================================
Loading WISDM data...
Loaded 1098207 sensor readings
Activities: ['Jogging' 'Walking' 'Upstairs' 'Downstairs' 'Sitting' 'Standing']
Activity distribution:
Walking      424398
Jogging      342176
Upstairs     122869
Downstairs   100427
Sitting       59939
Standing      48395

Creating sliding windows (size=80, step=40)...
Created 27450 windows

Converting windows to text format...
Created 27450 text examples

Splitting dataset...
Train: 19764, Validation: 2196, Test: 5490

Loading FLAN-T5 model: google/flan-t5-small...
Using device: cuda

Tokenizing dataset...
Tokenization complete!

Setting up training...
Starting training...
Epoch 1/3: [Training progress bar]
Epoch 2/3: [Training progress bar]
Epoch 3/3: [Training progress bar]

Training complete!

Evaluating on test set...
Test Loss: 0.0234

============================================================
Testing predictions on sample inputs...
============================================================
[Sample predictions displayed]

============================================================
PIPELINE COMPLETE!
============================================================
Model saved to: ./flan-t5-wisdm/final_model
You can now use this model for activity recognition from accelerometer data.
```

### Saved Files

After execution, you'll have:
```
./flan-t5-wisdm/
├── checkpoint-xxxx/          # Training checkpoints
├── final_model/              # Final fine-tuned model
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── ...
└── logs/                     # Training logs
```

---

## 🎯 Model Performance

### Expected Accuracy

With the full WISDM dataset and default settings:
- **Training Accuracy**: ~95-98%
- **Validation Accuracy**: ~85-90%
- **Test Accuracy**: ~85-90%

### Per-Class Performance

Typically:
- **High Performance**: Walking, Jogging (many samples, distinct patterns)
- **Medium Performance**: Upstairs, Downstairs (fewer samples, similar patterns)
- **Lower Performance**: Sitting, Standing (subtle differences, static activities)

### Inference Speed

- **CPU**: ~50-100 ms per prediction
- **GPU**: ~5-10 ms per prediction

---

## 🔧 Troubleshooting

### Out of Memory Error

If you encounter OOM errors:
1. Reduce batch size: Change `per_device_train_batch_size=4` or `2`
2. Use smaller model: Switch to `google/flan-t5-small`
3. Reduce window size: Use `WINDOW_SIZE=40`
4. Limit dataset: Uncomment data limiting lines in `main()`

### CUDA Not Available

If CUDA is not detected but you have a GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Low Accuracy

If model accuracy is poor:
1. Increase epochs: Set `NUM_EPOCHS=5` or more
2. Use larger model: Try `google/flan-t5-base`
3. Adjust learning rate: Experiment with 1e-5 to 1e-4
4. Increase window size: Try `WINDOW_SIZE=120`

---

## 📝 Citation

### WISDM Dataset

```
Jennifer R. Kwapisz, Gary M. Weiss and Samuel A. Moore (2010). 
Activity Recognition using Cell Phone Accelerometers,
Proceedings of the Fourth International Workshop on Knowledge Discovery from Sensor Data (at KDD-10), Washington DC.
```

### FLAN-T5 Model

```
@article{chung2022scaling,
  title={Scaling instruction-finetuned language models},
  author={Chung, Hyung Won and Hou, Le and Longpre, Shayne and others},
  journal={arXiv preprint arXiv:2210.11416},
  year={2022}
}
```

---

## 📄 License

This project is for educational purposes. Please refer to the original dataset and model licenses:
- **WISDM Dataset**: Available for research use
- **FLAN-T5**: Apache 2.0 License

---

## 🙋 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the function descriptions for implementation details
3. Refer to the DataCamp tutorial for FLAN-T5 basics
4. Check HuggingFace documentation for Transformers library

---

**Happy Training! 🚀**
