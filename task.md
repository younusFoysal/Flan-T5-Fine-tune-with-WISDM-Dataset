# Fine-tune FLAN-T5 on the WISDM Activity Recognition dataset

This is NOT a normal use-case, because FLAN-T5 is a text model, while WISDM is numeric time-series data.

However, we can make it work by converting each sequence window into text and treating the problem as sequence-to-label text generation, following the DataCamp tutorial approach.

**Reference Tutorial**: https://www.datacamp.com/tutorial/flan-t5-tutorial

---

## Implementation Steps

The implementation in `main.py` follows these steps:

### 1. ✅ Data Loading
- **Function**: `load_wisdm_data()`
- Loads WISDM raw accelerometer data from text file
- Parses user ID, activity label, timestamp, and x/y/z accelerometer values
- Returns pandas DataFrame with all sensor readings

### 2. ✅ Create Sliding Windows
- **Function**: `create_sliding_windows()`
- Groups data by user and activity to maintain temporal continuity
- Creates overlapping windows of 80 samples (~4 seconds at 20Hz)
- Uses 50% overlap (step size = 40) to generate more training examples
- Returns list of (window_data, activity_label) tuples

### 3. ✅ Convert Windows to Text
- **Function**: `window_to_text()` and `create_text_dataset()`
- Converts numeric sensor windows into text strings
- Format: "classify activity from accelerometer: x1 y1 z1, x2 y2 z2, ..."
- Samples every 4th reading to keep input length manageable
- Creates input-target pairs for T5 model

### 4. ✅ Prepare Dataset
- **Function**: `prepare_dataset()`
- Splits data into train (72%), validation (8%), test (20%)
- Uses stratified splitting to maintain class balance
- Creates HuggingFace Dataset objects for compatibility with Transformers

### 5. ✅ Tokenization
- **Functions**: `preprocess_function()` and `tokenize_dataset()`
- Loads FLAN-T5 tokenizer from HuggingFace
- Tokenizes input texts (max 512 tokens) and target labels (max 10 tokens)
- Applies tokenization to all dataset splits

### 6. ✅ Fine-tune FLAN-T5
- **Function**: `train_model()`
- Uses google/flan-t5-small model (faster training)
- Configures Seq2SeqTrainer with appropriate hyperparameters
- Training settings: 3 epochs, learning rate 5e-5, batch size 8
- Saves checkpoints and uses early stopping based on validation loss

### 7. ✅ Evaluate & Test
- **Functions**: `evaluate_model()` and `test_predictions()`
- Evaluates model on held-out test set
- Generates predictions for sample inputs
- Compares predictions with ground truth labels
- Saves final fine-tuned model for future use

---

## Configuration

The main pipeline uses these default settings (configurable in `main()` function):

- **Model**: `google/flan-t5-small`
- **Window Size**: 80 samples (~4 seconds)
- **Step Size**: 40 samples (50% overlap)
- **Epochs**: 3
- **Batch Size**: 8
- **Learning Rate**: 5e-5
- **Output Directory**: `./flan-t5-wisdm/`

---

## Activities

The WISDM dataset includes 6 activity classes:
1. Walking
2. Jogging
3. Upstairs
4. Downstairs
5. Sitting
6. Standing