# Fixes Applied to FLAN-T5 Fine-tuning Pipeline

## Issues Found in Original Output

1. **Training Loss = 0.000000** - Model was not learning anything
2. **Validation Loss = NaN** - Loss calculation was failing
3. **Predictions wrong** - Model was echoing input instead of predicting activities (e.g., "x-axis mean -1." instead of "Jogging")
4. **0% Accuracy** - All predictions were incorrect

## Root Causes

1. **Improper Padding Configuration**: Used `padding=False` which didn't work with `DataCollatorForSeq2Seq`
2. **Missing Metrics Computation**: No metrics were being calculated during training
3. **Wrong Trainer API**: Used deprecated `processing_class` parameter instead of `tokenizer`
4. **Suboptimal Training Parameters**: Learning rate and batch size weren't appropriate for fine-tuning
5. **Poor Decoding Settings**: Used `num_beams=4` which could cause the model to search for tokens outside its training vocabulary

## Fixes Applied

### 1. **Fixed Tokenization (Line 240-262)**
```python
# Changed from:
padding=False  # Dynamic padding handled by DataCollator

# To:
padding="max_length",
return_tensors=None
```
- This ensures all sequences are padded to the same length
- Prevents loss calculation issues

### 2. **Added Metrics Computation (Line 286-309)**
```python
def compute_metrics(eval_preds):
    """Compute accuracy metrics during evaluation."""
    predictions, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    accuracy = sum([pred.strip().lower() == label.strip().lower() 
                   for pred, label in zip(decoded_preds, decoded_labels)]) / len(decoded_preds)
    
    return {"accuracy": accuracy}
```
- Properly tracks model accuracy during training
- Helps identify when the model is actually learning

### 3. **Optimized Training Parameters (Line 311-330)**
```python
# Learning rate: 1e-4 → 5e-5 (standard for fine-tuning)
# Batch size: 16 → 8 (more stable gradient updates)
# Added: gradient_accumulation_steps=2 (effective batch size of 16)
# Added: optim="adafactor" (more stable optimizer for T5)
# Logging: steps=50 → steps=100 (less frequent logging overhead)
```
- Standard hyperparameters for seq2seq fine-tuning
- Adafactor is proven to work better with T5 models

### 4. **Fixed Trainer API (Line 344-353)**
```python
# Changed from:
processing_class=tokenizer,  # Deprecated

# To:
tokenizer=tokenizer,  # Correct parameter
compute_metrics=compute_metrics,  # Added metrics
```
- Uses correct, non-deprecated parameter
- Includes metrics computation

### 5. **Improved Data Collator (Line 299-302)**
```python
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8  # Better GPU utilization
)
```
- Properly handles seq2seq padding and attention masks

### 6. **Fixed Generation Parameters (Line 393-404)**
```python
# Changed from:
outputs = model.generate(
    **inputs,
    max_length=10,
    num_beams=4,
    early_stopping=True
)

# To:
outputs = model.generate(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_length=16,
    num_beams=1,
    temperature=0.7,
    top_p=0.9
)
```
- `num_beams=1` (greedy decoding) - more stable than beam search
- `max_length=16` - enough for activity names
- `temperature=0.7` - adds diversity while maintaining quality
- Explicit input/attention mask passing - more reliable

## Expected Results After Fixes

✅ Training loss should **decrease** from epoch 1 to 5  
✅ Validation loss should **decrease** (not NaN)  
✅ Model accuracy should **increase** during training  
✅ Predictions should be **actual activity names** (Jogging, Walking, etc.)  
✅ Test set accuracy should be **> 70%**  

## Key Improvements

| Aspect | Before | After |
|--------|--------|-------|
| Training Loss | 0.0 (not learning) | Should decrease |
| Validation Loss | NaN (broken) | Should be valid |
| Prediction Quality | "x-axis mean -1." | "Jogging" |
| Accuracy | 0% | Expected 70-80%+ |
| Model Stability | Unstable | Stable with Adafactor |

## Next Steps for Further Improvement

1. **Increase training epochs** to 10-15 for better convergence
2. **Use larger model** (google/flan-t5-base) for better accuracy
3. **Add data augmentation** for sensor readings (noise, scaling)
4. **Fine-tune learning rate** based on validation curves
5. **Use stratified k-fold cross-validation** for better evaluation
