import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    BitsAndBytesConfig
)
import evaluate
from peft import LoraConfig, get_peft_model, TaskType

model_name = "google/flan-t5-base"

# init the BitsAndBytesConfig for model quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load quantized model
tokenizer   = AutoTokenizer.from_pretrained(model_name)
model       = AutoModelForSeq2SeqLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")

tokenized_train_dataset_dir = "./Datasets/tokenized_train_dataset"
tokenized_eval_dataset_dir  = "./Datasets/tokenized_eval_dataset"

# Load tokenized train and validation datasets from disk
train_dataset   = load_from_disk(tokenized_train_dataset_dir)
eval_dataset    = load_from_disk(tokenized_eval_dataset_dir)

# Init LoraConfig for PEFT
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05
)
# Load PEFT model using base mode and LoraConfig
model = get_peft_model(model, lora_config)
# Show trainable parameters to make sure only adapter parameters are going to be trained
model.print_trainable_parameters()

# Load ROUGE metric
rouge = evaluate.load("rouge")

# Postprocessing for removing whitespaces
def postprocess_text(preds, labels):
    preds   = [pred.strip() for pred in preds]
    labels  = [label.strip() for label in labels]
    return preds, labels

# Use ROUGE to compute metrics during training and validation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    preds, refs = postprocess_text(predictions, labels)

    result = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    return {k: round(v * 100, 4) for k, v in result.items()}


training_checkpoints_dir    = "./training_checkpoints"
logs_dir                    = "./logs"

batch_size = 32  # safe default for A100 GPU

training_args = Seq2SeqTrainingArguments(
    output_dir=training_checkpoints_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=1,
    learning_rate=5e-4,
    weight_decay=0.01,
    num_train_epochs=4,
    warmup_steps=1000,
    logging_dir=logs_dir,
    logging_strategy="steps",
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=2000,
    save_strategy="steps",
    save_steps=2000,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    predict_with_generate=True,
    generation_max_length=64,
    generation_num_beams=1,
    fp16=True,
    seed=42
)


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train model
trainer.train()

model_dir = "./trained_model"

# Save model and tokenizer after trining
trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)