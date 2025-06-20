from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from evaluate import load
import torch

dataset_path = "sidovic/LearningQ-qg"

test_dataset = load_dataset(dataset_path, split="test")

model_name          = "flan-t5-qlora-learningq-qg"
remote_model_path   = f"elmehdiessalehy/{model_name}"

# Load model from huggingface hub
model       = AutoModelForSeq2SeqLM.from_pretrained(remote_model_path)
tokenizer   = AutoTokenizer.from_pretrained(remote_model_path)

# Load meterics
rouge   = load("rouge")
bleu    = load("bleu")
meteor  = load("meteor")

def generate_question(example):
    # Tokenize the input text
    context = "generate question: " + example["context"]
    inputs  = tokenizer(context, return_tensors='pt', padding=True, truncation=True)

    # Generate the questions
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=64)

    # Decode the generated tokens into text
    generated_question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_question

# Generate predictions for the test dataset
generated_questions = [generate_question(example) for example in test_dataset]

# Get the refence questions
references = test_dataset['question']  # Ground truth answers in the dataset

# Compute metrics
rouge_score     = rouge.compute(predictions=generated_questions, references=references)
bleu_score      = bleu.compute(predictions=generated_questions, references=references)
meteor_score    = meteor.compute(predictions=generated_questions, references=references)

# Display results
print(f"ROUGE:  {rouge_score}")
print(f"BLEU:   {bleu_score}")
print(f"METEOR: {meteor_score}")
