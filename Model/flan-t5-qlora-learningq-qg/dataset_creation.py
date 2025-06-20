from datasets import load_dataset
from transformers import T5Tokenizer

model_checkpoint    = "google/flan-t5-base"
dataset_path        = "sidovic/LearningQ-qg"

train_dataset_dir   = "./Datasets/train_dataset"
eval_dataset_dir    = "./Datasets/eval_dataset"
test_dataset_dir    = "./Datasets/test_dataset"

tokenized_train_dataset_dir = "./Datasets/tokenized_train_dataset"
tokenized_eval_dataset_dir  = "./Datasets/tokenized_eval_dataset"
tokenized_test_dataset_dir  = "./Datasets/tokenized_test_dataset"

learningq_dataset = load_dataset(dataset_path)

train_dataset   = learningq_dataset['train']
eval_dataset    = learningq_dataset['validation']
test_dataset    = learningq_dataset['test']

train_dataset.save_to_disk(train_dataset_dir)
eval_dataset.save_to_disk(eval_dataset_dir)
test_dataset.save_to_disk(test_dataset_dir)

tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)

max_input_length = 512
max_target_length = 64

def preprocess_function(examples):
    inputs  = ["generate question: " + c for c in examples['context']] # add prefix
    targets = examples['question']

    model_inputs    = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    labels          = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset  = eval_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset  = test_dataset.map(preprocess_function, batched=True)

tokenized_train_dataset.save_to_disk(tokenized_train_dataset_dir)
tokenized_eval_dataset.save_to_disk(tokenized_eval_dataset_dir)
tokenized_test_dataset.save_to_disk(tokenized_test_dataset_dir)