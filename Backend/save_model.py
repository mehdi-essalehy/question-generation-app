from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "flan-t5-qlora-learningq-qg"
remote_model_path = f"elmehdiessalehy/{model_name}"

# Load fine-tuned model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

local_model_path = "./model"

# Save the fine-tuned model locally
model.save_pretrained(local_model_path)
tokenizer.save_pretrained(local_model_path)