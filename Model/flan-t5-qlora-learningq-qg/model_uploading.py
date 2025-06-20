from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_dir = "/merged_model"

# Load from local folder
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

model_name = "flan-t5-qlora-learningq-qg"

# Push to your Hugging Face Hub
model.push_to_hub(f"elmehdiessalehy/{model_name}")
tokenizer.push_to_hub(f"elmehdiessalehy/{model_name}")
