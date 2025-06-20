from peft import PeftModel

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "google/flan-t5-base"

# Load base model
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Assuming PEFT directory only has adapter files
peft_model_path = "./peft_only"

# Load PEFT model from the base model and the trained adapters
model = PeftModel.from_pretrained(base_model, peft_model_path)

# Merge the adapters into the model
model = model.merge_and_unload()

merged_model_dir = "./merged_model"

# Save merged model
model.save_pretrained(merged_model_dir)
tokenizer.save_pretrained(merged_model_dir)
