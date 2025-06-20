from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

merged_model_dir = "./merged_model"

# Load merged model
model       = AutoModelForSeq2SeqLM.from_pretrained(merged_model_dir)
tokenizer   = AutoTokenizer.from_pretrained(merged_model_dir)

# Move model to device (CPU or GPU)
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define inference function
def generate_question(context):
    task_prefix = "generate question: "  # or any task prefix you trained with
    input_text  = task_prefix + context

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

    outputs = model.generate(**inputs, max_length=64, num_beams=1, temperature=0.7)

    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

# Example
context = """
The Declaration of Independence, formally The unanimous Declaration of the thirteen united States of America in the original printing, is the founding document of the United States.
On July 4, 1776, it was adopted unanimously by the Second Continental Congress, who convened at Pennsylvania State House, later renamed Independence Hall, in the colonial capital of Philadelphia.
These delegates became known as the nation's Founding Fathers.
The Declaration explains why the Thirteen Colonies regarded themselves as independent sovereign states no longer subject to British colonial rule, and has become one of the most circulated, reprinted, and influential documents in history.
"""

print(generate_question(context))