from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
# Initialize FastAPI app
app = FastAPI()

print("Loading model...")
# Load fine-tuned model and tokenizer
model_name = "./model"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model loaded successfully!")

# Ensure the model is in evaluation mode
model.eval()

# Input data format
class TextRequest(BaseModel):
    paragraphs: list

# Generate questions function
def generate_questions(paragraphs):
    questions = []
    for paragraph in paragraphs:
        try:
            # Preprocess the input text
            input_text = f"generate questions: {paragraph}"
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

            # Generate questions using the model
            with torch.no_grad():
                outputs = model.generate(inputs['input_ids'], max_length=128, num_return_sequences=1)

            # Decode the generated question(s)
            question = tokenizer.decode(outputs[0], skip_special_tokens=True)

            questions.append(question)
        except Exception as e:
            questions.append(f"Error generating question: {str(e)}")
    return questions

# API endpoint to receive text and return generated questions
@app.post("/generate_questions/")
async def generate_questions_endpoint(request: TextRequest):
    try:
        paragraphs = request.paragraphs
        if not paragraphs:
            raise HTTPException(status_code=400, detail="No paragraphs provided.")
        
        # Generate questions
        questions = generate_questions(paragraphs)
        return {"questions": questions}

    except HTTPException as http_err:
        # HTTPException for custom errors like missing paragraphs
        raise http_err
    except Exception as e:
        # Catch all other exceptions and return a generic error message
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
