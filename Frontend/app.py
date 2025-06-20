from csv import Error
import streamlit as st

import os
import requests

from text_extraction import extract_from_file, FileType
from text_preprocessing import segment_for_question_generation

from random import sample, choices, choice, shuffle

from sentence_transformers import SentenceTransformer
import spacy

# Get API URL from the environment variable API_URL
api_url = os.getenv('API_URL', "http://127.0.0.1:8000/generate_questions/")

# Page title
st.title("Question Generation App")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=None)

# Number of questions input
number_of_questions = st.number_input("Number of questions to generate: ", min_value=0, max_value=20, value=1, step=1)

# Topic input
topic = st.text_input("Topic: ")

# Check if SentenceTransformer model  is already loaded, and reload it if not
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    if "embedding_model" in st.session_state:
        st.success("SentenceTransformer model loaded successfully!")
    else:
        st.error("Failed to load SentenceTransformer model!")

# Check if Spacy model  is already loaded, and reload it if not
if "nlp" not in st.session_state:
    st.session_state.nlp = spacy.load("en_core_web_sm")
    if "nlp" in st.session_state:
        st.success("Spacy model loaded successfully!")
    else:
        st.error("Failed to load Spacy model!")

# Submit button
if st.button("Submit"):
    # Check if a file was uploaded sucessfully
    if uploaded_file:
        # Confirm successful file upload
        st.success("File Submitted successfully!")
        # Check if user entered a topic, and set it to the uploaded file's name if not
        if topic == "":
            topic = uploaded_file.name
        # Get filetype
        file_type = None
        try:
            if uploaded_file.type == "application/pdf":
                file_type = FileType.PDF
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                file_type = FileType.DOCX
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
                file_type = FileType.PPTX
            else:
                raise ValueError("Invalid filetype!")
        except Exception as e:
            print(f"Error checking filetype: {str(e)}")
        
        if file_type:
            body_text = ""
            paragraphs = []
            # Extract text from file
            try:
                body_text = extract_from_file(uploaded_file, file_type)
            except Exception as e:
                print(f"Error extracting text from file: {str(e)}")
            if len(body_text) > 0:
                # Segment the text into paragraphs and filter them
                try:
                    paragraphs = segment_for_question_generation(
                        raw_text=body_text,
                        embedding_model=st.session_state.embedding_model,
                        nlp=st.session_state.nlp,
                        topic=topic,
                        window_size=2,
                        overlap=1,
                        semantic_threshold=0.2,
                        entity_filtering=True
                    )
                except Exception as e:
                    print(f"Error segmenting and filtering text: {str(e)}")
                # Get number of paragraphs
                number_of_paragraphs = len(paragraphs)
                if number_of_paragraphs > 0:
                    # Get a number of paragraphs equal to the number of questions requested by the user (with repetition if necessary)
                    try:
                        data = {"paragraphs": []} # initialize data to be used for question generation
                        number_of_paragraphs_left = number_of_questions # keep track of the number that still need to be added to data
                        while number_of_paragraphs_left > number_of_paragraphs:
                            data["paragraphs"].extend(shuffle(paragraphs)) # as long as the number of paragraphs to be add is still higher the number of paragraphs we have, just add a shuffled version of all the paragraphs
                            number_of_paragraphs_left -= number_of_paragraphs # update number_of_paragraphs_left
                        data["paragraphs"].extend(sample(paragraphs, number_of_paragraphs_left)) # finally sample a subset of the paragraphs available to compllete the data
                    except Exception as e:
                        print(f"Error sampling paragraphs: {str(e)}")
                    # Send request to API to generate questions, and parse them to present them to the user
                    try:
                        response = requests.post(api_url, json=data)
                        response_content = response.json()
                        questions = response_content["questions"]
                        # Format the input output pairs
                        input_output = ["Context " + str(i+1) + ". " + paragraph + "\n" + "Question " + str(i+1) + ". " + question for i, paragraph, question in zip(range(number_of_questions), data["paragraphs"],questions)]
                        st.session_state.editable_output = "\n\n".join(input_output) # add the formatted pairs to session state
                    except Exception as e:
                        print(f"Error generating questions: {str(e)}")
                else:
                    st.warning("No paragraphs left after filtering.")
            else:
                st.warning("Unable to extract text from file.")
        else:
            st.warning("Invalid file type.")
    else:
        st.warning("Please upload a file before submitting.")

# Display the formatted output if it was successfully added to session state
if "editable_output" in st.session_state:
    st.subheader("Questions:")
    updated_text = st.text_area("Editable Text", st.session_state["editable_output"], height=200)

