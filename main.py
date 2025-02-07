from data_loader import load_documents
from splitter import split_documents
from dataBase import add_to_chroma, clear_database
from query import query_rag
import streamlit as st
import os
import torch

torch.classes.__path__ = []
selected_files = []
binary_semaphore = 1

def take_semaphore():
    global binary_semaphore
    binary_semaphore -=1
def release_semaphore():
    global binary_semaphore
    binary_semaphore +=1
        
# Set up the page configuration for the Streamlit app
st.set_page_config(
    page_title="Quiz PDF",
    layout="centered",
    page_icon="🏫"
)

# Define the folder where uploaded files will be stored
upload_folder = "Data/Documents"

#initializing arrays
uploaded_files = os.listdir(upload_folder)
existing_files = []
# Display the title of the application
st.title("Quiz PDF")



# Initialize session state for the file uploader key if not already present
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Function to update the file uploader key
def update_key():
    st.session_state.uploader_key += 1

# Add a header to the sidebar
st.sidebar.header("Uploaded Files")

# Display the list of uploaded files in the sidebar
if uploaded_files != existing_files:
    existing_files = uploaded_files
    selected_files = []
    for i, file in enumerate(uploaded_files, start=1):
        if st.sidebar.checkbox(f"{i}) {file}", value= True):
            selected_files.append(file)

# Add a button to delete all files in the folder
if st.sidebar.button("Delete Selected Files🗑️"):
    if selected_files:
        try:
            # Clear the database before deleting files
            clear_database(selected_files)

            # Delete all files in the folder
            for file in selected_files:
                file_path = os.path.join(upload_folder, file)
                os.remove(file_path)
            update_key()
            st.rerun()  # Rerun the app to reflect changes
        except Exception as e:
            st.warning(f"Can't delete files at the moment!\n{e}")
    else:
        st.warning("Select a File to Delete!")

# File uploader interface to allow multiple PDF uploads
uploaded_files = st.file_uploader(
    "Upload Your Documents", 
    accept_multiple_files=True, 
    key=f"uploader_{st.session_state.uploader_key}"
)
embModel = st.selectbox(
            "Choose the Emb Model",
            ("Gemini API","Nomic-Embed-Text")
        )
# Process uploaded files if any exist
if uploaded_files:
    try:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(upload_folder, uploaded_file.name)
            # Save each uploaded file to the specified folder
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        # Display a spinner while processing the files
        with st.spinner("Processing and Adding to DB..."):
            # Load, split, and add documents to the database
            documents = load_documents()  # Load the documents
            chunks = split_documents(documents)  # Split documents into chunks
            add_to_chroma(chunks,embModel)  # Add the chunks to the Chroma database

            update_key()  # Update the uploader key
        st.rerun()  # Rerun the app to reflect changes

    except Exception as e:
        st.warning(f"Error while uploading files: {e}")

# Create a form for user query input
with st.form("user_query_input",enter_to_submit=False):
    query = st.text_area("Enter Your Question:")
    col1,col2 = st.columns(2)
    
    
    with col1:
        num_questions = st.number_input(
            "Number of Questions Per Topic",
            value=5,
            max_value=50,
            min_value=1,
        )
    with col2:
        ai_selector = st.selectbox(
            "Choose the AI Model",
            ("LLAMA 3.3 API","LLAMA3.2")
        )
    # Add a button to submit the query
    if st.form_submit_button("Query with Selected Files🤖"): 
        if binary_semaphore:
            take_semaphore()
            # Check if there is files selected
            if  selected_files:
                # Query the database using RAG (Retrieval-Augmented Generation)
                response, sources = query_rag(query,selected_files,num_questions,ai_selector,embModel)
                st.write(response)  # Display the response from the query
                st.balloons()
                # Display the sources of the response
                st.write("Sources📖:")
                cleaned_source = "\n".join(sorted(set([reference[15:-2].replace(":", " | Page ") for reference in sources])))
                st.text(cleaned_source)  # Display cleaned and formatted sources
                release_semaphore()
                
            else:
                st.warning("Select a File to Search!")
        else:
            st.warning("AI🤖 is already running!")