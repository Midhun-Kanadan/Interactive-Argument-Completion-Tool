import streamlit as st
import vertexai
from vertexai.preview.language_models import TextGenerationModel
from google.oauth2 import service_account
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModel
import torch
import pickle
from datasets import load_dataset
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
collection_name = "summarized_arguments_collection"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
project_id = os.getenv("PROJECT_ID")
location = os.getenv("LOCATION")
vertex_model_name = os.getenv("VERTEX_MODEL_NAME")
embeddings_file = "summarized_embeddings.pkl"
credentials_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Initialize ChromaDB client
@st.cache_resource
def init_chroma_client():
    return chromadb.Client(Settings())

chroma_client = init_chroma_client()

# Load pre-trained model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# Function to vectorize user input
def vectorize_input(user_input):
    inputs = tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

# Load or create embeddings
@st.cache_data
def load_or_create_embeddings():
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            data = pickle.load(f)
        return data["summaries"], data["vectorized_summaries"]
    else:
        dataset = load_dataset("MidhunKanadan/arguments_and_summaries")
        summaries = [entry['summary'] for entry in dataset['train']]
        vectorized_summaries = [vectorize_input(summary) for summary in summaries]
        data = {"summaries": summaries, "vectorized_summaries": vectorized_summaries}
        with open(embeddings_file, 'wb') as f:
            pickle.dump(data, f)
        return summaries, vectorized_summaries

summaries, vectorized_summaries = load_or_create_embeddings()

# Initialize or get ChromaDB collection
@st.cache_resource
def init_or_get_collection():
    existing_collections = chroma_client.list_collections()
    if collection_name in existing_collections:
        return chroma_client.get_collection(name=collection_name)
    else:
        collection = chroma_client.create_collection(name=collection_name)
        for i, (summary, vector) in enumerate(zip(summaries, vectorized_summaries)):
            collection.add(
                documents=[summary],
                ids=[f"summary_{i}"],
                embeddings=[vector.tolist()],
                metadatas=[{"argument_id": i}]
            )
        return collection

collection = init_or_get_collection()

# Function to retrieve similar summaries
def retrieve_similar_summaries(user_input):
    if not user_input:
        return []
    user_vector = vectorize_input(user_input)
    results = collection.query(query_embeddings=[user_vector.tolist()], n_results=5)
    return list(zip(results['documents'][0], results['metadatas'][0]))

# Initialize Vertex AI
@st.cache_resource
def init_vertex_ai():
    credentials = service_account.Credentials.from_service_account_file(credentials_file)
    vertexai.init(project=project_id, location=location, credentials=credentials)
    return TextGenerationModel.from_pretrained(vertex_model_name)

vertex_model = init_vertex_ai()

def complete_argument(user_input, retrieved_summaries, sentiment="neutral"):
    # Hard-coded text for empty input
    hard_coded_text = "Machine learning and AI have the potential to revolutionize the world by enabling smarter decision-making, improving healthcare outcomes, and driving innovations that can lead to a more sustainable and efficient future."

    if not user_input:
        return hard_coded_text

    example = f"""Examples:
User input: "I enjoy reading books"
Good completion: "I enjoy reading books because they transport me to new worlds and expand my knowledge."
Bad completion: "because they transport me to new worlds and expand my knowledge."
Bad completion: "is a great hobby. Reading books is enjoyable for many people."

User input: "Artificial intelligence is rapidly advancing"
Good completion: "Artificial intelligence is rapidly advancing, leading to groundbreaking innovations in various fields."
Bad completion: "leading to groundbreaking innovations in various fields."
Bad completion: "and changing the world in unprecedented ways."
"""

    if retrieved_summaries:
        combined_input = "\n".join([doc for doc, _ in retrieved_summaries])
        prompt = f"""Complete the following partial statement by directly continuing from where it left off. The completion should form a single, coherent sentence when combined with the user's input. Use the provided related information if relevant, but prioritize a natural and logical continuation. Do not repeat the user input.

User's partial statement: "{user_input}"

Related information:
{combined_input}

Complete the statement (continue directly from the last word of the user's input without repeating it) in a {sentiment} sentiment:"""
    else:
        prompt = f"""Complete the following partial statement by directly continuing from where it left off. The completion should form a single, coherent sentence when combined with the user's input. Do not repeat the user input.

User's partial statement: "{user_input}"

Complete the statement (continue directly from the last word of the user's input without repeating it) in a {sentiment} sentiment:"""
    
    response = vertex_model.predict(prompt, temperature=0.7, max_output_tokens=60, top_k=40, top_p=0.9)
    completion = response.text.strip()
    
    # Remove any leading spaces or punctuation
    completion = completion.lstrip(' .,;:')
    
    # Ensure the completion does not duplicate the user input
    if completion.startswith(user_input):
        completion = completion[len(user_input):].lstrip()
    
    # Ensure the completion starts with a lowercase letter unless it's a proper noun
    if completion and completion[0].isalpha() and completion[0].isupper():
        completion = completion[0].lower() + completion[1:]
    
    return f"{user_input} {completion}"

def complete_argument_from_input(user_input):
    similar_summaries_with_metadata = retrieve_similar_summaries(user_input)
    positive_completion = complete_argument(user_input, similar_summaries_with_metadata, sentiment="positive")
    negative_completion = complete_argument(user_input, similar_summaries_with_metadata, sentiment="negative")
    neutral_completion = complete_argument(user_input, similar_summaries_with_metadata, sentiment="neutral")
    return positive_completion, negative_completion, neutral_completion, similar_summaries_with_metadata

# Streamlit UI
st.title("Interactive Argument Completion Tool")

# Initialize session state
if 'last_completion' not in st.session_state:
    st.session_state.last_completion = ""
if 'positive_completion' not in st.session_state:
    st.session_state.positive_completion = ""
if 'negative_completion' not in st.session_state:
    st.session_state.negative_completion = ""
if 'neutral_completion' not in st.session_state:
    st.session_state.neutral_completion = ""
if 'context' not in st.session_state:
    st.session_state.context = []
if 'example_selected' not in st.session_state:
    st.session_state.example_selected = ""

# Example sentences
example_sentences = {
    "Example 1": "Elon musk is great, but",
    "Example 2": "I don't like Taylor Swift,",
    "Example 3": "ML and AI can"
}

# Layout for input and examples dropdown
col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_area("Enter your statement:", key="input", value=st.session_state.last_completion, height=100)

with col2:
    selected_example = st.selectbox(
        "Choose an example:",
        options=list(example_sentences.keys()),
        format_func=lambda x: x.split(":")[0],
        key="example_select", label_visibility = "hidden",
        index=None,
        placeholder="Examples",
    )
    if selected_example and selected_example != st.session_state.example_selected:
        st.session_state.last_completion = example_sentences[selected_example]
        st.session_state.example_selected = selected_example
        st.rerun()

    st.write("")  # Add a small gap
    if st.button("Clear Conversation  "):
        st.session_state.last_completion = ""
        st.session_state.positive_completion = ""
        st.session_state.negative_completion = ""
        st.session_state.neutral_completion = ""
        st.session_state.context = []
        st.rerun()

if st.button("Complete"):
    if user_input:
        positive_completion, negative_completion, neutral_completion, context = complete_argument_from_input(user_input)
        st.session_state.positive_completion = positive_completion
        st.session_state.negative_completion = negative_completion
        st.session_state.neutral_completion = neutral_completion
        st.session_state.context = context
        st.session_state.example_selected = ""  # Reset example selection after completion
        st.rerun()

# Display the output in buttons with different sentiments and colors
if st.session_state.positive_completion:
    if st.button(f"Positive: {st.session_state.positive_completion}", key="positive_output", help="Click to select this positive completion"):
        st.session_state.last_completion = st.session_state.positive_completion
        st.rerun()

if st.session_state.negative_completion:
    if st.button(f"Negative: {st.session_state.negative_completion}", key="negative_output", help="Click to select this negative completion"):
        st.session_state.last_completion = st.session_state.negative_completion
        st.rerun()

if st.session_state.neutral_completion:
    if st.button(f"Neutral: {st.session_state.neutral_completion}", key="neutral_output", help="Click to select this neutral completion"):
        st.session_state.last_completion = st.session_state.neutral_completion
        st.rerun()

# Display context sentences and IDs
if st.session_state.context:
    if st.button("Show Context"):
        context_display = ""
        for i, (doc, meta) in enumerate(st.session_state.context[:3]):
            context_display += f"ID: {meta['argument_id']}, Context: {doc}\n"
        st.text_area("Context Sentences Used:", value=context_display, height=150, disabled=True)
