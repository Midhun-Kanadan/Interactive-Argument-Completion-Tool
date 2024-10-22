# Interactive-Argument-Completion-Tool

![Logo](images/logo.png)


This project aims to build an interactive argument completion tool using Retrieval-Augmented Generation (RAG) methodology. The tool leverages ChromaDB for vector storage and similarity search, and Vertex AI for text generation. The dataset used for generating arguments is from the HuggingFace dataset [MidhunKanadan/arguments_and_summaries](https://huggingface.co/datasets/MidhunKanadan/arguments_and_summaries).

## Methodology

The tool follows the Retrieval-Augmented Generation (RAG) approach, which involves the following steps:

1. **Retrieve**: Fetch relevant context sentences from the dataset based on the input query.
2. **Generate**: Use the retrieved context to generate coherent and contextually relevant completions.

## Steps Used

### 1. Load the Dataset
- The dataset is loaded from HuggingFace using the `datasets` library.
- Extract summaries from the dataset for further processing.

### 2. Vectorize Summaries
- Use a pre-trained model from the `sentence-transformers` library to vectorize the summaries.
- The `AutoTokenizer` and `AutoModel` from the `transformers` library are used for tokenizing and embedding the summaries.

### 3. Store Embeddings in ChromaDB
- Initialize ChromaDB client to manage and query vector embeddings.
- Store vectorized summaries along with metadata in a ChromaDB collection.

### 4. Save and Load Embeddings
- Create functions to save embeddings and their metadata to a file using the `pickle` library.
- Load embeddings from the file to avoid reprocessing in future sessions.

### 5. Retrieve Similar Summaries
- Implement a function to retrieve similar summaries from ChromaDB based on user input.
- Retrieved documents and metadata are used to provide context for completing arguments.

### 6. Argument Completion with Vertex AI
- Initialize Vertex AI to generate completions based on user input and retrieved summaries.
- Generate three different completions with positive, negative, and neutral sentiments.
- Handle empty input by returning a hard-coded sentence about machine learning and AI.

## Features

### Vector Storage
- Vectorized summaries are stored in ChromaDB, enabling efficient similarity searches.
- Metadata is stored along with the vectors to facilitate context retrieval.

### Argument Dataset
- The dataset [MidhunKanadan/arguments_and_summaries](https://huggingface.co/datasets/MidhunKanadan/arguments_and_summaries) is used for generating arguments.
- This dataset contains arguments and their summaries, which are used as the basis for generating completions.

### Three Different Outputs
- The tool generates three different completions for the input statement:
  - Positive sentiment
  - Negative sentiment
  - Neutral sentiment

### Vector Similarity
- The tool uses vector similarity to retrieve the most relevant context sentences from the dataset.
- These context sentences are used to generate coherent and contextually relevant completions.

### Options to Check the Context Used
- The tool provides an option to display the context sentences used for generating the completions.
- Users can see the context sentences along with their IDs.


## How to Run

### Prerequisites

- Docker installed on your machine.
- Google Cloud credentials JSON file.

### Steps to Run

1. **Pull the Docker image:**

    \`\`\`sh
    docker pull midhunkanadan/argument-completion-tool:latest
    \`\`\`

2. **Run the Docker container:**

    \`\`\`sh
    docker run -p 8501:8501 -v /path/to/your/credentials.json:/app/credentials/credentials.json midhunkanadan/argument-completion-tool
    \`\`\`

    Replace \`/path/to/your/credentials.json\` with the actual path to your Google Cloud credentials JSON file.

3. **Access the application:**

    Open your web browser and navigate to \`http://localhost:8501\`.

## Usage

- Enter your partial argument or statement in the text area.
- Click the \"Complete\" button to generate positive, negative, and neutral completions.
- Use the \"Show Context\" button to display similar summaries used for generating completions.
- Clear the conversation using the \"Clear Conversation\" button.

## Author

Midhun Kanadan

- [LinkedIn](https://www.linkedin.com/in/midhunkanadan/)
