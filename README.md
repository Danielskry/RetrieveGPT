# Semantic-document-retriever
A simple local large language model (LLM) [Flask](https://github.com/pallets/flask) app is employed to retrieve documents based on questions, utilizing Retrieval Augmented Generation (RAG) with [Llama2](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)  and [Langchain](https://github.com/langchain-ai/langchain).

## Setup
1. Clone [Llama2](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF):
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
```
2. Clone [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2):
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
```
3. Place both models in the `models` directory.
4. Install `requirements.txt`:
```bash
pip install -r requirements.txt
```
5. Run Flask application:
```
python3 app.py
```

## Ingest data

With the ingest functionality, you can transform raw and unstructured data into a structured format within a local vector database with ChromaDB. To utilize this feature, simply deposit your documents into the designated `source_documents` directory. The ingest route will subsequently extract data from this directory, process the documents by dividing them into manageable chunks, generate embeddings, and finally, integrate the resulting data into the Chroma Vector DB, conveniently stored in the `db` directory.

## Get answer
Retrieve insightful answers to your queries using the `/get_answer` endpoint. Input your question in the specified JSON format, and receive a detailed response providing relevant information.

**Input:**
```json
{
    "query" : "What is Git?"
}
```
**Output:**
```json
{
    "answer": " Git is a version control system used for source code management. It allows developers to track changes made to their codebase over time, collaborate with others, and manage different versions of their software.",
    "query": {
        "query": "What is Git?"
    },
    "source": [
        {
            "name": "source_documents\\git.pdf"
        }
    ]
}
```
