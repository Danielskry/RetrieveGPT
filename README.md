# RetrieveGPT
A simple and fast local large language model (LLM) [Flask](https://github.com/pallets/flask) app that can be employed to retrieve internal documents based on questions, utilizing Retrieval Augmented Generation (RAG) with [Llama2](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF) and [Langchain](https://github.com/langchain-ai/langchain).

![Untitled Diagram drawio(5)](https://github.com/Danielskry/RetrieveGPT/assets/15195014/a54d1970-c5db-466b-ad39-0749bec4221e)

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
python3 run.py
```

## Ingest data

With the `/ingest` endpoint, you can transform raw and unstructured data into a structured format within a local vector database with ChromaDB. To utilize this feature, simply deposit your documents into the designated `source_documents` directory. The ingest route will subsequently extract data from this directory, process the documents by dividing them into manageable chunks, generate embeddings, and finally, integrate the resulting data into the Chroma Vector DB, conveniently stored in the `db` directory.

## Get answer
Retrieve insightful answers to your queries using the `/get_answer` endpoint. Input your question in the specified JSON format, and receive a detailed response providing relevant information based on the data you have ingested. The example below is based on ingesting the Git documentation from the `git.pdf` found in `source_documents`.

**Input:**
```json
{
    "query" : "What is Git?"
}
```
**Output:**
```json
{
    "answer": " Git is a version control system used for source code management in software development. It allows developers to track changes made to their codebase over time, collaborate with others on the same project, and easily revert back to previous versions if necessary.",
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
