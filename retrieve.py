#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import time
from flask import jsonify, request
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores.chroma import Chroma
from ingest import CHROMA_SETTINGS

def get_answer(shared_components: object) -> jsonify:
    """
    Retrieve an answer based on the input query using shared components.

    Args:
        shared_components (object): An object containing shared components like embeddings and llm.

    Returns:
        jsonify: A JSON response containing the query, answer, and source information.
    """

    try:
        query = request.json
        logging.info('Get answer triggered with query: %s', query)
        embeddings = shared_components.get_embeddings()
        llm = shared_components.get_llm()

        logging.info('Received request with query %s', query)

        if llm is None:
            logging.error("Model not found!")
            return jsonify("Model not found!"), 400

        chroma_db = Chroma(
            persist_directory="db",
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS
        )

        retriever = chroma_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.3}
        )

        docs = retriever.get_relevant_documents(query)
        logging.info('Retrieved %s amount of docs with the set relevance score.', len(docs))

        if not docs:
            logging.info('Could not find any docs for query: %s', query)
            return jsonify(answer="Could not find any docs!", source="")

        prompt_template = """Use the following knowledge triplets to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        \n\n{context}\n\nQuestion: {question}\nHelpful Answer:"""

        prompt_template = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        chain_type_kwargs = {
            "prompt": prompt_template,
            'verbose': True,
        }

        start = time.time()  # Track time for debugging purposes
        retrieval_qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )

        if query is not None and query != "":
            result = retrieval_qa(query)
            answer, docs = result['result'], result['source_documents']
            source_data = []

            for document in docs:
                source_data.append({"name": document.metadata["source"]})

            end = time.time()
            response = {
                "query": query,
                "answer": answer,
                "source": source_data
            }
            print(jsonify(response))
            print(f"From source documents: {docs}")
            print(f"Query executed in: {end - start}")
            return jsonify(response)

        return jsonify("Empty Query"), 400

    except Exception as e:
        logging.error('An error occurred: %s', e)
        return jsonify("Internal Server Error"), 500
