''' Retrieve answer module '''
import logging
from typing import Any
from langchain.vectorstores.chroma import Chroma
from flask import jsonify, request
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from app.utils.tracktime import tracktime
from app.ingest import CHROMA_SETTINGS, chroma_config
from app.config import BaseConfig as app_config

logger = logging.getLogger(app_config.APP_NAME)

def create_chroma_retriever(embeddings : Any) -> object:
    ''' Chroma retriever '''

    chroma_db = Chroma(
        persist_directory=chroma_config['chroma_config']['persist_directory'],
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS
    )

    retriever = chroma_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": chroma_config['chroma_config']['similarity_score_threshold']
        }
    )

    return retriever

def create_prompt_template() -> object:
    ''' Prompt template for LLM '''

    template = """Use the following knowledge triplets to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    \n\n{context}\n\nQuestion: {question}\nHelpful Answer:"""

    return PromptTemplate(template=template, input_variables=["context", "question"])

@tracktime
def get_answer(shared_components: object) -> jsonify:
    """
    Retrieve an answer based on the input query from JSON request using shared components.

    Args:
        shared_components (object): An object containing shared components with embeddings and llm.

    Returns:
        jsonify: A JSON response containing the query, answer, and source information.
    """
    try:
        query = request.json
        logger.info('Get answer triggered with query: %s', query)

        embeddings = shared_components.get_embeddings()
        llm = shared_components.get_llm()

        logger.info('Received request with query %s', query)

        if llm is None:
            logger.error("Model not found!")
            return jsonify("Model not found!"), 400

        retriever = create_chroma_retriever(embeddings)

        docs = retriever.get_relevant_documents(query)
        logger.info('Retrieved %s docs with the set relevance score.', len(docs))

        if not docs:
            logger.info('Could not find any docs for query: %s', query)
            return jsonify(answer="Could not find any docs!", source="")

        prompt_template = create_prompt_template()

        chain_type_kwargs = {
            "prompt": prompt_template,
            'verbose': True,
        }

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

            response = {
                "query": query,
                "answer": answer,
                "source": source_data
            }

            logger.info("Returning response: %s", response)
            return jsonify(response)

        return jsonify("Empty Query"), 400

    except Exception as e:
        logger.error('An error occurred: %s', e)
        return jsonify("Internal Server Error"), 500
