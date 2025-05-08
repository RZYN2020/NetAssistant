from typing import Optional

from net.llm_services import BaseLLMService, get_llm_service
from net.rag_builder import build_rag_chain
from net.config import settings
from net.vector_store import get_vector_store
from net.document_loader import load_and_split_documents

if __name__ == "__main__":
    # terminal repl 
    llm_service: Optional[BaseLLMService] = get_llm_service(settings)

    if not llm_service:
        print("Error: LLM service could not be initialized.")
        exit(1)

    if settings.ADD_KNOWLEDGE:
        split_docs = load_and_split_documents(
            settings.KNOWLEDGE_BASE_DIR,
            settings.CHUNK_SIZE,
            settings.CHUNK_OVERLAP
        )
    else:
        print("No knowledge base documents to load. Skipping document loading.")
        split_docs = []


    vector_store = get_vector_store(split_docs, 
                                    llm_service.get_embedding_model(), 
                                    settings.VECTOR_DB_PATH)
    if not vector_store:
        print("Error: Vector store could not be initialized.")
        exit(1)

    rag_chain = build_rag_chain(llm_service.get_chat_model(), 
                                vector_store, 
                                settings.SEARCH_K_DOCS)

    while True:
        user_input = input("Enter your question (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        try:
            response = rag_chain({"query": user_input})
            print("Response:", response)
        except Exception as e:
            print(f"Error during query: {e}")