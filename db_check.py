from net.config import settings
from net.vector_store import get_vector_store
from net.llm_services import LLMServiceManager

if __name__ == "__main__":
    llm_service = LLMServiceManager(settings)
    embedding_service = llm_service.get_embedding_model()    
    
    # Example usage
    vector_store = get_vector_store(
        [],
        embedding_service,
        settings.VECTOR_DB_PATH,
        force_recreate=False  # Set to True to force recreation of the vector store
    )

    if vector_store:
        print("Vector store initialized successfully.")
    else:
        print("Failed to initialize vector store.")
        exit(1)
        
    # vector store statics
    print("Vector store statistics:")
    print(f"Number of vectors: {vector_store.index}")
    print(f"Vector dimension: {vector_store.index.d}")
    print(f"Vector store path: {settings.VECTOR_DB_PATH}")    
    
    # check vecotor store
    print("Check vector store interactively:")
    while True:
        query = input("Enter a query: ")
        if query == "exit":
            break
        results = vector_store.similarity_search(query, k=3)
        for i, result in enumerate(results):
            print(f"Result {i+1}:")
            print(result.page_content)
            print("-" * 40)