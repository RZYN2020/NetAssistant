import os
from typing import List, Any, Optional
from langchain_core.embeddings import Embeddings


def get_vector_store(
    docs: List[Any],
    embedding_service: Embeddings,
    vector_db_path: str,
    force_recreate: bool = False
) -> Optional[FAISS]:
    """Creates or loads a FAISS vector store."""
    if force_recreate and os.path.exists(vector_db_path):
        print(f"Force recreate: Deleting existing vector store at '{vector_db_path}'")
        import shutil
        shutil.rmtree(vector_db_path)

    if os.path.exists(vector_db_path) and os.listdir(vector_db_path):
        print(f"Loading existing vector store from '{vector_db_path}'...")
        try:
            # Note: allow_dangerous_deserialization can be a security risk.
            # For production, consider safer alternatives or ensure the source of db is trusted.
            vector_store = FAISS.load_local(
                vector_db_path,
                embedding_service,
                allow_dangerous_deserialization=True
            )
            print("Vector store loaded successfully.")
            return vector_store
        except Exception as e:
            print(f"Failed to load vector store: {e}. Will attempt to recreate.")
    
    if not docs:
        print("No documents provided to create a new vector store.")
        return None

    print(f"Creating new vector store at '{vector_db_path}'...")
    try:
        vector_store = FAISS.from_documents(docs, embedding_service)
        vector_store.save_local(vector_db_path)
        print("New vector store created and saved.")
        return vector_store
    except Exception as e:
        print(f"Failed to create vector store: {e}")
        raise HTTPException(status_code=500, detail=f"Vector store creation failed: {e}")
