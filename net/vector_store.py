import os
from typing import List, Any, Optional
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from fastapi import HTTPException

def get_vector_store(
    docs: List[Any], # only used for creating a new vector store
    embedding_service: Embeddings,
    vector_db_path: str,
    force_recreate: bool = False
) -> Optional[FAISS]:
    """Creates or loads a FAISS vector store."""
    if force_recreate and os.path.exists(vector_db_path):
        print(f"Force recreate: Deleting existing vector store at '{vector_db_path}'")
        import shutil
        shutil.rmtree(vector_db_path)

    vector_store = None
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
            return vector_store
        except Exception as e:
            print(f"Failed to load vector store: {e}. Will attempt to recreate.")

    if len(docs) == 0:
        print("No documents provided and no existing vector store found.")
    try:
        vector_store = FAISS.from_documents(docs, embedding_service)
        vector_store.save_local(vector_db_path)
        print("New vector store created and saved.")
        return vector_store
    except Exception as e:
        print(f"Failed to create vector store: {e}")
        raise HTTPException(status_code=500, detail=f"Vector store creation failed: {e}")
