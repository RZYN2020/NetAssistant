import os
from typing import List, Any

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_and_split_documents(knowledge_base_dir: str, chunk_size: int, chunk_overlap: int) -> List[Any]:
    """Loads documents from the specified directory and splits them into chunks."""
    print(f"Loading documents from '{knowledge_base_dir}'...")
    if not os.path.exists(knowledge_base_dir):
        os.makedirs(knowledge_base_dir, exist_ok=True)
        print(f"Warning: Knowledge base directory '{knowledge_base_dir}' was created.")
        placeholder_path = os.path.join(knowledge_base_dir, "placeholder_content.txt")
        if not os.path.exists(placeholder_path):
            with open(placeholder_path, "w", encoding="utf-8") as f:
                f.write("This is a placeholder file for the knowledge base.\n"
                        "Computer networks connect devices to share resources.\n"
                        "The OSI model has 7 layers, TCP/IP has 4 or 5.")
            print(f"Created a placeholder file: '{placeholder_path}'")

    # Configure loaders for different file types
    loader_kwargs_map = {
        ".pdf": {"loader_cls": PyPDFLoader, "kwargs": {"extract_images": False}},
        ".txt": {"loader_cls": TextLoader, "kwargs": {"encoding": "utf-8"}},
        ".md": {"loader_cls": TextLoader, "kwargs": {"encoding": "utf-8"}},
    }
    
    # Using DirectoryLoader to handle multiple file types
    # It will try to match file extensions to the loader_kwargs_map implicitly if loader_cls is not set
    # However, being explicit with loader_mapping is more robust.
    # For simplicity, we'll let DirectoryLoader try its best with common extensions.
    # For more control, you could iterate and use specific loaders.
    
    all_documents = []

    # Manually specify how to load which files for more control
    # This part is a bit tricky with DirectoryLoader's default behavior.
    # A simpler approach for this example: load known types explicitly.
    
    loaded_docs_count = 0
    for ext, loader_info in loader_kwargs_map.items():
        try:
            current_loader = DirectoryLoader(
                knowledge_base_dir,
                glob=f"**/*{ext}",
                loader_cls=loader_info["loader_cls"],
                loader_kwargs=loader_info["kwargs"],
                use_multithreading=True,
                show_progress=False, # Avoid multiple progress bars
                silent_errors=True
            )
            docs_for_ext = current_loader.load()
            if docs_for_ext:
                all_documents.extend(docs_for_ext)
                loaded_docs_count += len(docs_for_ext)
                print(f"Loaded {len(docs_for_ext)} document(s)/page(s) from *{ext} files.")
        except Exception as e:
            print(f"Error loading *{ext} files: {e}")


    if not all_documents:
        print("Warning: No documents were loaded from the knowledge base. The RAG assistant may not function correctly.")
        return []
    
    print(f"Total documents/pages loaded: {loaded_docs_count}.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(all_documents)
    print(f"Documents split into {len(split_docs)} chunks.")
    return split_docs