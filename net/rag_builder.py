from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.vectorstores import FAISS
from .config import settings

def build_rag_chain(
    llm: BaseChatModel,
    vector_store: FAISS,
    k_docs_to_retrieve: int
) -> RetrievalQA:
    """Builds the RAG chain."""
    # read prompt template from file
    with open(settings.PROMPT_PATH, "r") as file:
        prompt_template_str = file.read()
    
    QA_PROMPT = PromptTemplate.from_template(prompt_template_str)

    retriever = vector_store.as_retriever(search_kwargs={"k": k_docs_to_retrieve})
    
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" is a common chain type for RAG
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True # Useful for debugging or showing sources
    )
    print("RAG chain built successfully.")
    return rag_chain
