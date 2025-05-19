from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.vectorstores import FAISS
from .config import settings
from typing import Dict, List, Any, Optional
from .llm_services import LLMServiceManager
from .vector_store import get_vector_store
from .document import load_and_split_documents
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import List
from langchain.schema.retriever import BaseRetriever
from langchain.schema import Document
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

def classify_query(llm: BaseChatModel, query: str) -> bool:
    """Determines if the query requires external knowledge retrieval."""
    classify_prompt = PromptTemplate.from_template(
        "Determine if this query is related to computer networking topics (like TCP/IP, protocols, networking hardware, network security, etc.), networking courses, learning materials, or course arrangements/schedules. "
        "If it's a computer networking related question, or about course schedules, or arrangements that might need externel knowledge, answer 'yes'. "
        "For non-networking questions, answer 'no'.\n"
        "Answer with only 'yes' or 'no'.\nQuery: {query}"
    )
    chain = classify_prompt | llm
    response = chain.invoke({"query": query})
    # Check if the response is 'yes' or 'no'
    # See if last word is 'yes' or 'no'
    return response.content.strip()[-3:].lower() == "yes"

def transform_query(llm: BaseChatModel, query: str) -> str:
    """Transforms the query to improve retrieval accuracy."""
    transform_prompt = PromptTemplate.from_template(
        "Rewrite this query to be more specific and searchable while preserving its original intent and language. "
        "Keep the response in the same language as the input query: {query}"
    )
    chain = transform_prompt | llm
    response = chain.invoke({"query": query})
    return response.content

def rerank_documents(llm: BaseChatModel, query: str, documents: List[Document]) -> List[Document]:
    """Re-ranks documents based on their relevance to the query using LLM scoring."""
    print(f"DEBUG: Reranking {len(documents)} documents for query: '{query}'")
    if not documents:
        return []

    scoring_prompt = PromptTemplate.from_template(
        "On a scale of 0-10, rate how relevant this document is to the query.\n"
        "Query: {query}\n"
        "Document: {document}\n"
        "Output only the numeric score, nothing else."
    )
    
    chain = scoring_prompt | llm
    scored_documents = []
    for doc in documents:
        try:
            score = float(chain.invoke({"query": query, "document": doc.page_content}).content.strip())
            scored_documents.append({"doc": doc, "score": score})
        except ValueError:
            scored_documents.append({"doc": doc, "score": 0})
    
    reranked_docs = sorted(scored_documents, key=lambda x: x["score"], reverse=True)
    reranked_documents = [item["doc"] for item in reranked_docs]
    
    return reranked_documents

def summarize_documents(llm: BaseChatModel, docs: List[str]) -> List[str]:
    """Summarizes retrieved documents to reduce redundancy."""
    summarize_prompt = PromptTemplate.from_template(
        "Summarize the following document while preserving key information and keep the language not change:\n{document}"
    )
    chain = summarize_prompt | llm
    return [chain.invoke({"document": doc}).content for doc in docs]

def debug(message: str):
    """Debugging utility to print messages."""
    if settings.DEBUG:
        print(f"DEBUG: {message}")

class EnhancedCustomRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    llm: BaseChatModel

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        debug(f"EnhancedCustomRetriever received query: '{query}'")
        
        # 1. Classify Query
        if not classify_query(self.llm, query):
            debug("Query classified as not requiring external knowledge.")
            return []

        # 2. Transform Query
        transformed_query = transform_query(self.llm, query)
        debug(f"Transformed query: '{transformed_query}'")

        # 3. Retrieve Initial Documents
        initial_docs = self.base_retriever.get_relevant_documents(
            transformed_query, callbacks=run_manager.get_child(), **kwargs
        )
        print(f"DEBUG: Retrieved {len(initial_docs)} initial documents after transformation.")
        if not initial_docs:
            return []

        # 4. Re-rank Retrieved Documents
        # The transformed_query is likely better for reranking than the original query
        reranked_docs = rerank_documents(
            self.llm, transformed_query, initial_docs
        )
        
        # Select top N documents after re-ranking for summarization ("文档打包" - part 1: selection)
        docs_to_summarize = reranked_docs[:settings.TOP_K]
        print(f"DEBUG: Selected top {len(docs_to_summarize)} re-ranked documents for summarization.")

        # 5. Summarize Selected Documents
        doc_page_contents = [doc.page_content for doc in docs_to_summarize]
        summarized_contents = summarize_documents(self.llm, doc_page_contents)
        if not summarized_contents:
            print("DEBUG: No summaries generated.")
            return []

        # 6. Package Summaries into Document objects ("文档打包" - part 2: formatting for LLM)
        # This is the "optimized information presentation" part for the RAG chain
        final_documents: List[Document] = []
        for i, summary_text in enumerate(summarized_contents):
            # Carry over metadata from the re-ranked and selected document
            original_doc_metadata = docs_to_summarize[i].metadata if i < len(docs_to_summarize) else {}
            # Add information about the summarization and reranking if desired
            final_metadata = original_doc_metadata.copy()
            final_metadata["retrieval_step"] = "summarized_after_reranking"
            final_metadata["original_reranked_doc_source"] = original_doc_metadata.get("source", "unknown")


            final_documents.append(
                Document(
                    page_content=summary_text,
                    metadata=final_metadata
                )
            )
        print(f"DEBUG: Returning {len(final_documents)} summarized and packaged documents to the RAG chain.")
        return final_documents

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        debug(f"EnhancedCustomRetriever received async query: '{query}'")
        
        # 1. Classify Query
        if not await self._async_classify_query(self.llm, query):
            debug("Query classified as not requiring external knowledge.")
            return []

        # 2. Transform Query
        transformed_query = await self._async_transform_query(self.llm, query)
        debug(f"Transformed query: '{transformed_query}'")

        # 3. Retrieve Initial Documents
        initial_docs = await self.base_retriever.aget_relevant_documents(
            transformed_query, callbacks=run_manager.get_child(), **kwargs
        )
        print(f"DEBUG: Retrieved {len(initial_docs)} initial documents after transformation.")
        if not initial_docs:
            return []

        # 4. Re-rank Retrieved Documents
        reranked_docs = await self._async_rerank_documents(
            self.llm, transformed_query, initial_docs
        )
        
        # Select top N documents after re-ranking
        docs_to_summarize = reranked_docs[:settings.TOP_K]
        print(f"DEBUG: Selected top {len(docs_to_summarize)} re-ranked documents for summarization.")

        # 5. Summarize Selected Documents
        doc_page_contents = [doc.page_content for doc in docs_to_summarize]
        summarized_contents = await self._async_summarize_documents(self.llm, doc_page_contents)
        if not summarized_contents:
            print("DEBUG: No summaries generated.")
            return []

        # 6. Package Summaries into Document objects
        final_documents: List[Document] = []
        for i, summary_text in enumerate(summarized_contents):
            original_doc_metadata = docs_to_summarize[i].metadata if i < len(docs_to_summarize) else {}
            final_metadata = original_doc_metadata.copy()
            final_metadata["retrieval_step"] = "summarized_after_reranking"
            final_metadata["original_reranked_doc_source"] = original_doc_metadata.get("source", "unknown")

            final_documents.append(
                Document(
                    page_content=summary_text,
                    metadata=final_metadata
                )
            )
        print(f"DEBUG: Returning {len(final_documents)} summarized and packaged documents to the RAG chain.")
        return final_documents


    # Placeholder async helper wrappers
    async def _async_classify_query(self, llm: BaseChatModel, query: str) -> bool:
        # Replace with: return await llm.ainvoke(...) or similar for actual async LLM call
        return classify_query(llm, query)
    async def _async_transform_query(self, llm: BaseChatModel, query: str) -> str:
        return transform_query(llm, query)
    async def _async_rerank_documents(self, llm: BaseChatModel, query: str, documents: List[Document]) -> List[Document]:
        # Replace with actual async reranking logic
        return rerank_documents(llm, query, documents) # top_n for rerank_documents is different from k_after_rerank_to_summarize
    async def _async_summarize_documents(self, llm: BaseChatModel, contents: List[str]) -> List[str]:
        return summarize_documents(llm, contents)

def build_rag_chain(
    llm: BaseChatModel,
    help_model: BaseChatModel,
    vector_store: FAISS,
    k_docs_to_retrieve: int
) -> RetrievalQA:
    """Builds the RAG chain with enhanced query processing."""
    # Read prompt template from file
    with open(settings.PROMPT_PATH, "r") as file:
        prompt_template_str = file.read()
    
    QA_PROMPT = PromptTemplate.from_template(prompt_template_str)
    retriever = vector_store.as_retriever(search_kwargs={"k": k_docs_to_retrieve})
    
    custom_enhanced_retriever = EnhancedCustomRetriever(
        base_retriever=retriever,
        llm=help_model,
    )
    
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=custom_enhanced_retriever,
        chain_type_kwargs={
            "prompt": QA_PROMPT,
            "document_prompt": PromptTemplate.from_template("{page_content}")
        },
        return_source_documents=True
    )
    print("Enhanced RAG chain built successfully.")
    return rag_chain
    
class Rag:
    """RAG system for document retrieval and question answering.""" 
    llm_service_manager = None
    rag_chain_manager: Dict[str, RetrievalQA] = {}
    is_initialized: bool = False
    vector_store = None
    
    def __init__(self):
        print("Application startup: Initializing RAG system...")
        try:
            # 1. Initialize LLM Service (Ollama or OpenAI)
            self.llm_service_manager = LLMServiceManager(settings)
            embedding_model = self.llm_service_manager.get_embedding_model()
            help_model = self.llm_service_manager.get_help_model()

            # 2. Load and process documents
            if settings.RECRATE_KNOWLEDGE:
                split_docs = load_and_split_documents(
                    settings.KNOWLEDGE_BASE_DIR,
                    settings.CHUNK_SIZE,
                    settings.CHUNK_OVERLAP
                )
            else:
                print("No knowledge base documents to load. Skipping document loading.")
                split_docs = []

            # 3. Create or load vector store
            self.vector_store = get_vector_store(
                split_docs,
                embedding_model,
                settings.VECTOR_DB_PATH,
                force_recreate=settings.RECRATE_KNOWLEDGE
            )

            if not self.vector_store:
                print("Error: Vector store could not be initialized.")
                self.is_initialized = False
                return

            # 4. Build RAG chain
            self.rag_chain_manager = {name: build_rag_chain(llm, help_model, self.vector_store, settings.SEARCH_K_DOCS)
                            for name, llm in self.llm_service_manager.get_chat_models().items()}
            self.is_initialized = True
            print(f"RAG system initialized successfully.")

        except Exception as e:
            self.is_initialized = False
            print(f"FATAL: RAG system initialization failed: {e}")

    def get_models(self) -> List[str]:
        """Returns the list of available models."""
        return list(self.rag_chain_manager.keys())

    async def invoke(self, model_name: str, query: str):
        """Invokes the RAG chain for a given model and query."""
        rag_chain = self.rag_chain_manager.get(model_name)
        if not rag_chain:
            raise ValueError(f"Model {model_name} not found in RAG chain manager.")
        print(f"Invoking RAG chain for model: {model_name} with query: {query}")
        response = await rag_chain.arun(query)
        print(f"Response from RAG chain: {response}")
        return response
    def get_rag_chain(self, model_name: str) -> RetrievalQA:
        """Returns the RAG chain for a given model name."""
        return self.rag_chain_manager.get(model_name)