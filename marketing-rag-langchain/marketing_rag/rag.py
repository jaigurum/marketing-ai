"""
Marketing RAG System

Production-inspired RAG pipeline for marketing analytics Q&A.
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document


@dataclass
class RAGResponse:
    """Response from RAG query"""
    content: str
    sources: List[Dict[str, Any]]
    confidence: float
    tokens_used: int


class MarketingRAG:
    """
    Marketing-focused RAG system using LangChain.
    
    Features:
    - Multi-format document ingestion (PDF, CSV, TXT)
    - Semantic chunking with metadata preservation
    - MMR retrieval for diverse results
    - Source citation in responses
    
    Usage:
        rag = MarketingRAG()
        rag.ingest_documents("data/reports/")
        answer = rag.query("What was our Q4 ROAS?")
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model: str = "text-embedding-3-small",
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the RAG system.
        
        Args:
            persist_directory: Where to store the vector database
            embedding_model: OpenAI embedding model to use
            llm_provider: "openai" or "anthropic"
            llm_model: Specific model name (default: gpt-4 or claude-3-sonnet)
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Initialize LLM
        if llm_provider == "openai":
            self.llm = ChatOpenAI(
                model=llm_model or "gpt-4",
                temperature=0.1
            )
        elif llm_provider == "anthropic":
            self.llm = ChatAnthropic(
                model=llm_model or "claude-3-sonnet-20240229",
                temperature=0.1
            )
        else:
            raise ValueError(f"Unknown LLM provider: {llm_provider}")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " "],
            length_function=len
        )
        
        # Initialize or load vector store
        self.vectorstore = self._init_vectorstore()
        
        # Build RAG chain
        self.rag_chain = self._build_rag_chain()
    
    def _init_vectorstore(self) -> Chroma:
        """Initialize or load the vector store."""
        if os.path.exists(self.persist_directory):
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
    
    def _build_rag_chain(self):
        """Build the RAG chain with prompt and retriever."""
        
        # System prompt for marketing analytics
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior marketing analytics expert helping stakeholders 
understand campaign performance and make data-driven decisions.

INSTRUCTIONS:
1. Answer questions based ONLY on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Always cite your sources using [Source: filename] format
4. When discussing metrics, be precise with numbers
5. Highlight any caveats or limitations in the data
6. If asked for recommendations, base them on the data provided

CONTEXT:
{context}

Remember: Never make up data. If you're uncertain, express that uncertainty."""),
            ("user", "{question}")
        ])
        
        # Create retriever with MMR for diversity
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 20,
                "lambda_mult": 0.7  # Balance relevance vs diversity
            }
        )
        
        # Build chain
        chain = (
            {
                "context": retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents for context."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            
            source_info = f"[Source {i}: {Path(source).name}"
            if page:
                source_info += f", Page {page}"
            source_info += "]"
            
            formatted.append(f"{source_info}\n{doc.page_content}")
        
        return "\n\n---\n\n".join(formatted)
    
    def ingest_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Ingest a single file into the vector store.
        
        Args:
            file_path: Path to the file
            metadata: Additional metadata to attach to chunks
            
        Returns:
            Number of chunks created
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Select appropriate loader
        suffix = path.suffix.lower()
        
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix == ".csv":
            loader = CSVLoader(str(path))
        elif suffix in [".txt", ".md"]:
            loader = TextLoader(str(path))
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        # Load and split
        documents = loader.load()
        
        # Add custom metadata
        if metadata:
            for doc in documents:
                doc.metadata.update(metadata)
        
        # Extract marketing-specific metadata
        for doc in documents:
            doc.metadata.update(self._extract_marketing_metadata(doc))
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Add to vector store
        self.vectorstore.add_documents(chunks)
        
        print(f"Ingested {len(chunks)} chunks from {path.name}")
        return len(chunks)
    
    def ingest_documents(self, directory: str, glob: str = "**/*.*") -> int:
        """
        Ingest all documents from a directory.
        
        Args:
            directory: Path to directory
            glob: Glob pattern for files
            
        Returns:
            Total number of chunks created
        """
        path = Path(directory)
        
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        total_chunks = 0
        
        # Find all supported files
        for file_path in path.glob(glob):
            if file_path.suffix.lower() in [".pdf", ".csv", ".txt", ".md"]:
                try:
                    chunks = self.ingest_file(str(file_path))
                    total_chunks += chunks
                except Exception as e:
                    print(f"Error ingesting {file_path}: {e}")
        
        print(f"\nTotal: {total_chunks} chunks from {directory}")
        return total_chunks
    
    def _extract_marketing_metadata(self, doc: Document) -> Dict[str, Any]:
        """
        Extract marketing-relevant metadata from document content.
        
        Looks for:
        - Channel mentions (search, social, display, etc.)
        - Date references
        - Metric types (ROAS, CPA, CTR, etc.)
        - Campaign names
        """
        content = doc.page_content.lower()
        metadata = {}
        
        # Detect channels
        channels = []
        channel_keywords = {
            "search": ["search", "sem", "ppc", "google ads"],
            "social": ["social", "facebook", "instagram", "meta", "tiktok"],
            "display": ["display", "banner", "programmatic", "dv360"],
            "video": ["video", "youtube", "ctv", "ott"],
            "email": ["email", "newsletter", "crm"]
        }
        
        for channel, keywords in channel_keywords.items():
            if any(kw in content for kw in keywords):
                channels.append(channel)
        
        if channels:
            metadata["channels"] = channels
        
        # Detect metrics mentioned
        metrics = []
        metric_keywords = ["roas", "cpa", "cpc", "ctr", "cvr", "roi", "impressions", "clicks", "conversions"]
        
        for metric in metric_keywords:
            if metric in content:
                metrics.append(metric)
        
        if metrics:
            metadata["metrics_mentioned"] = metrics
        
        # Detect report type
        if any(word in content for word in ["performance", "results", "analysis"]):
            metadata["report_type"] = "performance"
        elif any(word in content for word in ["budget", "allocation", "spend"]):
            metadata["report_type"] = "budget"
        elif any(word in content for word in ["audience", "segment", "targeting"]):
            metadata["report_type"] = "audience"
        
        return metadata
    
    def query(
        self,
        question: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> RAGResponse:
        """
        Query the RAG system.
        
        Args:
            question: Natural language question
            filters: Metadata filters for retrieval
            top_k: Number of documents to retrieve
            
        Returns:
            RAGResponse with answer and sources
        """
        # Get retriever (with filters if specified)
        if filters:
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": top_k,
                    "fetch_k": top_k * 4,
                    "filter": filters
                }
            )
            
            # Rebuild chain with filtered retriever
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a senior marketing analytics expert helping stakeholders 
understand campaign performance and make data-driven decisions.

INSTRUCTIONS:
1. Answer questions based ONLY on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Always cite your sources using [Source: filename] format
4. When discussing metrics, be precise with numbers
5. Highlight any caveats or limitations in the data

CONTEXT:
{context}"""),
                ("user", "{question}")
            ])
            
            chain = (
                {
                    "context": retriever | self._format_docs,
                    "question": RunnablePassthrough()
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            chain = self.rag_chain
        
        # Get relevant documents for source tracking
        docs = self.vectorstore.similarity_search(question, k=top_k)
        
        # Generate answer
        answer = chain.invoke(question)
        
        # Extract sources
        sources = []
        for doc in docs:
            sources.append({
                "file": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page"),
                "channels": doc.metadata.get("channels", []),
                "preview": doc.page_content[:200] + "..."
            })
        
        # Estimate confidence based on retrieval scores
        # (In production, we'd use actual similarity scores)
        confidence = min(0.95, 0.6 + (len(docs) * 0.07))
        
        return RAGResponse(
            content=answer,
            sources=sources,
            confidence=confidence,
            tokens_used=0  # Would track in production
        )
    
    def clear(self):
        """Clear the vector store."""
        self.vectorstore.delete_collection()
        self.vectorstore = self._init_vectorstore()
        print("Vector store cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        collection = self.vectorstore._collection
        
        return {
            "total_documents": collection.count(),
            "persist_directory": self.persist_directory,
            "embedding_model": "text-embedding-3-small",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }


# Convenience function
def create_marketing_rag(
    documents_path: Optional[str] = None,
    **kwargs
) -> MarketingRAG:
    """
    Create and optionally populate a MarketingRAG instance.
    
    Args:
        documents_path: Optional path to documents to ingest
        **kwargs: Arguments passed to MarketingRAG
        
    Returns:
        Initialized MarketingRAG instance
    """
    rag = MarketingRAG(**kwargs)
    
    if documents_path:
        rag.ingest_documents(documents_path)
    
    return rag
