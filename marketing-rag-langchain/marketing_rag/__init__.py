"""
Marketing RAG System

A production-inspired RAG pipeline for marketing analytics Q&A using LangChain.

Example:
    from marketing_rag import MarketingRAG, create_marketing_rag
    
    # Quick setup with documents
    rag = create_marketing_rag(documents_path="data/reports/")
    
    # Query
    answer = rag.query("What was our ROAS on social campaigns in Q4?")
    print(answer.content)
    print(answer.sources)
    
    # Or step by step
    rag = MarketingRAG()
    rag.ingest_file("data/q4_report.pdf")
    rag.ingest_documents("data/campaign_exports/")
    
    response = rag.query(
        "Compare search vs display performance",
        filters={"channels": ["search", "display"]}
    )
"""

from .rag import MarketingRAG, RAGResponse, create_marketing_rag

__version__ = "0.1.0"
__author__ = "Jaiguru Thevar"

__all__ = [
    "MarketingRAG",
    "RAGResponse",
    "create_marketing_rag"
]
