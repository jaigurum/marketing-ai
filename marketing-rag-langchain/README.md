# Marketing Insights RAG System (LangChain)

**RAG pipeline for answering questions about marketing campaign performance**

This project demonstrates a production-inspired Retrieval-Augmented Generation system that enables natural language Q&A over marketing documents, reports, and performance data.

---

## What It Does

```
Input:  Marketing documents (PDFs, reports, data exports) + User question
Output: Grounded answer with source citations
```

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│  Documents  →  Chunking  →  Embedding  →  Vector Store      │
│  (PDF/CSV)     (Semantic)   (OpenAI)      (ChromaDB)        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                          │
├─────────────────────────────────────────────────────────────┤
│  Question  →  Query      →  Retrieval  →  Reranking         │
│              Analysis       (Top-K)       (Relevance)       │
│                              │                               │
│                              ▼                               │
│              Context   →   LLM       →   Answer             │
│              Assembly      Generation    + Citations         │
└─────────────────────────────────────────────────────────────┘
```

---

## Production Parallel

This is a simplified version of the RAG system deployed for Colgate and Unilever:

| This Demo | Production System |
|-----------|-------------------|
| Local ChromaDB | AWS OpenSearch with 10M+ vectors |
| Sample marketing docs | Live AMC, Google Ads, Meta API data |
| Single-user | Multi-tenant with client isolation |
| Console output | Slack bot + Dashboard integration |
| Basic chunking | Semantic chunking with metadata preservation |

**Production Impact:** 85% user satisfaction, passed Unilever/Colgate security reviews

---

## Installation

```bash
cd marketing-rag-langchain
pip install -r requirements.txt
```

## Quick Start

### 1. Ingest Documents

```python
from marketing_rag import MarketingRAG

# Initialize
rag = MarketingRAG()

# Ingest documents
rag.ingest_documents("data/reports/")

# Or ingest specific files
rag.ingest_file("data/q4_campaign_report.pdf")
rag.ingest_file("data/performance_data.csv")
```

### 2. Ask Questions

```python
# Simple query
answer = rag.query("What was our ROAS on social campaigns in Q4?")
print(answer.content)
print(answer.sources)

# Query with filters
answer = rag.query(
    "Compare search vs display performance",
    filters={"channel": ["search", "display"]},
    top_k=10
)
```

### 3. Command Line

```bash
# Ingest documents
python main.py ingest --path data/reports/

# Interactive Q&A
python main.py chat

# Single query
python main.py query "What drove the ROAS improvement in November?"
```

---

## Project Structure

```
marketing-rag-langchain/
├── README.md
├── requirements.txt
├── main.py                      # CLI entry point
├── marketing_rag/
│   ├── __init__.py
│   ├── rag.py                   # Main RAG class
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py            # Document loaders
│   │   ├── chunker.py           # Text chunking strategies
│   │   └── metadata.py          # Metadata extraction
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vectorstore.py       # ChromaDB interface
│   │   ├── retriever.py         # Retrieval strategies
│   │   └── reranker.py          # Result reranking
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompts.py           # Prompt templates
│   │   └── chains.py            # LangChain chains
│   └── utils/
│       ├── __init__.py
│       └── citations.py         # Citation formatting
├── data/
│   └── sample_reports/          # Sample marketing documents
└── tests/
    └── test_rag.py
```

---

## Key LangChain Concepts Demonstrated

### 1. Document Loading & Chunking

```python
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load documents
loader = PyPDFLoader("report.pdf")
documents = loader.load()

# Semantic chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " "]
)
chunks = splitter.split_documents(documents)
```

### 2. Vector Store & Embeddings

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance
    search_kwargs={"k": 5, "fetch_k": 20}
)
```

### 3. RAG Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a marketing analytics expert. Answer questions 
    based ONLY on the provided context. If the context doesn't contain 
    the answer, say so.
    
    Context: {context}
    
    Always cite your sources using [Source: filename, page X] format."""),
    ("user", "{question}")
])

# RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Query
answer = rag_chain.invoke("What was our Q4 ROAS?")
```

### 4. Metadata Filtering

```python
# Add metadata during ingestion
for doc in documents:
    doc.metadata["channel"] = extract_channel(doc)
    doc.metadata["date"] = extract_date(doc)
    doc.metadata["report_type"] = classify_report(doc)

# Filter during retrieval
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"channel": "social", "date": {"$gte": "2024-10-01"}}
    }
)
```

---

## Sample Queries

```
"What was our ROAS across all channels in Q4?"
→ Retrieves performance summaries, calculates weighted average

"Why did social performance decline in November?"
→ Finds trend analysis, identifies contributing factors

"Compare our CPA to industry benchmarks"
→ Retrieves both internal data and benchmark reports

"What recommendations did the last campaign review include?"
→ Finds recommendation sections, summarizes action items

"How much did we spend on video advertising?"
→ Retrieves budget allocation data, sums across reports
```

---

## Extending the System

### Add Custom Document Types

```python
# In loader.py
class GoogleAdsReportLoader(BaseLoader):
    """Load Google Ads performance reports"""
    
    def load(self) -> List[Document]:
        # Parse Google Ads export format
        # Extract metrics, campaigns, ad groups
        # Return as Document objects with metadata
        pass
```

### Add Hybrid Search

```python
# Combine vector + keyword search
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(documents)
vector_retriever = vectorstore.as_retriever()

ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]
)
```

### Add Query Routing

```python
# Route to different retrievers based on query type
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

router_prompt = PromptTemplate.from_template("""
Classify this query into one of: performance, budget, creative, audience

Query: {query}
Classification:""")

router = router_prompt | llm | StrOutputParser()
```

---

## License

MIT
