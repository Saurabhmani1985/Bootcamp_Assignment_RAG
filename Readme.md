#Bootcamp Assignment - Multimodal Rag - Automotive Application
# Bootcamp Assignment — Diagnostic Multimodal RAG

---

## 1. Domain Identification

**Domain:** Automotive Engineering — Diesel Engine ECU Fault Diagnostics

**Sub-domain:** Technical Document Intelligence / Condition-Based Maintenance Support

**Industry Context:**
The automotive service and repair industry relies heavily on technical
documentation to diagnose and resolve vehicle faults. This project
specifically targets common-rail diesel engine ECU (Engine Control Unit)
diagnostic manuals — a category of highly structured technical documents
that combine narrative text, tabular fault code data, and embedded
wiring schematic images within a single PDF.

---

## 2. Problem Description

Modern diesel engine vehicles use Engine Control Units to monitor hundreds
of sensor parameters simultaneously. When a parameter falls outside its
acceptable range, the ECU logs a Diagnostic Trouble Code (DTC), also
called a P-code (e.g., P0087, P0191, P0251). Each fault code maps to a
specific fault condition and requires a structured diagnostic procedure
to resolve it.

The reference document used in this project is an 86-page ECU Diagnostic
Manual for a 1.05L three-cylinder common-rail diesel engine (BS IV
emission specification), covering 129 distinct fault codes across all
major ECU subsystems including:

- Fuel system faults (rail pressure, injector circuit, IMV)
- Sensor faults (coolant temperature, pedal position, rail pressure sensor)
- Actuator faults (EGR valve, glow plugs, throttle body)
- Communication faults (CAN bus, immobiliser)

**The Core Problem:**
When a vehicle presents a fault code in a workshop, the service engineer
must:

1. Identify the correct fault code in the index table (Page 3–7)
2. Navigate to the specific fault code section (scattered across 86 pages)
3. Understand the fault condition, lamp status, and recovery mode
4. Cross-reference the environment variables table for that fault
5. Follow the diagnostic procedure — which may reference a wiring
   schematic on a completely different page
6. Interpret the wiring diagram to identify which physical ECU pin,
   connector, or wire to test

This process requires simultaneously reasoning over **text**, **tables**,
and **images** — three fundamentally different content types — across a
dense, unstructured PDF. For a junior technician, this may take 20–40
minutes per fault. For complex multi-fault scenarios, it can be longer.

There is currently no intelligent system that can answer the question
*"What does P0087 mean and what should I check?"* by retrieving and
synthesising information from all three content types in the document
simultaneously.

---

## 3. Why This Problem Is Unique

Most RAG systems are built for text-only documents — research papers,
legal contracts, product documentation. This problem is uniquely
challenging for three reasons:

**Reason 1 — True Multimodality Is Required**
The diagnostic answer for a single fault code may be distributed across:
- A text paragraph describing the fault condition (text chunk)
- A lookup table showing the lamp status and recovery mode (table chunk)
- A wiring schematic showing which ECU pin connects to which sensor
  terminal (image chunk)

A system that only searches text will miss the wiring diagram.
A system that only searches tables will miss the diagnostic procedure.
Only a system that can retrieve and reason across all three types
simultaneously gives a complete and actionable answer.

**Reason 2 — Images Cannot Be Searched Without VLM Pre-processing**
Wiring schematics are the most technically critical content in the
document. They show exact ECU pin numbers, wire colours, connector
designations, and sensor terminal assignments. This information exists
nowhere in the text. It is only in the images.

Standard RAG systems cannot search images. This system solves that by
passing every extracted image through Claude Vision before ingestion,
converting each schematic into a rich text description. The description
— containing specific pin numbers, sensor names, and connection details —
is then embedded and stored like any other chunk.

**Reason 3 — Structured Tables Require Preservation**
Automotive diagnostic manuals use tables extensively. The fault code
index table maps 129 P-codes to fault names, lamp statuses, and recovery
modes across multiple columns. A naive text chunker would destroy this
structure, breaking rows across chunks and losing column alignment.

This system uses pdfplumber specifically for table extraction, converting
each table to Markdown format that preserves row and column relationships.

---

## 4. Why RAG Is the Right Approach

**Why Not Fine-tuning?**
Fine-tuning a language model on this document would require training data
(question-answer pairs), significant compute resources, and would produce
a model frozen at the time of training. When the vehicle manufacturer
releases an updated diagnostic manual with revised fault procedures, the
fine-tuned model would be outdated. RAG retrieves from the current
document version at query time.

**Why Not Prompt Stuffing?**
The document is 86 pages. At approximately 500 words per page, that is
43,000 words — far exceeding what can be reliably processed in a single
prompt context window. Even if it fit, sending the entire document with
every query would be extremely expensive and slow.

**Why RAG Works Here:**
RAG (Retrieval-Augmented Generation) solves both problems:
- It stores all 312 chunks in a vector index, searchable in milliseconds
- At query time, only the 5–6 most relevant chunks (a few hundred words)
  are sent to Claude — keeping cost and latency low
- The retrieved chunks contain exactly the right information from text,
  tables, and images for that specific fault code query
- Claude generates a grounded, structured answer from real document content

**The Grounding Requirement:**
The SYSTEM_PROMPT in this implementation explicitly instructs Claude:
*"Base your answer EXCLUSIVELY on the provided context. If the answer is
not in the context, state that clearly."* This prevents hallucination —
Claude cannot invent fault procedures that do not exist in the document.

---

## 5. Expected Outcomes

After successfully completing this implementation, the system will be
able to answer the following classes of questions from the diagnostic
manual:

**Fault Code Lookup (Text Retrieval)**
> "What does fault code P0087 mean?"
> "What is the recovery mode when P0191 is active?"
> "Which lamp activates for the rail pressure over-pressure fault?"

**Procedure and Check Queries (Text Retrieval)**
> "What diagnostic checks should I perform for P0087?"
> "What are the steps to diagnose the IMV open circuit fault?"

**Fault Group Queries (Table Retrieval)**
> "List all fault codes in the STOP_ENGINE_GROUP_FAULT"
> "Which faults trigger torque reduction mode?"
> "Show me the environment variables table for P0251"

**Wiring and Connection Queries (Image Retrieval)**
> "Describe the ECU pin connections for the rail pressure sensor"
> "What ECU pins connect to the coolant temperature sensor?"
> "Which connector terminal is the IMV signal wire?"

**Cross-modal Queries (All Three Types)**
> "For fault P0087, what are the diagnostic steps and which ECU pin
>  should I check on the rail pressure sensor wiring?"

---

## 6. Architecture Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                           │
│                                                                 │
│  PDF File                                                       │
│     │                                                           │
│     ├──► PyMuPDF ──────────────► Text Pages ──► TextChunker ──┐│
│     │                                                          ││
│     ├──► pdfplumber ───────────► Tables ──────► Markdown ─────┤│
│     │                                                          ││
│     └──► PyMuPDF image extract ► PIL Images                   ││
│                                        │                       ││
│                                        ▼                       ││
│                              Claude Vision (Opus 4.5)          ││
│                              "ECU pin 40 connects to           ││
│                               Rail Pressure Signal..."         ││
│                                        │                       ││
│                                   Text Summary ───────────────►┤│
│                                                                 ││
│                                sentence-transformers           ││
│                                all-MiniLM-L6-v2               ││
│                                        ▼                       ││
│                              Embedding Vectors ◄───────────────┘│
│                                        │                        │
│                                        ▼                        │
│                              ChromaDB (persistent)              │
│                         {text | table | image chunks}           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                             │
│                                                                 │
│  User Question                                                  │
│     │                                                           │
│     ├──► sentence-transformers embed query                      │
│     │                                                           │
│     ├──► ChromaDB cosine similarity search                      │
│     │        ├── top-k text chunks  (score + page + filename)  │
│     │        ├── top-k table chunks (Markdown tables)          │
│     │        └── top-k image chunks (VLM descriptions)         │
│     │                                                           │
│     ├──► Context Assembly (custom RAG prompt template)          │
│     │        "## Text Excerpts\n[Source | page | score]\n..."  │
│     │                                                           │
│     └──► Claude Sonnet 4.6                                      │
│               │                                                 │
│               ▼                                                 │
│        Grounded Answer + Source References                      │
│        {answer, sources[{filename, page, type, score}]}        │
└─────────────────────────────────────────────────────────────────┘
```

### 6.1 Ingestion and Query Pipeline — Runtime Components
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   PDFParser  │────►│ TextChunker  │────►│   Embedder   │
│              │     │              │     │              │
│ PyMuPDF      │     │ Recursive    │     │ all-MiniLM   │
│ + pdfplumber │     │ Character    │     │ L6-v2        │
│ + VLM Vision │     │ Splitter     │     │ (local)      │
│              │     │ 800 tok/100  │     │ dim = 384    │
│ Output:      │     │ overlap      │     │              │
│  text chunks │     │              │     │ Output:      │
│  table chunks│     │ Tables/images│     │  np.ndarray  │
│  image chunks│     │ pass through │     │  (N, 384)    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Retriever  │◄────│ VectorStore  │◄────│  ChromaDB    │
│              │     │              │     │              │
│ 1. embed Q   │     │ upsert()     │     │ Persistent   │
│ 2. search    │     │ query()      │     │ HNSW index   │
│ 3. format ctx│     │ metadata     │     │ cosine sim   │
│ 4. generate  │     │ filtering    │     │              │
│              │     │              │     │ Filter by:   │
│ Output:      │     │ chunk_type   │     │  chunk_type  │
│  QueryResult │     │ source       │     │  source      │
└──────────────┘     └──────────────┘     └──────────────┘
        │
        ▼
┌──────────────┐     ┌──────────────┐
│  LLMClient   │────►│ Claude Sonnet│
│              │     │    4.6       │
│ SYSTEM_PROMPT│     │              │
│ RAG_TEMPLATE │     │ Grounded     │
│ format_ctx() │     │ answer with  │
│              │     │ source refs  │
└──────────────┘     └──────────────┘
```

---

## 7. Technology Choices

| Component | Choice | Justification |
|---|---|---|
| **PDF Text Parser** | PyMuPDF (fitz) | Fastest Python PDF library for text extraction. Handles complex layouts and multi-column pages reliably. Also provides direct access to embedded images via xref. |
| **PDF Table Parser** | pdfplumber | Purpose-built for table extraction with reliable cell boundary detection. Returns structured row/column data that PyMuPDF's text extraction would destroy. Both libraries used together for best-of-breed results. |
| **Vision Language Model** | Claude Opus 4.5 | Produces the most accurate technical descriptions of ECU wiring schematics — correctly identifying pin numbers, sensor labels, connector designations, and electrical connections. This is the critical quality gate: poor image descriptions mean poor image retrieval. |
| **Embedding Model** | sentence-transformers/all-MiniLM-L6-v2 | Runs entirely locally — no external API calls, no cost per embed, no latency. 384-dimensional vectors with fast cosine similarity. Strong semantic performance on technical English. All three chunk types (text, table, image summaries) share the same embedding space, enabling unified semantic search. |
| **Vector Store** | ChromaDB | Native metadata filtering by `chunk_type`, `source`, and `page` without needing a separate filtering layer. Persistent local storage out-of-the-box — no separate server process needed. FAISS was considered but lacks metadata filtering. Pinecone/Weaviate require external services. |
| **Answer Generator** | Claude Sonnet 4.6 | Excellent technical reasoning on automotive diagnostic content. Follows complex system prompts reliably. Cost-effective compared to Opus for generation. Custom RAG prompt template ensures domain-appropriate, grounded responses. |
| **Web Framework** | FastAPI (custom, no LangChain) | Full transparency and control over every pipeline step. Auto-generates OpenAPI/Swagger UI from Pydantic schemas. Async support for concurrent request handling. LangChain abstraction adds complexity without benefit at this scale. |
| **Async Runtime** | uvicorn + ThreadPoolExecutor | Standard production ASGI stack. CPU-bound ingestion tasks (parsing, embedding) run in a thread pool, keeping the async event loop free for concurrent requests. |

---

## 8. Setup Instructions

### Clone Repository
git clone https://github.com/YOUR-USERNAME/Bootcamp_Assignment_Diagnostic_Multimodal_Rag.git
cd Bootcamp_Assignment_Diagnostic_Multimodal_Rag
```

### Create Environment

pip install --upgrade pip
pip install -r requirements.txt

```

### Configure Environment Variables

cp .env.example .env

```
ANTHROPIC_API_KEY=sk-ant-api03-YOUR-REAL-KEY-HERE
```

### Run FastAPI Server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — ingest the sample PDF
curl -X POST http://localhost:8000/ingest \
  -F "file=@sample_documents/Diagnostic_Document.pdf" \
  -F "collection=diagnostic_rag" \
  -F "use_vision=true"

# Query — text retrieval
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does fault code P0087 mean?"}'

# Query — table retrieval only
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "List STOP_ENGINE group faults",
       "chunk_type_filter": "table"}'

# Query — image retrieval only (VLM wiring descriptions)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "ECU pin connections for rail pressure sensor",
       "chunk_type_filter": "image"}'
```

### Open API Docs

```
http://localhost:8000/docs
```

## 9. API Documentation

### GET /health

Returns system status for all components, server uptime, number of indexed
documents, total chunk count, and approximate index size on disk.

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "uptime_seconds": 142.3,
  "indexed_documents": 1,
  "total_chunks": 312,
  "index_size_mb": 4.2,
  "components": {
    "vector_store": {"status": "ok", "detail": "ChromaDB 312 chunks"},
    "embedding_model": {"status": "ok", "detail": "all-MiniLM-L6-v2 dim=384"},
    "llm": {"status": "ok", "detail": "claude-sonnet-4-6"},
    "vision_model": {"status": "ok", "detail": "claude-opus-4-5"}
  }
}
```

---

### POST /ingest

Upload a PDF document. The pipeline extracts text, tables, and images
(each image is summarised by Claude Vision), embeds all chunks using
sentence-transformers, and stores them in ChromaDB.

**Request:** `multipart/form-data`

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | PDF to ingest (max 100 MB) |
| `collection` | string | `diagnostic_rag` | ChromaDB collection name |
| `use_vision` | boolean | `true` | Run images through Claude Vision |

**Response:**
```json
{
  "status": "success",
  "filename": "Diagnostic_Document.pdf",
  "collection": "diagnostic_rag",
  "total_pages": 86,
  "chunk_counts": {
    "text": 248,
    "table": 41,
    "image": 23,
    "total": 312
  },
  "processing_time_s": 94.7,
  "warnings": [],
  "message": "Indexed 312 chunks from 86 pages."
}
```

---

### POST /query

Submit a natural language question. The system embeds the question,
retrieves the top-k most similar chunks from ChromaDB, assembles a
structured context, and generates a grounded answer via Claude Sonnet.

**Request:**
```json
{
  "question": "What does fault code P0087 mean?",
  "collection": "diagnostic_rag",
  "top_k": 6,
  "chunk_type_filter": null
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `question` | string | required | Natural language question (3–2000 chars) |
| `collection` | string | `diagnostic_rag` | Collection to query |
| `top_k` | integer | 6 | Chunks to retrieve (1–20) |
| `chunk_type_filter` | string or null | null | Restrict to `"text"`, `"table"`, or `"image"` |

**Response:**
```json
{
  "question": "What does fault code P0087 mean?",
  "answer": "Fault code `P0087` is the **Rail Pressure Build Fault**...",
  "sources": [
    {
      "filename": "Diagnostic_Document.pdf",
      "page": 28,
      "chunk_type": "text",
      "score": 0.912,
      "excerpt": "P0087 Rail Pressure Build Error. Rail pressure does not..."
    }
  ],
  "retrieval_stats": {
    "total_retrieved": 6,
    "by_type": {"text": 4, "table": 2},
    "collection": "diagnostic_rag",
    "top_k_requested": 6,
    "llm_input_tokens": 1842,
    "llm_output_tokens": 387,
    "llm_latency_ms": 2140.5
  },
  "latency_ms": 2341.2,
  "model": "claude-sonnet-4-6-..."
}
```

---

### GET /docs

FastAPI auto-generated interactive Swagger UI. Accessible at:
```
http://localhost:8000/docs
```
---

## 10. Screenshots

The screenshots below are captured from the live local API and stored in the `screenshots/` folder.

### 13. Swagger UI
`/docs` page showing all endpoints.

![Swagger UI -- /docs](screenshots/01_swagger_ui.png)

### 14. Successful Ingestion
`POST /ingest` call using the multimodal PDF and its response.

![Successful Ingestion -- POST /ingest](screenshots/02_ingest_response.png)

### 15. Text Query Result
Query retrieving text-based chunks.

![Text Query Result](screenshots/03_text_query.png)

### 16. Table Query Result
Query retrieving table-based chunks.

![Table Query Result](screenshots/04_table_query.png)

### 17. Image Query Result
Query retrieving image-summary chunks.

![Image Query Result](screenshots/05_image_query.png)

### 18. Health Endpoint
`GET /health` response showing indexed document count.

![Health Endpoint](screenshots/06_health.png)

---

## 11. Limitations and Future Work

### Current Limitations

**Ingestion Time**
Processing the 86-page sample document takes 90–150 seconds because
each of the 23 images makes a separate Claude Vision API call
(approximately 3–5 seconds per image). This is acceptable for batch
ingestion but would be too slow for real-time uploads.

**Character-Based Chunking**
Text is split using a character-count heuristic. For diagnostic manuals,
a better approach would be semantic or structure-aware chunking that
keeps an entire fault code entry (code + environment table + recovery
mode + diagnostic checks) as one chunk rather than splitting it.

**No Re-ranking**
Retrieved chunks are ranked purely by cosine similarity. A cross-encoder
re-ranker as a second stage would significantly improve precision for
queries that have many partially-relevant results.

**No Conversation Memory**
Each query is independent. A diagnostic session in the real world is
iterative: "What does P0087 mean?" → "What should I check first?" →
"The IMV resistance is 8 ohms — is that within spec?" The system
currently cannot maintain this context between queries.

**OCR Not Implemented**
Scanned PDFs (raster pages with no text layer) would produce empty
text extraction. Adding Tesseract OCR or using Claude Vision on every
page would handle scanned documents.

### Future Work

1. **Streaming responses** — stream Claude's answer token-by-token via
   Server-Sent Events for better perceived performance
2. **Semantic chunking** — detect fault code section boundaries and
   keep complete fault entries as single chunks
3. **Conversation history** — add session ID and rolling message history
   for multi-turn diagnostic workflows
4. **Cross-encoder re-ranking** — second-stage result improvement using
   `cross-encoder/ms-marco-MiniLM-L-6-v2`
5. **Multi-document corpus** — query across multiple manuals with
   source-level metadata filtering
6. **Evaluation suite** — RAGAS evaluation dataset with
   question/expected-answer/expected-source-page triples
7. **Parallel VLM processing** — batch image summarisation using
   asyncio to reduce ingestion latency
8. **Authentication** — API key middleware for production deployment

---

## 12. Repository Layout
```
Bootcamp_Assignment_Diagnostic_Multimodal_Rag/
│
├── main.py                        ← FastAPI app entry point
│                                    Lifespan, CORS, timing middleware
│
├── requirements.txt               ← Pinned Python dependencies
├── pyproject.toml                 ← Project metadata and pytest config
├── .env.example                   ← Configuration template (safe to commit)
├── .env                           ← Real secrets (blocked by .gitignore)
├── .gitignore                     ← Patterns excluded from Git
├── README.md                      ← This document
│
├── sample_documents/
│   └── Diagnostic_Document.pdf   ← 86-page diesel ECU manual (input data)
│
├── screenshots/
│   ├── 01_swagger_ui.png
│   ├── 02_ingest_response.png
│   ├── 03_text_query.png
│   ├── 04_table_query.png
│   ├── 05_image_query.png
│   └── 06_health.png
│
└── src/
    │
    ├── config.py                  ← Pydantic-settings; reads .env
    │                                All settings typed and validated
    │
    ├── models/
    │   ├── vision.py              ← Claude Vision VLM wrapper
    │   │                            Images → text summaries (at ingest)
    │   └── llm.py                 ← Claude Sonnet wrapper
    │                                SYSTEM_PROMPT + RAG_PROMPT_TEMPLATE
    │                                format_context() groups by chunk type
    │
    ├── ingestion/
    │   ├── pdf_parser.py          ← Multimodal PDF extraction
    │   │                            PyMuPDF (text + images)
    │   │                            pdfplumber (tables → Markdown)
    │   │                            VisionModel.summarise() per image
    │   ├── chunker.py             ← RecursiveCharacterTextSplitter
    │   │                            Text only; tables/images pass through
    │   ├── embedder.py            ← all-MiniLM-L6-v2 (local, no API cost)
    │   │                            Unified space for all chunk types
    │   └── pipeline.py            ← Orchestrator: parse→chunk→embed→store
    │                                async wrapper for FastAPI
    │
    ├── retrieval/
    │   ├── vector_store.py        ← ChromaDB PersistentClient wrapper
    │   │                            upsert(), query(), metadata filtering
    │   │                            list_indexed_documents(), delete()
    │   └── retriever.py           ← Full RAG chain
    │                                embed query → retrieve → format →
    │                                generate → return QueryResult
    │
    └── api/
        ├── schemas.py             ← Pydantic v2 request/response models
        │                            Powers OpenAPI/Swagger auto-docs
        └── routes.py              ← All 5 FastAPI route handlers
                                     GET  /health
                                     POST /ingest
                                     POST /query
                                     GET  /documents
                                     DELETE /delete
```

---
