# Agentic RAG System

An intelligent Retrieval-Augmented Generation (RAG) system built with LangGraph that uses a **plan-execute-replan** architecture to answer questions from book content. The system intelligently breaks down complex queries into executable steps, retrieves relevant information from multiple sources, and dynamically replans based on accumulated context.

## 🎯 Features

- **Intelligent Planning**: Breaks down user questions into step-by-step execution plans
- **Multi-Source Retrieval**: Searches both book chunks and quotes for comprehensive answers
- **Dynamic Replanning**: Adapts execution strategy based on retrieved context
- **Context Aggregation**: Accumulates information across multiple retrieval steps
- **Hallucination Detection**: Validates answers against retrieved context
- **Hash-Based Deduplication**: Efficient ingestion pipeline that prevents re-processing of documents
- **Persistent Storage**: Qdrant vector database with persistent data storage

## 🏗️ Architecture

### System Components

1. **Vector Database**: Qdrant for storing and retrieving embeddings
2. **Embeddings**: Jina Cloud API for generating text embeddings
3. **LLM**: Local Ollama (Llama 3.1 8B) for planning, task handling, and answer generation
4. **Graph Workflow**: LangGraph for orchestrating the agent workflow

### Workflow Overview

The system follows a sophisticated plan-execute-replan cycle:

1. **Planning**: Analyzes the question and creates an initial execution plan
2. **Plan Refinement**: Breaks down the plan into executable retrieval/QA tasks
3. **Task Handling**: Decides which tool to use (chunks, quotes, or context-based QA)
4. **Retrieval**: Fetches relevant information from vector stores
5. **Replanning**: Updates the plan based on accumulated context
6. **Answer Generation**: Produces final answer when sufficient context is gathered
7. **Validation**: Checks if answers are grounded in retrieved context

### LangGraph Workflow

<!-- TODO: Add workflow diagram image here -->
![LangGraph Workflow](docs/workflow-diagram.png)

*The workflow diagram shows the complete agent execution flow with all nodes and conditional edges.*

## 📁 Project Structure

```
agentic-rag/
├── app/
│   ├── config.py                 # Configuration settings
│   ├── data/                      # Source documents (PDFs)
│   ├── scripts/
│   │   ├── ingestion.py          # Data ingestion pipeline
│   │   └── preprocessor.py       # Document preprocessing
│   ├── src/agent/
│   │   ├── graph.py              # Main LangGraph definition
│   │   └── utils/
│   │       ├── nodes.py          # Graph node implementations
│   │       ├── prompts.py        # LLM prompt templates
│   │       ├── state.py          # State schemas
│   │       ├── tools.py          # Retrieval tools
│   │       ├── retrieval_nodes.py  # Retrieval workflow nodes
│   │       └── workflow.py       # Workflow builders
│   └── utils/
│       └── embeddings.py         # Jina embeddings wrapper
├── docker-compose.yml            # Qdrant container configuration
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installation
└── .env                          # Environment variables (create this)

```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose
- Ollama installed locally ([Install here](https://ollama.ai/))
- Jina API key ([Get one here](https://jina.ai/))

### Step 1: Clone and Navigate

```bash
cd /Users/syedtaha/Desktop/agentic-rag
```

### Step 2: Install and Setup Ollama

Install Ollama from [ollama.ai](https://ollama.ai/) and pull the Llama 3.1 8B model:

```bash
# Install Ollama (if not already installed)
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.ai/install.sh | sh
# Windows: Download from https://ollama.ai/

# Pull the Llama 3.1 8B model
ollama pull llama3.1:8b

# Verify installation
ollama list
```

**Note**: Make sure Ollama is running before starting the application. The default port is `11434`.

### Step 3: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install --upgrade pip setuptools
pip install -r requirements.txt
pip install -e .  # Install project in editable mode
```

### Step 5: Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Required API Keys
JINA_API_KEY=your_jina_api_key_here

# Optional Configuration (defaults provided)
QDRANT_URL=http://localhost:6333
OLLAMA_BASE_URL=http://localhost:11434  # Ollama API endpoint
BATCH_SIZE=10
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MIN_QUOTE_LENGTH=50
```

**Note**: No Groq API key needed - we're using local Ollama!

### Step 6: Start Qdrant Database

```bash
docker-compose up -d
```

Verify Qdrant is running:
```bash
curl http://localhost:6333/health
```

### Step 7: Ingest Documents

First-time setup (creates collections and ingests data):
```bash
python -m app.scripts.ingestion --setup --cleanup --data-type all
```

Subsequent runs (only new documents will be ingested):
```bash
python -m app.scripts.ingestion --data-type all
```

**Ingestion Options:**
- `--setup`: Create collections before ingestion
- `--cleanup`: Delete existing collections first
- `--data-type`: Choose `chunks`, `quotes`, or `all`
- `--batch-size`: Override default batch size

## 🎮 Running the Application

### Prerequisites Before Starting

Make sure both services are running:

1. **Qdrant** (if not already running):
   ```bash
   docker-compose up -d
   ```

2. **Ollama** (verify it's running):
   ```bash
   ollama list  # Should show llama3.1:8b if working
   ```

### Start LangGraph Server

From the `app/` directory:

```bash
cd app
langgraph dev
```

The server will start at `http://localhost:8123`

**Note**: The agent uses local Ollama, so make sure it's running before making requests!

### Using the API

Once the server is running, you can interact with the agent:

**Via HTTP:**
```bash
curl -X POST http://localhost:8123/threads \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "agent",
    "input": {
      "question": "What is the main theme of the book?"
    }
  }'
```

**Via Python:**
```python
from app.src.agent.graph import graph

result = graph.invoke({
    "question": "What is the main theme of the book?"
})

print(result["response"])
```

## 🔧 Configuration

### Key Configuration Options

Edit `app/config.py` or set environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `JINA_API_KEY` | Required | Jina embeddings API key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `VECTOR_SIZE` | `1024` | Embedding dimension (Jina v3) |
| `BATCH_SIZE` | `10` | Documents per batch during ingestion |
| `CHUNK_SIZE` | `1000` | Text chunk size for splitting |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |

## 📊 Data Ingestion Pipeline

### Features

- **Hash-Based Deduplication**: Uses MD5 hashing to prevent re-ingestion of existing documents
- **Batch Processing**: Efficient batch processing with configurable batch sizes
- **Progress Tracking**: Real-time progress updates during ingestion
- **Multiple Collections**: Supports separate collections for chunks and quotes

### How It Works

1. **Preprocessing**: Extracts text from PDF, splits into chunks, and extracts quotes
2. **Hash Generation**: Computes MD5 hash for each document
3. **Duplicate Check**: Fetches existing hashes from Qdrant (fast, no API calls)
4. **Embedding**: Only new documents are embedded (saves API costs)
5. **Storage**: Documents stored with metadata including content hash

### Example Usage

```bash
# Ingest only chunks
python -m app.scripts.ingestion --data-type chunks

# Ingest only quotes
python -m app.scripts.ingestion --data-type quotes

# Ingest everything with custom batch size
python -m app.scripts.ingestion --data-type all --batch-size 20
```

## 🔍 Understanding the Workflow

### State Schema

The `PlanExecute` state contains:

- `question`: Original user question
- `plan`: List of execution steps
- `past_steps`: Completed steps history
- `aggregated_context`: Accumulated context from retrievals
- `curr_context`: Current step's context
- `tool`: Selected tool (retrieve_chunks, retrieve_quotes, answer_from_context)
- `response`: Final answer

### Node Descriptions

1. **planner_node**: Creates initial execution plan from question
2. **break_down_plan_node**: Refines plan into executable retrieval/QA tasks
3. **task_handler_node**: Selects appropriate tool for current task
4. **retrieve_chunks**: Retrieves relevant book chunks
5. **retrieve_quotes**: Retrieves relevant book quotes
6. **answer_question_from_context_node**: Answers question using aggregated context
7. **replanner_node**: Updates plan based on progress and context
8. **get_final_answer_node**: Generates final response
9. **can_question_be_answered**: Conditional check if enough context exists
10. **is_answer_grounded_on_context**: Validates answer against context

## 🐛 Troubleshooting

### Qdrant Connection Refused

**Error**: `Connection refused` when starting LangGraph server

**Solution**:
```bash
# Check if Qdrant is running
docker-compose ps

# Start Qdrant if not running
docker-compose up -d

# Verify health
curl http://localhost:6333/health
```

### Missing API Keys

**Error**: `JINA_API_KEY environment variable is not set`

**Solution**: Create `.env` file in project root with your Jina API key (see Step 5 above)

### Ollama Connection Issues

**Error**: `Connection refused` or `Ollama not responding`

**Solution**:
```bash
# Check if Ollama is running
ollama list

# If not running, start Ollama service
# macOS/Linux: ollama serve (or it may run as a service)
# Windows: Start Ollama from Start Menu

# Verify Ollama is accessible
curl http://localhost:11434/api/tags
```

### Ollama Model Not Found

**Error**: Model `llama3.1:8b` not found

**Solution**:
```bash
# Pull the required model
ollama pull llama3.1:8b

# Verify it's available
ollama list
```

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'app'`

**Solution**:
```bash
# Install project in editable mode
pip install -e .
```

### Collection Not Found

**Error**: Collection doesn't exist when running agent

**Solution**:
```bash
# Create collections and ingest data
python -m app.scripts.ingestion --setup --data-type all
```

### Dependency Conflicts

**Error**: Version conflicts during installation

**Solution**:
```bash
# Upgrade pip and setuptools first
pip install --upgrade pip setuptools

# Then install requirements
pip install -r requirements.txt
```

## 📝 Development

### Running Tests

```bash
# Add test commands here when tests are added
pytest tests/
```

### Code Formatting

```bash
ruff check app/
ruff format app/
```

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues and questions, please open an issue on the repository.

---

**Built with**: LangGraph, Qdrant, Jina AI, Ollama (Llama 3.1 8B), and Python 🚀

