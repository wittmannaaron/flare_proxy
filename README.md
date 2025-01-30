# FLARE-Proxy for AnythingLLM (and other RAG systems)

This project implements a FLARE (Forward-Looking Active REtrieval) proxy for AnythingLLM. The proxy enhances the standard RAG (Retrieval Augmented Generation) process by dynamically retrieving additional context when the LLM's confidence falls below a certain threshold.

## Project Overview

The FLARE-Proxy acts as an intermediary between AnythingLLM and the configured LLM service. It implements the OpenAI API specification to seamlessly integrate with AnythingLLM while adding FLARE capabilities to improve response quality.

### Key Features

- Compatible with AnythingLLM's API expectations
- Implements FLARE algorithm for dynamic context retrieval
- Modular architecture supporting multiple retrieval sources
- Streaming support for real-time responses
- Configurable confidence thresholds
- Extensible retriever interface

## Architecture

The implementation consists of four main components, each handling specific responsibilities:

### 1. API Layer (`main.py`)
- FastAPI server implementing OpenAI's chat completions endpoint
- Handles request/response formatting and error management
- Manages environment configuration through .env file
- Provides OpenAI-compatible interface for AnythingLLM integration

```python
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # Implementation
```

### 2. FLARE Engine (`flare/engine.py`)
- Implements core FLARE algorithm for dynamic context retrieval
- Manages confidence scoring for LLM responses
- Coordinates between LLM client and retrievers
- Handles response generation with additional context

Note: The implementation uses a configurable embedding model (defaulting to OpenAI's text-embedding-3-large) for generating embeddings, which are then used by ChromaDB for vector similarity search. For future implementations with other databases (e.g., SQLite), this embedding generation functionality can be reused.

Key interfaces:
```python
class FlareProcessor:
    async def process_message()
    async def get_prediction_and_scores()
```

### 3. Retrieval System (`retrievers/`)

The retrieval system uses a modular architecture for document retrieval:

#### Base Interface (`retrievers/base.py`)
```python
class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str) -> List[str]:
        pass
```

#### Implemented Retrievers
- ChromaRetriever (`retrievers/chroma.py`): Interfaces with AnythingLLM's ChromaDB
  - Connects to ChromaDB endpoint
  - Generates embeddings using OpenAI's text-embedding-3-large model
  - Performs semantic search with configurable similarity thresholds
  - Processes and filters documents based on relevance scores
  - Handles JSON document parsing and content extraction
  - Manages multiple collections with automatic relevance filtering
  - Provides comprehensive error handling and logging

### 4. LLM Interface (`llm/anthropic_client.py`)
- Implements Anthropic Claude API integration
- Handles message formatting and API communication
- Manages model responses and confidence scoring
- Provides async interface for FLARE engine

## Development Roadmap

1. Phase 1: Completed ✓
   - Set up project structure
   - Implement API layer
   - Basic FLARE processing
   - Chroma integration
   - Streaming support
   - Configuration management
   - Error handling
   - Logging system

2. Phase 2: In Progress
   - Complete confidence scoring implementation
   - Add SQLite retriever
   - Implement caching layer
   - Add performance optimizations
   - Enhance error recovery mechanisms

3. Phase 3: Planned
   - Multi-threading support
   - Additional LLM providers
   - Advanced retrieval strategies
   - Real-time monitoring dashboard
   - Automated testing suite

## Getting Started

### Prerequisites
- Python 3.8+
- FastAPI
- ChromaDB client
- Transformers library
- Required LLM dependencies

### Installation
```bash
# Clone the repository
git clone [repository-url]

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Configuration

The proxy requires several environment variables to be set in the .env file:

```env
# LLM Configuration
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-anthropic-api-key
MODEL_NAME=claude-3-sonnet-20240229

# OpenAI Configuration (for embeddings)
OPEN_AI_KEY=your-openai-key
EMBEDDING_MODEL_PREF=text-embedding-3-large

# Chroma Configuration
CHROMA_ENDPOINT=http://localhost:1523  # AnythingLLM's ChromaDB endpoint
EMBEDDING_MODEL_PREF=text-embedding-3-large  # Model to use for embeddings
N_RESULTS=3        # Number of results to fetch per collection
MAX_RESULTS=5      # Maximum total results to return
DISTANCE_THRESHOLD=0.5  # Similarity threshold for filtering results

# Proxy Settings
PORT=3128  # Standard proxy port
HOST=0.0.0.0
LOG_PATH=flare_proxy.log

# FLARE Configuration
CONFIDENCE_THRESHOLD=0.7  # Threshold for additional context retrieval
MAX_RETRIEVAL_ROUNDS=3   # Maximum number of retrieval attempts
```

Note: The proxy uses OpenAI's text-embedding-3-large model for generating embeddings to match AnythingLLM's vector search capabilities.

### Running the Proxy

```bash
# Activate virtual environment if not already activated
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start the proxy server in normal mode
python main.py

# Start the proxy server in debug mode
python main.py --debug
```

### Logging

The proxy implements comprehensive logging with two modes:

1. Operational Logging (Default)
   - Logs are written to the path specified in LOG_PATH
   - Records important operational events:
     * Server startup and shutdown
     * Component initialization
     * Request processing
     * Error conditions
   - Format: `timestamp - level - message`

2. Debug Logging (With --debug flag)
   - Includes all operational logs
   - Additional detailed information:
     * Request and response data
     * Component initialization details
     * Processing steps and decisions
     * ChromaDB interactions
   - Output to both file and console
   - Format: `timestamp - level - [file:line] - message`

Example log output:
```
# Normal mode
2024-03-14 10:15:30 - INFO - Initializing FLARE Proxy components...
2024-03-14 10:15:31 - INFO - All components initialized successfully
2024-03-14 10:15:32 - INFO - Received chat completion request
2024-03-14 10:15:33 - INFO - Successfully processed chat completion request

# Debug mode
2024-03-14 10:15:30 - INFO - [main.py:45] - Initializing FLARE Proxy components...
2024-03-14 10:15:30 - DEBUG - [main.py:50] - Initialized Anthropic client
2024-03-14 10:15:30 - DEBUG - [main.py:53] - Initialized ChromaDB retriever
2024-03-14 10:15:30 - DEBUG - [main.py:55] - ChromaDB connection established
2024-03-14 10:15:30 - INFO - [main.py:63] - All components initialized successfully
```

The proxy validates the log directory specified in LOG_PATH during startup:
- Creates the directory if it doesn't exist
- Verifies write permissions
- Exits with an error if the directory cannot be accessed

### Integration with AnythingLLM

To use the FLARE proxy with AnythingLLM, configure the following settings in AnythingLLM:

1. LLM Provider: Select "OpenAI" (The proxy implements OpenAI's API specification)
2. Base URL: Set to `http://localhost:3128`
3. API Key: Can be any value as authentication is handled by the proxy

### API Usage

The proxy implements the LM Studio compatible chat completions endpoint with streaming support. Here's an example of how to use it:

```bash
curl http://localhost:3128/v1/chat/completions \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "model": "claude-3-sonnet-20240229",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "What can you tell me about this document?"
      }
    ],
    "temperature": 0.7,
    "stream": true
  }'
```

Note: The `-N` flag enables streaming mode in curl. The proxy will send incremental updates as server-sent events (SSE) when `stream: true` is set.

The proxy will:
1. Process the initial request through Claude
2. If confidence is below threshold, retrieve additional context from ChromaDB
3. Refine the response with the new context
4. Repeat until confidence threshold is met or max rounds reached

### Error Handling

The proxy implements comprehensive error handling:

- Invalid requests return 400 status codes with error details
- Authentication failures return 401 status codes
- Internal errors return 500 status codes with error messages
- ChromaDB connection issues are handled gracefully with retries

### Advanced Features

1. Text Processing
   - HTML/XML tag cleaning for better content extraction
   - German character normalization support
   - Intelligent whitespace handling
   - JSON document parsing with fallback mechanisms

2. Traffic Logging (Debug Mode)
   - Complete request/response logging
   - Component interaction tracking
   - Detailed timing information
   - Separate traffic.log file for analysis

3. Confidence Scoring (In Development)
   - Dynamic confidence threshold adjustment
   - Multi-round retrieval optimization
   - Response quality assessment
   - Confidence-based context injection

### Limitations

- Currently only supports Anthropic's Claude as the LLM provider
- Requires AnythingLLM's ChromaDB to be accessible
- No caching implementation yet
- Single-threaded processing of requests
- Confidence scoring system still under development

## Technical Details

### Vector Search and Embeddings

The FLARE-Proxy implements sophisticated vector search through ChromaDB:

1. Document Storage:
   - AnythingLLM stores documents as embeddings in ChromaDB
   - Each document is converted into a high-dimensional vector representation (3072 dimensions)
   - Collections in ChromaDB organize documents by workspace

2. Query Processing:
   - When the FLARE engine requests additional context:
     * Query text is converted to embeddings using OpenAI's text-embedding-3-large
     * ChromaDB performs vector similarity search
     * Results are filtered based on similarity scores (distance threshold: 0.5)
     * Most relevant documents (top 3 per collection) are returned

3. Integration Benefits:
   - Consistent embeddings between AnythingLLM and FLARE-Proxy
   - High-quality semantic search through modern embedding model
   - Efficient similarity search with configurable thresholds
   - Automatic handling of multiple collections

This architecture allows for efficient semantic search without requiring direct handling of embeddings in the proxy. For future implementations with other databases (e.g., SQLite), additional embedding generation functionality would need to be added.

## Development Guide

### Adding a New Retriever

1. Create a new class in `retrievers/`
2. Implement the BaseRetriever interface
3. Add configuration in `config/`
4. Register the retriever in the FLARE processor

Example:
```python
class NewRetriever(BaseRetriever):
    async def retrieve(self, query: str) -> List[str]:
        # Implementation
        pass
```

### Testing

Each module should have corresponding tests in the `tests/` directory:

```python
def test_retriever():
    retriever = ChromaRetriever()
    results = await retriever.retrieve("test query")
    assert len(results) > 0
```

## API Reference

### Chat Completions

```http
POST /v1/chat/completions
Content-Type: application/json

{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant..."
    },
    {
      "role": "user",
      "content": "Context:\n[Document content]\n\nQuestion: User question"
    }
  ],
  "temperature": 0.7,
  "stream": true
}
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## License

MIT

## Technical Summary

The FLARE-Proxy implements a sophisticated retrieval-augmented generation system with the following technical characteristics:

### Architecture and Data Flow

1. Request Processing:
   - Incoming requests are received through an OpenAI-compatible endpoint
   - Requests are validated and transformed into the internal message format
   - The FLARE engine coordinates the interaction between components

2. FLARE Algorithm Implementation:
   - Initial query is processed by Claude with confidence scoring
   - If confidence is below threshold (default 0.7):
     * Query is sent to ChromaDB for context retrieval
     * Retrieved context is injected into the conversation
     * Process repeats until confidence threshold is met or max rounds reached
   - Final response is formatted according to OpenAI API specification

3. Vector Search Integration:
   - Uses OpenAI's text-embedding-3-large for embedding generation
   - Performs vector similarity search through ChromaDB
   - Results are filtered by configurable distance thresholds
   - Multiple collections are searched with relevance ranking

4. Asynchronous Processing:
   - All operations are implemented asynchronously
   - FastAPI provides high-performance async HTTP handling
   - Concurrent processing of LLM requests and context retrieval
   - Efficient connection management for ChromaDB

5. Error Handling and Reliability:
   - Comprehensive error handling at each layer
   - Automatic retry mechanisms for transient failures
   - Graceful degradation when services are unavailable
   - Detailed error reporting for debugging

This implementation provides a robust foundation for dynamic context retrieval while maintaining compatibility with existing AnythingLLM infrastructure.
