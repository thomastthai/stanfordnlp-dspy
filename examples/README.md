# DSPy Go Examples

This directory contains examples demonstrating various features of the DSPy Go implementation.

## Available Examples

### RAG (Retrieval-Augmented Generation)
**Location:** `rag/main.go`

Demonstrates:
- Using different retrievers (Dummy, ColBERTv2, Weaviate, ChromaDB)
- Document retrieval with scores
- Usage tracking and cost estimation

**Run:**
```bash
go run examples/rag/main.go
```

## Features Demonstrated

### Retrievers
- **ColBERTv2**: Neural retriever with vector search
- **Weaviate**: Graph-based vector database
- **Qdrant**: Vector similarity search
- **ChromaDB**: Embedding-based retrieval
- **Marqo**: Multimodal search
- **DummyRM**: Mock retriever for testing

### Datasets
- **DataLoader**: Generic dataset loading with batching
- **GSM8K**: Grade school math problems
- **HotPotQA**: Multi-hop question answering
- **MMLU**: Massive multitask language understanding
- **HellaSwag**: Commonsense reasoning

### Utilities
- **Asyncify**: Convert sync functions to async with worker pools
- **Syncify**: Convert async to sync with timeouts
- **Callback**: Event system for LM/retriever hooks
- **Usage Tracker**: Token counting and cost estimation
- **Streaming**: Channel-based streaming utilities

### Advanced
- **Python Interpreter**: Secure Python code execution
- **DSP Legacy**: Backward compatibility layer

## Requirements

Most examples can run standalone with dummy/mock implementations. To use real services:

- **ColBERTv2**: Requires a running ColBERTv2 server
- **Weaviate**: Requires a Weaviate instance
- **Qdrant**: Requires a Qdrant server
- **ChromaDB**: Requires a ChromaDB instance
- **Marqo**: Requires a Marqo server
- **Python Interpreter**: Requires Python 3 installed

## Running Examples

```bash
# Run from project root
go run examples/rag/main.go

# Or build first
go build -o bin/rag examples/rag/main.go
./bin/rag
```

## Example Structure

Each example demonstrates:
1. Setting up the necessary components
2. Performing core operations
3. Handling results and errors
4. Best practices for the Go implementation

## Additional Examples

More examples are planned:
- **agents/**: ReAct agent with tools
- **optimization/**: MIPROv2 optimization
- **multimodal/**: Multimodal reasoning with images

Contributions welcome!
