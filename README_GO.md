# DSPy-Go

A complete Go reimplementation of [DSPy](https://dspy.ai/), following Terraform's architectural patterns and Go best practices.

## Overview

DSPy-Go is a framework for programming—rather than prompting—language models. It allows you to iterate fast on building modular AI systems and offers algorithms for optimizing their prompts and weights.

This Go implementation provides:
- **Interface-based design** similar to Terraform providers
- **Context-based execution** for cancellation and timeouts
- **Goroutines and channels** for concurrency
- **Functional options pattern** for configuration
- **Clean separation of concerns**
- **Complete feature parity** with Python DSPy

## Project Status

🚧 **Work in Progress** 🚧

This is a complete refactoring effort to port the entire DSPy codebase to idiomatic Go. Current progress:

- [x] Project structure and foundation
- [x] Core primitives (BaseModule, Module, Example, Prediction, Parameter)
- [x] Signature system with parser
- [x] Basic Predict module
- [ ] LM clients (OpenAI, Anthropic, etc.)
- [ ] Adapters (Chat, JSON, XML, etc.)
- [ ] Predict modules (ChainOfThought, ReAct, etc.)
- [ ] Optimizers (Bootstrap, MIPROv2, COPRO, etc.)
- [ ] Retrievers (ColBERT, Weaviate, etc.)
- [ ] Evaluation framework
- [ ] Comprehensive tests

## Installation

### Prerequisites
- Go 1.21 or later

### Install from source

```bash
# Clone the repository
git clone https://github.com/stanfordnlp/dspy.git
cd dspy

# Build the project
make build

# Run tests
make test
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"
    
    "github.com/stanfordnlp/dspy/internal/predict"
    "github.com/stanfordnlp/dspy/pkg/dspy"
)

func main() {
    // Configure DSPy
    dspy.Configure(
        dspy.WithTemperature(0.0),
        dspy.WithMaxTokens(1000),
    )
    
    // Create a prediction module
    predictor, err := predict.New("question -> answer")
    if err != nil {
        log.Fatal(err)
    }
    
    // Execute prediction
    result, err := predictor.Forward(context.Background(), map[string]interface{}{
        "question": "What is DSPy?",
    })
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Println("Answer:", result.Get("answer"))
}
```

## Cache Configuration

DSPy-Go supports configurable cache for datasets, making it suitable for containerized environments (Kubernetes, Docker) where `/tmp` is ephemeral.

### Environment Variables

```bash
export DSPY_CACHE_DIR=/persistent/cache   # Cache directory
export DSPY_CACHE_SIZE_MB=5120            # Max cache size (5GB)
export DSPY_CACHE_TTL=48h                 # Cache TTL
export DSPY_CACHE_ENABLED=true            # Enable/disable
```

### Programmatic Configuration

```go
dspy.Configure(
    dspy.WithCacheDir("/persistent/cache"),
    dspy.WithCacheSize(5120),  // 5GB
    dspy.WithCacheTTL(48*time.Hour),
)
```

### Kubernetes Example

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: dspy-cache
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 10Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dspy-app
spec:
  template:
    spec:
      containers:
      - name: app
        env:
        - name: DSPY_CACHE_DIR
          value: "/cache"
        volumeMounts:
        - name: cache
          mountPath: /cache
      volumes:
      - name: cache
        persistentVolumeClaim:
          claimName: dspy-cache
```

See [docs/CACHE_CONFIGURATION.md](docs/CACHE_CONFIGURATION.md) for comprehensive documentation.

## Architecture

The project follows a clean architecture inspired by Terraform:

```
dspy-go/
├── cmd/dspy/              # CLI entry point
├── pkg/dspy/              # Public API
│   ├── dspy.go            # Main package
│   ├── settings.go        # Configuration
│   ├── configure.go       # Setup functions
│   └── version.go         # Version info
├── internal/              # Internal implementation
│   ├── primitives/        # Core types (Module, Example, Prediction, etc.)
│   ├── signatures/        # Signature system
│   ├── predict/           # Prediction modules
│   ├── clients/           # LM clients (provider pattern)
│   ├── adapters/          # Format adapters
│   ├── teleprompt/        # Optimizers
│   ├── retrievers/        # Retrieval systems
│   ├── evaluate/          # Evaluation framework
│   └── utils/             # Utilities
└── examples/              # Example programs
```

## Design Principles

1. **Idiomatic Go**: Not a direct Python translation, but proper Go code
2. **Interface-based**: Clean abstractions for extensibility
3. **Context-aware**: Use context.Context for cancellation and timeouts
4. **Concurrent**: Leverage goroutines for parallel execution
5. **Type-safe**: Strong typing with clear interfaces
6. **Testable**: Table-driven tests with comprehensive coverage
7. **Documented**: godoc for all exported types and functions

## Building

```bash
# Build the CLI
make build

# Run tests
make test

# Run linters
make lint

# Format code
make fmt

# Run all checks
make check
```

## Examples

See the `examples/` directory for complete working examples:

- `quickstart/` - Basic prediction usage
- `cache_config/` - Cache configuration for containers
- `rag/` - RAG system with retriever (TODO)
- `agents/` - ReAct agent with tools (TODO)
- `optimization/` - Optimizing with BootstrapFewShot (TODO)

## Contributing

Contributions are welcome! This is a large refactoring effort and we appreciate:

- Bug reports and feature requests
- Code contributions (please discuss large changes first)
- Documentation improvements
- Example programs

## Compatibility

DSPy-Go aims to maintain compatibility with Python DSPy:
- Compatible version: Python DSPy v3.1.0
- JSON serialization format is compatible
- Can load models saved by Python DSPy

## Performance

Go implementation is expected to be:
- **2-5x faster** than Python for most operations
- **Lower memory usage** due to efficient Go runtime
- **Single binary deployment** - no dependencies needed
- **Better concurrency** with goroutines

## License

Apache 2.0 - See LICENSE file

## Acknowledgments

This is a Go reimplementation of [DSPy](https://github.com/stanfordnlp/dspy) by the Stanford NLP Group. Original Python implementation and research by Omar Khattab and the DSPy team.

## Status

Currently in active development. Not yet ready for production use.
