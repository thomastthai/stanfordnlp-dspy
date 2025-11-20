# DSPy Go Datasets

This package provides dataset loading and processing utilities for DSPy in Go, including core dataset infrastructure and implementations of popular benchmarks.

## Features

- **Base Dataset Infrastructure**: Generic dataset interface with train/dev/test splits
- **DataLoader**: Efficient batching, shuffling, and iteration over datasets
- **Dataset Implementations**: HotPotQA, GSM8K, Colors, MATH, HellaSwag, MMLU
- **File Loading**: JSON, JSONL support for custom datasets
- **Reproducibility**: Configurable seeds for shuffling and splitting

## Quick Start

### Using the Colors Dataset

```go
package main

import (
    "fmt"
    "github.com/stanfordnlp/dspy/internal/datasets"
)

func main() {
    // Create a Colors dataset with default options
    opts := datasets.DefaultColorsOptions()
    colors := datasets.NewColors(opts)
    
    fmt.Printf("Dataset: %s\n", colors.Name())
    fmt.Printf("Total examples: %d\n", colors.Len())
    fmt.Printf("Training examples: %d\n", len(colors.Train()))
    fmt.Printf("Dev examples: %d\n", len(colors.Dev()))
    
    // Get a sample color
    if len(colors.Train()) > 0 {
        ex := colors.Train()[0]
        if color, ok := ex.Get("color"); ok {
            fmt.Printf("Sample color: %s\n", color)
        }
    }
}
```

### Using the MATH Dataset

```go
package main

import (
    "context"
    "fmt"
    "os"
    "github.com/stanfordnlp/dspy/internal/datasets"
)

func main() {
    // Load MATH examples from a JSON file
    file, err := os.Open("math_test.json")
    if err != nil {
        panic(err)
    }
    defer file.Close()
    
    examples, err := datasets.LoadMATHFromJSON(file)
    if err != nil {
        panic(err)
    }
    
    // Create dataset from examples
    opts := datasets.DefaultMATHOptions()
    mathDataset := datasets.CreateMATHDatasetFromExamples(examples, opts)
    
    fmt.Printf("MATH dataset loaded with %d examples\n", mathDataset.Len())
    
    // Access splits
    for i, ex := range mathDataset.Train() {
        if i >= 3 { break } // Show first 3
        question, _ := ex.Get("question")
        answer, _ := ex.Get("answer")
        fmt.Printf("Q: %v\nA: %v\n\n", question, answer)
    }
}
```

### Using DataLoader for Batching

```go
package main

import (
    "fmt"
    "github.com/stanfordnlp/dspy/internal/datasets"
)

func main() {
    // Create a Colors dataset
    colors := datasets.NewColors(datasets.DefaultColorsOptions())
    
    // Create a DataLoader with batching
    opts := datasets.DataLoaderOptions{
        BatchSize:  8,
        Shuffle:    true,
        Seed:       42,
        NumWorkers: 1,
        DropLast:   false,
    }
    
    loader := datasets.NewDataLoader(colors.Train(), opts)
    
    fmt.Printf("Total examples: %d\n", loader.Len())
    fmt.Printf("Number of batches: %d\n", loader.NumBatches())
    
    // Iterate over batches
    batchNum := 0
    for {
        batch := loader.Next()
        if batch == nil {
            break
        }
        batchNum++
        fmt.Printf("Batch %d: %d examples\n", batchNum, len(batch))
    }
    
    // Reset for another epoch
    loader.Reset()
}
```

### Loading Custom Datasets from JSON

```go
package main

import (
    "fmt"
    "github.com/stanfordnlp/dspy/internal/datasets"
    "github.com/stanfordnlp/dspy/internal/primitives"
)

func main() {
    // Load from JSON file
    examples, err := datasets.LoadFromJSON("mydata.json")
    if err != nil {
        panic(err)
    }
    
    // Create a custom dataset
    opts := datasets.DefaultDatasetOptions()
    opts.TrainSeed = 42
    dataset := datasets.NewBaseDataset("custom", opts)
    
    // Split into train/dev/test (60/20/20)
    train, dev, test := datasets.SplitData(examples, 0.6, 0.2)
    
    dataset.SetTrain(train)
    dataset.SetDev(dev)
    dataset.SetTest(test)
    
    fmt.Printf("Train: %d, Dev: %d, Test: %d\n", 
        len(train), len(dev), len(test))
}
```

### Loading from JSONL

```go
package main

import (
    "fmt"
    "github.com/stanfordnlp/dspy/internal/datasets"
)

func main() {
    // Load from JSONL file (one JSON object per line)
    examples, err := datasets.LoadFromJSONL("data.jsonl")
    if err != nil {
        panic(err)
    }
    
    fmt.Printf("Loaded %d examples\n", len(examples))
    
    // Process examples
    for _, ex := range examples {
        fmt.Println(ex.String())
    }
}
```

## Dataset Descriptions

### Colors Dataset

The Colors dataset contains 138+ color names from matplotlib, useful for simple classification and reasoning tasks.

**Features:**
- 60/40 train/dev split
- Colors sorted by suffix to prevent similar colors in different splits
- Shuffled with seed 0 for reproducibility

**Example:**
```go
{
    "color": "alice blue"
}
```

### MATH Dataset

The MATH dataset contains competition-level mathematics problems with LaTeX formatting.

**Features:**
- Automatic answer extraction from LaTeX `\boxed{}` commands
- Problem, solution, level, and type fields
- Difficulty-based filtering (levels 1-5)
- Problem types: algebra, counting, geometry, etc.

**Example:**
```go
{
    "question": "Solve for x: 2x + 3 = 7",
    "reasoning": "Subtract 3 from both sides...",
    "answer": "2",
    "level": "2",
    "type": "algebra"
}
```

### HotPotQA Dataset

Multi-hop question answering dataset (implementation requires HuggingFace integration).

**Features:**
- Hard examples only
- 75/25 train/dev split from training data
- Gold titles extraction for retrieval evaluation

### GSM8K Dataset

Grade school math word problems (implementation requires HuggingFace integration).

**Features:**
- Automatic numerical answer extraction
- Format: "#### [number]" parsing
- Train/test splits

## Dataset Interface

All datasets implement the following interface:

```go
type Dataset interface {
    Train() []*primitives.Example
    Dev() []*primitives.Example
    Test() []*primitives.Example
    Name() string
    Len() int
}
```

## DataLoader Options

```go
type DataLoaderOptions struct {
    BatchSize  int   // Number of examples per batch
    Shuffle    bool  // Whether to shuffle data
    Seed       int64 // Random seed for shuffling
    NumWorkers int   // Number of parallel workers (future use)
    DropLast   bool  // Drop incomplete final batch
}
```

## Dataset Options

```go
type DatasetOptions struct {
    TrainSize int   // Max training examples (-1 for all)
    DevSize   int   // Max dev examples (-1 for all)
    TestSize  int   // Max test examples (-1 for all)
    TrainSeed int64 // Random seed for train shuffling
    EvalSeed  int64 // Random seed for eval shuffling
    Shuffle   bool  // Whether to shuffle splits
}
```

## Notes

- **HuggingFace Integration**: Some datasets (HotPotQA, GSM8K, MATH) require pre-downloaded data or HuggingFace API integration
- **Pre-downloaded Data**: For datasets requiring external data, download JSON files and use the provided loading functions
- **Reproducibility**: Use consistent seeds across runs for reproducible results
- **Memory Management**: Use DataLoader for large datasets to avoid loading everything into memory

## Testing

Run the dataset tests:

```bash
go test ./internal/datasets/... -v
```

## Contributing

When adding new datasets:

1. Implement the `Dataset` interface
2. Extend `BaseDataset` for common functionality
3. Add comprehensive unit tests
4. Document the dataset format and usage
5. Include example code in this README
