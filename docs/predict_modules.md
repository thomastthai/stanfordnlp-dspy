# Predict Modules Documentation

This document provides detailed documentation and usage examples for the DSPy predict modules.

## Table of Contents
- [ProgramOfThought](#programofthought)
- [MultiChainComparison](#multichaincomparison)
- [Aggregation](#aggregation)

---

## ProgramOfThought

The `ProgramOfThought` module generates executable code (Python/pseudocode) to solve reasoning problems. It follows the Program-of-Thought prompting paradigm where the LM generates code that can be executed to compute the answer.

### Features
- **Code Generation**: Automatically generates Python code to solve problems
- **Error Recovery**: Retries code generation if execution fails
- **Code Execution**: Simulated code execution (real Python execution can be added via Interpreter)
- **Result Extraction**: Extracts final answers from code output

### Usage Example

```go
import (
    "context"
    "fmt"
    "github.com/stanfordnlp/dspy/internal/predict"
    "github.com/stanfordnlp/dspy/pkg/dspy"
)

func main() {
    // Configure DSPy with an LM
    dspy.Configure(
        dspy.WithLM("openai/gpt-4"),
        dspy.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
    )

    // Create a ProgramOfThought module
    pot, err := predict.NewProgramOfThought("question -> answer", 3)
    if err != nil {
        panic(err)
    }

    // Execute the module
    ctx := context.Background()
    result, err := pot.Forward(ctx, map[string]interface{}{
        "question": "What is the sum of all even numbers from 1 to 100?",
    })
    if err != nil {
        panic(err)
    }

    // Access the result
    answer := result.Fields()["answer"]
    fmt.Printf("Answer: %v\n", answer)

    // Access metadata
    code, _ := result.GetMetadata("generated_code")
    fmt.Printf("Generated Code:\n%s\n", code)
}
```

### Configuration

- **signature**: Input/output signature (e.g., "question -> answer")
- **maxIters**: Maximum number of code generation retries (default: 3)

### Metadata

The module adds the following metadata to predictions:
- `generated_code`: The final generated code
- `code_output`: Output from executing the code
- `code_attempts`: Number of attempts needed to generate working code

### Extending with Real Python Execution

To add real Python code execution, implement the `Interpreter` interface:

```go
type PythonInterpreter struct {
    // Your Python execution implementation
}

pot.Interpreter = &PythonInterpreter{}
```

---

## MultiChainComparison

The `MultiChainComparison` module generates and compares multiple reasoning chains to select the best answer. This implements the Multi-Chain Comparison technique where diverse reasoning paths are generated and then compared to find the most accurate solution.

### Features
- **Multiple Chain Generation**: Generates M different reasoning chains
- **Chain Comparison**: Uses LM to compare and select the best chain
- **Diversity Control**: Temperature parameter for generating diverse chains
- **Rationale Extraction**: Extracts corrected reasoning that compares all attempts

### Usage Example

```go
import (
    "context"
    "fmt"
    "github.com/stanfordnlp/dspy/internal/predict"
    "github.com/stanfordnlp/dspy/internal/primitives"
    "github.com/stanfordnlp/dspy/pkg/dspy"
)

func main() {
    // Configure DSPy
    dspy.Configure(
        dspy.WithLM("openai/gpt-4"),
        dspy.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
    )

    // Create a MultiChainComparison module for 3 chains
    mcc, err := predict.NewMultiChainComparison("question -> answer", 3)
    if err != nil {
        panic(err)
    }

    // First, generate multiple reasoning chains (using ChainOfThought or similar)
    cot, _ := predict.NewChainOfThought("question -> answer")
    ctx := context.Background()
    
    completions := []*primitives.Prediction{}
    for i := 0; i < 3; i++ {
        pred, _ := cot.Forward(ctx, map[string]interface{}{
            "question": "What is the capital of France?",
        })
        completions = append(completions, pred)
    }

    // Now compare the chains
    result, err := mcc.Forward(ctx, map[string]interface{}{
        "question":    "What is the capital of France?",
        "completions": completions,
    })
    if err != nil {
        panic(err)
    }

    // Access the result
    rationale := result.Fields()["rationale"]
    answer := result.Fields()["answer"]
    fmt.Printf("Rationale: %v\n", rationale)
    fmt.Printf("Answer: %v\n", answer)
}
```

### Configuration

- **signature**: Input/output signature (e.g., "question -> answer")
- **m**: Number of reasoning chains to compare (default: 3)
- **Temperature**: Temperature for diverse chain generation (default: 0.7)

### Input Format

The module expects:
- Original input fields from the signature
- `completions`: Slice of `*primitives.Prediction` containing M reasoning chains

### Metadata

The module adds the following metadata to predictions:
- `num_chains`: Number of chains compared
- `comparison_method`: Set to "multi_chain"

---

## Aggregation

The `Aggregation` module combines multiple predictions using various voting strategies. This is useful for ensemble methods and consensus-based decision making.

### Features
- **Majority Voting**: Select the most common prediction
- **Weighted Voting**: Weight predictions by confidence scores
- **Consensus Voting**: Require all predictions to agree
- **Customizable Normalization**: Custom text normalization for comparison

### Voting Strategies

#### 1. Majority Vote
Selects the prediction that appears most frequently across all predictions.

```go
agg := predict.NewAggregation("majority")
```

#### 2. Weighted Vote
Weights predictions by confidence scores (currently uses majority vote as base).

```go
agg := predict.NewAggregation("weighted")
```

#### 3. Consensus Vote
Requires all predictions to agree; returns error if there's no consensus.

```go
agg := predict.NewAggregation("consensus")
```

### Usage Example

```go
import (
    "context"
    "fmt"
    "github.com/stanfordnlp/dspy/internal/predict"
    "github.com/stanfordnlp/dspy/internal/primitives"
)

func main() {
    // Create predictions to aggregate
    predictions := []*primitives.Prediction{
        primitives.NewPrediction(map[string]interface{}{"answer": "Paris"}),
        primitives.NewPrediction(map[string]interface{}{"answer": "paris"}),
        primitives.NewPrediction(map[string]interface{}{"answer": "London"}),
        primitives.NewPrediction(map[string]interface{}{"answer": "Paris"}),
    }

    // Create aggregation module with majority voting
    agg := predict.NewAggregation("majority")

    // Aggregate predictions
    ctx := context.Background()
    result, err := agg.Forward(ctx, map[string]interface{}{
        "predictions": predictions,
        "field":       "answer", // Field to aggregate (optional)
    })
    if err != nil {
        panic(err)
    }

    // Access the result
    answer := result.Fields()["answer"]
    fmt.Printf("Aggregated Answer: %v\n", answer)

    // Check metadata
    majorityCount, _ := result.GetMetadata("majority_count")
    totalPreds, _ := result.GetMetadata("total_predictions")
    fmt.Printf("Won with %d out of %d votes\n", majorityCount, totalPreds)
}
```

### Custom Normalization

You can provide a custom normalization function:

```go
agg := predict.NewAggregation("majority")
agg.NormalizeFunc = func(s string) string {
    // Custom normalization logic
    s = strings.ToLower(s)
    s = strings.TrimSpace(s)
    s = strings.ReplaceAll(s, ".", "")
    return s
}
```

### Input Format

The module expects:
- `predictions`: Slice of `*primitives.Prediction` to aggregate
- `field` (optional): Specific field name to aggregate (uses first field if not specified)

### Metadata

The module adds the following metadata:
- **Majority Vote**:
  - `majority_count`: Number of votes for the winning prediction
  - `total_predictions`: Total number of predictions
  - `aggregation_strategy`: Set to "majority"

- **Consensus Vote**:
  - `consensus`: Set to `true` if consensus achieved
  - `total_predictions`: Total number of predictions
  - `aggregation_strategy`: Set to "consensus"

---

## Best Practices

### ProgramOfThought
- Use for mathematical or computational problems
- Provide clear problem descriptions in the question
- Set appropriate `maxIters` based on problem complexity
- Consider implementing a real Python interpreter for production use

### MultiChainComparison
- Use for complex reasoning tasks where multiple perspectives help
- Generate chains with temperature > 0 for diversity
- Typically 3-5 chains provide good balance of coverage and efficiency
- Can be combined with other modules like ChainOfThought or ReAct

### Aggregation
- Use `majority` for most cases
- Use `consensus` when high confidence is required
- Provide enough predictions for meaningful voting (at least 3)
- Customize normalization function based on your domain

---

## Integration with Other Modules

### Combining ProgramOfThought with BestOfN

```go
// Create base ProgramOfThought module
pot, _ := predict.NewProgramOfThought("question -> answer", 3)

// Wrap with BestOfN for multiple attempts
rewardFunc := func(inputs map[string]interface{}, pred *primitives.Prediction) float64 {
    // Custom reward logic
    return 1.0
}
bestPot := predict.NewBestOfN(pot, 5, rewardFunc, 0.8)

result, _ := bestPot.Forward(ctx, inputs)
```

### Using MultiChainComparison with Parallel

```go
// Generate chains in parallel
parallel := predict.NewParallel(3)
cot, _ := predict.NewChainOfThought("question -> answer")

pairs := []predict.ExecutionPair{}
for i := 0; i < 3; i++ {
    pairs = append(pairs, predict.ExecutionPair{
        Module: cot.Copy(),
        Inputs: inputs,
    })
}

parallelResult, _ := parallel.Forward(ctx, map[string]interface{}{
    "execution_pairs": pairs,
})

// Extract completions and compare
completions := parallelResult.Fields()["predictions"].([]*primitives.Prediction)
mcc, _ := predict.NewMultiChainComparison("question -> answer", 3)
result, _ := mcc.Forward(ctx, map[string]interface{}{
    "question":    inputs["question"],
    "completions": completions,
})
```

---

## Error Handling

All modules follow consistent error handling patterns:

```go
result, err := module.Forward(ctx, inputs)
if err != nil {
    // Handle specific errors
    switch {
    case strings.Contains(err.Error(), "no language model configured"):
        // Configure LM first
    case strings.Contains(err.Error(), "missing"):
        // Check required input fields
    default:
        // Handle general errors
    }
}
```

---

## Testing

All modules include comprehensive test suites. To run tests:

```bash
go test ./internal/predict/... -v
```

Test coverage includes:
- Module creation and configuration
- Forward pass with various inputs
- Error handling and edge cases
- Module copying and parameter management
- Integration with mock LM

---

## Performance Considerations

- **ProgramOfThought**: Code generation can be slower than direct prediction. Use for problems where code provides clear benefits.
- **MultiChainComparison**: Requires M+1 LM calls (M for chains, 1 for comparison). Use when improved accuracy justifies the cost.
- **Aggregation**: Very fast (no LM calls). Use liberally for ensemble methods.

---

## Future Enhancements

### Planned Features
- Real Python code execution for ProgramOfThought
- Parallel chain generation in MultiChainComparison
- Advanced confidence scoring in Aggregation
- Support for custom comparison strategies
- Caching and optimization for repeated queries
