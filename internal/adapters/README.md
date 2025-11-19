# DSPy Adapters

This package provides format adapters that bridge DSPy signatures and Language Model (LM) APIs. Adapters handle the translation between structured DSPy inputs/outputs and the specific formats required by different LM providers.

## Available Adapters

### ChatAdapter

The default adapter that works with all language models. Uses field-based formatting with special markers `[[ ## field_name ## ]]`.

**Usage:**
```go
adapter := adapters.NewChatAdapter()
request, err := adapter.Format(signature, inputs, demos)
outputs, err := adapter.Parse(signature, response)
```

**Best for:**
- Universal compatibility across all models
- Default choice when no specific requirements exist
- Automatic fallback protection

### JSONAdapter

Prompts LMs to return JSON data containing output fields. Leverages native JSON generation for models supporting `response_format`.

**Usage:**
```go
// Basic JSON adapter
adapter := adapters.NewJSONAdapter()

// With strict schema validation
adapter := adapters.NewJSONAdapterWithSchema()
```

**Features:**
- JSON parsing and validation
- Schema validation support
- Error recovery for malformed JSON
- Field extraction with type conversion
- Handles JSON wrapped in markdown code blocks

**Best for:**
- Models with structured output support
- Low latency requirements (minimal boilerplate)
- Strict type validation needs

### BAMLAdapter

Extends JSONAdapter with BAML-inspired formatting for improved structured output generation. Creates human-readable, token-efficient schema representations.

**Usage:**
```go
// Default BAML adapter (uses # for comments)
adapter := adapters.NewBAMLAdapter()

// Custom comment symbol
adapter := adapters.NewBAMLAdapterWithCommentSymbol("//")
```

**Features:**
- Simplified schema rendering
- Field descriptions with comment syntax
- Support for nested structures
- Type conversion (string, int, float, boolean, arrays)
- Inherits JSON parsing from JSONAdapter

**Best for:**
- Complex nested structures
- Better model comprehension with schema comments
- Token-efficient schema representation

### TwoStepAdapter

Two-phase adapter for improved structured output extraction. Phase 1 uses natural prompts, Phase 2 extracts structured data.

**Usage:**
```go
// Create extraction model for phase 2
extractionModel, err := clients.NewLM(clients.LMOptions{
    Model: "openai/gpt-4o-mini",
})

// Create two-step adapter
adapter := adapters.NewTwoStepAdapter(extractionModel)
```

**Features:**
- Natural language prompts for main LM (phase 1)
- Structured extraction with smaller model (phase 2)
- Automatic extractor signature creation
- Detailed task descriptions
- Cost optimization (expensive model for reasoning, cheap for extraction)

**Best for:**
- Reasoning models (o1, o3) that struggle with structured output
- Complex reasoning tasks requiring natural language
- Cost optimization scenarios

### XMLAdapter

Parses XML-structured outputs from LM responses with error recovery.

**Usage:**
```go
// Default XML adapter (root tag: "response")
adapter := adapters.NewXMLAdapter()

// Custom root tag
adapter := adapters.NewXMLAdapterWithRootTag("result")
```

**Features:**
- XML parsing and validation
- Tag-based field extraction
- Error recovery for malformed XML
- XML escaping/unescaping
- Case-insensitive field matching

**Best for:**
- Legacy systems requiring XML
- Models that prefer XML structure
- Hierarchical data representation

## Adapter Interface

All adapters implement the `Adapter` interface:

```go
type Adapter interface {
    // Format converts a signature and inputs into an LM request
    Format(sig *signatures.Signature, inputs map[string]interface{}, demos []map[string]interface{}) (*clients.Request, error)
    
    // Parse extracts outputs from an LM response
    Parse(sig *signatures.Signature, response *clients.Response) (map[string]interface{}, error)
    
    // Name returns the adapter name
    Name() string
}
```

## Testing

Each adapter has comprehensive unit tests. Run tests with:

```bash
go test ./internal/adapters/... -v
```

## Error Handling

All adapters include robust error handling:

- **Format errors**: Invalid signatures, missing required fields
- **Parse errors**: Malformed responses, missing output fields
- **Recovery mechanisms**: JSON/XML extraction from markdown, schema repair

## Examples

### Basic Usage

```go
import (
    "github.com/stanfordnlp/dspy/internal/adapters"
    "github.com/stanfordnlp/dspy/internal/signatures"
)

// Create signature
sig, _ := signatures.NewSignature("question -> answer")
sig.Instructions = "Answer concisely"

// Create adapter
adapter := adapters.NewJSONAdapter()

// Format request
inputs := map[string]interface{}{
    "question": "What is Go?",
}
request, _ := adapter.Format(sig, inputs, nil)

// Parse response (after LM call)
outputs, _ := adapter.Parse(sig, response)
answer := outputs["answer"]
```

### With Demonstrations

```go
demos := []map[string]interface{}{
    {
        "question": "What is Python?",
        "answer": "A programming language",
    },
}

request, _ := adapter.Format(sig, inputs, demos)
```

### Two-Step with Reasoning Model

```go
// Main reasoning model
mainLM, _ := clients.NewLM(clients.LMOptions{
    Model: "openai/o3-mini",
})

// Extraction model
extractionLM, _ := clients.NewLM(clients.LMOptions{
    Model: "openai/gpt-4o-mini",
})

// Create adapter
adapter := adapters.NewTwoStepAdapter(extractionLM)

// Use with main model
request, _ := adapter.Format(sig, inputs, nil)
response, _ := mainLM.Call(ctx, request)
outputs, _ := adapter.Parse(sig, response)
```

## Choosing an Adapter

| Use Case | Recommended Adapter |
|----------|-------------------|
| Default/Universal | ChatAdapter |
| Structured Output | JSONAdapter |
| Complex Nested Types | BAMLAdapter |
| Reasoning Models | TwoStepAdapter |
| Legacy XML Systems | XMLAdapter |
| Low Latency | JSONAdapter |
| Token Efficiency | BAMLAdapter |

## Contributing

When adding new adapters:

1. Implement the `Adapter` interface
2. Extend `BaseAdapter` for common functionality
3. Add comprehensive unit tests
4. Document usage patterns and edge cases
5. Include error handling and recovery
6. Follow existing naming conventions

## References

- [DSPy Adapter Documentation](../../docs/docs/learn/programming/adapters.md)
- [Python DSPy Adapters](../../dspy/adapters/)
- [Signature System](../signatures/)
