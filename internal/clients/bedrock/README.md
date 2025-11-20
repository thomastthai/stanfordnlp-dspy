# AWS Bedrock Client

This package provides a comprehensive client for AWS Bedrock foundation models with support for multiple model families, streaming, and automatic AWS IAM authentication.

## Features

- ✅ Multi-model support (Anthropic Claude, Amazon Titan, Meta Llama, AI21, Cohere)
- ✅ Streaming support for real-time responses
- ✅ AWS SDK v2 integration with automatic credential management
- ✅ Model-specific request/response formatting
- ✅ Automatic retries with exponential backoff
- ✅ Region configuration
- ✅ Comprehensive error handling

## Installation

```bash
go get github.com/stanfordnlp/dspy
go get github.com/aws/aws-sdk-go-v2/config
go get github.com/aws/aws-sdk-go-v2/service/bedrockruntime
```

## Quick Start

### Basic Usage

```go
package main

import (
    "context"
    "fmt"
    "log"
    
    "github.com/stanfordnlp/dspy/internal/clients"
    "github.com/stanfordnlp/dspy/internal/clients/bedrock"
)

func main() {
    // Create Bedrock client (uses default AWS credentials)
    client, err := bedrock.NewClient(bedrock.ClientOptions{
        Region: "us-east-1",
    })
    if err != nil {
        log.Fatal(err)
    }
    
    // Create a request
    request := &clients.Request{
        Messages: []clients.Message{
            {Role: "user", Content: "What is AWS Bedrock?"},
        },
        Temperature: 0.7,
        MaxTokens:   500,
    }
    
    // Call Claude 3 on Bedrock
    modelID := "anthropic.claude-3-sonnet-20240229-v1:0"
    response, err := client.Call(context.Background(), request, modelID)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Println(response.Choices[0].Message.Content)
}
```

### Using the Provider Pattern

```go
package main

import (
    "context"
    "fmt"
    "log"
    
    "github.com/stanfordnlp/dspy/internal/clients"
    _ "github.com/stanfordnlp/dspy/internal/clients/bedrock" // Register provider
)

func main() {
    // Create LM through provider
    provider := clients.GetProvider("bedrock")
    lm, err := provider.Create(map[string]interface{}{
        "model":  "anthropic.claude-3-opus-20240229-v1:0",
        "region": "us-west-2", // Optional, defaults to us-east-1
    })
    if err != nil {
        log.Fatal(err)
    }
    
    // Use the LM
    request := clients.NewRequest().
        WithMessages(clients.NewMessage("user", "Explain quantum computing")).
        WithTemperature(0.5).
        WithMaxTokens(1000)
    
    response, err := lm.Call(context.Background(), request)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Println(response.Choices[0].Message.Content)
}
```

## Advanced Features

### Streaming

Stream responses for real-time interaction:

```go
// Stream from Claude
modelID := "anthropic.claude-3-sonnet-20240229-v1:0"
chunkCh, errCh := client.Stream(context.Background(), request, modelID)

// Process chunks
for {
    select {
    case chunk, ok := <-chunkCh:
        if !ok {
            // Stream complete
            return
        }
        fmt.Print(chunk.Delta) // Print each chunk as it arrives
        if chunk.Done {
            fmt.Println("\n[Stream completed]")
        }
        
    case err := <-errCh:
        if err != nil {
            log.Fatal(err)
        }
    }
}
```

### Model-Specific Adapters

Use model adapters for fine-grained control:

```go
// Get adapter for a specific model
adapter := bedrock.GetModelAdapter("anthropic.claude-3-opus-20240229-v1:0")

// Build custom request
requestBody, err := adapter.BuildRequest(request)
if err != nil {
    log.Fatal(err)
}

// Parse custom response
response, err := adapter.ParseResponse(responseBody, modelID)
if err != nil {
    log.Fatal(err)
}
```

## Supported Models

### Anthropic Claude

Best for: Complex reasoning, analysis, coding, creative writing

```go
// Claude 3 Opus - Most capable
"anthropic.claude-3-opus-20240229-v1:0"

// Claude 3 Sonnet - Balanced performance and speed
"anthropic.claude-3-sonnet-20240229-v1:0"
"anthropic.claude-3-5-sonnet-20240620-v1:0"

// Claude 3 Haiku - Fastest
"anthropic.claude-3-haiku-20240307-v1:0"

// Claude 2
"anthropic.claude-v2:1"
"anthropic.claude-v2"
"anthropic.claude-instant-v1"
```

**Features**: Streaming ✅, Chat ✅, Long context (200K tokens)

### Amazon Titan

Best for: Summarization, text generation, embeddings

```go
// Titan Text Express
"amazon.titan-text-express-v1"

// Titan Text Lite
"amazon.titan-text-lite-v1"

// Titan Text Premier
"amazon.titan-text-premier-v1:0"
```

**Features**: Streaming ✅, Embeddings ✅

### Meta Llama

Best for: Open-source, instruction following, chat

```go
// Llama 3
"meta.llama3-70b-instruct-v1:0"
"meta.llama3-8b-instruct-v1:0"

// Llama 2
"meta.llama2-70b-chat-v1"
"meta.llama2-13b-chat-v1"
```

**Features**: Streaming ✅, Chat ✅, Open-source

### AI21 Labs Jurassic

Best for: Long-form content, creative writing

```go
"ai21.j2-ultra-v1"
"ai21.j2-mid-v1"
```

**Features**: Long context, Instruction following

### Cohere

Best for: Embeddings, classification, search

```go
"cohere.command-text-v14"
"cohere.command-light-text-v14"
```

**Features**: Embeddings ✅, Classification ✅

## Configuration

### ClientOptions

```go
type ClientOptions struct {
    Region  string        // AWS region (default: "us-east-1")
    Timeout time.Duration // Request timeout (default: 60s)
}
```

### AWS Credentials

The client uses standard AWS credential chain:

1. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
2. AWS credentials file (`~/.aws/credentials`)
3. IAM role (when running on EC2, ECS, Lambda, etc.)
4. AWS SSO

```bash
# Set via environment
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_REGION=us-west-2

# Or configure via AWS CLI
aws configure
```

### Available Regions

Bedrock is available in select AWS regions:

- `us-east-1` (N. Virginia)
- `us-west-2` (Oregon)
- `ap-southeast-1` (Singapore)
- `ap-southeast-2` (Sydney)
- `ap-northeast-1` (Tokyo)
- `eu-central-1` (Frankfurt)
- `eu-west-1` (Ireland)
- `eu-west-2` (London)
- `eu-west-3` (Paris)

Check AWS documentation for the latest region availability.

## Model Capabilities

Query model capabilities programmatically:

```go
// Get capabilities for a model family
caps := bedrock.GetModelCapabilities(bedrock.ModelFamilyAnthropic)

fmt.Printf("Chat: %v\n", caps.ChatCompletion)
fmt.Printf("Streaming: %v\n", caps.Streaming)
fmt.Printf("Embeddings: %v\n", caps.Embedding)
```

## Error Handling

Handle AWS Bedrock-specific errors:

```go
response, err := client.Call(ctx, request, modelID)
if err != nil {
    // Check if error is retryable
    if bedrock.isRetryableAWSError(err) {
        // Implement retry logic
        log.Println("Retryable error, will retry...")
    } else {
        log.Fatal("Non-retryable error:", err)
    }
}
```

### Common Error Types

- `ThrottlingException`: Rate limit exceeded (retryable)
- `ModelNotReadyException`: Model not ready (retryable)
- `ValidationException`: Invalid request parameters (not retryable)
- `AccessDeniedException`: Insufficient permissions (not retryable)
- `ServiceUnavailableException`: Service temporarily unavailable (retryable)

## Throttling and Rate Limits

Bedrock has per-model rate limits. Configure throttling:

```go
config := bedrock.DefaultThrottlingConfig()
config.MaxRetries = 5
config.InitialBackoff = 1000 // milliseconds
config.MaxBackoff = 60000    // milliseconds
config.BackoffMultiplier = 2.0
```

## Best Practices

1. **Model Selection**: Choose the right model for your use case
   - Claude 3 Opus for complex reasoning
   - Claude 3 Sonnet for balanced performance
   - Claude 3 Haiku for speed
   - Titan for cost-effective generation
   - Llama for open-source requirements

2. **Streaming**: Use streaming for long responses to improve UX

3. **Error Handling**: Implement proper retry logic for throttling

4. **Region Selection**: Choose a region close to your users

5. **Credentials**: Use IAM roles when running on AWS infrastructure

6. **Request Limits**: Be aware of per-model token limits:
   - Claude 3: 200K tokens input
   - Llama 2: 4K tokens
   - Titan: 8K tokens

7. **Cost Optimization**: Use appropriate model sizes for your needs

## Pricing Considerations

Bedrock pricing varies by model. Generally:

- **Claude 3 Opus**: Highest cost, best quality
- **Claude 3 Sonnet**: Medium cost, balanced
- **Claude 3 Haiku**: Lowest cost, fastest
- **Titan**: Cost-effective for high-volume
- **Llama**: Open-source, pay for infrastructure only

Check AWS Bedrock pricing page for current rates.

## Examples

Complete examples are available in the `examples/` directory:

- `examples/bedrock_basic/` - Basic model usage
- `examples/bedrock_streaming/` - Streaming responses
- `examples/bedrock_multi_model/` - Using multiple models
- `examples/bedrock_batch/` - Batch processing

## Troubleshooting

### Model Access

If you get `AccessDeniedException`:

1. Request model access in AWS Console
2. Go to Bedrock → Model access
3. Request access for specific models
4. Wait for approval (usually instant for most models)

### Credentials

If you get authentication errors:

```bash
# Check your AWS credentials
aws sts get-caller-identity

# Verify region configuration
echo $AWS_REGION
```

### Rate Limits

If you're hitting rate limits:

1. Request quota increases in AWS Service Quotas
2. Implement exponential backoff (built into client)
3. Use batch processing where possible
4. Consider using multiple regions

## License

This project is licensed under the MIT License - see the LICENSE file for details.
