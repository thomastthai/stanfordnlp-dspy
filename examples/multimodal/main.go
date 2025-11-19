// Package main demonstrates multimodal capabilities with images.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/stanfordnlp/dspy/internal/adapters/types"
	"github.com/stanfordnlp/dspy/internal/clients"
	"github.com/stanfordnlp/dspy/internal/clients/openai"
)

func main() {
	// Check for API key
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is required for this example")
	}

	fmt.Println("DSPy-Go Multimodal Example")
	fmt.Println("===========================")
	fmt.Println()

	// Example 1: Image from URL
	fmt.Println("Example 1: Analyzing an image from URL")
	if err := analyzeImageFromURL(apiKey); err != nil {
		log.Printf("Error in Example 1: %v", err)
	}

	fmt.Println()
	fmt.Println(string('='))
	fmt.Println()

	// Example 2: Working with code blocks
	fmt.Println("Example 2: Parsing and formatting code blocks")
	demonstrateCodeBlocks()

	fmt.Println()
	fmt.Println(string('='))
	fmt.Println()

	// Example 3: Tool calling
	fmt.Println("Example 3: Tool calling and orchestration")
	demonstrateToolCalling()

	fmt.Println("\nMultimodal Example Complete!")
}

// analyzeImageFromURL demonstrates analyzing an image from a URL.
func analyzeImageFromURL(apiKey string) error {
	// Create an image from URL
	imageURL := "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
	img := types.NewImageFromURL(imageURL)

	// Convert to message content
	imgContent, err := img.ToMessageContent()
	if err != nil {
		return fmt.Errorf("failed to create image content: %w", err)
	}

	fmt.Printf("Image URL: %s\n", imageURL)
	fmt.Printf("Image Content Type: %v\n", imgContent["type"])

	// Note: This is a demonstration of the types.
	// To actually send multimodal requests, you would need to use the OpenAI client
	// with vision-capable models like gpt-4o or gpt-4o-mini

	// Create OpenAI client
	client, err := openai.NewClient(openai.ClientOptions{
		APIKey: apiKey,
	})
	if err != nil {
		return fmt.Errorf("failed to create client: %w", err)
	}

	// Create a simple text request (vision requires special message format)
	req := clients.NewRequest().
		WithMessages(
			clients.NewMessage("user", "What can you tell me about analyzing images with AI?"),
		).
		WithTemperature(0.7).
		WithMaxTokens(200)

	ctx := context.Background()
	resp, err := client.Call(ctx, req, "gpt-4o-mini")
	if err != nil {
		return fmt.Errorf("API call failed: %w", err)
	}

	if len(resp.Choices) > 0 {
		fmt.Printf("\nResponse: %s\n", resp.Choices[0].Text)
		fmt.Printf("Tokens used: %d\n", resp.Usage.TotalTokens)
	}

	return nil
}

// demonstrateCodeBlocks shows code block parsing and formatting.
func demonstrateCodeBlocks() {
	// Create a Python code block
	pythonCode := `def hello_world():
    print("Hello, DSPy!")`

	block := types.NewCodeBlock(pythonCode, types.LanguagePython)
	block.FileName = "example.py"

	fmt.Println("Code Block:")
	fmt.Println(block.Format())

	fmt.Println("\nWith Context:")
	fmt.Println(block.FormatWithContext())

	// Parse a markdown code block
	markdown := "```go\nfunc main() {\n    fmt.Println(\"Hello\")\n}\n```"
	parsed, err := types.ParseCodeBlock(markdown)
	if err != nil {
		log.Printf("Failed to parse code block: %v", err)
		return
	}

	fmt.Printf("\nParsed Language: %s\n", parsed.Language)
	fmt.Printf("Code:\n%s\n", parsed.Code)
}

// demonstrateToolCalling shows tool calling functionality.
func demonstrateToolCalling() {
	// Create a tool orchestrator
	orchestrator := types.NewToolCallOrchestrator()

	// Register a sample tool
	orchestrator.RegisterTool("get_weather", func(args map[string]interface{}) (interface{}, error) {
		location, ok := args["location"].(string)
		if !ok {
			return nil, fmt.Errorf("location parameter required")
		}
		// Mock weather data
		return map[string]interface{}{
			"location":    location,
			"temperature": 72,
			"condition":   "sunny",
		}, nil
	})

	// Create a tool call
	toolCall := types.ToolCall{
		ID:   "call_123",
		Type: "function",
		Function: types.FunctionCall{
			Name:      "get_weather",
			Arguments: `{"location": "San Francisco"}`,
		},
	}

	// Add and execute
	orchestrator.AddToolCall(toolCall)
	if err := orchestrator.ExecuteAll(); err != nil {
		log.Printf("Tool execution failed: %v", err)
		return
	}

	// Display results
	results := orchestrator.GetResults()
	for _, result := range results {
		fmt.Printf("Tool Call ID: %s\n", result.ToolCallID)
		fmt.Printf("Result: %s\n", result.Content)
		if result.Error != nil {
			fmt.Printf("Error: %v\n", result.Error)
		}
	}
}
