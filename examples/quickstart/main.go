// Package main demonstrates basic DSPy-Go usage.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/stanfordnlp/dspy/internal/predict"
	"github.com/stanfordnlp/dspy/pkg/dspy"
)

func main() {
	// Configure DSPy with OpenAI (or mock if no API key)
	// To use real OpenAI, set the OPENAI_API_KEY environment variable
	apiKey := os.Getenv("OPENAI_API_KEY")

	if apiKey != "" {
		fmt.Println("Using OpenAI API (gpt-4o-mini)")
		// Configure with real OpenAI
		dspy.Configure(
			dspy.WithLM("openai/gpt-4o-mini"),
			dspy.WithTemperature(0.7),
			dspy.WithMaxTokens(1000),
		)
	} else {
		fmt.Println("Using mock LM (set OPENAI_API_KEY to use real OpenAI)")
		// Configure with mock LM
		dspy.Configure(
			dspy.WithTemperature(0.0),
			dspy.WithMaxTokens(1000),
		)
	}

	// Create a simple predict module
	// Signature: "question -> answer"
	predictor, err := predict.New("question -> answer")
	if err != nil {
		log.Fatalf("Failed to create predictor: %v", err)
	}

	// Execute a prediction
	ctx := context.Background()
	inputs := map[string]interface{}{
		"question": "What is DSPy and why is it useful?",
	}

	result, err := predictor.Forward(ctx, inputs)
	if err != nil {
		log.Fatalf("Prediction failed: %v", err)
	}

	// Print the result
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("Question:", inputs["question"])
	fmt.Println(strings.Repeat("=", 60))
	if answer, ok := result.Get("answer"); ok {
		fmt.Println("Answer:", answer)
	}
	fmt.Println(strings.Repeat("=", 60))

	fmt.Println("\nDSPy-Go Quickstart Example Complete!")
}
