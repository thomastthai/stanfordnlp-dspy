// Package main demonstrates basic DSPy-Go usage.
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/stanfordnlp/dspy/internal/predict"
	"github.com/stanfordnlp/dspy/pkg/dspy"
)

func main() {
	// Configure DSPy with default settings
	dspy.Configure(
		dspy.WithTemperature(0.0),
		dspy.WithMaxTokens(1000),
	)

	// Create a simple predict module
	// Signature: "question -> answer"
	predictor, err := predict.New("question -> answer")
	if err != nil {
		log.Fatalf("Failed to create predictor: %v", err)
	}

	// Execute a prediction
	ctx := context.Background()
	inputs := map[string]interface{}{
		"question": "What is DSPy?",
	}

	result, err := predictor.Forward(ctx, inputs)
	if err != nil {
		log.Fatalf("Prediction failed: %v", err)
	}

	// Print the result
	fmt.Println("Question:", inputs["question"])
	if answer, ok := result.Get("answer"); ok {
		fmt.Println("Answer:", answer)
	}

	fmt.Println("\nDSPy-Go Quickstart Example Complete!")
	fmt.Println("Note: This is using a mock LM. Configure with a real LM for actual predictions.")
}
