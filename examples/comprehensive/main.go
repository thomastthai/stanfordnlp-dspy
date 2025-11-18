// Package main demonstrates comprehensive DSPy-Go features.
package main

import (
	"context"
	"fmt"
	"log"
	
	"github.com/stanfordnlp/dspy/internal/adapters"
	"github.com/stanfordnlp/dspy/internal/clients"
	"github.com/stanfordnlp/dspy/internal/evaluate"
	"github.com/stanfordnlp/dspy/internal/predict"
	"github.com/stanfordnlp/dspy/internal/primitives"
	"github.com/stanfordnlp/dspy/internal/signatures"
	"github.com/stanfordnlp/dspy/internal/teleprompt"
	"github.com/stanfordnlp/dspy/pkg/dspy"
)

func main() {
	fmt.Println("=== DSPy-Go Comprehensive Example ===")
	fmt.Println()
	
	// 1. Configure DSPy
	fmt.Println("1. Configuring DSPy...")
	dspy.Configure(
		dspy.WithTemperature(0.0),
		dspy.WithMaxTokens(500),
	)
	
	// 2. Create a signature
	fmt.Println("2. Creating signature...")
	sig, err := signatures.NewSignature("question, context -> answer")
	if err != nil {
		log.Fatal(err)
	}
	sig.Instructions = "Answer the question based on the context."
	fmt.Printf("   Signature: %s\n\n", sig)
	
	// 3. Create a Predict module
	fmt.Println("3. Creating Predict module...")
	predictor, err := predict.New(sig)
	if err != nil {
		log.Fatal(err)
	}
	
	// 4. Create a ChainOfThought module
	fmt.Println("4. Creating ChainOfThought module...")
	cot, err := predict.NewChainOfThought("question -> answer")
	if err != nil {
		log.Fatal(err)
	}
	
	// 5. Test the modules
	fmt.Println("5. Testing modules...")
	ctx := context.Background()
	inputs := map[string]interface{}{
		"question": "What is DSPy?",
		"context":  "DSPy is a framework for programming language models.",
	}
	
	// Basic predict
	result, err := predictor.Forward(ctx, inputs)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("   Predict result: %v\n", result.Fields())
	
	// Chain of thought
	cotResult, err := cot.Forward(ctx, map[string]interface{}{
		"question": "What is DSPy?",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("   CoT result: %v\n\n", cotResult.Fields())
	
	// 6. Test with mock LM
	fmt.Println("6. Testing with Mock LM...")
	mockLM := clients.NewMockLM("mock-gpt-4")
	mockLM.ResponseFunc = func(req *clients.Request) (*clients.Response, error) {
		return &clients.Response{
			Choices: []clients.Choice{
				{
					Message: clients.NewMessage("assistant", "DSPy is a framework for programming language models."),
					Text:    "DSPy is a framework for programming language models.",
				},
			},
		}, nil
	}
	
	request := clients.NewRequest().
		WithMessages(clients.NewMessage("user", "What is DSPy?")).
		WithTemperature(0.7)
	
	response, err := mockLM.Call(ctx, request)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("   Mock LM response: %s\n\n", response.Choices[0].Text)
	
	// 7. Test adapter
	fmt.Println("7. Testing ChatAdapter...")
	adapter := adapters.NewChatAdapter()
	adapterReq, err := adapter.Format(sig, inputs, nil)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("   Formatted %d messages\n", len(adapterReq.Messages))
	
	mockResponse := &clients.Response{
		Choices: []clients.Choice{
			{
				Message: clients.NewMessage("assistant", "answer: DSPy is a framework"),
			},
		},
	}
	outputs, err := adapter.Parse(sig, mockResponse)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("   Parsed outputs: %v\n\n", outputs)
	
	// 8. Test evaluation
	fmt.Println("8. Testing evaluation framework...")
	
	// Create test dataset (matching the signature: question, context -> answer)
	dataset := []*primitives.Example{
		primitives.NewExample(
			map[string]interface{}{
				"question": "What is Go?",
				"context":  "Go is a programming language created by Google.",
			},
			map[string]interface{}{"answer": "A programming language"},
		),
		primitives.NewExample(
			map[string]interface{}{
				"question": "What is Python?",
				"context":  "Python is a high-level programming language.",
			},
			map[string]interface{}{"answer": "A programming language"},
		),
	}
	
	// Create metric
	metric := evaluate.ExactMatch("answer")
	
	// Create evaluator
	evaluator := evaluate.NewEvaluator(metric).WithDisplayProgress(false)
	
	// Evaluate (with mock predictor)
	evalResult, err := evaluator.Evaluate(ctx, predictor, dataset)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("   Average score: %.2f\n", evalResult.AverageScore)
	fmt.Printf("   Total evaluated: %d\n\n", evalResult.Count)
	
	// 9. Test bootstrap optimizer
	fmt.Println("9. Testing BootstrapFewShot optimizer...")
	optimizer := teleprompt.NewBootstrapFewShot(5).
		WithMaxLabeledDemos(3).
		WithMaxRounds(1)
	
	optimizedModule, err := optimizer.Compile(ctx, predictor, dataset, metric)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("   Optimized module type: %T\n\n", optimizedModule)
	
	// 10. Test serialization
	fmt.Println("10. Testing module serialization...")
	data, err := predictor.Save()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("   Serialized %d bytes\n", len(data))
	
	// Load into new module
	newPredictor, err := predict.New(sig)
	if err != nil {
		log.Fatal(err)
	}
	if err := newPredictor.Load(data); err != nil {
		log.Fatal(err)
	}
	fmt.Println("   Successfully loaded module")
	fmt.Println()
	
	fmt.Println("=== All tests passed! ===")
}
