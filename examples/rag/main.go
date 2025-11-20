// Package main demonstrates a simple RAG (Retrieval-Augmented Generation) example.
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/stanfordnlp/dspy/internal/retrievers"
	"github.com/stanfordnlp/dspy/internal/utils"
)

func main() {
	fmt.Println("=== DSPy RAG Example ===")

	// Example: Using a dummy retriever for demonstration
	ctx := context.Background()

	// Create sample documents
	documents := []string{
		"Paris is the capital of France. It is known for the Eiffel Tower.",
		"London is the capital of the United Kingdom. It has Big Ben.",
		"Berlin is the capital of Germany. The Berlin Wall was historic.",
		"Madrid is the capital of Spain. It's known for its museums.",
		"Rome is the capital of Italy. It has the Colosseum.",
	}

	// Create a dummy retriever with scores
	scores := []float64{0.95, 0.85, 0.75, 0.65, 0.55}
	retriever := utils.NewDummyRMWithScores(documents, scores)

	// Perform retrieval
	query := "What is the capital of France?"
	k := 3

	fmt.Printf("\nQuery: %s\n", query)
	fmt.Printf("Retrieving top %d documents...\n\n", k)

	results, err := retriever.RetrieveWithScores(ctx, query, k)
	if err != nil {
		log.Fatalf("Retrieval failed: %v", err)
	}

	// Display results
	fmt.Println("Retrieved Documents:")
	for i, doc := range results {
		fmt.Printf("%d. [Score: %.2f] %s\n", i+1, doc.Score, doc.Content)
	}

	// Example: ColBERTv2 (would need a running server)
	fmt.Println("\n=== ColBERTv2 Example (requires server) ===")
	colbert := retrievers.NewColBERTv2(retrievers.ColBERTv2Options{
		URL:  "http://localhost:8080",
		Port: "",
	})
	fmt.Printf("ColBERTv2 retriever created: %s\n", colbert.Name())

	// Example: Weaviate (would need a running instance)
	fmt.Println("\n=== Weaviate Example (requires instance) ===")
	weaviate := retrievers.NewWeaviate(retrievers.WeaviateOptions{
		URL:            "http://localhost:8080",
		CollectionName: "Documents",
		TextKey:        "content",
	})
	fmt.Printf("Weaviate retriever created: %s\n", weaviate.Name())

	// Example: ChromaDB (would need a running instance)
	fmt.Println("\n=== ChromaDB Example (requires instance) ===")
	chroma := retrievers.NewChromaDB(retrievers.ChromaDBOptions{
		URL:            "http://localhost:8000",
		CollectionName: "documents",
	})
	fmt.Printf("ChromaDB retriever created: %s\n", chroma.Name())

	// Demonstrate usage tracking
	fmt.Println("\n=== Usage Tracking Example ===")
	tracker := utils.NewUsageTracker()

	// Add some mock usage
	tracker.AddUsage("gpt-4", &utils.UsageEntry{
		PromptTokens:     100,
		CompletionTokens: 50,
		TotalTokens:      150,
	})
	tracker.AddUsage("gpt-4", &utils.UsageEntry{
		PromptTokens:     200,
		CompletionTokens: 100,
		TotalTokens:      300,
	})

	// Get total usage
	totals := tracker.GetTotalUsage()
	for model, usage := range totals {
		fmt.Printf("%s: %d total tokens (%d prompt + %d completion)\n",
			model, usage.TotalTokens, usage.PromptTokens, usage.CompletionTokens)
	}

	// Estimate costs
	pricing := utils.DefaultPricing()
	tracker.UpdateCosts(pricing)

	grand := tracker.GetGrandTotal()
	fmt.Printf("Total estimated cost: $%.4f\n", grand.EstimatedCost)

	fmt.Println("\n=== Example Complete ===")
}
