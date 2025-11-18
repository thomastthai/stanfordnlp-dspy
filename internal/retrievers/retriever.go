// Package retrievers provides retrieval implementations for DSPy.
package retrievers

import "context"

// Retriever defines the interface for retrieving relevant documents.
type Retriever interface {
	// Retrieve returns the top-k most relevant documents for the query.
	Retrieve(ctx context.Context, query string, k int) ([]string, error)

	// RetrieveWithScores returns documents with their relevance scores.
	RetrieveWithScores(ctx context.Context, query string, k int) ([]Document, error)

	// Name returns the retriever name.
	Name() string
}

// Document represents a retrieved document with metadata.
type Document struct {
	// Content is the document text
	Content string

	// Score is the relevance score
	Score float64

	// ID is the document identifier
	ID string

	// Metadata contains additional document metadata
	Metadata map[string]interface{}
}

// BaseRetriever provides common functionality for retrievers.
type BaseRetriever struct {
	name string
}

// NewBaseRetriever creates a new base retriever.
func NewBaseRetriever(name string) *BaseRetriever {
	return &BaseRetriever{name: name}
}

// Name implements Retriever.Name.
func (r *BaseRetriever) Name() string {
	return r.name
}
