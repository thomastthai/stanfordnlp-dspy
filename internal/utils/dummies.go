package utils

import (
	"context"
	"fmt"
	"strings"
	"sync"
)

// DummyLM is a mock language model for testing purposes.
type DummyLM struct {
	responses     []map[string]interface{}
	currentIndex  int
	followExample bool
	mu            sync.Mutex
	history       []map[string]interface{}
}

// DummyLMOptions configures a DummyLM.
type DummyLMOptions struct {
	// Responses is a list of responses to return
	Responses []map[string]interface{}
	// FollowExample determines whether to follow examples in the prompt
	FollowExample bool
}

// NewDummyLM creates a new DummyLM for testing.
func NewDummyLM(opts DummyLMOptions) *DummyLM {
	return &DummyLM{
		responses:     opts.Responses,
		followExample: opts.FollowExample,
		history:       make([]map[string]interface{}, 0),
	}
}

// Call simulates a language model call.
func (d *DummyLM) Call(ctx context.Context, messages []map[string]interface{}) (map[string]interface{}, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	var response map[string]interface{}

	if d.followExample {
		// Try to extract response from examples in the messages
		response = d.extractFromExamples(messages)
	}

	if response == nil {
		// Use predefined responses
		if d.currentIndex < len(d.responses) {
			response = d.responses[d.currentIndex]
			d.currentIndex++
		} else {
			response = map[string]interface{}{
				"answer": "No more responses",
			}
		}
	}

	// Record in history
	d.history = append(d.history, map[string]interface{}{
		"messages": messages,
		"response": response,
	})

	return response, nil
}

// extractFromExamples tries to extract a response from example demonstrations.
func (d *DummyLM) extractFromExamples(messages []map[string]interface{}) map[string]interface{} {
	// This is a simplified implementation
	// In the real implementation, this would parse the prompt structure
	// and extract matching examples
	return nil
}

// Reset resets the DummyLM to its initial state.
func (d *DummyLM) Reset() {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.currentIndex = 0
	d.history = make([]map[string]interface{}, 0)
}

// GetHistory returns the call history.
func (d *DummyLM) GetHistory() []map[string]interface{} {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.history
}

// DummyRM is a mock retrieval model for testing purposes.
type DummyRM struct {
	passages []string
}

// NewDummyRM creates a new DummyRM with the given passages.
func NewDummyRM(passages []string) *DummyRM {
	return &DummyRM{
		passages: passages,
	}
}

// Retrieve returns the top-k passages based on simple string matching.
func (d *DummyRM) Retrieve(ctx context.Context, query string, k int) ([]string, error) {
	if k <= 0 {
		return nil, fmt.Errorf("k must be positive, got %d", k)
	}

	if k > len(d.passages) {
		k = len(d.passages)
	}

	// Simple scoring based on keyword overlap
	scores := make([]float64, len(d.passages))
	queryWords := strings.Fields(strings.ToLower(query))

	for i, passage := range d.passages {
		passageWords := strings.Fields(strings.ToLower(passage))
		overlap := 0
		for _, qw := range queryWords {
			for _, pw := range passageWords {
				if qw == pw {
					overlap++
					break
				}
			}
		}
		scores[i] = float64(overlap)
	}

	// Get top-k passages
	indices := argsort(scores)
	results := make([]string, 0, k)
	for i := 0; i < k && i < len(indices); i++ {
		results = append(results, d.passages[indices[i]])
	}

	return results, nil
}

// argsort returns indices that would sort the array in descending order.
func argsort(arr []float64) []int {
	indices := make([]int, len(arr))
	for i := range indices {
		indices[i] = i
	}

	// Simple bubble sort (good enough for small test arrays)
	for i := 0; i < len(indices); i++ {
		for j := i + 1; j < len(indices); j++ {
			if arr[indices[j]] > arr[indices[i]] {
				indices[i], indices[j] = indices[j], indices[i]
			}
		}
	}

	return indices
}
