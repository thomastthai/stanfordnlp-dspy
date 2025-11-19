package utils

import (
	"context"
	"fmt"
	"math/rand"
	"time"

	"github.com/stanfordnlp/dspy/internal/retrievers"
)

// DummyLM is an enhanced mock language model with configurable responses.
type DummyLM struct {
	responses     []string
	currentIndex  int
	followExample bool
	pattern       string
	randomize     bool
	rng           *rand.Rand
}

// NewDummyLM creates a new dummy language model.
func NewDummyLM(responses []string) *DummyLM {
	return &DummyLM{
		responses:    responses,
		currentIndex: 0,
		rng:          rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// NewDummyLMWithPattern creates a dummy LM with a pattern-based response.
func NewDummyLMWithPattern(pattern string) *DummyLM {
	return &DummyLM{
		pattern: pattern,
		rng:     rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// SetFollowExample sets whether to follow example responses.
func (d *DummyLM) SetFollowExample(follow bool) {
	d.followExample = follow
}

// SetRandomize sets whether to randomize response selection.
func (d *DummyLM) SetRandomize(randomize bool) {
	d.randomize = randomize
}

// GetResponse returns the next response based on configuration.
func (d *DummyLM) GetResponse(input string) string {
	if d.pattern != "" {
		return fmt.Sprintf(d.pattern, input)
	}
	
	if len(d.responses) == 0 {
		return "[Mock response]"
	}
	
	if d.randomize {
		idx := d.rng.Intn(len(d.responses))
		return d.responses[idx]
	}
	
	response := d.responses[d.currentIndex%len(d.responses)]
	d.currentIndex++
	return response
}

// Reset resets the response index.
func (d *DummyLM) Reset() {
	d.currentIndex = 0
}

// DummyRM is a mock retriever for testing.
type DummyRM struct {
	retrievers.BaseRetriever
	documents []string
	scores    []float64
	errorOn   int // Return error on this call number (0 = never)
	callCount int
}

// NewDummyRM creates a new dummy retriever.
func NewDummyRM(documents []string) *DummyRM {
	return &DummyRM{
		BaseRetriever: *retrievers.NewBaseRetriever("dummy"),
		documents:     documents,
		scores:        make([]float64, len(documents)),
	}
}

// NewDummyRMWithScores creates a dummy retriever with predefined scores.
func NewDummyRMWithScores(documents []string, scores []float64) *DummyRM {
	if len(scores) < len(documents) {
		// Pad with default scores
		for i := len(scores); i < len(documents); i++ {
			scores = append(scores, 1.0)
		}
	}
	
	return &DummyRM{
		BaseRetriever: *retrievers.NewBaseRetriever("dummy"),
		documents:     documents,
		scores:        scores,
	}
}

// Retrieve implements retrievers.Retriever.Retrieve.
func (d *DummyRM) Retrieve(ctx context.Context, query string, k int) ([]string, error) {
	d.callCount++
	
	if d.errorOn > 0 && d.callCount == d.errorOn {
		return nil, fmt.Errorf("dummy retriever error on call %d", d.callCount)
	}
	
	if k > len(d.documents) {
		k = len(d.documents)
	}
	
	results := make([]string, k)
	copy(results, d.documents[:k])
	return results, nil
}

// RetrieveWithScores implements retrievers.Retriever.RetrieveWithScores.
func (d *DummyRM) RetrieveWithScores(ctx context.Context, query string, k int) ([]retrievers.Document, error) {
	d.callCount++
	
	if d.errorOn > 0 && d.callCount == d.errorOn {
		return nil, fmt.Errorf("dummy retriever error on call %d", d.callCount)
	}
	
	if k > len(d.documents) {
		k = len(d.documents)
	}
	
	results := make([]retrievers.Document, k)
	for i := 0; i < k; i++ {
		score := 1.0
		if i < len(d.scores) {
			score = d.scores[i]
		}
		
		results[i] = retrievers.Document{
			Content:  d.documents[i],
			Score:    score,
			ID:       fmt.Sprintf("doc_%d", i),
			Metadata: map[string]interface{}{"index": i},
		}
	}
	
	return results, nil
}

// SetErrorOn sets the call number to return an error on.
func (d *DummyRM) SetErrorOn(callNumber int) {
	d.errorOn = callNumber
}

// GetCallCount returns the number of times the retriever has been called.
func (d *DummyRM) GetCallCount() int {
	return d.callCount
}

// Reset resets the call count.
func (d *DummyRM) Reset() {
	d.callCount = 0
}

// DummyDataGenerator generates dummy data for testing.
type DummyDataGenerator struct {
	rng *rand.Rand
}

// NewDummyDataGenerator creates a new dummy data generator.
func NewDummyDataGenerator(seed int64) *DummyDataGenerator {
	return &DummyDataGenerator{
		rng: rand.New(rand.NewSource(seed)),
	}
}

// GenerateText generates random text of the specified length.
func (d *DummyDataGenerator) GenerateText(length int) string {
	const letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
	
	result := make([]byte, length)
	for i := range result {
		result[i] = letters[d.rng.Intn(len(letters))]
	}
	
	return string(result)
}

// GenerateSentence generates a random sentence.
func (d *DummyDataGenerator) GenerateSentence() string {
	words := []string{
		"the", "quick", "brown", "fox", "jumps", "over",
		"lazy", "dog", "cat", "bird", "runs", "fast",
	}
	
	numWords := 5 + d.rng.Intn(10)
	sentence := ""
	
	for i := 0; i < numWords; i++ {
		if i > 0 {
			sentence += " "
		}
		sentence += words[d.rng.Intn(len(words))]
	}
	
	return sentence + "."
}

// GenerateDocument generates a random document with multiple sentences.
func (d *DummyDataGenerator) GenerateDocument(numSentences int) string {
	doc := ""
	for i := 0; i < numSentences; i++ {
		if i > 0 {
			doc += " "
		}
		doc += d.GenerateSentence()
	}
	return doc
}

// GenerateDocuments generates multiple random documents.
func (d *DummyDataGenerator) GenerateDocuments(count int, sentencesPerDoc int) []string {
	docs := make([]string, count)
	for i := 0; i < count; i++ {
		docs[i] = d.GenerateDocument(sentencesPerDoc)
	}
	return docs
}

// GenerateNumber generates a random number in the specified range.
func (d *DummyDataGenerator) GenerateNumber(min, max int) int {
	return min + d.rng.Intn(max-min+1)
}

// GenerateFloat generates a random float in the specified range.
func (d *DummyDataGenerator) GenerateFloat(min, max float64) float64 {
	return min + d.rng.Float64()*(max-min)
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
