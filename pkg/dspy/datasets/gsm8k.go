package datasets

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
)

// GSM8K represents the Grade School Math 8K dataset.
// Reference: https://github.com/openai/grade-school-math
type GSM8K struct {
	mu            sync.RWMutex
	trainExamples []Example
	testExamples  []Example
	cacheDir      string
	lazyLoad      bool
	loaded        bool
}

// GSM8KConfig configures the GSM8K dataset loader.
type GSM8KConfig struct {
	CacheDir string
	LazyLoad bool
}

// gsm8kRaw represents a raw GSM8K example from JSON.
type gsm8kRaw struct {
	Question string `json:"question"`
	Answer   string `json:"answer"`
}

// NewGSM8K creates a new GSM8K dataset loader.
func NewGSM8K(config GSM8KConfig) *GSM8K {
	if config.CacheDir == "" {
		config.CacheDir = filepath.Join(os.TempDir(), "dspy", "gsm8k")
	}

	return &GSM8K{
		cacheDir: config.CacheDir,
		lazyLoad: config.LazyLoad,
	}
}

// Load loads all splits of the dataset.
func (g *GSM8K) Load(ctx context.Context) ([]Example, error) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.loaded && !g.lazyLoad {
		return append(g.trainExamples, g.testExamples...), nil
	}

	// Load training data
	if err := g.loadSplit(ctx, "train"); err != nil {
		return nil, fmt.Errorf("failed to load train split: %w", err)
	}

	// Load test data
	if err := g.loadSplit(ctx, "test"); err != nil {
		return nil, fmt.Errorf("failed to load test split: %w", err)
	}

	g.loaded = true
	return append(g.trainExamples, g.testExamples...), nil
}

// Train returns training examples.
func (g *GSM8K) Train() ([]Example, error) {
	g.mu.RLock()
	if len(g.trainExamples) > 0 {
		defer g.mu.RUnlock()
		return g.trainExamples, nil
	}
	g.mu.RUnlock()

	if err := g.loadSplit(context.Background(), "train"); err != nil {
		return nil, err
	}

	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.trainExamples, nil
}

// Test returns test examples.
func (g *GSM8K) Test() ([]Example, error) {
	g.mu.RLock()
	if len(g.testExamples) > 0 {
		defer g.mu.RUnlock()
		return g.testExamples, nil
	}
	g.mu.RUnlock()

	if err := g.loadSplit(context.Background(), "test"); err != nil {
		return nil, err
	}

	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.testExamples, nil
}

// Len returns the total number of examples.
func (g *GSM8K) Len() int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return len(g.trainExamples) + len(g.testExamples)
}

// loadSplit loads a specific split from cache or downloads it.
func (g *GSM8K) loadSplit(ctx context.Context, split string) error {
	cachePath := filepath.Join(g.cacheDir, fmt.Sprintf("%s.jsonl", split))

	// Check if cached
	if _, err := os.Stat(cachePath); err == nil {
		return g.loadFromCache(cachePath, split)
	}

	// Download if not cached
	if err := g.download(ctx, split, cachePath); err != nil {
		return fmt.Errorf("failed to download %s split: %w", split, err)
	}

	return g.loadFromCache(cachePath, split)
}

// loadFromCache loads examples from a cached file.
func (g *GSM8K) loadFromCache(path string, split string) error {
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("failed to open cache file: %w", err)
	}
	defer file.Close()

	examples, err := g.parseJSONL(file)
	if err != nil {
		return fmt.Errorf("failed to parse JSONL: %w", err)
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	switch split {
	case "train":
		g.trainExamples = examples
	case "test":
		g.testExamples = examples
	}

	return nil
}

// download downloads the dataset from Hugging Face or GitHub.
func (g *GSM8K) download(ctx context.Context, split string, destPath string) error {
	// Create cache directory
	if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
		return fmt.Errorf("failed to create cache directory: %w", err)
	}

	// URL for GSM8K dataset (HuggingFace datasets)
	var url string
	if split == "train" {
		url = "https://huggingface.co/datasets/gsm8k/resolve/main/train.jsonl"
	} else if split == "test" {
		url = "https://huggingface.co/datasets/gsm8k/resolve/main/test.jsonl"
	} else {
		return fmt.Errorf("unknown split: %s", split)
	}

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to download: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download failed with status: %d", resp.StatusCode)
	}

	// Save to cache
	out, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("failed to create cache file: %w", err)
	}
	defer out.Close()

	if _, err := io.Copy(out, resp.Body); err != nil {
		return fmt.Errorf("failed to save file: %w", err)
	}

	return nil
}

// parseJSONL parses GSM8K JSONL data.
func (g *GSM8K) parseJSONL(r io.Reader) ([]Example, error) {
	examples := make([]Example, 0)
	scanner := bufio.NewScanner(r)

	lineNum := 0
	for scanner.Scan() {
		lineNum++
		var raw gsm8kRaw
		if err := json.Unmarshal(scanner.Bytes(), &raw); err != nil {
			return nil, fmt.Errorf("failed to decode JSON at line %d: %w", lineNum, err)
		}

		// Extract numeric answer from the chain-of-thought
		answer := extractAnswer(raw.Answer)

		ex := Example{
			ID: fmt.Sprintf("gsm8k_%d", lineNum),
			Inputs: map[string]interface{}{
				"question": raw.Question,
			},
			Labels: map[string]interface{}{
				"answer":           answer,
				"chain_of_thought": raw.Answer,
			},
		}

		examples = append(examples, ex)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading JSONL: %w", err)
	}

	return examples, nil
}

// extractAnswer extracts the numeric answer from GSM8K chain-of-thought.
// GSM8K answers end with "#### <number>"
func extractAnswer(chainOfThought string) string {
	// Look for #### followed by a number
	re := regexp.MustCompile(`####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)`)
	matches := re.FindStringSubmatch(chainOfThought)
	if len(matches) > 1 {
		// Remove commas from numbers like "1,000"
		return strings.ReplaceAll(matches[1], ",", "")
	}

	// Fallback: try to find last number in the text
	re = regexp.MustCompile(`(-?\d+(?:,\d+)*(?:\.\d+)?)`)
	allMatches := re.FindAllString(chainOfThought, -1)
	if len(allMatches) > 0 {
		return strings.ReplaceAll(allMatches[len(allMatches)-1], ",", "")
	}

	return ""
}

// Split implements Dataset.Split.
func (g *GSM8K) Split(ratios ...float64) []Dataset {
	g.mu.RLock()
	defer g.mu.RUnlock()

	allExamples := append(g.trainExamples, g.testExamples...)
	ds := NewInMemoryDataset("gsm8k", allExamples)
	return ds.Split(ratios...)
}

// Batch implements Dataset.Batch.
func (g *GSM8K) Batch(size int) [][]Example {
	g.mu.RLock()
	defer g.mu.RUnlock()

	allExamples := append(g.trainExamples, g.testExamples...)
	ds := NewInMemoryDataset("gsm8k", allExamples)
	return ds.Batch(size)
}

// Shuffle implements Dataset.Shuffle.
func (g *GSM8K) Shuffle(seed int64) Dataset {
	g.mu.RLock()
	defer g.mu.RUnlock()

	allExamples := append(g.trainExamples, g.testExamples...)
	ds := NewInMemoryDataset("gsm8k", allExamples)
	return ds.Shuffle(seed)
}

// Filter implements Dataset.Filter.
func (g *GSM8K) Filter(predicate func(Example) bool) Dataset {
	g.mu.RLock()
	defer g.mu.RUnlock()

	allExamples := append(g.trainExamples, g.testExamples...)
	ds := NewInMemoryDataset("gsm8k", allExamples)
	return ds.Filter(predicate)
}

// LoadFromFile loads GSM8K examples from a local JSONL file.
func LoadGSM8KFromFile(path string) ([]Example, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	g := &GSM8K{}
	return g.parseJSONL(bufio.NewReader(file))
}
