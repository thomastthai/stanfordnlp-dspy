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
	"strings"
	"sync"
)

// HellaSwag represents the HellaSwag commonsense NLI dataset.
// Reference: https://rowanzellers.com/hellaswag/
type HellaSwag struct {
	mu            sync.RWMutex
	trainExamples []Example
	valExamples   []Example
	testExamples  []Example
	cacheDir      string
	lazyLoad      bool
	loaded        bool
}

// HellaSwagConfig configures the HellaSwag dataset loader.
type HellaSwagConfig struct {
	CacheDir string
	LazyLoad bool
}

// hellaSwagRaw represents a raw HellaSwag example from JSON.
type hellaSwagRaw struct {
	IndActivityLabel string   `json:"ind_activity_label"`
	Context          string   `json:"ctx"`
	ContextA         string   `json:"ctx_a"`
	ContextB         string   `json:"ctx_b"`
	Activity         string   `json:"activity_label"`
	Endings          []string `json:"endings"`
	Label            int      `json:"label,omitempty"` // Only in train/val
	SourceID         string   `json:"source_id"`
	SplitType        string   `json:"split_type"`
}

// NewHellaSwag creates a new HellaSwag dataset loader.
func NewHellaSwag(config HellaSwagConfig) *HellaSwag {
	if config.CacheDir == "" {
		config.CacheDir = filepath.Join(os.TempDir(), "dspy", "hellaswag")
	}

	return &HellaSwag{
		cacheDir: config.CacheDir,
		lazyLoad: config.LazyLoad,
	}
}

// Load loads all splits of the dataset.
func (h *HellaSwag) Load(ctx context.Context) ([]Example, error) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.loaded && !h.lazyLoad {
		return append(append(h.trainExamples, h.valExamples...), h.testExamples...), nil
	}

	// Load training data
	if err := h.loadSplit(ctx, "train"); err != nil {
		return nil, fmt.Errorf("failed to load train split: %w", err)
	}

	// Load validation data
	if err := h.loadSplit(ctx, "val"); err != nil {
		return nil, fmt.Errorf("failed to load val split: %w", err)
	}

	h.loaded = true
	return append(h.trainExamples, h.valExamples...), nil
}

// Train returns training examples.
func (h *HellaSwag) Train() ([]Example, error) {
	h.mu.RLock()
	if len(h.trainExamples) > 0 {
		defer h.mu.RUnlock()
		return h.trainExamples, nil
	}
	h.mu.RUnlock()

	if err := h.loadSplit(context.Background(), "train"); err != nil {
		return nil, err
	}

	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.trainExamples, nil
}

// Val returns validation examples.
func (h *HellaSwag) Val() ([]Example, error) {
	h.mu.RLock()
	if len(h.valExamples) > 0 {
		defer h.mu.RUnlock()
		return h.valExamples, nil
	}
	h.mu.RUnlock()

	if err := h.loadSplit(context.Background(), "val"); err != nil {
		return nil, err
	}

	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.valExamples, nil
}

// Len returns the total number of examples.
func (h *HellaSwag) Len() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.trainExamples) + len(h.valExamples) + len(h.testExamples)
}

// loadSplit loads a specific split from cache or downloads it.
func (h *HellaSwag) loadSplit(ctx context.Context, split string) error {
	cachePath := filepath.Join(h.cacheDir, fmt.Sprintf("hellaswag_%s.jsonl", split))

	// Check if cached
	if _, err := os.Stat(cachePath); err == nil {
		return h.loadFromCache(cachePath, split)
	}

	// Download if not cached
	if err := h.download(ctx, split, cachePath); err != nil {
		return fmt.Errorf("failed to download %s split: %w", split, err)
	}

	return h.loadFromCache(cachePath, split)
}

// loadFromCache loads examples from a cached file.
func (h *HellaSwag) loadFromCache(path string, split string) error {
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("failed to open cache file: %w", err)
	}
	defer file.Close()

	examples, err := h.parseJSONL(file)
	if err != nil {
		return fmt.Errorf("failed to parse JSONL: %w", err)
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	switch split {
	case "train":
		h.trainExamples = examples
	case "val":
		h.valExamples = examples
	case "test":
		h.testExamples = examples
	}

	return nil
}

// download downloads the dataset from GitHub or HuggingFace.
func (h *HellaSwag) download(ctx context.Context, split string, destPath string) error {
	// Create cache directory
	if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
		return fmt.Errorf("failed to create cache directory: %w", err)
	}

	// URL for HellaSwag dataset
	var url string
	switch split {
	case "train":
		url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl"
	case "val":
		url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
	case "test":
		url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl"
	default:
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

// parseJSONL parses HellaSwag JSONL data.
func (h *HellaSwag) parseJSONL(r io.Reader) ([]Example, error) {
	examples := make([]Example, 0)
	scanner := bufio.NewScanner(r)

	lineNum := 0
	for scanner.Scan() {
		lineNum++
		var raw hellaSwagRaw
		if err := json.Unmarshal(scanner.Bytes(), &raw); err != nil {
			return nil, fmt.Errorf("failed to decode JSON at line %d: %w", lineNum, err)
		}

		// Build full context from parts
		context := raw.Context
		if raw.ContextA != "" {
			context = context + " " + raw.ContextA
		}
		if raw.ContextB != "" {
			context = context + " " + raw.ContextB
		}
		context = strings.TrimSpace(context)

		ex := Example{
			ID: raw.SourceID,
			Inputs: map[string]interface{}{
				"context":  context,
				"endings":  raw.Endings,
				"activity": raw.Activity,
			},
			Labels: map[string]interface{}{
				"label": raw.Label,
			},
		}

		examples = append(examples, ex)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading JSONL: %w", err)
	}

	return examples, nil
}

// Split implements Dataset.Split.
func (h *HellaSwag) Split(ratios ...float64) []Dataset {
	h.mu.RLock()
	defer h.mu.RUnlock()

	allExamples := append(append(h.trainExamples, h.valExamples...), h.testExamples...)
	ds := NewInMemoryDataset("hellaswag", allExamples)
	return ds.Split(ratios...)
}

// Batch implements Dataset.Batch.
func (h *HellaSwag) Batch(size int) [][]Example {
	h.mu.RLock()
	defer h.mu.RUnlock()

	allExamples := append(append(h.trainExamples, h.valExamples...), h.testExamples...)
	ds := NewInMemoryDataset("hellaswag", allExamples)
	return ds.Batch(size)
}

// Shuffle implements Dataset.Shuffle.
func (h *HellaSwag) Shuffle(seed int64) Dataset {
	h.mu.RLock()
	defer h.mu.RUnlock()

	allExamples := append(append(h.trainExamples, h.valExamples...), h.testExamples...)
	ds := NewInMemoryDataset("hellaswag", allExamples)
	return ds.Shuffle(seed)
}

// Filter implements Dataset.Filter.
func (h *HellaSwag) Filter(predicate func(Example) bool) Dataset {
	h.mu.RLock()
	defer h.mu.RUnlock()

	allExamples := append(append(h.trainExamples, h.valExamples...), h.testExamples...)
	ds := NewInMemoryDataset("hellaswag", allExamples)
	return ds.Filter(predicate)
}
