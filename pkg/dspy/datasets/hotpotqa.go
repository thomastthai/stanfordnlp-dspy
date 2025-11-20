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
	"sync"
)

// HotPotQA represents the HotPotQA multi-hop question answering dataset.
// Reference: https://hotpotqa.github.io/
type HotPotQA struct {
	mu            sync.RWMutex
	trainExamples []Example
	devExamples   []Example
	testExamples  []Example
	cacheDir      string
	lazyLoad      bool
	loaded        bool
	setting       string // "distractor" or "fullwiki"
}

// HotPotQAConfig configures the HotPotQA dataset loader.
type HotPotQAConfig struct {
	CacheDir string
	LazyLoad bool
	Setting  string // "distractor" or "fullwiki"
	OnlyHard bool
}

// hotPotQARaw represents a raw HotPotQA example from JSON.
type hotPotQARaw struct {
	ID              string          `json:"_id"`
	Question        string          `json:"question"`
	Answer          string          `json:"answer"`
	Type            string          `json:"type"`
	Level           string          `json:"level"`
	SupportingFacts [][]interface{} `json:"supporting_facts"`
	Context         [][]string      `json:"context"`
}

// NewHotPotQA creates a new HotPotQA dataset loader.
func NewHotPotQA(config HotPotQAConfig) *HotPotQA {
	if config.CacheDir == "" {
		config.CacheDir = getCacheDir("hotpotqa")
	}
	if config.Setting == "" {
		config.Setting = "distractor"
	}

	return &HotPotQA{
		cacheDir: config.CacheDir,
		lazyLoad: config.LazyLoad,
		setting:  config.Setting,
	}
}

// Load loads all splits of the dataset.
func (h *HotPotQA) Load(ctx context.Context) ([]Example, error) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.loaded && !h.lazyLoad {
		return append(append(h.trainExamples, h.devExamples...), h.testExamples...), nil
	}

	// Load training data
	if err := h.loadSplit(ctx, "train"); err != nil {
		return nil, fmt.Errorf("failed to load train split: %w", err)
	}

	// Load dev data
	if err := h.loadSplit(ctx, "dev"); err != nil {
		return nil, fmt.Errorf("failed to load dev split: %w", err)
	}

	h.loaded = true
	return append(h.trainExamples, h.devExamples...), nil
}

// Train returns training examples.
func (h *HotPotQA) Train() ([]Example, error) {
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

// Dev returns development examples.
func (h *HotPotQA) Dev() ([]Example, error) {
	h.mu.RLock()
	if len(h.devExamples) > 0 {
		defer h.mu.RUnlock()
		return h.devExamples, nil
	}
	h.mu.RUnlock()

	if err := h.loadSplit(context.Background(), "dev"); err != nil {
		return nil, err
	}

	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.devExamples, nil
}

// Len returns the total number of examples.
func (h *HotPotQA) Len() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.trainExamples) + len(h.devExamples) + len(h.testExamples)
}

// loadSplit loads a specific split from cache or downloads it.
func (h *HotPotQA) loadSplit(ctx context.Context, split string) error {
	cachePath := filepath.Join(h.cacheDir, fmt.Sprintf("%s_%s.json", split, h.setting))

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
func (h *HotPotQA) loadFromCache(path string, split string) error {
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("failed to open cache file: %w", err)
	}
	defer file.Close()

	examples, err := h.parseJSON(file)
	if err != nil {
		return fmt.Errorf("failed to parse JSON: %w", err)
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	switch split {
	case "train":
		h.trainExamples = examples
	case "dev":
		h.devExamples = examples
	case "test":
		h.testExamples = examples
	}

	return nil
}

// download downloads the dataset from HotPotQA.
func (h *HotPotQA) download(ctx context.Context, split string, destPath string) error {
	// Create cache directory
	if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
		return fmt.Errorf("failed to create cache directory: %w", err)
	}

	// URL for HotPotQA dataset (this is a placeholder - actual URLs may vary)
	var url string
	if split == "train" {
		url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
	} else if split == "dev" {
		url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
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

// parseJSON parses HotPotQA JSON data.
func (h *HotPotQA) parseJSON(r io.Reader) ([]Example, error) {
	var rawExamples []hotPotQARaw

	decoder := json.NewDecoder(r)
	if err := decoder.Decode(&rawExamples); err != nil {
		return nil, fmt.Errorf("failed to decode JSON: %w", err)
	}

	examples := make([]Example, 0, len(rawExamples))
	for _, raw := range rawExamples {
		// Convert context to string format
		contextStr := ""
		for _, ctx := range raw.Context {
			if len(ctx) > 0 {
				contextStr += ctx[0] + ": "
				if len(ctx) > 1 {
					contextStr += ctx[1]
				}
				contextStr += "\n"
			}
		}

		// Extract supporting fact titles
		goldTitles := make([]string, 0)
		for _, fact := range raw.SupportingFacts {
			if len(fact) > 0 {
				if title, ok := fact[0].(string); ok {
					goldTitles = append(goldTitles, title)
				}
			}
		}

		ex := Example{
			ID: raw.ID,
			Inputs: map[string]interface{}{
				"question": raw.Question,
				"context":  contextStr,
			},
			Labels: map[string]interface{}{
				"answer":      raw.Answer,
				"gold_titles": goldTitles,
				"type":        raw.Type,
				"level":       raw.Level,
			},
		}

		examples = append(examples, ex)
	}

	return examples, nil
}

// Split implements Dataset.Split.
func (h *HotPotQA) Split(ratios ...float64) []Dataset {
	h.mu.RLock()
	defer h.mu.RUnlock()

	allExamples := append(append(h.trainExamples, h.devExamples...), h.testExamples...)
	ds := NewInMemoryDataset("hotpotqa", allExamples)
	return ds.Split(ratios...)
}

// Batch implements Dataset.Batch.
func (h *HotPotQA) Batch(size int) [][]Example {
	h.mu.RLock()
	defer h.mu.RUnlock()

	allExamples := append(append(h.trainExamples, h.devExamples...), h.testExamples...)
	ds := NewInMemoryDataset("hotpotqa", allExamples)
	return ds.Batch(size)
}

// Shuffle implements Dataset.Shuffle.
func (h *HotPotQA) Shuffle(seed int64) Dataset {
	h.mu.RLock()
	defer h.mu.RUnlock()

	allExamples := append(append(h.trainExamples, h.devExamples...), h.testExamples...)
	ds := NewInMemoryDataset("hotpotqa", allExamples)
	return ds.Shuffle(seed)
}

// Filter implements Dataset.Filter.
func (h *HotPotQA) Filter(predicate func(Example) bool) Dataset {
	h.mu.RLock()
	defer h.mu.RUnlock()

	allExamples := append(append(h.trainExamples, h.devExamples...), h.testExamples...)
	ds := NewInMemoryDataset("hotpotqa", allExamples)
	return ds.Filter(predicate)
}

// LoadFromFile loads HotPotQA examples from a local JSON file.
func LoadHotPotQAFromFile(path string) ([]Example, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	h := &HotPotQA{}
	return h.parseJSON(bufio.NewReader(file))
}
