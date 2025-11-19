package datasets

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sync"
)

// SQuAD represents the Stanford Question Answering Dataset (v1.1 and v2.0).
// Reference: https://rajpurkar.github.io/SQuAD-explorer/
type SQuAD struct {
	mu            sync.RWMutex
	trainExamples []Example
	devExamples   []Example
	cacheDir      string
	lazyLoad      bool
	loaded        bool
	version       string // "v1.1" or "v2.0"
}

// SQuADConfig configures the SQuAD dataset loader.
type SQuADConfig struct {
	CacheDir string
	LazyLoad bool
	Version  string // "v1.1" or "v2.0"
}

// squadRaw represents the SQuAD JSON structure.
type squadRaw struct {
	Data    []squadData `json:"data"`
	Version string      `json:"version,omitempty"`
}

type squadData struct {
	Title      string           `json:"title"`
	Paragraphs []squadParagraph `json:"paragraphs"`
}

type squadParagraph struct {
	Context string     `json:"context"`
	QAs     []squadQA `json:"qas"`
}

type squadQA struct {
	Question      string        `json:"question"`
	ID            string        `json:"id"`
	Answers       []squadAnswer `json:"answers"`
	IsImpossible  bool          `json:"is_impossible,omitempty"` // v2.0 only
	PlausibleAnswers []squadAnswer `json:"plausible_answers,omitempty"` // v2.0 only
}

type squadAnswer struct {
	Text       string `json:"text"`
	AnswerStart int   `json:"answer_start"`
}

// NewSQuAD creates a new SQuAD dataset loader.
func NewSQuAD(config SQuADConfig) *SQuAD {
	if config.CacheDir == "" {
		config.CacheDir = filepath.Join(os.TempDir(), "dspy", "squad")
	}
	if config.Version == "" {
		config.Version = "v2.0"
	}
	
	return &SQuAD{
		cacheDir: config.CacheDir,
		lazyLoad: config.LazyLoad,
		version:  config.Version,
	}
}

// Load loads all splits of the dataset.
func (s *SQuAD) Load(ctx context.Context) ([]Example, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	if s.loaded && !s.lazyLoad {
		return append(s.trainExamples, s.devExamples...), nil
	}
	
	// Load training data
	if err := s.loadSplit(ctx, "train"); err != nil {
		return nil, fmt.Errorf("failed to load train split: %w", err)
	}
	
	// Load dev data
	if err := s.loadSplit(ctx, "dev"); err != nil {
		return nil, fmt.Errorf("failed to load dev split: %w", err)
	}
	
	s.loaded = true
	return append(s.trainExamples, s.devExamples...), nil
}

// Train returns training examples.
func (s *SQuAD) Train() ([]Example, error) {
	s.mu.RLock()
	if len(s.trainExamples) > 0 {
		defer s.mu.RUnlock()
		return s.trainExamples, nil
	}
	s.mu.RUnlock()
	
	if err := s.loadSplit(context.Background(), "train"); err != nil {
		return nil, err
	}
	
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.trainExamples, nil
}

// Dev returns development examples.
func (s *SQuAD) Dev() ([]Example, error) {
	s.mu.RLock()
	if len(s.devExamples) > 0 {
		defer s.mu.RUnlock()
		return s.devExamples, nil
	}
	s.mu.RUnlock()
	
	if err := s.loadSplit(context.Background(), "dev"); err != nil {
		return nil, err
	}
	
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.devExamples, nil
}

// Len returns the total number of examples.
func (s *SQuAD) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.trainExamples) + len(s.devExamples)
}

// Version returns the SQuAD version.
func (s *SQuAD) Version() string {
	return s.version
}

// loadSplit loads a specific split from cache or downloads it.
func (s *SQuAD) loadSplit(ctx context.Context, split string) error {
	cachePath := filepath.Join(s.cacheDir, s.version, fmt.Sprintf("%s.json", split))
	
	// Check if cached
	if _, err := os.Stat(cachePath); err == nil {
		return s.loadFromCache(cachePath, split)
	}
	
	// Download if not cached
	if err := s.download(ctx, split, cachePath); err != nil {
		return fmt.Errorf("failed to download %s split: %w", split, err)
	}
	
	return s.loadFromCache(cachePath, split)
}

// loadFromCache loads examples from a cached file.
func (s *SQuAD) loadFromCache(path string, split string) error {
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("failed to open cache file: %w", err)
	}
	defer file.Close()
	
	examples, err := s.parseJSON(file)
	if err != nil {
		return fmt.Errorf("failed to parse JSON: %w", err)
	}
	
	s.mu.Lock()
	defer s.mu.Unlock()
	
	switch split {
	case "train":
		s.trainExamples = examples
	case "dev":
		s.devExamples = examples
	}
	
	return nil
}

// download downloads the dataset from the official source.
func (s *SQuAD) download(ctx context.Context, split string, destPath string) error {
	// Create cache directory
	if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
		return fmt.Errorf("failed to create cache directory: %w", err)
	}
	
	// URLs for SQuAD dataset
	var url string
	if s.version == "v1.1" {
		if split == "train" {
			url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
		} else {
			url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
		}
	} else { // v2.0
		if split == "train" {
			url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
		} else {
			url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
		}
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

// parseJSON parses SQuAD JSON data.
func (s *SQuAD) parseJSON(r io.Reader) ([]Example, error) {
	var raw squadRaw
	decoder := json.NewDecoder(r)
	if err := decoder.Decode(&raw); err != nil {
		return nil, fmt.Errorf("failed to decode JSON: %w", err)
	}
	
	examples := make([]Example, 0)
	
	for _, data := range raw.Data {
		for _, para := range data.Paragraphs {
			for _, qa := range para.QAs {
				// Extract answer text (use first answer for training)
				answerText := ""
				answerStart := -1
				if len(qa.Answers) > 0 {
					answerText = qa.Answers[0].Text
					answerStart = qa.Answers[0].AnswerStart
				}
				
				// Collect all answer texts
				allAnswers := make([]string, 0, len(qa.Answers))
				for _, ans := range qa.Answers {
					allAnswers = append(allAnswers, ans.Text)
				}
				
				ex := Example{
					ID: qa.ID,
					Inputs: map[string]interface{}{
						"question": qa.Question,
						"context":  para.Context,
						"title":    data.Title,
					},
					Labels: map[string]interface{}{
						"answer":        answerText,
						"answer_start":  answerStart,
						"all_answers":   allAnswers,
						"is_impossible": qa.IsImpossible,
					},
				}
				
				examples = append(examples, ex)
			}
		}
	}
	
	return examples, nil
}

// Split implements Dataset.Split.
func (s *SQuAD) Split(ratios ...float64) []Dataset {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	allExamples := append(s.trainExamples, s.devExamples...)
	ds := NewInMemoryDataset("squad", allExamples)
	return ds.Split(ratios...)
}

// Batch implements Dataset.Batch.
func (s *SQuAD) Batch(size int) [][]Example {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	allExamples := append(s.trainExamples, s.devExamples...)
	ds := NewInMemoryDataset("squad", allExamples)
	return ds.Batch(size)
}

// Shuffle implements Dataset.Shuffle.
func (s *SQuAD) Shuffle(seed int64) Dataset {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	allExamples := append(s.trainExamples, s.devExamples...)
	ds := NewInMemoryDataset("squad", allExamples)
	return ds.Shuffle(seed)
}

// Filter implements Dataset.Filter.
func (s *SQuAD) Filter(predicate func(Example) bool) Dataset {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	allExamples := append(s.trainExamples, s.devExamples...)
	ds := NewInMemoryDataset("squad", allExamples)
	return ds.Filter(predicate)
}
