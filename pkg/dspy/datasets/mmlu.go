package datasets

import (
	"bufio"
	"context"
	"encoding/csv"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// MMLU represents the Massive Multitask Language Understanding dataset.
// Reference: https://github.com/hendrycks/test
type MMLU struct {
	mu            sync.RWMutex
	examples      map[string][]Example // subject -> examples
	cacheDir      string
	lazyLoad      bool
	loaded        bool
	subjects      []string
	filterSubject string
}

// MMLUConfig configures the MMLU dataset loader.
type MMLUConfig struct {
	CacheDir      string
	LazyLoad      bool
	FilterSubject string // Optional: load only specific subject
}

// All 57 MMLU subjects
var MMLUSubjects = []string{
	"abstract_algebra", "anatomy", "astronomy", "business_ethics",
	"clinical_knowledge", "college_biology", "college_chemistry",
	"college_computer_science", "college_mathematics", "college_medicine",
	"college_physics", "computer_security", "conceptual_physics",
	"econometrics", "electrical_engineering", "elementary_mathematics",
	"formal_logic", "global_facts", "high_school_biology",
	"high_school_chemistry", "high_school_computer_science",
	"high_school_european_history", "high_school_geography",
	"high_school_government_and_politics", "high_school_macroeconomics",
	"high_school_mathematics", "high_school_microeconomics",
	"high_school_physics", "high_school_psychology",
	"high_school_statistics", "high_school_us_history",
	"high_school_world_history", "human_aging", "human_sexuality",
	"international_law", "jurisprudence", "logical_fallacies",
	"machine_learning", "management", "marketing", "medical_genetics",
	"miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
	"philosophy", "prehistory", "professional_accounting",
	"professional_law", "professional_medicine", "professional_psychology",
	"public_relations", "security_studies", "sociology",
	"us_foreign_policy", "virology", "world_religions",
}

// NewMMLU creates a new MMLU dataset loader.
func NewMMLU(config MMLUConfig) *MMLU {
	if config.CacheDir == "" {
		config.CacheDir = filepath.Join(os.TempDir(), "dspy", "mmlu")
	}
	
	subjects := MMLUSubjects
	if config.FilterSubject != "" {
		subjects = []string{config.FilterSubject}
	}
	
	return &MMLU{
		cacheDir:      config.CacheDir,
		lazyLoad:      config.LazyLoad,
		subjects:      subjects,
		filterSubject: config.FilterSubject,
		examples:      make(map[string][]Example),
	}
}

// Load loads all subjects of the dataset.
func (m *MMLU) Load(ctx context.Context) ([]Example, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	if m.loaded && !m.lazyLoad {
		return m.getAllExamples(), nil
	}
	
	for _, subject := range m.subjects {
		if err := m.loadSubject(ctx, subject); err != nil {
			return nil, fmt.Errorf("failed to load subject %s: %w", subject, err)
		}
	}
	
	m.loaded = true
	return m.getAllExamples(), nil
}

// LoadSubject loads examples for a specific subject.
func (m *MMLU) LoadSubject(ctx context.Context, subject string) ([]Example, error) {
	m.mu.RLock()
	if examples, ok := m.examples[subject]; ok && len(examples) > 0 {
		m.mu.RUnlock()
		return examples, nil
	}
	m.mu.RUnlock()
	
	if err := m.loadSubject(ctx, subject); err != nil {
		return nil, err
	}
	
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.examples[subject], nil
}

// Subjects returns all available subjects.
func (m *MMLU) Subjects() []string {
	return m.subjects
}

// Len returns the total number of examples across all subjects.
func (m *MMLU) Len() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	count := 0
	for _, examples := range m.examples {
		count += len(examples)
	}
	return count
}

// loadSubject loads a specific subject from cache or downloads it.
func (m *MMLU) loadSubject(ctx context.Context, subject string) error {
	// MMLU has dev, val, and test splits
	splits := []string{"dev", "val", "test"}
	allExamples := make([]Example, 0)
	
	for _, split := range splits {
		cachePath := filepath.Join(m.cacheDir, subject, fmt.Sprintf("%s_%s.csv", split, subject))
		
		// Check if cached
		if _, err := os.Stat(cachePath); err != nil {
			// Download if not cached
			if err := m.download(ctx, subject, split, cachePath); err != nil {
				// Some subjects may not have all splits, continue
				continue
			}
		}
		
		examples, err := m.loadFromCache(cachePath, subject, split)
		if err != nil {
			return fmt.Errorf("failed to load %s split: %w", split, err)
		}
		
		allExamples = append(allExamples, examples...)
	}
	
	m.mu.Lock()
	m.examples[subject] = allExamples
	m.mu.Unlock()
	
	return nil
}

// loadFromCache loads examples from a cached CSV file.
func (m *MMLU) loadFromCache(path string, subject string, split string) ([]Example, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open cache file: %w", err)
	}
	defer file.Close()
	
	return m.parseCSV(file, subject, split)
}

// download downloads the dataset from GitHub.
func (m *MMLU) download(ctx context.Context, subject string, split string, destPath string) error {
	// Create cache directory
	if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
		return fmt.Errorf("failed to create cache directory: %w", err)
	}
	
	// URL for MMLU dataset
	url := fmt.Sprintf("https://raw.githubusercontent.com/hendrycks/test/master/data/%s/%s_%s.csv",
		split, split, subject)
	
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

// parseCSV parses MMLU CSV data.
// Format: question,choice_a,choice_b,choice_c,choice_d,answer
func (m *MMLU) parseCSV(r io.Reader, subject string, split string) ([]Example, error) {
	reader := csv.NewReader(bufio.NewReader(r))
	examples := make([]Example, 0)
	
	lineNum := 0
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read CSV: %w", err)
		}
		
		lineNum++
		
		// Skip if not enough fields
		if len(record) < 6 {
			continue
		}
		
		question := strings.TrimSpace(record[0])
		choices := []string{
			strings.TrimSpace(record[1]),
			strings.TrimSpace(record[2]),
			strings.TrimSpace(record[3]),
			strings.TrimSpace(record[4]),
		}
		answer := strings.TrimSpace(record[5])
		
		// Convert answer letter to index (A=0, B=1, C=2, D=3)
		answerIndex := -1
		if len(answer) > 0 {
			answerIndex = int(answer[0] - 'A')
		}
		
		ex := Example{
			ID: fmt.Sprintf("mmlu_%s_%s_%d", subject, split, lineNum),
			Inputs: map[string]interface{}{
				"question": question,
				"choices":  choices,
				"subject":  subject,
			},
			Labels: map[string]interface{}{
				"answer":       answer,
				"answer_index": answerIndex,
			},
		}
		
		examples = append(examples, ex)
	}
	
	return examples, nil
}

// getAllExamples returns all examples across all subjects.
func (m *MMLU) getAllExamples() []Example {
	allExamples := make([]Example, 0)
	for _, examples := range m.examples {
		allExamples = append(allExamples, examples...)
	}
	return allExamples
}

// Split implements Dataset.Split.
func (m *MMLU) Split(ratios ...float64) []Dataset {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	allExamples := m.getAllExamples()
	ds := NewInMemoryDataset("mmlu", allExamples)
	return ds.Split(ratios...)
}

// Batch implements Dataset.Batch.
func (m *MMLU) Batch(size int) [][]Example {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	allExamples := m.getAllExamples()
	ds := NewInMemoryDataset("mmlu", allExamples)
	return ds.Batch(size)
}

// Shuffle implements Dataset.Shuffle.
func (m *MMLU) Shuffle(seed int64) Dataset {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	allExamples := m.getAllExamples()
	ds := NewInMemoryDataset("mmlu", allExamples)
	return ds.Shuffle(seed)
}

// Filter implements Dataset.Filter.
func (m *MMLU) Filter(predicate func(Example) bool) Dataset {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	allExamples := m.getAllExamples()
	ds := NewInMemoryDataset("mmlu", allExamples)
	return ds.Filter(predicate)
}
