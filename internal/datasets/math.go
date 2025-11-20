package datasets

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"regexp"
	"strings"

	"github.com/stanfordnlp/dspy/internal/primitives"
)

// MATH represents the MATH competition mathematics dataset.
type MATH struct {
	*BaseDataset
	subset string
}

// MATHOptions extends DatasetOptions with MATH-specific options.
type MATHOptions struct {
	DatasetOptions
	Subset string // Subset/difficulty level to load
}

// DefaultMATHOptions returns default MATH options.
func DefaultMATHOptions() MATHOptions {
	return MATHOptions{
		DatasetOptions: DefaultDatasetOptions(),
		Subset:         "", // Empty means all subsets
	}
}

// MATHExample represents a single MATH problem from the dataset.
type MATHExample struct {
	Problem  string `json:"problem"`
	Solution string `json:"solution"`
	Level    string `json:"level"`
	Type     string `json:"type"`
}

// NewMATH creates a new MATH dataset loader.
// Note: This implementation splits the test set into train/dev/test since
// current LLMs are often trained on MATH's training set.
func NewMATH(ctx context.Context, opts MATHOptions) (*MATH, error) {
	base := NewBaseDataset("math", opts.DatasetOptions)
	dataset := &MATH{
		BaseDataset: base,
		subset:      opts.Subset,
	}

	// Load dataset - this is a placeholder that requires actual data loading
	if err := dataset.load(ctx); err != nil {
		return nil, err
	}

	return dataset, nil
}

// load loads the MATH dataset.
// This is a placeholder implementation that requires HuggingFace integration.
func (m *MATH) load(ctx context.Context) error {
	// Placeholder: In production, this would load from HuggingFace datasets
	return fmt.Errorf("MATH requires pre-downloaded data or HuggingFace API integration")
}

// ProcessMATHExample converts a raw MATH example to a DSPy Example.
func ProcessMATHExample(raw MATHExample) *primitives.Example {
	// Extract the answer from the solution using \boxed{}
	answer := ExtractAnswer(raw.Solution)

	data := map[string]interface{}{
		"question":  raw.Problem,
		"reasoning": raw.Solution,
		"answer":    answer,
		"level":     raw.Level,
		"type":      raw.Type,
	}

	ex := primitives.NewExample(nil, data)
	return ex.WithInputs("question")
}

// ExtractAnswer extracts the final answer from a MATH solution.
// The answer is typically enclosed in \boxed{...} LaTeX command.
func ExtractAnswer(solution string) string {
	start := strings.Index(solution, "\\boxed{")
	if start == -1 {
		return ""
	}

	idx := start + len("\\boxed{")
	braceLevel := 1
	answer := ""

	for idx < len(solution) && braceLevel > 0 {
		c := solution[idx]
		if c == '{' {
			braceLevel++
		} else if c == '}' {
			braceLevel--
			if braceLevel == 0 {
				break
			}
		}
		answer += string(c)
		idx++
	}

	// Clean up the answer
	// Remove \text{...} commands
	re := regexp.MustCompile(`\\text\{[^}]*\}`)
	answer = re.ReplaceAllString(answer, "")

	// Remove \! commands
	answer = strings.ReplaceAll(answer, "\\!", "")

	return strings.TrimSpace(answer)
}

// LoadMATHFromJSON loads MATH examples from a JSON file.
func LoadMATHFromJSON(r io.Reader) ([]*primitives.Example, error) {
	var rawExamples []MATHExample
	decoder := json.NewDecoder(r)
	if err := decoder.Decode(&rawExamples); err != nil {
		return nil, fmt.Errorf("failed to decode JSON: %w", err)
	}

	examples := make([]*primitives.Example, 0, len(rawExamples))
	for _, raw := range rawExamples {
		ex := ProcessMATHExample(raw)
		examples = append(examples, ex)
	}

	return examples, nil
}

// CreateMATHDatasetFromExamples creates a MATH dataset from pre-loaded examples.
// This splits the data into train/dev/test (1/3 each) with shuffling.
func CreateMATHDatasetFromExamples(examples []*primitives.Example, opts MATHOptions) *MATH {
	base := NewBaseDataset("math", opts.DatasetOptions)
	dataset := &MATH{
		BaseDataset: base,
		subset:      opts.Subset,
	}

	// Shuffle with seed 0
	shuffled := make([]*primitives.Example, len(examples))
	copy(shuffled, examples)
	r := rand.New(rand.NewSource(0))
	r.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})

	// Split into thirds (or use size parameter from examples)
	size := min(350, len(shuffled)/3)
	if size > len(shuffled) {
		size = len(shuffled) / 3
	}

	dataset.SetTrain(shuffled[:size])
	dataset.SetDev(shuffled[size : 2*size])
	dataset.SetTest(shuffled[2*size:])

	return dataset
}

// min returns the minimum of two integers.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// MATHMetric evaluates if the prediction matches the gold answer.
// This uses mathematical equivalence checking if available.
func MATHMetric(gold, pred *primitives.Example) bool {
	goldAnswer, ok := gold.Get("answer")
	if !ok {
		return false
	}

	predAnswer, ok := pred.Get("answer")
	if !ok {
		return false
	}

	goldStr, ok := goldAnswer.(string)
	if !ok {
		return false
	}

	predStr, ok := predAnswer.(string)
	if !ok {
		return false
	}

	// Simple string comparison (could be extended with mathematical equivalence)
	return strings.TrimSpace(goldStr) == strings.TrimSpace(predStr)
}
