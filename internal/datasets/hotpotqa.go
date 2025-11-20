package datasets

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/stanfordnlp/dspy/internal/primitives"
)

// HotPotQA represents the HotPotQA multi-hop question answering dataset.
type HotPotQA struct {
	*BaseDataset
	onlyHardExamples bool
	keepDetails      string
}

// HotPotQAOptions extends DatasetOptions with HotPotQA-specific options.
type HotPotQAOptions struct {
	DatasetOptions
	OnlyHardExamples bool
	KeepDetails      string // "dev_titles", true, or false
	UnofficialDev    bool
}

// DefaultHotPotQAOptions returns default HotPotQA options.
func DefaultHotPotQAOptions() HotPotQAOptions {
	return HotPotQAOptions{
		DatasetOptions:   DefaultDatasetOptions(),
		OnlyHardExamples: true,
		KeepDetails:      "dev_titles",
		UnofficialDev:    true,
	}
}

// HotPotQARawExample represents a raw example from the dataset.
type HotPotQARawExample struct {
	ID              string          `json:"_id"`
	Question        string          `json:"question"`
	Answer          string          `json:"answer"`
	Type            string          `json:"type"`
	Level           string          `json:"level"`
	SupportingFacts SupportingFacts `json:"supporting_facts"`
	Context         [][]string      `json:"context"`
}

// SupportingFacts contains the titles and sentence indices of supporting facts.
type SupportingFacts struct {
	Title   []string `json:"title"`
	SentIdx []int    `json:"sent_id"`
}

// NewHotPotQA creates a new HotPotQA dataset loader.
func NewHotPotQA(ctx context.Context, opts HotPotQAOptions) (*HotPotQA, error) {
	if !opts.OnlyHardExamples {
		return nil, fmt.Errorf("only hard examples are currently supported")
	}

	base := NewBaseDataset("hotpotqa", opts.DatasetOptions)
	dataset := &HotPotQA{
		BaseDataset:      base,
		onlyHardExamples: opts.OnlyHardExamples,
		keepDetails:      opts.KeepDetails,
	}

	// Load training data
	trainExamples, err := dataset.loadSplit(ctx, "train", opts)
	if err != nil {
		return nil, fmt.Errorf("failed to load train split: %w", err)
	}

	// Split training data into train/dev
	if opts.UnofficialDev {
		splitPoint := len(trainExamples) * 75 / 100
		dataset.SetTrain(trainExamples[:splitPoint])
		dataset.SetDev(trainExamples[splitPoint:])
	} else {
		dataset.SetTrain(trainExamples)
	}

	// Load test data (official validation set)
	testExamples, err := dataset.loadSplit(ctx, "validation", opts)
	if err != nil {
		return nil, fmt.Errorf("failed to load validation split: %w", err)
	}
	dataset.SetTest(testExamples)

	return dataset, nil
}

// loadSplit loads a specific split from the dataset.
func (h *HotPotQA) loadSplit(ctx context.Context, split string, opts HotPotQAOptions) ([]*primitives.Example, error) {
	// This is a placeholder. In production, you'd integrate with HuggingFace datasets
	// or load from pre-downloaded files.
	return nil, fmt.Errorf("HotPotQA requires pre-downloaded data or HuggingFace API integration")
}

// ProcessRawExample converts a raw HotPotQA example to a DSPy Example.
func ProcessRawExample(raw HotPotQARawExample, keepDetails string) *primitives.Example {
	data := make(map[string]interface{})

	// Determine which fields to keep
	var keys []string
	if keepDetails == "true" {
		keys = []string{"id", "question", "answer", "type", "supporting_facts", "context"}
	} else if keepDetails == "dev_titles" {
		keys = []string{"question", "answer", "supporting_facts"}
	} else {
		keys = []string{"question", "answer"}
	}

	// Extract requested fields
	for _, key := range keys {
		switch key {
		case "id":
			data["id"] = raw.ID
		case "question":
			data["question"] = raw.Question
		case "answer":
			data["answer"] = raw.Answer
		case "type":
			data["type"] = raw.Type
		case "supporting_facts":
			// Extract gold titles from supporting facts
			goldTitles := make([]string, len(raw.SupportingFacts.Title))
			copy(goldTitles, raw.SupportingFacts.Title)
			data["gold_titles"] = goldTitles
		case "context":
			data["context"] = raw.Context
		}
	}

	ex := primitives.NewExample(nil, data)
	return ex.WithInputs("question")
}

// LoadHotPotQAFromJSON loads HotPotQA examples from a JSON file.
func LoadHotPotQAFromJSON(r io.Reader, keepDetails string, onlyHard bool) ([]*primitives.Example, error) {
	var rawExamples []HotPotQARawExample
	decoder := json.NewDecoder(r)
	if err := decoder.Decode(&rawExamples); err != nil {
		return nil, fmt.Errorf("failed to decode JSON: %w", err)
	}

	examples := make([]*primitives.Example, 0, len(rawExamples))
	for _, raw := range rawExamples {
		// Filter by difficulty level
		if onlyHard && raw.Level != "hard" {
			continue
		}

		ex := ProcessRawExample(raw, keepDetails)
		examples = append(examples, ex)
	}

	return examples, nil
}

// HotPotQAMetric evaluates if the prediction matches the gold answer.
// For multi-hop QA, we typically do exact match or F1 score.
func HotPotQAMetric(gold, pred *primitives.Example) float64 {
	goldAnswer, ok := gold.Get("answer")
	if !ok {
		return 0.0
	}

	predAnswer, ok := pred.Get("answer")
	if !ok {
		return 0.0
	}

	goldStr, ok := goldAnswer.(string)
	if !ok {
		return 0.0
	}

	predStr, ok := predAnswer.(string)
	if !ok {
		return 0.0
	}

	// Simple exact match (could be extended with F1 score)
	if goldStr == predStr {
		return 1.0
	}
	return 0.0
}
