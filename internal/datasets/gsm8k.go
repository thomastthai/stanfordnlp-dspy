package datasets

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"

	"github.com/stanfordnlp/dspy/internal/primitives"
)

// GSM8K represents the Grade School Math 8K dataset.
type GSM8K struct {
	*BaseDataset
}

// GSM8KExample represents a single GSM8K example from the dataset.
type GSM8KExample struct {
	Question string `json:"question"`
	Answer   string `json:"answer"`
}

// NewGSM8K creates a new GSM8K dataset loader.
// It loads the dataset from HuggingFace datasets hub.
func NewGSM8K(ctx context.Context, opts DatasetOptions) (*GSM8K, error) {
	base := NewBaseDataset("gsm8k", opts)
	dataset := &GSM8K{BaseDataset: base}
	
	// Load training data
	trainExamples, err := dataset.loadSplit(ctx, "train")
	if err != nil {
		return nil, fmt.Errorf("failed to load train split: %w", err)
	}
	
	// Load test data
	testExamples, err := dataset.loadSplit(ctx, "test")
	if err != nil {
		return nil, fmt.Errorf("failed to load test split: %w", err)
	}
	
	// Split training data into train/dev (similar to Python implementation)
	// Use first 200 for train, next 300 for dev
	if len(trainExamples) >= 500 {
		dataset.SetTrain(trainExamples[:200])
		dataset.SetDev(trainExamples[200:500])
	} else {
		// If not enough data, use what we have
		splitPoint := len(trainExamples) * 2 / 3
		dataset.SetTrain(trainExamples[:splitPoint])
		dataset.SetDev(trainExamples[splitPoint:])
	}
	
	dataset.SetTest(testExamples)
	
	return dataset, nil
}

// loadSplit loads a specific split from the HuggingFace dataset API.
func (g *GSM8K) loadSplit(ctx context.Context, split string) ([]*primitives.Example, error) {
	// This is a simplified loader. In production, you'd want to use the HuggingFace datasets library
	// or download and cache the dataset locally.
	// For now, we return an error indicating that data needs to be pre-downloaded.
	return nil, fmt.Errorf("GSM8K requires pre-downloaded data or HuggingFace API integration")
}

// ParseAnswer extracts the integer answer from a GSM8K answer string.
// The answer format is typically: "reasoning text #### 42"
func ParseAnswer(answer string) (reasoning string, finalAnswer int, err error) {
	parts := strings.Split(strings.TrimSpace(answer), "####")
	if len(parts) != 2 {
		return "", 0, fmt.Errorf("invalid answer format: expected '####' separator")
	}
	
	reasoning = strings.TrimSpace(parts[0])
	answerStr := strings.TrimSpace(parts[1])
	
	// Remove commas from numbers
	answerStr = strings.ReplaceAll(answerStr, ",", "")
	
	finalAnswer, err = strconv.Atoi(answerStr)
	if err != nil {
		return reasoning, 0, fmt.Errorf("failed to parse answer as integer: %w", err)
	}
	
	return reasoning, finalAnswer, nil
}

// ParseIntegerAnswer extracts an integer from a model's response.
// This is useful for evaluating model predictions against gold answers.
func ParseIntegerAnswer(answer string, onlyFirstLine bool) int {
	if onlyFirstLine {
		lines := strings.Split(strings.TrimSpace(answer), "\n")
		if len(lines) > 0 {
			answer = lines[0]
		}
	}
	
	// Find tokens with digits
	tokens := strings.Fields(answer)
	var lastNumberToken string
	for _, token := range tokens {
		if containsDigit(token) {
			lastNumberToken = token
		}
	}
	
	if lastNumberToken == "" {
		return 0
	}
	
	// Extract just the numeric part
	lastNumberToken = strings.Split(lastNumberToken, ".")[0]
	
	// Remove non-digit characters
	re := regexp.MustCompile(`\d+`)
	matches := re.FindAllString(lastNumberToken, -1)
	if len(matches) == 0 {
		return 0
	}
	
	numStr := strings.Join(matches, "")
	num, err := strconv.Atoi(numStr)
	if err != nil {
		return 0
	}
	
	return num
}

func containsDigit(s string) bool {
	for _, c := range s {
		if c >= '0' && c <= '9' {
			return true
		}
	}
	return false
}

// GSM8KMetric compares a prediction against the gold answer.
func GSM8KMetric(gold, pred *primitives.Example) bool {
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
	
	goldInt := ParseIntegerAnswer(goldStr, false)
	predInt := ParseIntegerAnswer(predStr, true)
	
	return goldInt == predInt
}

// LoadGSM8KFromJSON loads GSM8K examples from a JSON file.
// This is a helper function for loading pre-downloaded dataset files.
func LoadGSM8KFromJSON(r io.Reader) ([]*primitives.Example, error) {
	var rawExamples []GSM8KExample
	decoder := json.NewDecoder(r)
	if err := decoder.Decode(&rawExamples); err != nil {
		return nil, fmt.Errorf("failed to decode JSON: %w", err)
	}
	
	examples := make([]*primitives.Example, 0, len(rawExamples))
	for _, raw := range rawExamples {
		reasoning, answer, err := ParseAnswer(raw.Answer)
		if err != nil {
			// Skip malformed examples
			continue
		}
		
		ex := primitives.NewExample(nil, map[string]interface{}{
			"question":       raw.Question,
			"gold_reasoning": reasoning,
			"answer":         strconv.Itoa(answer),
		})
		ex = ex.WithInputs("question")
		examples = append(examples, ex)
	}
	
	return examples, nil
}

// DownloadGSM8K downloads GSM8K data from HuggingFace (placeholder implementation).
func DownloadGSM8K(ctx context.Context, split string) ([]GSM8KExample, error) {
	// This would need to be implemented with proper HuggingFace API integration
	// For now, return an error
	url := fmt.Sprintf("https://huggingface.co/datasets/gsm8k/resolve/main/%s.jsonl", split)
	
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to download: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}
	
	var examples []GSM8KExample
	decoder := json.NewDecoder(resp.Body)
	
	// Read JSONL format (one JSON object per line)
	for {
		var ex GSM8KExample
		if err := decoder.Decode(&ex); err == io.EOF {
			break
		} else if err != nil {
			return nil, fmt.Errorf("failed to decode example: %w", err)
		}
		examples = append(examples, ex)
	}
	
	return examples, nil
}
