package datasets

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// GSM8KExample represents a single GSM8K math problem.
type GSM8KExample struct {
	Question string `json:"question"`
	Answer   string `json:"answer"`
}

// LoadGSM8K loads the GSM8K dataset from a JSONL file.
func LoadGSM8K(path string) (Dataset, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening file: %w", err)
	}
	defer file.Close()

	var examples []map[string]interface{}
	decoder := json.NewDecoder(file)

	for {
		var ex GSM8KExample
		if err := decoder.Decode(&ex); err == io.EOF {
			break
		} else if err != nil {
			return nil, fmt.Errorf("decoding JSONL line: %w", err)
		}

		example := map[string]interface{}{
			"question": ex.Question,
			"answer":   ex.Answer,
		}
		examples = append(examples, example)
	}

	return NewInMemoryDataset(examples), nil
}
