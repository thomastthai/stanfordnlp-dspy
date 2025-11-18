package datasets

import (
	"fmt"
)

// DataLoader provides utilities for loading datasets from various sources.
type DataLoader struct {
	// inputKeys specifies which fields are inputs (vs outputs/labels)
	inputKeys []string
}

// NewDataLoader creates a new DataLoader.
func NewDataLoader(inputKeys ...string) *DataLoader {
	return &DataLoader{
		inputKeys: inputKeys,
	}
}

// FromJSON loads a dataset from a JSON file.
func (dl *DataLoader) FromJSON(path string, fields []string) (Dataset, error) {
	return LoadFromJSON(path, fields)
}

// FromCSV loads a dataset from a CSV file.
func (dl *DataLoader) FromCSV(path string, fields []string) (Dataset, error) {
	return LoadFromCSV(path, fields)
}

// FromJSONL loads a dataset from a JSONL file.
func (dl *DataLoader) FromJSONL(path string, fields []string) (Dataset, error) {
	return LoadFromJSONL(path, fields)
}

// Batch creates batches of examples from a dataset.
func (dl *DataLoader) Batch(dataset Dataset, batchSize int) ([][]map[string]interface{}, error) {
	all, err := dataset.GetAll()
	if err != nil {
		return nil, err
	}

	if batchSize <= 0 {
		return nil, fmt.Errorf("batch size must be positive, got %d", batchSize)
	}

	numBatches := (len(all) + batchSize - 1) / batchSize
	batches := make([][]map[string]interface{}, 0, numBatches)

	for i := 0; i < len(all); i += batchSize {
		end := i + batchSize
		if end > len(all) {
			end = len(all)
		}
		batches = append(batches, all[i:end])
	}

	return batches, nil
}

// Split splits a dataset into train, validation, and test sets.
func (dl *DataLoader) Split(dataset Dataset, trainRatio, valRatio float64) (train, val, test Dataset, err error) {
	if trainRatio < 0 || trainRatio > 1 || valRatio < 0 || valRatio > 1 {
		return nil, nil, nil, fmt.Errorf("ratios must be between 0 and 1")
	}
	if trainRatio+valRatio > 1 {
		return nil, nil, nil, fmt.Errorf("train and val ratios must sum to at most 1")
	}

	all, err := dataset.GetAll()
	if err != nil {
		return nil, nil, nil, err
	}

	n := len(all)
	trainSize := int(float64(n) * trainRatio)
	valSize := int(float64(n) * valRatio)

	train = NewInMemoryDataset(all[:trainSize])
	val = NewInMemoryDataset(all[trainSize : trainSize+valSize])
	test = NewInMemoryDataset(all[trainSize+valSize:])

	return train, val, test, nil
}
