package datasets

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"os"

	"github.com/stanfordnlp/dspy/internal/primitives"
)

// DataLoader provides generic dataset loading capabilities.
type DataLoader struct {
	batchSize    int
	shuffle      bool
	seed         int64
	numWorkers   int
	dropLast     bool
	examples     []*primitives.Example
	currentIndex int
}

// DataLoaderOptions configures the data loader.
type DataLoaderOptions struct {
	BatchSize  int
	Shuffle    bool
	Seed       int64
	NumWorkers int
	DropLast   bool // Drop the last incomplete batch
}

// DefaultDataLoaderOptions returns default data loader options.
func DefaultDataLoaderOptions() DataLoaderOptions {
	return DataLoaderOptions{
		BatchSize:  32,
		Shuffle:    false,
		Seed:       0,
		NumWorkers: 1,
		DropLast:   false,
	}
}

// NewDataLoader creates a new data loader for the given examples.
func NewDataLoader(examples []*primitives.Example, opts DataLoaderOptions) *DataLoader {
	dl := &DataLoader{
		batchSize:    opts.BatchSize,
		shuffle:      opts.Shuffle,
		seed:         opts.Seed,
		numWorkers:   opts.NumWorkers,
		dropLast:     opts.DropLast,
		examples:     examples,
		currentIndex: 0,
	}

	if dl.shuffle {
		dl.shuffleExamples()
	}

	return dl
}

// shuffleExamples shuffles the examples using the configured seed.
func (dl *DataLoader) shuffleExamples() {
	if dl.seed != 0 {
		r := rand.New(rand.NewSource(dl.seed))
		r.Shuffle(len(dl.examples), func(i, j int) {
			dl.examples[i], dl.examples[j] = dl.examples[j], dl.examples[i]
		})
	}
}

// Next returns the next batch of examples.
// Returns nil when all batches have been consumed.
func (dl *DataLoader) Next() []*primitives.Example {
	if dl.currentIndex >= len(dl.examples) {
		return nil
	}

	endIndex := dl.currentIndex + dl.batchSize
	if endIndex > len(dl.examples) {
		endIndex = len(dl.examples)

		// Drop last incomplete batch if configured
		if dl.dropLast && endIndex-dl.currentIndex < dl.batchSize {
			return nil
		}
	}

	batch := dl.examples[dl.currentIndex:endIndex]
	dl.currentIndex = endIndex

	return batch
}

// Reset resets the data loader to the beginning.
func (dl *DataLoader) Reset() {
	dl.currentIndex = 0
	if dl.shuffle {
		dl.shuffleExamples()
	}
}

// Len returns the total number of examples.
func (dl *DataLoader) Len() int {
	return len(dl.examples)
}

// NumBatches returns the number of batches.
func (dl *DataLoader) NumBatches() int {
	if dl.dropLast {
		return len(dl.examples) / dl.batchSize
	}
	return (len(dl.examples) + dl.batchSize - 1) / dl.batchSize
}

// StreamingDataLoader provides streaming dataset loading for large datasets.
type StreamingDataLoader struct {
	reader    io.ReadCloser
	batchSize int
	decoder   *json.Decoder
	closed    bool
}

// NewStreamingDataLoader creates a streaming data loader from a reader.
func NewStreamingDataLoader(reader io.ReadCloser, batchSize int) *StreamingDataLoader {
	return &StreamingDataLoader{
		reader:    reader,
		batchSize: batchSize,
		decoder:   json.NewDecoder(reader),
		closed:    false,
	}
}

// NextBatch reads the next batch from the stream.
func (sdl *StreamingDataLoader) NextBatch() ([]*primitives.Example, error) {
	if sdl.closed {
		return nil, io.EOF
	}

	batch := make([]*primitives.Example, 0, sdl.batchSize)

	for i := 0; i < sdl.batchSize; i++ {
		var data map[string]interface{}
		if err := sdl.decoder.Decode(&data); err != nil {
			if err == io.EOF {
				sdl.closed = true
				if len(batch) > 0 {
					return batch, nil
				}
				return nil, io.EOF
			}
			return nil, fmt.Errorf("failed to decode example: %w", err)
		}

		batch = append(batch, primitives.NewExample(nil, data))
	}

	return batch, nil
}

// Close closes the underlying reader.
func (sdl *StreamingDataLoader) Close() error {
	if !sdl.closed {
		sdl.closed = true
		return sdl.reader.Close()
	}
	return nil
}

// LoadFromJSONL loads examples from a JSONL file (one JSON object per line).
func LoadFromJSONL(path string) ([]*primitives.Example, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	var examples []*primitives.Example
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		var data map[string]interface{}
		if err := json.Unmarshal(scanner.Bytes(), &data); err != nil {
			return nil, fmt.Errorf("failed to parse JSON line: %w", err)
		}
		examples = append(examples, primitives.NewExample(nil, data))
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file: %w", err)
	}

	return examples, nil
}

// LoadFromJSON loads examples from a JSON file (array of objects).
func LoadFromJSON(path string) ([]*primitives.Example, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	var data []map[string]interface{}
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&data); err != nil {
		return nil, fmt.Errorf("failed to decode JSON: %w", err)
	}

	examples := make([]*primitives.Example, 0, len(data))
	for _, item := range data {
		examples = append(examples, primitives.NewExample(nil, item))
	}

	return examples, nil
}

// SaveToJSONL saves examples to a JSONL file.
func SaveToJSONL(path string, examples []*primitives.Example) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	for _, ex := range examples {
		if err := encoder.Encode(ex.ToMap()); err != nil {
			return fmt.Errorf("failed to encode example: %w", err)
		}
	}

	return nil
}

// HuggingFaceDatasetLoader provides integration with HuggingFace datasets.
type HuggingFaceDatasetLoader struct {
	datasetName string
	split       string
	config      string
}

// NewHuggingFaceDatasetLoader creates a loader for HuggingFace datasets.
func NewHuggingFaceDatasetLoader(datasetName, split, config string) *HuggingFaceDatasetLoader {
	return &HuggingFaceDatasetLoader{
		datasetName: datasetName,
		split:       split,
		config:      config,
	}
}

// Load loads the dataset from HuggingFace.
// This is a placeholder - actual implementation would need Python interop or HTTP API.
func (hf *HuggingFaceDatasetLoader) Load(ctx context.Context) ([]*primitives.Example, error) {
	return nil, fmt.Errorf("HuggingFace integration requires Python interop or API access")
}
