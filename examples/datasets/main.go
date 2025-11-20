package main

import (
	"fmt"

	"github.com/stanfordnlp/dspy/internal/datasets"
)

func main() {
	fmt.Println("=== DSPy Go Datasets Example ===\n")

	// Example 1: Colors Dataset
	fmt.Println("1. Colors Dataset:")
	colorsOpts := datasets.DefaultColorsOptions()
	colorsOpts.TrainSize = 10 // Limit to 10 for demo
	colorsOpts.DevSize = 5    // Limit to 5 for demo

	colors := datasets.NewColors(colorsOpts)
	fmt.Printf("   Name: %s\n", colors.Name())
	fmt.Printf("   Total: %d examples\n", colors.Len())
	fmt.Printf("   Train: %d examples\n", len(colors.Train()))
	fmt.Printf("   Dev: %d examples\n", len(colors.Dev()))

	// Show a few color examples
	fmt.Println("   Sample colors:")
	for i, ex := range colors.Train() {
		if i >= 3 {
			break
		}
		if color, ok := ex.Get("color"); ok {
			fmt.Printf("   - %s\n", color)
		}
	}
	fmt.Println()

	// Example 2: DataLoader with Batching
	fmt.Println("2. DataLoader with Batching:")
	loaderOpts := datasets.DataLoaderOptions{
		BatchSize:  3,
		Shuffle:    false,
		Seed:       0,
		NumWorkers: 1,
		DropLast:   false,
	}

	loader := datasets.NewDataLoader(colors.Train(), loaderOpts)
	fmt.Printf("   Total examples: %d\n", loader.Len())
	fmt.Printf("   Batch size: %d\n", loaderOpts.BatchSize)
	fmt.Printf("   Number of batches: %d\n", loader.NumBatches())

	fmt.Println("   Batches:")
	batchNum := 0
	for {
		batch := loader.Next()
		if batch == nil {
			break
		}
		batchNum++
		fmt.Printf("   - Batch %d: %d examples\n", batchNum, len(batch))
	}
	fmt.Println()

	// Example 3: Dataset Splitting
	fmt.Println("3. Dataset Splitting:")
	fullColors := datasets.NewColors(datasets.DefaultColorsOptions())
	allExamples := append(fullColors.Train(), fullColors.Dev()...)

	train, dev, test := datasets.SplitData(allExamples, 0.6, 0.2)
	fmt.Printf("   Original: %d examples\n", len(allExamples))
	fmt.Printf("   Train (60%%): %d examples\n", len(train))
	fmt.Printf("   Dev (20%%): %d examples\n", len(dev))
	fmt.Printf("   Test (20%%): %d examples\n", len(test))
	fmt.Println()

	// Example 4: MATH Answer Extraction
	fmt.Println("4. MATH Answer Extraction:")
	testSolutions := []string{
		"The answer is \\boxed{42}",
		"We get x = \\boxed{\\frac{1}{2}}",
		"Therefore, the solution is \\boxed{3.14}",
	}

	for i, solution := range testSolutions {
		answer := datasets.ExtractAnswer(solution)
		fmt.Printf("   Solution %d: %s\n", i+1, solution)
		fmt.Printf("   Extracted: %s\n", answer)
	}
	fmt.Println()

	fmt.Println("=== Example Complete ===")
}
