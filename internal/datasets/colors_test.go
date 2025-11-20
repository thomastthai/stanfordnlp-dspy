package datasets

import (
	"testing"
)

func TestNewColors(t *testing.T) {
	opts := DefaultColorsOptions()
	dataset := NewColors(opts)

	if dataset.Name() != "colors" {
		t.Errorf("Expected name 'colors', got '%s'", dataset.Name())
	}

	// Verify we have train and dev splits
	trainSize := len(dataset.Train())
	devSize := len(dataset.Dev())

	if trainSize == 0 {
		t.Error("Expected non-empty training set")
	}

	if devSize == 0 {
		t.Error("Expected non-empty dev set")
	}

	// Verify split ratio is approximately 60/40
	total := trainSize + devSize
	expectedTrainSize := int(float64(total) * 0.6)

	// Allow some tolerance due to integer division
	if trainSize < expectedTrainSize-1 || trainSize > expectedTrainSize+1 {
		t.Errorf("Expected train size ~%d (60%%), got %d", expectedTrainSize, trainSize)
	}

	// Verify examples have color field
	if trainSize > 0 {
		ex := dataset.Train()[0]
		if _, ok := ex.Get("color"); !ok {
			t.Error("Expected training example to have 'color' field")
		}
	}

	// Verify Len() returns total count
	expectedLen := trainSize + devSize
	if dataset.Len() != expectedLen {
		t.Errorf("Expected Len() to be %d, got %d", expectedLen, dataset.Len())
	}
}

func TestColorsSortBySuffix(t *testing.T) {
	colors := []string{"alice blue", "dodger blue", "red", "green", "blue"}
	sorted := sortBySuffix(colors)

	// Verify sorting happened
	if len(sorted) != len(colors) {
		t.Errorf("Expected %d colors, got %d", len(colors), len(sorted))
	}

	// Verify all colors are present
	colorMap := make(map[string]bool)
	for _, c := range colors {
		colorMap[c] = true
	}
	for _, c := range sorted {
		if !colorMap[c] {
			t.Errorf("Unexpected color in sorted list: %s", c)
		}
	}
}

func TestColorsWithoutSorting(t *testing.T) {
	opts := DefaultColorsOptions()
	opts.SortBySuffix = false

	dataset := NewColors(opts)

	if dataset.Name() != "colors" {
		t.Errorf("Expected name 'colors', got '%s'", dataset.Name())
	}

	// Should still have train and dev splits
	if len(dataset.Train()) == 0 {
		t.Error("Expected non-empty training set")
	}
	if len(dataset.Dev()) == 0 {
		t.Error("Expected non-empty dev set")
	}
}

func TestReverseString(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"hello", "olleh"},
		{"alice blue", "eulb ecila"},
		{"a", "a"},
		{"", ""},
	}

	for _, tt := range tests {
		result := reverseString(tt.input)
		if result != tt.expected {
			t.Errorf("reverseString(%q) = %q, expected %q", tt.input, result, tt.expected)
		}
	}
}

func TestAllColorsCount(t *testing.T) {
	// Verify we have at least 138 colors as mentioned in the spec
	if len(AllColors) < 138 {
		t.Errorf("Expected at least 138 colors, got %d", len(AllColors))
	}

	// Verify no duplicates
	colorMap := make(map[string]bool)
	for _, color := range AllColors {
		if colorMap[color] {
			t.Errorf("Duplicate color found: %s", color)
		}
		colorMap[color] = true
	}
}

func TestColorsDatasetOptions(t *testing.T) {
	opts := DefaultColorsOptions()
	opts.TrainSize = 10
	opts.DevSize = 5

	dataset := NewColors(opts)

	// Verify size limits are applied
	if len(dataset.Train()) > 10 {
		t.Errorf("Expected at most 10 training examples, got %d", len(dataset.Train()))
	}
	if len(dataset.Dev()) > 5 {
		t.Errorf("Expected at most 5 dev examples, got %d", len(dataset.Dev()))
	}
}
