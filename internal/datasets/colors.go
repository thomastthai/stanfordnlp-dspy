package datasets

import (
	"math/rand"
	"sort"

	"github.com/stanfordnlp/dspy/internal/primitives"
)

// AllColors is a list of 144 color names from matplotlib.
var AllColors = []string{
	"alice blue", "dodger blue", "light sky blue", "deep sky blue",
	"sky blue", "steel blue", "light steel blue", "medium blue",
	"navy blue", "blue", "royal blue", "cadet blue",
	"cornflower blue", "medium slate blue", "slate blue", "dark slate blue",
	"powder blue", "turquoise", "dark turquoise", "medium turquoise",
	"pale turquoise", "light sea green", "medium sea green", "sea green",
	"forest green", "green yellow", "lime green", "dark green",
	"green", "lime", "chartreuse", "lawn green",
	"yellow green", "olive green", "dark olive green", "medium spring green",
	"spring green", "medium aquamarine", "aquamarine", "aqua",
	"cyan", "dark cyan", "teal", "medium orchid",
	"dark orchid", "orchid", "blue violet", "violet",
	"dark violet", "plum", "thistle", "magenta",
	"fuchsia", "dark magenta", "medium purple", "purple",
	"rebecca purple", "dark red", "fire brick", "indian red",
	"light coral", "dark salmon", "light salmon", "salmon",
	"red", "crimson", "tomato", "coral",
	"orange red", "dark orange", "orange", "yellow",
	"gold", "light goldenrod yellow", "pale goldenrod", "goldenrod",
	"dark goldenrod", "beige", "moccasin", "blanched almond",
	"navajo white", "antique white", "bisque", "burlywood",
	"dark khaki", "khaki", "tan", "wheat",
	"snow", "floral white", "old lace", "ivory",
	"linen", "seashell", "honeydew", "mint cream",
	"azure", "lavender", "ghost white", "white smoke",
	"gainsboro", "light gray", "silver", "dark gray",
	"gray", "dim gray", "slate gray", "light slate gray",
	"dark slate gray", "black", "medium violet red", "pale violet red",
	"deep pink", "hot pink", "light pink", "pink",
	"peach puff", "rosy brown", "saddle brown", "sandy brown",
	"chocolate", "peru", "sienna", "brown",
	"maroon", "white", "misty rose", "lavender blush",
	"papaya whip", "lemon chiffon", "light yellow", "corn silk",
	"pale green", "light green", "olive drab", "olive",
	"dark sea green",
	"medium aqua",
}

// Colors represents the Colors reasoning dataset.
type Colors struct {
	*BaseDataset
	sortBySuffix bool
}

// ColorsOptions extends DatasetOptions with Colors-specific options.
type ColorsOptions struct {
	DatasetOptions
	SortBySuffix bool
}

// DefaultColorsOptions returns default Colors options.
func DefaultColorsOptions() ColorsOptions {
	return ColorsOptions{
		DatasetOptions: DefaultDatasetOptions(),
		SortBySuffix:   true,
	}
}

// NewColors creates a new Colors dataset.
// The dataset splits colors into train (60%) and dev (40%) sets.
// Colors are sorted by suffix (reversed string) to ensure similar colors
// aren't repeated between train and dev sets.
func NewColors(opts ColorsOptions) *Colors {
	base := NewBaseDataset("colors", opts.DatasetOptions)
	dataset := &Colors{
		BaseDataset:  base,
		sortBySuffix: opts.SortBySuffix,
	}

	// Sort colors by suffix if enabled
	colors := make([]string, len(AllColors))
	copy(colors, AllColors)
	if opts.SortBySuffix {
		colors = sortBySuffix(colors)
	}

	// Split into train (60%) and dev (40%)
	trainSize := int(float64(len(colors)) * 0.6)
	trainColors := colors[:trainSize]
	devColors := colors[trainSize:]

	// Convert to examples
	trainExamples := make([]*primitives.Example, len(trainColors))
	for i, color := range trainColors {
		trainExamples[i] = primitives.NewExample(
			map[string]interface{}{"color": color},
			nil,
		)
	}

	devExamples := make([]*primitives.Example, len(devColors))
	for i, color := range devColors {
		devExamples[i] = primitives.NewExample(
			map[string]interface{}{"color": color},
			nil,
		)
	}

	// Shuffle with seed 0
	r := rand.New(rand.NewSource(0))
	r.Shuffle(len(trainExamples), func(i, j int) {
		trainExamples[i], trainExamples[j] = trainExamples[j], trainExamples[i]
	})
	r.Shuffle(len(devExamples), func(i, j int) {
		devExamples[i], devExamples[j] = devExamples[j], devExamples[i]
	})

	dataset.SetTrain(trainExamples)
	dataset.SetDev(devExamples)

	return dataset
}

// sortBySuffix sorts colors by their reversed string (suffix).
// This ensures similar colors (e.g., "alice blue", "dodger blue") are grouped together.
func sortBySuffix(colors []string) []string {
	sorted := make([]string, len(colors))
	copy(sorted, colors)

	sort.Slice(sorted, func(i, j int) bool {
		return reverseString(sorted[i]) < reverseString(sorted[j])
	})

	return sorted
}

// reverseString reverses a string.
func reverseString(s string) string {
	runes := []rune(s)
	for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
		runes[i], runes[j] = runes[j], runes[i]
	}
	return string(runes)
}
