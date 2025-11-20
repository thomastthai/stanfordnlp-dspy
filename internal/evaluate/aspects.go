package evaluate

// EvaluationAspect defines a specific dimension of evaluation with weighting and criteria.
type EvaluationAspect struct {
	Name        string
	Description string
	Weight      float64
	ScoreRange  [2]float64 // [min, max]
	Criteria    []string
}

// Predefined evaluation aspects with default weights
var (
	AspectAccuracy = &EvaluationAspect{
		Name:        "accuracy",
		Description: "Correctness of the output against ground truth",
		Weight:      1.0,
		ScoreRange:  [2]float64{0.0, 1.0},
		Criteria: []string{
			"Does the output match the expected answer?",
			"Is the information factually correct?",
		},
	}

	AspectFluency = &EvaluationAspect{
		Name:        "fluency",
		Description: "Natural language quality and grammatical correctness",
		Weight:      1.0,
		ScoreRange:  [2]float64{0.0, 1.0},
		Criteria: []string{
			"Is the output grammatically correct?",
			"Does it read naturally?",
			"Is the language clear and well-formed?",
		},
	}

	AspectCoherence = &EvaluationAspect{
		Name:        "coherence",
		Description: "Logical consistency and organization of the output",
		Weight:      1.0,
		ScoreRange:  [2]float64{0.0, 1.0},
		Criteria: []string{
			"Is the output logically consistent?",
			"Do ideas flow naturally?",
			"Is the structure well-organized?",
		},
	}

	AspectRelevance = &EvaluationAspect{
		Name:        "relevance",
		Description: "How well the output addresses the input query",
		Weight:      1.0,
		ScoreRange:  [2]float64{0.0, 1.0},
		Criteria: []string{
			"Does the output address the question?",
			"Is it on-topic and focused?",
			"Does it contain relevant information?",
		},
	}

	AspectCompleteness = &EvaluationAspect{
		Name:        "completeness",
		Description: "Coverage of all necessary information",
		Weight:      1.0,
		ScoreRange:  [2]float64{0.0, 1.0},
		Criteria: []string{
			"Does the output cover all key points?",
			"Is any critical information missing?",
			"Does it fully answer the question?",
		},
	}
)

// NewCustomAspect creates a custom evaluation aspect with specified properties.
func NewCustomAspect(name, description string, weight float64, criteria []string) *EvaluationAspect {
	if weight <= 0 {
		weight = 1.0
	}
	if criteria == nil {
		criteria = []string{}
	}

	return &EvaluationAspect{
		Name:        name,
		Description: description,
		Weight:      weight,
		ScoreRange:  [2]float64{0.0, 1.0},
		Criteria:    criteria,
	}
}
