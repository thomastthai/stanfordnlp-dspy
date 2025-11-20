package evaluate

import (
	"testing"
)

func TestPredefinedAspects(t *testing.T) {
	aspects := []*EvaluationAspect{
		AspectAccuracy,
		AspectFluency,
		AspectCoherence,
		AspectRelevance,
		AspectCompleteness,
	}

	for _, aspect := range aspects {
		if aspect.Name == "" {
			t.Error("Aspect name should not be empty")
		}

		if aspect.Description == "" {
			t.Error("Aspect description should not be empty")
		}

		if aspect.Weight != 1.0 {
			t.Errorf("Expected default weight 1.0, got %f", aspect.Weight)
		}

		if aspect.ScoreRange[0] != 0.0 || aspect.ScoreRange[1] != 1.0 {
			t.Errorf("Expected score range [0.0, 1.0], got [%f, %f]",
				aspect.ScoreRange[0], aspect.ScoreRange[1])
		}

		if len(aspect.Criteria) == 0 {
			t.Error("Aspect should have at least one criterion")
		}
	}
}

func TestNewCustomAspect(t *testing.T) {
	criteria := []string{"criterion 1", "criterion 2"}
	aspect := NewCustomAspect("custom", "Custom aspect", 2.0, criteria)

	if aspect.Name != "custom" {
		t.Errorf("Expected name 'custom', got %s", aspect.Name)
	}

	if aspect.Description != "Custom aspect" {
		t.Errorf("Expected description 'Custom aspect', got %s", aspect.Description)
	}

	if aspect.Weight != 2.0 {
		t.Errorf("Expected weight 2.0, got %f", aspect.Weight)
	}

	if len(aspect.Criteria) != 2 {
		t.Errorf("Expected 2 criteria, got %d", len(aspect.Criteria))
	}
}

func TestNewCustomAspect_DefaultWeight(t *testing.T) {
	aspect := NewCustomAspect("test", "Test aspect", 0.0, nil)

	if aspect.Weight != 1.0 {
		t.Errorf("Expected default weight 1.0, got %f", aspect.Weight)
	}

	if aspect.Criteria == nil {
		t.Error("Expected non-nil criteria slice")
	}
}

func TestNewCustomAspect_NegativeWeight(t *testing.T) {
	aspect := NewCustomAspect("test", "Test aspect", -1.0, nil)

	if aspect.Weight != 1.0 {
		t.Errorf("Expected default weight 1.0 for negative weight, got %f", aspect.Weight)
	}
}
