package predict

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/Knetic/govaluate"
)

// ToolInterface defines the interface for executable tools in ReAct.
type ToolInterface interface {
	Name() string
	Description() string
	Execute(ctx context.Context, input string) (string, error)
}

// SearchTool implements a search tool (mock implementation).
type SearchTool struct{}

// Name returns the tool name.
func (s *SearchTool) Name() string {
	return "search"
}

// Description returns the tool description.
func (s *SearchTool) Description() string {
	return "Search the web for information. Input should be a search query string."
}

// Execute performs a web search (mock implementation).
func (s *SearchTool) Execute(ctx context.Context, query string) (string, error) {
	// Mock implementation - in production, this would call a real search API
	return fmt.Sprintf("Search results for '%s': [Mock search results would appear here]", query), nil
}

// CalculatorTool implements a calculator tool for mathematical expressions.
type CalculatorTool struct{}

// Name returns the tool name.
func (s *CalculatorTool) Name() string {
	return "calculator"
}

// Description returns the tool description.
func (s *CalculatorTool) Description() string {
	return "Perform mathematical calculations. Input should be a mathematical expression like '2 + 2' or 'sqrt(16)'."
}

// Execute evaluates a mathematical expression safely.
func (s *CalculatorTool) Execute(ctx context.Context, expression string) (string, error) {
	// Clean up the expression
	expression = strings.TrimSpace(expression)
	if expression == "" {
		return "", fmt.Errorf("empty expression")
	}

	// Try to evaluate using govaluate
	result, err := evaluateExpression(expression)
	if err != nil {
		return "", fmt.Errorf("failed to evaluate expression: %w", err)
	}

	return result, nil
}

// evaluateExpression safely evaluates a mathematical expression.
func evaluateExpression(expression string) (string, error) {
	// Replace common mathematical functions with govaluate-compatible names
	expression = strings.ReplaceAll(expression, "sqrt", "sqrt")
	expression = strings.ReplaceAll(expression, "^", "**")
	
	// Create the evaluator with math functions
	expr, err := govaluate.NewEvaluableExpressionWithFunctions(
		expression,
		map[string]govaluate.ExpressionFunction{
			"sqrt": func(args ...interface{}) (interface{}, error) {
				if len(args) != 1 {
					return nil, fmt.Errorf("sqrt requires exactly 1 argument")
				}
				val, err := toFloat64(args[0])
				if err != nil {
					return nil, err
				}
				if val < 0 {
					return nil, fmt.Errorf("cannot take square root of negative number")
				}
				return math.Sqrt(val), nil
			},
			"pow": func(args ...interface{}) (interface{}, error) {
				if len(args) != 2 {
					return nil, fmt.Errorf("pow requires exactly 2 arguments")
				}
				base, err := toFloat64(args[0])
				if err != nil {
					return nil, err
				}
				exp, err := toFloat64(args[1])
				if err != nil {
					return nil, err
				}
				return math.Pow(base, exp), nil
			},
			"abs": func(args ...interface{}) (interface{}, error) {
				if len(args) != 1 {
					return nil, fmt.Errorf("abs requires exactly 1 argument")
				}
				val, err := toFloat64(args[0])
				if err != nil {
					return nil, err
				}
				return math.Abs(val), nil
			},
		},
	)
	if err != nil {
		return "", fmt.Errorf("invalid expression: %w", err)
	}

	// Evaluate the expression
	result, err := expr.Evaluate(nil)
	if err != nil {
		return "", fmt.Errorf("evaluation error: %w", err)
	}

	// Format the result
	switch v := result.(type) {
	case float64:
		// Check if it's an integer value
		if v == math.Floor(v) {
			return fmt.Sprintf("%.0f", v), nil
		}
		return fmt.Sprintf("%g", v), nil
	case int:
		return fmt.Sprintf("%d", v), nil
	case int64:
		return fmt.Sprintf("%d", v), nil
	default:
		return fmt.Sprintf("%v", v), nil
	}
}

// toFloat64 converts various numeric types to float64.
func toFloat64(val interface{}) (float64, error) {
	switch v := val.(type) {
	case float64:
		return v, nil
	case float32:
		return float64(v), nil
	case int:
		return float64(v), nil
	case int64:
		return float64(v), nil
	case int32:
		return float64(v), nil
	case string:
		return strconv.ParseFloat(v, 64)
	default:
		return 0, fmt.Errorf("cannot convert %T to float64", val)
	}
}

// WikipediaTool implements a Wikipedia lookup tool (mock implementation).
type WikipediaTool struct{}

// Name returns the tool name.
func (w *WikipediaTool) Name() string {
	return "wikipedia"
}

// Description returns the tool description.
func (w *WikipediaTool) Description() string {
	return "Look up information on Wikipedia. Input should be a topic or article title."
}

// Execute looks up information on Wikipedia (mock implementation).
func (w *WikipediaTool) Execute(ctx context.Context, topic string) (string, error) {
	// Mock implementation - in production, this would call Wikipedia API
	return fmt.Sprintf("Wikipedia summary for '%s': [Mock Wikipedia content would appear here]", topic), nil
}

// FinishTool signals task completion.
type FinishTool struct{}

// Name returns the tool name.
func (f *FinishTool) Name() string {
	return "finish"
}

// Description returns the tool description.
func (f *FinishTool) Description() string {
	return "Signal that the task is complete and you have gathered all necessary information."
}

// Execute marks the task as complete.
func (f *FinishTool) Execute(ctx context.Context, input string) (string, error) {
	return "Task marked as complete.", nil
}

// NewDefaultTools returns a set of commonly used tools.
func NewDefaultTools() []ToolInterface {
	return []ToolInterface{
		&SearchTool{},
		&CalculatorTool{},
		&WikipediaTool{},
		&FinishTool{},
	}
}
