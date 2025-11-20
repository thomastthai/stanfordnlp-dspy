package model

// Example represents a training or evaluation example.
type Example struct {
	ID     string
	Inputs map[string]Field
	Labels map[string]Field
}
