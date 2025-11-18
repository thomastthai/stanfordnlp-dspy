package signatures

import (
	"fmt"
	"strings"
)

// Signature defines the input and output fields for a DSPy module.
// It acts like a type signature for a function, specifying what goes in and what comes out.
type Signature struct {
	// Name is an optional name for this signature
	Name string
	
	// Instructions provide guidance to the LM about the task
	Instructions string
	
	// InputFields are the input fields for this signature
	InputFields []*Field
	
	// OutputFields are the output fields for this signature
	OutputFields []*Field
}

// NewSignature creates a new Signature from a string specification.
// The format is: "field1, field2 -> output1, output2"
// Example: "question, context -> answer"
func NewSignature(spec string) (*Signature, error) {
	return ParseSignature(spec)
}

// NewSignatureWithFields creates a new Signature with explicit fields.
func NewSignatureWithFields(inputFields, outputFields []*Field) *Signature {
	return &Signature{
		InputFields:  inputFields,
		OutputFields: outputFields,
	}
}

// WithInstructions adds instructions to the signature.
func (s *Signature) WithInstructions(instructions string) *Signature {
	s.Instructions = instructions
	return s
}

// WithName sets the signature name.
func (s *Signature) WithName(name string) *Signature {
	s.Name = name
	return s
}

// GetInputField returns the input field with the given name.
func (s *Signature) GetInputField(name string) (*Field, bool) {
	for _, field := range s.InputFields {
		if field.Name == name {
			return field, true
		}
	}
	return nil, false
}

// GetOutputField returns the output field with the given name.
func (s *Signature) GetOutputField(name string) (*Field, bool) {
	for _, field := range s.OutputFields {
		if field.Name == name {
			return field, true
		}
	}
	return nil, false
}

// GetField returns the field with the given name (input or output).
func (s *Signature) GetField(name string) (*Field, bool) {
	if field, ok := s.GetInputField(name); ok {
		return field, true
	}
	return s.GetOutputField(name)
}

// InputFieldNames returns the names of all input fields.
func (s *Signature) InputFieldNames() []string {
	names := make([]string, len(s.InputFields))
	for i, field := range s.InputFields {
		names[i] = field.Name
	}
	return names
}

// OutputFieldNames returns the names of all output fields.
func (s *Signature) OutputFieldNames() []string {
	names := make([]string, len(s.OutputFields))
	for i, field := range s.OutputFields {
		names[i] = field.Name
	}
	return names
}

// String returns a string representation of the signature.
func (s *Signature) String() string {
	inputs := strings.Join(s.InputFieldNames(), ", ")
	outputs := strings.Join(s.OutputFieldNames(), ", ")
	return fmt.Sprintf("%s -> %s", inputs, outputs)
}

// Validate checks if the signature is valid.
func (s *Signature) Validate() error {
	if len(s.InputFields) == 0 {
		return fmt.Errorf("signature must have at least one input field")
	}
	if len(s.OutputFields) == 0 {
		return fmt.Errorf("signature must have at least one output field")
	}
	
	// Check for duplicate field names
	seen := make(map[string]bool)
	for _, field := range s.InputFields {
		if seen[field.Name] {
			return fmt.Errorf("duplicate field name: %s", field.Name)
		}
		seen[field.Name] = true
	}
	for _, field := range s.OutputFields {
		if seen[field.Name] {
			return fmt.Errorf("duplicate field name: %s", field.Name)
		}
		seen[field.Name] = true
	}
	
	return nil
}
