package signatures

import (
	"fmt"
	"strings"
)

// ParseSignature parses a signature string into a Signature object.
// Format: "field1, field2 -> output1, output2"
// Example: "question, context -> answer"
func ParseSignature(spec string) (*Signature, error) {
	// Split on "->"
	parts := strings.Split(spec, "->")
	if len(parts) != 2 {
		return nil, fmt.Errorf("invalid signature format: expected 'inputs -> outputs', got %q", spec)
	}
	
	inputPart := strings.TrimSpace(parts[0])
	outputPart := strings.TrimSpace(parts[1])
	
	// Parse input fields
	inputFields, err := parseFieldList(inputPart, true)
	if err != nil {
		return nil, fmt.Errorf("failed to parse input fields: %w", err)
	}
	
	// Parse output fields
	outputFields, err := parseFieldList(outputPart, false)
	if err != nil {
		return nil, fmt.Errorf("failed to parse output fields: %w", err)
	}
	
	sig := &Signature{
		InputFields:  inputFields,
		OutputFields: outputFields,
	}
	
	if err := sig.Validate(); err != nil {
		return nil, err
	}
	
	return sig, nil
}

// parseFieldList parses a comma-separated list of field names.
func parseFieldList(fieldList string, isInput bool) ([]*Field, error) {
	if fieldList == "" {
		return nil, fmt.Errorf("field list cannot be empty")
	}
	
	// Split by comma
	fieldNames := strings.Split(fieldList, ",")
	fields := make([]*Field, 0, len(fieldNames))
	
	for _, name := range fieldNames {
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}
		
		// Check for type annotations (e.g., "field:type")
		var field *Field
		if isInput {
			field = NewInputField(name)
		} else {
			field = NewOutputField(name)
		}
		
		// Parse type if present
		if strings.Contains(name, ":") {
			parts := strings.Split(name, ":")
			field.Name = strings.TrimSpace(parts[0])
			if len(parts) > 1 {
				field.Type = strings.TrimSpace(parts[1])
			}
		}
		
		fields = append(fields, field)
	}
	
	if len(fields) == 0 {
		return nil, fmt.Errorf("no valid fields found in list")
	}
	
	return fields, nil
}
