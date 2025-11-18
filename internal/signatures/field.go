package signatures

// Field represents a field in a signature (input or output).
type Field struct {
	// Name is the field name
	Name string

	// Type is the field type (string, int, etc.)
	Type string

	// Description provides context about what this field contains
	Description string

	// Prefix is the prefix to use when formatting this field in prompts
	Prefix string

	// IsInput indicates if this is an input field (vs output)
	IsInput bool

	// Required indicates if this field is required
	Required bool

	// Format specifies the expected format (e.g., "json", "xml")
	Format string
}

// NewInputField creates a new input field with the given name.
func NewInputField(name string) *Field {
	return &Field{
		Name:     name,
		Type:     "string",
		IsInput:  true,
		Required: true,
		Prefix:   name + ":",
	}
}

// NewOutputField creates a new output field with the given name.
func NewOutputField(name string) *Field {
	return &Field{
		Name:     name,
		Type:     "string",
		IsInput:  false,
		Required: true,
		Prefix:   name + ":",
	}
}

// WithDescription adds a description to the field.
func (f *Field) WithDescription(desc string) *Field {
	f.Description = desc
	return f
}

// WithType sets the field type.
func (f *Field) WithType(typ string) *Field {
	f.Type = typ
	return f
}

// WithPrefix sets the field prefix.
func (f *Field) WithPrefix(prefix string) *Field {
	f.Prefix = prefix
	return f
}

// WithFormat sets the field format.
func (f *Field) WithFormat(format string) *Field {
	f.Format = format
	return f
}

// WithRequired sets whether the field is required.
func (f *Field) WithRequired(required bool) *Field {
	f.Required = required
	return f
}
