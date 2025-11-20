package model

// Field represents a typed field value.
type Field struct {
	Name  string
	Value interface{}
	Type  FieldType
}

type FieldType string

const (
	FieldTypeString FieldType = "string"
	FieldTypeInt    FieldType = "int"
	FieldTypeFloat  FieldType = "float"
	FieldTypeBool   FieldType = "bool"
	FieldTypeList   FieldType = "list"
	FieldTypeObject FieldType = "object"
)

// String returns the string representation of the field.
func (f *Field) String() string {
	if s, ok := f.Value.(string); ok {
		return s
	}
	return ""
}
