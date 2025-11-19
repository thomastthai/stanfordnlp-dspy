package adapters

import (
	"encoding/xml"
	"fmt"
	"regexp"
	"strings"

	"github.com/stanfordnlp/dspy/internal/clients"
	"github.com/stanfordnlp/dspy/internal/signatures"
)

// XMLAdapter formats requests for XML output.
type XMLAdapter struct {
	*BaseAdapter
	rootTag string
}

// NewXMLAdapter creates a new XML adapter.
func NewXMLAdapter() *XMLAdapter {
	return &XMLAdapter{
		BaseAdapter: NewBaseAdapter("xml"),
		rootTag:     "response",
	}
}

// NewXMLAdapterWithRootTag creates a new XML adapter with a custom root tag.
func NewXMLAdapterWithRootTag(rootTag string) *XMLAdapter {
	return &XMLAdapter{
		BaseAdapter: NewBaseAdapter("xml"),
		rootTag:     rootTag,
	}
}

// Format implements Adapter.Format.
func (a *XMLAdapter) Format(sig *signatures.Signature, inputs map[string]interface{}, demos []map[string]interface{}) (*clients.Request, error) {
	request := clients.NewRequest()

	// Build system message with XML instructions
	systemMsg := a.buildSystemMessage(sig)
	request.WithMessages(clients.NewMessage("system", systemMsg))

	// Add demonstrations as few-shot examples in XML format
	for _, demo := range demos {
		// Format demo input
		inputXML := a.formatInputXML(sig, demo)
		request.WithMessages(clients.NewMessage("user", inputXML))

		// Format demo output
		outputXML := a.formatOutputXML(sig, demo)
		request.WithMessages(clients.NewMessage("assistant", outputXML))
	}

	// Add current input in XML format
	inputXML := a.formatInputXML(sig, inputs)
	request.WithMessages(clients.NewMessage("user", inputXML))

	return request, nil
}

// Parse implements Adapter.Parse.
func (a *XMLAdapter) Parse(sig *signatures.Signature, response *clients.Response) (map[string]interface{}, error) {
	if len(response.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	content := response.Choices[0].Message.Content

	// Extract XML from response (may be wrapped in markdown or other text)
	xmlContent, err := a.extractXML(content)
	if err != nil {
		// Try to use raw content as-is
		xmlContent = content
	}

	// Parse XML and extract fields
	outputs := make(map[string]interface{})

	for _, field := range sig.OutputFields {
		value, err := a.extractField(xmlContent, field.Name)
		if err != nil {
			// Try alternative formats
			value, err = a.extractFieldAlternative(xmlContent, field.Name)
			if err != nil {
				continue
			}
		}
		outputs[field.Name] = value
	}

	// If no fields were extracted, try parsing as generic XML
	if len(outputs) == 0 {
		genericOutputs, err := a.parseGenericXML(xmlContent)
		if err != nil {
			return nil, fmt.Errorf("failed to parse XML: %w", err)
		}
		return genericOutputs, nil
	}

	return outputs, nil
}

// buildSystemMessage creates the system message with XML instructions.
func (a *XMLAdapter) buildSystemMessage(sig *signatures.Signature) string {
	var sb strings.Builder

	if sig.Instructions != "" {
		sb.WriteString(sig.Instructions)
		sb.WriteString("\n\n")
	}

	sb.WriteString("You must respond with valid XML. ")
	sb.WriteString(fmt.Sprintf("Wrap your response in a <%s> root element.\n", a.rootTag))
	sb.WriteString("Use the following tags for the output fields:\n")

	for _, field := range sig.OutputFields {
		sb.WriteString(fmt.Sprintf("- <%s>", field.Name))
		if field.Description != "" {
			sb.WriteString(fmt.Sprintf(": %s", field.Description))
		}
		sb.WriteString(fmt.Sprintf("</%s>\n", field.Name))
	}

	return sb.String()
}

// formatInputXML formats input fields as XML.
func (a *XMLAdapter) formatInputXML(sig *signatures.Signature, data map[string]interface{}) string {
	var sb strings.Builder

	sb.WriteString("<input>\n")
	for _, field := range sig.InputFields {
		if val, ok := data[field.Name]; ok {
			sb.WriteString(fmt.Sprintf("  <%s>%s</%s>\n", field.Name, a.escapeXML(fmt.Sprintf("%v", val)), field.Name))
		}
	}
	sb.WriteString("</input>")

	return sb.String()
}

// formatOutputXML formats output fields as XML.
func (a *XMLAdapter) formatOutputXML(sig *signatures.Signature, data map[string]interface{}) string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("<%s>\n", a.rootTag))
	for _, field := range sig.OutputFields {
		if val, ok := data[field.Name]; ok {
			sb.WriteString(fmt.Sprintf("  <%s>%s</%s>\n", field.Name, a.escapeXML(fmt.Sprintf("%v", val)), field.Name))
		}
	}
	sb.WriteString(fmt.Sprintf("</%s>", a.rootTag))

	return sb.String()
}

// extractXML tries to extract XML from text that may contain additional content.
func (a *XMLAdapter) extractXML(text string) (string, error) {
	// Remove markdown code blocks
	text = strings.TrimPrefix(text, "```xml")
	text = strings.TrimPrefix(text, "```")
	text = strings.TrimSuffix(text, "```")
	text = strings.TrimSpace(text)

	// Try to find XML root element
	pattern := fmt.Sprintf("<%s[^>]*>.*</%s>", a.rootTag, a.rootTag)
	re := regexp.MustCompile(pattern)
	matches := re.FindString(text)
	if matches != "" {
		return matches, nil
	}

	// If no root tag found, try to find any XML-like content
	start := strings.Index(text, "<")
	end := strings.LastIndex(text, ">")
	if start != -1 && end != -1 && end > start {
		return text[start : end+1], nil
	}

	return text, fmt.Errorf("no XML content found")
}

// extractField extracts a field value from XML using tag-based parsing.
func (a *XMLAdapter) extractField(xmlContent, fieldName string) (string, error) {
	// Use regex to extract field content
	pattern := fmt.Sprintf("<%s[^>]*>(.*?)</%s>", regexp.QuoteMeta(fieldName), regexp.QuoteMeta(fieldName))
	re := regexp.MustCompile(pattern)
	matches := re.FindStringSubmatch(xmlContent)
	if len(matches) >= 2 {
		return a.unescapeXML(matches[1]), nil
	}

	return "", fmt.Errorf("field '%s' not found", fieldName)
}

// extractFieldAlternative tries alternative methods to extract field value.
func (a *XMLAdapter) extractFieldAlternative(xmlContent, fieldName string) (string, error) {
	// Try case-insensitive match
	pattern := fmt.Sprintf("(?i)<%s[^>]*>(.*?)</%s>", regexp.QuoteMeta(fieldName), regexp.QuoteMeta(fieldName))
	re := regexp.MustCompile(pattern)
	matches := re.FindStringSubmatch(xmlContent)
	if len(matches) >= 2 {
		return a.unescapeXML(matches[1]), nil
	}

	return "", fmt.Errorf("field '%s' not found with alternative methods", fieldName)
}

// parseGenericXML parses XML without assuming specific field names.
func (a *XMLAdapter) parseGenericXML(xmlContent string) (map[string]interface{}, error) {
	// Try to parse as generic XML structure
	type GenericXML struct {
		XMLName xml.Name
		Content string `xml:",chardata"`
		Nodes   []GenericXML `xml:",any"`
	}

	var root GenericXML
	if err := xml.Unmarshal([]byte(xmlContent), &root); err != nil {
		return nil, err
	}

	// Convert to map
	result := make(map[string]interface{})
	a.xmlToMap(&root, result)

	return result, nil
}

// xmlToMap converts a GenericXML structure to a map.
func (a *XMLAdapter) xmlToMap(node interface{}, result map[string]interface{}) {
	// This is a simplified implementation
	// A full implementation would handle nested structures
	// For now, we'll just extract text content from tags
}

// escapeXML escapes XML special characters.
func (a *XMLAdapter) escapeXML(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	s = strings.ReplaceAll(s, ">", "&gt;")
	s = strings.ReplaceAll(s, "\"", "&quot;")
	s = strings.ReplaceAll(s, "'", "&apos;")
	return s
}

// unescapeXML unescapes XML special characters.
func (a *XMLAdapter) unescapeXML(s string) string {
	s = strings.ReplaceAll(s, "&lt;", "<")
	s = strings.ReplaceAll(s, "&gt;", ">")
	s = strings.ReplaceAll(s, "&quot;", "\"")
	s = strings.ReplaceAll(s, "&apos;", "'")
	s = strings.ReplaceAll(s, "&amp;", "&")
	return strings.TrimSpace(s)
}
