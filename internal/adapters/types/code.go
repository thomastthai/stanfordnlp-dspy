package types

import (
	"fmt"
	"strings"
)

// Language represents a programming language.
type Language string

const (
	LanguagePython     Language = "python"
	LanguageJavaScript Language = "javascript"
	LanguageTypeScript Language = "typescript"
	LanguageGo         Language = "go"
	LanguageRust       Language = "rust"
	LanguageJava       Language = "java"
	LanguageCPP        Language = "cpp"
	LanguageC          Language = "c"
	LanguageRuby       Language = "ruby"
	LanguagePHP        Language = "php"
	LanguageSwift      Language = "swift"
	LanguageKotlin     Language = "kotlin"
	LanguageSQL        Language = "sql"
	LanguageBash       Language = "bash"
	LanguageYAML       Language = "yaml"
	LanguageJSON       Language = "json"
	LanguageXML        Language = "xml"
	LanguageMarkdown   Language = "markdown"
	LanguageHTML       Language = "html"
	LanguageCSS        Language = "css"
)

// CodeBlock represents a code block with optional language and metadata.
type CodeBlock struct {
	// Code is the code content
	Code string

	// Language is the programming language
	Language Language

	// FileName is an optional filename
	FileName string

	// StartLine is the starting line number (for partial code)
	StartLine int

	// EndLine is the ending line number (for partial code)
	EndLine int

	// Metadata contains additional metadata
	Metadata map[string]interface{}
}

// NewCodeBlock creates a new code block.
func NewCodeBlock(code string, language Language) *CodeBlock {
	return &CodeBlock{
		Code:     code,
		Language: language,
		Metadata: make(map[string]interface{}),
	}
}

// Format formats the code block as a markdown code block.
func (cb *CodeBlock) Format() string {
	var sb strings.Builder

	sb.WriteString("```")
	if cb.Language != "" {
		sb.WriteString(string(cb.Language))
	}
	sb.WriteString("\n")
	sb.WriteString(cb.Code)
	if !strings.HasSuffix(cb.Code, "\n") {
		sb.WriteString("\n")
	}
	sb.WriteString("```")

	return sb.String()
}

// FormatWithContext formats the code block with additional context.
func (cb *CodeBlock) FormatWithContext() string {
	var sb strings.Builder

	if cb.FileName != "" {
		sb.WriteString(fmt.Sprintf("File: %s\n", cb.FileName))
	}

	if cb.StartLine > 0 && cb.EndLine > 0 {
		sb.WriteString(fmt.Sprintf("Lines: %d-%d\n", cb.StartLine, cb.EndLine))
	}

	sb.WriteString(cb.Format())

	return sb.String()
}

// ParseCodeBlock parses a markdown code block.
func ParseCodeBlock(text string) (*CodeBlock, error) {
	text = strings.TrimSpace(text)

	// Check for markdown code block
	if !strings.HasPrefix(text, "```") {
		return &CodeBlock{Code: text}, nil
	}

	// Remove opening ```
	text = strings.TrimPrefix(text, "```")

	// Extract language
	lines := strings.Split(text, "\n")
	if len(lines) < 2 {
		return nil, fmt.Errorf("invalid code block format")
	}

	language := strings.TrimSpace(lines[0])
	code := strings.Join(lines[1:], "\n")

	// Remove closing ```
	code = strings.TrimSuffix(code, "```")
	code = strings.TrimSpace(code)

	return &CodeBlock{
		Code:     code,
		Language: Language(language),
		Metadata: make(map[string]interface{}),
	}, nil
}

// ParseMultipleCodeBlocks parses multiple code blocks from text.
func ParseMultipleCodeBlocks(text string) ([]*CodeBlock, error) {
	var blocks []*CodeBlock

	// Find all code blocks
	parts := strings.Split(text, "```")

	// Code blocks are at odd indices (1, 3, 5, ...)
	for i := 1; i < len(parts); i += 2 {
		if i < len(parts) {
			blockText := parts[i]
			block, err := ParseCodeBlock("```" + blockText + "```")
			if err != nil {
				continue // Skip invalid blocks
			}
			blocks = append(blocks, block)
		}
	}

	return blocks, nil
}

// DetectLanguage attempts to detect the programming language from code.
func DetectLanguage(code string) Language {
	code = strings.TrimSpace(code)

	// Simple heuristics for language detection
	if strings.Contains(code, "def ") || strings.Contains(code, "import ") {
		return LanguagePython
	}
	if strings.Contains(code, "function ") || strings.Contains(code, "const ") || strings.Contains(code, "let ") {
		return LanguageJavaScript
	}
	if strings.Contains(code, "package ") && strings.Contains(code, "func ") {
		return LanguageGo
	}
	if strings.Contains(code, "public class ") || strings.Contains(code, "public static void main") {
		return LanguageJava
	}
	if strings.HasPrefix(code, "SELECT ") || strings.HasPrefix(code, "select ") {
		return LanguageSQL
	}
	if strings.HasPrefix(code, "#!/bin/bash") || strings.HasPrefix(code, "#!/bin/sh") {
		return LanguageBash
	}
	if strings.HasPrefix(code, "{") && strings.HasSuffix(code, "}") {
		return LanguageJSON
	}
	if strings.HasPrefix(code, "<") && strings.HasSuffix(code, ">") {
		if strings.Contains(code, "<!DOCTYPE html>") || strings.Contains(code, "<html") {
			return LanguageHTML
		}
		return LanguageXML
	}

	return "" // Unknown
}

// Validate validates the code block.
func (cb *CodeBlock) Validate() error {
	if cb.Code == "" {
		return fmt.Errorf("code content is empty")
	}

	if cb.StartLine > 0 && cb.EndLine > 0 && cb.StartLine > cb.EndLine {
		return fmt.Errorf("start line (%d) is greater than end line (%d)", cb.StartLine, cb.EndLine)
	}

	return nil
}

// LineCount returns the number of lines in the code.
func (cb *CodeBlock) LineCount() int {
	return len(strings.Split(cb.Code, "\n"))
}
