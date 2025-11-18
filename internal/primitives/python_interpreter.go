package primitives

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"time"
)

// PythonInterpreter provides safe Python code execution.
type PythonInterpreter struct {
	timeout time.Duration
	python  string
}

// PythonInterpreterOptions configures the Python interpreter.
type PythonInterpreterOptions struct {
	// Timeout is the maximum execution time (default: 30s)
	Timeout time.Duration
	// Python is the path to the Python executable (default: "python3")
	Python string
}

// NewPythonInterpreter creates a new Python interpreter.
func NewPythonInterpreter(opts PythonInterpreterOptions) *PythonInterpreter {
	if opts.Timeout == 0 {
		opts.Timeout = 30 * time.Second
	}
	if opts.Python == "" {
		opts.Python = "python3"
	}

	return &PythonInterpreter{
		timeout: opts.Timeout,
		python:  opts.Python,
	}
}

// PythonResult represents the result of Python code execution.
type PythonResult struct {
	Stdout   string
	Stderr   string
	ExitCode int
	Error    error
}

// Execute executes Python code and returns the result.
// The code is executed in a subprocess with timeout protection.
func (pi *PythonInterpreter) Execute(ctx context.Context, code string) PythonResult {
	// Create a context with timeout
	execCtx, cancel := context.WithTimeout(ctx, pi.timeout)
	defer cancel()

	// Create command
	cmd := exec.CommandContext(execCtx, pi.python, "-c", code)

	// Capture stdout and stderr
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	// Execute
	err := cmd.Run()

	result := PythonResult{
		Stdout: stdout.String(),
		Stderr: stderr.String(),
	}

	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			result.ExitCode = exitErr.ExitCode()
		}
		result.Error = err
	}

	return result
}

// ExecuteFile executes a Python file and returns the result.
func (pi *PythonInterpreter) ExecuteFile(ctx context.Context, filepath string) PythonResult {
	// Create a context with timeout
	execCtx, cancel := context.WithTimeout(ctx, pi.timeout)
	defer cancel()

	// Create command
	cmd := exec.CommandContext(execCtx, pi.python, filepath)

	// Capture stdout and stderr
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	// Execute
	err := cmd.Run()

	result := PythonResult{
		Stdout: stdout.String(),
		Stderr: stderr.String(),
	}

	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			result.ExitCode = exitErr.ExitCode()
		}
		result.Error = err
	}

	return result
}

// EvaluateExpression evaluates a Python expression and returns the result as a string.
func (pi *PythonInterpreter) EvaluateExpression(ctx context.Context, expr string) (string, error) {
	code := fmt.Sprintf("print(%s)", expr)
	result := pi.Execute(ctx, code)

	if result.Error != nil {
		return "", fmt.Errorf("execution error: %w (stderr: %s)", result.Error, result.Stderr)
	}

	if result.ExitCode != 0 {
		return "", fmt.Errorf("exit code %d: %s", result.ExitCode, result.Stderr)
	}

	return result.Stdout, nil
}

// CheckSyntax checks if the Python code has valid syntax.
func (pi *PythonInterpreter) CheckSyntax(ctx context.Context, code string) error {
	// Use Python's compile to check syntax
	checkCode := fmt.Sprintf("compile(%q, '<string>', 'exec')", code)
	result := pi.Execute(ctx, checkCode)

	if result.Error != nil || result.ExitCode != 0 {
		return fmt.Errorf("syntax error: %s", result.Stderr)
	}

	return nil
}
