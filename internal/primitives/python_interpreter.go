package primitives

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"strings"
	"time"
)

// PythonInterpreter provides safe execution of Python code using subprocess.
type PythonInterpreter struct {
	timeout        time.Duration
	pythonPath     string
	maxOutputSize  int
	restrictedMode bool
	workingDir     string
}

// PythonOptions configures the Python interpreter.
type PythonOptions struct {
	Timeout        time.Duration
	PythonPath     string
	MaxOutputSize  int
	RestrictedMode bool
	WorkingDir     string
}

// DefaultPythonOptions returns default Python interpreter options.
func DefaultPythonOptions() PythonOptions {
	return PythonOptions{
		Timeout:        30 * time.Second,
		PythonPath:     "python3",
		MaxOutputSize:  1024 * 1024, // 1MB
		RestrictedMode: true,
		WorkingDir:     "",
	}
}

// NewPythonInterpreter creates a new Python interpreter.
func NewPythonInterpreter(opts PythonOptions) *PythonInterpreter {
	if opts.Timeout == 0 {
		opts.Timeout = 30 * time.Second
	}
	if opts.PythonPath == "" {
		opts.PythonPath = "python3"
	}
	if opts.MaxOutputSize == 0 {
		opts.MaxOutputSize = 1024 * 1024
	}
	
	return &PythonInterpreter{
		timeout:        opts.Timeout,
		pythonPath:     opts.PythonPath,
		maxOutputSize:  opts.MaxOutputSize,
		restrictedMode: opts.RestrictedMode,
		workingDir:     opts.WorkingDir,
	}
}

// Execute executes Python code and returns the output.
func (pi *PythonInterpreter) Execute(ctx context.Context, code string) (string, error) {
	// Validate code in restricted mode
	if pi.restrictedMode {
		if err := pi.validateCode(code); err != nil {
			return "", fmt.Errorf("code validation failed: %w", err)
		}
	}
	
	// Create context with timeout
	execCtx, cancel := context.WithTimeout(ctx, pi.timeout)
	defer cancel()
	
	// Prepare Python command
	cmd := exec.CommandContext(execCtx, pi.pythonPath, "-c", code)
	
	if pi.workingDir != "" {
		cmd.Dir = pi.workingDir
	}
	
	// Capture stdout and stderr
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	
	// Execute command
	err := cmd.Run()
	
	// Check for timeout
	if execCtx.Err() == context.DeadlineExceeded {
		return "", fmt.Errorf("execution timed out after %v", pi.timeout)
	}
	
	// Get output
	output := stdout.String()
	errorOutput := stderr.String()
	
	// Check output size
	if len(output) > pi.maxOutputSize {
		output = output[:pi.maxOutputSize] + "\n... (output truncated)"
	}
	
	// Return error if execution failed
	if err != nil {
		if errorOutput != "" {
			return output, fmt.Errorf("execution failed: %s", errorOutput)
		}
		return output, fmt.Errorf("execution failed: %w", err)
	}
	
	return output, nil
}

// ExecuteWithResult executes code and returns both stdout and the last expression value.
func (pi *PythonInterpreter) ExecuteWithResult(ctx context.Context, code string) (output string, result string, err error) {
	// Wrap code to capture result
	wrappedCode := fmt.Sprintf(`
import sys
from io import StringIO

# Capture stdout
_stdout = StringIO()
_old_stdout = sys.stdout
sys.stdout = _stdout

# Execute code and capture result
_result = None
try:
    _code = %s
    exec(_code, globals())
    # Try to get the last expression
    import ast
    tree = ast.parse(_code)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        _result = eval(compile(ast.Expression(tree.body[-1].value), '<string>', 'eval'))
finally:
    sys.stdout = _old_stdout

# Print result
print("__OUTPUT__")
print(_stdout.getvalue())
print("__RESULT__")
if _result is not None:
    print(_result)
`, fmt.Sprintf("%q", code))
	
	fullOutput, err := pi.Execute(ctx, wrappedCode)
	if err != nil {
		return "", "", err
	}
	
	// Parse output and result
	parts := strings.Split(fullOutput, "__OUTPUT__")
	if len(parts) < 2 {
		return fullOutput, "", nil
	}
	
	parts = strings.Split(parts[1], "__RESULT__")
	if len(parts) < 2 {
		return strings.TrimSpace(parts[0]), "", nil
	}
	
	output = strings.TrimSpace(parts[0])
	result = strings.TrimSpace(parts[1])
	
	return output, result, nil
}

// validateCode performs basic validation of Python code in restricted mode.
func (pi *PythonInterpreter) validateCode(code string) error {
	// Check for potentially dangerous operations
	dangerous := []string{
		"__import__",
		"eval",
		"exec",
		"compile",
		"open(",
		"file(",
		"input(",
		"raw_input(",
		"os.",
		"sys.",
		"subprocess",
		"socket",
	}
	
	lowerCode := strings.ToLower(code)
	for _, danger := range dangerous {
		if strings.Contains(lowerCode, strings.ToLower(danger)) {
			return fmt.Errorf("potentially dangerous operation detected: %s", danger)
		}
	}
	
	return nil
}

// ExecuteFile executes a Python file.
func (pi *PythonInterpreter) ExecuteFile(ctx context.Context, filePath string, args ...string) (string, error) {
	// Create context with timeout
	execCtx, cancel := context.WithTimeout(ctx, pi.timeout)
	defer cancel()
	
	// Prepare command
	cmdArgs := append([]string{filePath}, args...)
	cmd := exec.CommandContext(execCtx, pi.pythonPath, cmdArgs...)
	
	if pi.workingDir != "" {
		cmd.Dir = pi.workingDir
	}
	
	// Capture output
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	
	// Execute
	err := cmd.Run()
	
	// Check for timeout
	if execCtx.Err() == context.DeadlineExceeded {
		return "", fmt.Errorf("execution timed out after %v", pi.timeout)
	}
	
	// Get output
	output := stdout.String()
	errorOutput := stderr.String()
	
	// Check output size
	if len(output) > pi.maxOutputSize {
		output = output[:pi.maxOutputSize] + "\n... (output truncated)"
	}
	
	// Return error if execution failed
	if err != nil {
		if errorOutput != "" {
			return output, fmt.Errorf("execution failed: %s", errorOutput)
		}
		return output, fmt.Errorf("execution failed: %w", err)
	}
	
	return output, nil
}

// CheckPythonAvailable checks if Python is available.
func (pi *PythonInterpreter) CheckPythonAvailable() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	cmd := exec.CommandContext(ctx, pi.pythonPath, "--version")
	err := cmd.Run()
	return err == nil
}

// GetPythonVersion returns the Python version.
func (pi *PythonInterpreter) GetPythonVersion(ctx context.Context) (string, error) {
	cmd := exec.CommandContext(ctx, pi.pythonPath, "--version")
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(output)), nil
}

// InterpreterSession represents a persistent Python interpreter session.
// This is a placeholder for future implementation with persistent state.
type InterpreterSession struct {
	interpreter *PythonInterpreter
	variables   map[string]interface{}
}

// NewInterpreterSession creates a new interpreter session.
func NewInterpreterSession(opts PythonOptions) *InterpreterSession {
	return &InterpreterSession{
		interpreter: NewPythonInterpreter(opts),
		variables:   make(map[string]interface{}),
	}
}

// Execute executes code in the session context.
// Note: This is a simplified implementation. For true persistence,
// you would need to use a REPL or maintain a long-running Python process.
func (is *InterpreterSession) Execute(ctx context.Context, code string) (string, error) {
	return is.interpreter.Execute(ctx, code)
}

// SetVariable sets a variable in the session (placeholder).
func (is *InterpreterSession) SetVariable(name string, value interface{}) {
	is.variables[name] = value
}

// GetVariable gets a variable from the session (placeholder).
func (is *InterpreterSession) GetVariable(name string) (interface{}, bool) {
	val, ok := is.variables[name]
	return val, ok
}

// Close closes the session.
func (is *InterpreterSession) Close() {
	// Cleanup if needed
	is.variables = nil
}
