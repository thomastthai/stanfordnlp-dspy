// Package utils provides utility functions for DSPy.
package utils

import (
	"io"
	"log"
	"os"
	"sync"
)

// LogLevel represents the logging level.
type LogLevel int

const (
	// LevelDebug is for debug messages
	LevelDebug LogLevel = iota
	// LevelInfo is for informational messages
	LevelInfo
	// LevelWarn is for warning messages
	LevelWarn
	// LevelError is for error messages
	LevelError
)

// Logger provides structured logging for DSPy.
type Logger struct {
	level  LogLevel
	debug  *log.Logger
	info   *log.Logger
	warn   *log.Logger
	error  *log.Logger
	mu     sync.RWMutex
}

var (
	defaultLogger *Logger
	loggerMux     sync.RWMutex
)

func init() {
	defaultLogger = NewLogger(LevelInfo)
}

// NewLogger creates a new logger with the specified level.
func NewLogger(level LogLevel) *Logger {
	return &Logger{
		level: level,
		debug: log.New(os.Stdout, "DEBUG: ", log.LstdFlags),
		info:  log.New(os.Stdout, "INFO: ", log.LstdFlags),
		warn:  log.New(os.Stdout, "WARN: ", log.LstdFlags),
		error: log.New(os.Stderr, "ERROR: ", log.LstdFlags),
	}
}

// SetLevel sets the logging level.
func (l *Logger) SetLevel(level LogLevel) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.level = level
}

// SetOutput sets the output writer for all log levels.
func (l *Logger) SetOutput(w io.Writer) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.debug.SetOutput(w)
	l.info.SetOutput(w)
	l.warn.SetOutput(w)
	l.error.SetOutput(w)
}

// Debug logs a debug message.
func (l *Logger) Debug(v ...interface{}) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	if l.level <= LevelDebug {
		l.debug.Println(v...)
	}
}

// Debugf logs a formatted debug message.
func (l *Logger) Debugf(format string, v ...interface{}) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	if l.level <= LevelDebug {
		l.debug.Printf(format, v...)
	}
}

// Info logs an info message.
func (l *Logger) Info(v ...interface{}) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	if l.level <= LevelInfo {
		l.info.Println(v...)
	}
}

// Infof logs a formatted info message.
func (l *Logger) Infof(format string, v ...interface{}) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	if l.level <= LevelInfo {
		l.info.Printf(format, v...)
	}
}

// Warn logs a warning message.
func (l *Logger) Warn(v ...interface{}) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	if l.level <= LevelWarn {
		l.warn.Println(v...)
	}
}

// Warnf logs a formatted warning message.
func (l *Logger) Warnf(format string, v ...interface{}) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	if l.level <= LevelWarn {
		l.warn.Printf(format, v...)
	}
}

// Error logs an error message.
func (l *Logger) Error(v ...interface{}) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	if l.level <= LevelError {
		l.error.Println(v...)
	}
}

// Errorf logs a formatted error message.
func (l *Logger) Errorf(format string, v ...interface{}) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	if l.level <= LevelError {
		l.error.Printf(format, v...)
	}
}

// GetLogger returns the default logger.
func GetLogger() *Logger {
	loggerMux.RLock()
	defer loggerMux.RUnlock()
	return defaultLogger
}

// SetLogger sets the default logger.
func SetLogger(logger *Logger) {
	loggerMux.Lock()
	defer loggerMux.Unlock()
	defaultLogger = logger
}

// Package-level convenience functions
func Debug(v ...interface{})                 { GetLogger().Debug(v...) }
func Debugf(format string, v ...interface{}) { GetLogger().Debugf(format, v...) }
func Info(v ...interface{})                  { GetLogger().Info(v...) }
func Infof(format string, v ...interface{})  { GetLogger().Infof(format, v...) }
func Warn(v ...interface{})                  { GetLogger().Warn(v...) }
func Warnf(format string, v ...interface{})  { GetLogger().Warnf(format, v...) }
func Error(v ...interface{})                 { GetLogger().Error(v...) }
func Errorf(format string, v ...interface{}) { GetLogger().Errorf(format, v...) }
