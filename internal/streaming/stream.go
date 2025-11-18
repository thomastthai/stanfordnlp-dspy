// Package streaming provides streaming utilities for DSPy.
package streaming

import (
	"context"
	"errors"
	"io"
)

// StreamItem represents a single item in a stream.
type StreamItem struct {
	Data  interface{}
	Error error
	Done  bool
}

// Stream represents a stream of data.
type Stream struct {
	ch     chan StreamItem
	ctx    context.Context
	cancel context.CancelFunc
}

// NewStream creates a new stream.
func NewStream(ctx context.Context, bufferSize int) *Stream {
	streamCtx, cancel := context.WithCancel(ctx)
	return &Stream{
		ch:     make(chan StreamItem, bufferSize),
		ctx:    streamCtx,
		cancel: cancel,
	}
}

// Send sends an item to the stream.
func (s *Stream) Send(data interface{}) error {
	select {
	case s.ch <- StreamItem{Data: data}:
		return nil
	case <-s.ctx.Done():
		return s.ctx.Err()
	}
}

// SendError sends an error to the stream.
func (s *Stream) SendError(err error) error {
	select {
	case s.ch <- StreamItem{Error: err}:
		return nil
	case <-s.ctx.Done():
		return s.ctx.Err()
	}
}

// Close closes the stream.
func (s *Stream) Close() {
	s.ch <- StreamItem{Done: true}
	close(s.ch)
	s.cancel()
}

// Channel returns the underlying channel.
func (s *Stream) Channel() <-chan StreamItem {
	return s.ch
}

// Next returns the next item from the stream.
func (s *Stream) Next() (interface{}, error) {
	select {
	case item, ok := <-s.ch:
		if !ok {
			return nil, io.EOF
		}
		if item.Done {
			return nil, io.EOF
		}
		if item.Error != nil {
			return nil, item.Error
		}
		return item.Data, nil
	case <-s.ctx.Done():
		return nil, s.ctx.Err()
	}
}

// Collect collects all items from the stream into a slice.
func (s *Stream) Collect() ([]interface{}, error) {
	var items []interface{}
	for {
		item, err := s.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return items, err
		}
		items = append(items, item)
	}
	return items, nil
}

// StreamTransformer transforms stream items.
type StreamTransformer func(interface{}) (interface{}, error)

// Transform applies a transformation to each item in the stream.
func (s *Stream) Transform(transformer StreamTransformer) *Stream {
	output := NewStream(s.ctx, cap(s.ch))
	
	go func() {
		defer output.Close()
		
		for {
			item, err := s.Next()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				output.SendError(err)
				break
			}
			
			transformed, err := transformer(item)
			if err != nil {
				output.SendError(err)
				break
			}
			
			if err := output.Send(transformed); err != nil {
				break
			}
		}
	}()
	
	return output
}

// Filter filters stream items based on a predicate.
func (s *Stream) Filter(predicate func(interface{}) bool) *Stream {
	output := NewStream(s.ctx, cap(s.ch))
	
	go func() {
		defer output.Close()
		
		for {
			item, err := s.Next()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				output.SendError(err)
				break
			}
			
			if predicate(item) {
				if err := output.Send(item); err != nil {
					break
				}
			}
		}
	}()
	
	return output
}

// Merge merges multiple streams into one.
func Merge(ctx context.Context, streams ...*Stream) *Stream {
	output := NewStream(ctx, 10)
	
	go func() {
		defer output.Close()
		
		// Create a channel to collect items from all streams
		merged := make(chan StreamItem, len(streams))
		done := make(chan struct{})
		
		// Start a goroutine for each stream
		for _, stream := range streams {
			go func(s *Stream) {
				for {
					item, err := s.Next()
					if errors.Is(err, io.EOF) {
						break
					}
					
					var streamItem StreamItem
					if err != nil {
						streamItem = StreamItem{Error: err}
					} else {
						streamItem = StreamItem{Data: item}
					}
					
					select {
					case merged <- streamItem:
					case <-done:
						return
					case <-ctx.Done():
						return
					}
				}
			}(stream)
		}
		
		// Collect items from merged channel
		activeStreams := len(streams)
		for activeStreams > 0 {
			select {
			case item := <-merged:
				if item.Error != nil {
					output.SendError(item.Error)
				} else if item.Done {
					activeStreams--
				} else {
					output.Send(item.Data)
				}
			case <-ctx.Done():
				close(done)
				return
			}
		}
	}()
	
	return output
}

// Batch batches stream items.
func (s *Stream) Batch(batchSize int) *Stream {
	output := NewStream(s.ctx, cap(s.ch)/batchSize+1)
	
	go func() {
		defer output.Close()
		
		batch := make([]interface{}, 0, batchSize)
		
		for {
			item, err := s.Next()
			if errors.Is(err, io.EOF) {
				if len(batch) > 0 {
					output.Send(batch)
				}
				break
			}
			if err != nil {
				output.SendError(err)
				break
			}
			
			batch = append(batch, item)
			if len(batch) >= batchSize {
				if err := output.Send(batch); err != nil {
					break
				}
				batch = make([]interface{}, 0, batchSize)
			}
		}
	}()
	
	return output
}

// Tee splits a stream into two identical streams.
func (s *Stream) Tee() (*Stream, *Stream) {
	out1 := NewStream(s.ctx, cap(s.ch))
	out2 := NewStream(s.ctx, cap(s.ch))
	
	go func() {
		defer out1.Close()
		defer out2.Close()
		
		for {
			item, err := s.Next()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				out1.SendError(err)
				out2.SendError(err)
				break
			}
			
			// Send to both outputs
			if err := out1.Send(item); err != nil {
				break
			}
			if err := out2.Send(item); err != nil {
				break
			}
		}
	}()
	
	return out1, out2
}
