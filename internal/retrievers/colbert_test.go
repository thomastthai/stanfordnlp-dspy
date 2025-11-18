package retrievers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestNewColBERTv2(t *testing.T) {
	tests := []struct {
		name string
		opts ColBERTv2Options
		want string
	}{
		{
			name: "default URL",
			opts: ColBERTv2Options{},
			want: "http://0.0.0.0",
		},
		{
			name: "custom URL",
			opts: ColBERTv2Options{URL: "http://localhost"},
			want: "http://localhost",
		},
		{
			name: "URL with port",
			opts: ColBERTv2Options{URL: "http://localhost", Port: 8080},
			want: "http://localhost:8080",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			retriever := NewColBERTv2(tt.opts)
			if retriever.url != tt.want {
				t.Errorf("NewColBERTv2() url = %v, want %v", retriever.url, tt.want)
			}
			if retriever.Name() != "colbertv2" {
				t.Errorf("NewColBERTv2() name = %v, want %v", retriever.Name(), "colbertv2")
			}
		})
	}
}

func TestColBERTv2_Retrieve_GET(t *testing.T) {
	// Create a mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" {
			t.Errorf("Expected GET request, got %s", r.Method)
		}

		query := r.URL.Query().Get("query")
		if query != "test query" {
			t.Errorf("Expected query 'test query', got '%s'", query)
		}

		response := colbertResponse{
			TopK: []colbertDocument{
				{
					Text:     "Short text 1",
					LongText: "This is document 1",
					PID:      1,
					Score:    0.95,
				},
				{
					Text:     "Short text 2",
					LongText: "This is document 2",
					PID:      2,
					Score:    0.85,
				},
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	retriever := NewColBERTv2(ColBERTv2Options{
		URL:     server.URL,
		UsePost: false,
		Timeout: 5 * time.Second,
	})

	ctx := context.Background()
	docs, err := retriever.Retrieve(ctx, "test query", 2)
	if err != nil {
		t.Fatalf("Retrieve() error = %v", err)
	}

	if len(docs) != 2 {
		t.Errorf("Expected 2 documents, got %d", len(docs))
	}

	if docs[0] != "This is document 1" {
		t.Errorf("Expected first doc to be 'This is document 1', got '%s'", docs[0])
	}
}

func TestColBERTv2_RetrieveWithScores_POST(t *testing.T) {
	// Create a mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("Expected POST request, got %s", r.Method)
		}

		var payload map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("Failed to decode request body: %v", err)
		}

		if payload["query"] != "test query" {
			t.Errorf("Expected query 'test query', got '%v'", payload["query"])
		}

		response := colbertResponse{
			TopK: []colbertDocument{
				{
					Text:     "Short text 1",
					LongText: "This is document 1 with scores",
					PID:      "doc1",
					Score:    0.95,
					Metadata: map[string]interface{}{"source": "test"},
				},
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	retriever := NewColBERTv2(ColBERTv2Options{
		URL:     server.URL,
		UsePost: true,
		Timeout: 5 * time.Second,
	})

	ctx := context.Background()
	docs, err := retriever.RetrieveWithScores(ctx, "test query", 1)
	if err != nil {
		t.Fatalf("RetrieveWithScores() error = %v", err)
	}

	if len(docs) != 1 {
		t.Errorf("Expected 1 document, got %d", len(docs))
	}

	doc := docs[0]
	if doc.Content != "This is document 1 with scores" {
		t.Errorf("Expected content 'This is document 1 with scores', got '%s'", doc.Content)
	}
	if doc.Score != 0.95 {
		t.Errorf("Expected score 0.95, got %f", doc.Score)
	}
	if doc.ID != "doc1" {
		t.Errorf("Expected ID 'doc1', got '%s'", doc.ID)
	}
}

func TestColBERTv2_MaxK(t *testing.T) {
	retriever := NewColBERTv2(ColBERTv2Options{
		URL:  "http://localhost",
		MaxK: 50,
	})

	ctx := context.Background()
	_, err := retriever.RetrieveWithScores(ctx, "test", 100)
	if err == nil {
		t.Error("Expected error for k > maxK, got nil")
	}
}

func TestColBERTv2_ErrorHandling(t *testing.T) {
	// Create a server that returns an error
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("Internal server error"))
	}))
	defer server.Close()

	retriever := NewColBERTv2(ColBERTv2Options{
		URL:     server.URL,
		Timeout: 5 * time.Second,
	})

	ctx := context.Background()
	_, err := retriever.Retrieve(ctx, "test", 10)
	if err == nil {
		t.Error("Expected error from server, got nil")
	}
}
