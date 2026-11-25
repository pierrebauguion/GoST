package gost_test

import (
	"os"
	"testing"
	"time"

	"gost/mps"
)

func TestOneLP(t *testing.T) {
	f := os.Getenv("GOST_LPFILE")
	if f == "" {
		t.Skip()
	}
	m, err := mps.ReadFile(f)
	if err != nil {
		t.Fatal(err)
	}
	r := m.SolveRelaxation(20 * time.Second)
	t.Logf("%s obj=%.10g iters=%d time=%v", r.Status, r.Objective, r.Iterations, r.Time)
}
