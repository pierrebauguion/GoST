package gost_test

import (
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"gost/internal/simplex"
	"gost/mps"
)

// GOST_LPBENCH=<dir> go test -run TestLPBench -v .
func TestLPBench(t *testing.T) {
	dir := os.Getenv("GOST_LPBENCH")
	if dir == "" {
		t.Skip()
	}
	files, _ := filepath.Glob(filepath.Join(dir, "*.mps.gz"))
	if only := os.Getenv("GOST_ONLY"); only != "" {
		files = []string{filepath.Join(dir, only+".mps.gz")}
	}
	sort.Strings(files)
	total := 0.0
	for _, f := range files {
		m, err := mps.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		simplex.FillStats = struct{ Refactors, L, U, Eta, M int }{}
		r := m.SolveRelaxation(60 * time.Second)
		if os.Getenv("GOST_REPEAT") != "" {
			for rep := 1; rep < 3; rep++ {
				r2 := m.SolveRelaxation(60 * time.Second)
				if r2.Time < r.Time {
					r = r2
				}
			}
		}
		if fs := simplex.FillStats; fs.Refactors > 0 {
			t.Logf("   refactors=%d avg m=%d L=%d U=%d eta=%d", fs.Refactors, fs.M/fs.Refactors, fs.L/fs.Refactors, fs.U/fs.Refactors, fs.Eta/fs.Refactors)
		}
		total += r.Time.Seconds()
		t.Logf("%-10s %-10s obj=%.10g iters=%d time=%.3fs", filepath.Base(f[:len(f)-7]), r.Status, r.Objective, r.Iterations, r.Time.Seconds())
	}
	t.Logf("total %.2fs", total)
}
