package gost_test

import (
	"os"
	"testing"
	"time"

	"gost"
	"gost/mps"
)

func TestProfileInstance(t *testing.T) {
	name := os.Getenv("GOST_PROF")
	if name == "" {
		t.Skip()
	}
	m, err := mps.ReadFile("testdata/" + name + ".mps.gz")
	if err != nil {
		t.Fatal(err)
	}
	p := gost.DefaultParams()
	p.TimeLimit = 20 * time.Second
	r, _ := m.Solve(p)
	t.Logf("%v obj %v bound %v nodes %d", r.Status, r.Objective, r.Bound, r.Nodes)
}
