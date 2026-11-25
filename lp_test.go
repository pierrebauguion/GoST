package gost_test

import (
	"math"
	"path/filepath"
	"testing"

	"gost/mps"
)

var lpOpt = map[string]float64{
	"afiro": -464.7531428571, "brandy": 1518.509896, "e226": -18.751929066 + 7.113, "finnis": 172791.0655,
	"p0033": 2520.57, "p0201": 6875.0, "p0282": 176867.50, "p0548": 315.2549019607843, "p2756": 2688.75,
	"lseu": 834.68, "bell5": 8608417.95, "bell3a": 862578.64, "egout": 149.589, "enigma": 0,
	"flugpl": 1167185.73, "gt2": 13460.233074, "stein27": 13.0, "misc03": 1910.0, "vpm1": 15.4167,
	"vpm2": 9.8892645972, "mod008": 290.93, "khb05250": 95919464.0, "pk1": 0, "rgn": 48.7999,
	"dcmulti": 183975.5397, "fixnet6": 1200.88, "qnet1": 14274.102667, "mas76": 38893.9036,
	"blend2": 6.9156751140, "misc07": 1415.0, "10teams": 917, "air03": 338864.25,
	"gesa2": 25476489.678, "fiber": 156082.51759,
}

func TestLPRelaxations(t *testing.T) {
	for name, want := range lpOpt {
		m, err := mps.ReadFile(filepath.Join("testdata", name+".mps.gz"))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		r := m.SolveRelaxation()
		rel := math.Abs(r.Objective-want) / math.Max(1, math.Abs(want))
		ok := r.Status == "optimal" && rel < 1e-4
		t.Logf("%-10s %-10s obj=%.6f want=%.6f iters=%d time=%v ok=%v", name, r.Status, r.Objective, want, r.Iterations, r.Time, ok)
		if !ok {
			t.Errorf("%s mismatch", name)
		}
	}
}
