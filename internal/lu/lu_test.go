package lu

import (
	"math"
	"math/rand"
	"testing"
)

type dense struct {
	m    int
	cols [][]float64
}

func randomBasis(rng *rand.Rand, m int, density float64) *dense {
	d := &dense{m: m, cols: make([][]float64, m)}
	for k := 0; k < m; k++ {
		c := make([]float64, m)
		if rng.Float64() < 0.3 {
			c[k] = -1 // logical-like
		} else {
			c[k] = rng.NormFloat64() + 3
			for i := 0; i < m; i++ {
				if i != k && rng.Float64() < density {
					c[i] = rng.NormFloat64()
				}
			}
		}
		d.cols[k] = c
	}
	// shuffle columns to break diagonal structure
	rng.Shuffle(m, func(a, b int) { d.cols[a], d.cols[b] = d.cols[b], d.cols[a] })
	return d
}

func (d *dense) colFunc() ColumnFunc {
	return func(k int) ([]int, []float64) {
		var idx []int
		var val []float64
		for i, v := range d.cols[k] {
			if v != 0 {
				idx = append(idx, i)
				val = append(val, v)
			}
		}
		return idx, val
	}
}

func (d *dense) mul(x []float64) []float64 { // B x
	b := make([]float64, d.m)
	for k, c := range d.cols {
		for i, v := range c {
			b[i] += v * x[k]
		}
	}
	return b
}

func (d *dense) mulT(y []float64) []float64 { // B^T y
	r := make([]float64, d.m)
	for k, c := range d.cols {
		for i, v := range c {
			r[k] += v * y[i]
		}
	}
	return r
}

func maxDiff(a, b []float64) float64 {
	m := 0.0
	for i := range a {
		m = math.Max(m, math.Abs(a[i]-b[i]))
	}
	return m
}

func TestFactorSolve(t *testing.T) {
	for _, ft := range []bool{false, true} {
		UseFT = ft
		testFactorSolve(t)
	}
	UseFT = true
}

func testFactorSolve(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for trial := 0; trial < 50; trial++ {
		m := 1 + rng.Intn(60)
		d := randomBasis(rng, m, 0.1)
		f := New(m)
		if s := f.Factorize(m, d.colFunc()); s != nil {
			continue // random singular matrix; skip
		}
		// with updates
		for upd := 0; upd <= 60; upd++ {
			b := make([]float64, m)
			for i := range b {
				b[i] = rng.NormFloat64()
			}
			x := append([]float64(nil), b...)
			f.FTRAN(x)
			xm := 0.0
			for _, v := range x {
				xm = math.Max(xm, math.Abs(v))
			}
			if e := maxDiff(d.mul(x), b); e > 1e-8*(1+xm) {
				t.Fatalf("FT=%v trial %d upd %d ftran err %g xm %g", UseFT, trial, upd, e, xm)
			}
			y := append([]float64(nil), b...)
			f.BTRAN(y)
			ym := 0.0
			for _, v := range y {
				ym = math.Max(ym, math.Abs(v))
			}
			if e := maxDiff(d.mulT(y), b); e > 1e-8*(1+ym) {
				t.Fatalf("trial %d upd %d btran err %g", trial, upd, e)
			}
			// replace a random column
			pos := rng.Intn(m)
			nc := make([]float64, m)
			for i := range nc {
				if rng.Float64() < 0.2 {
					nc[i] = rng.NormFloat64()
				}
			}
			nc[rng.Intn(m)] += 2
			alpha := append([]float64(nil), nc...)
			f.FTRANCol(alpha)
			if math.Abs(alpha[pos]) < 1e-2 {
				continue
			}
			if !f.Update(pos, alpha) {
				t.Fatalf("update failed (FT=%v)", UseFT)
			}
			d.cols[pos] = nc
		}
	}
}

func TestSingular(t *testing.T) {
	d := &dense{m: 3, cols: [][]float64{{1, 0, 0}, {2, 0, 0}, {0, 0, 1}}}
	f := New(3)
	s := f.Factorize(3, d.colFunc())
	if s == nil || len(s.Positions) != 1 || len(s.Rows) != 1 || s.Rows[0] != 1 {
		t.Fatalf("expected singularity, got %+v", s)
	}
}
