package simplex

import (
	"math"
	"math/rand"
	"testing"
)

type denseLP struct {
	n, m   int
	c      []float64
	lb, ub []float64
	a      [][]float64 // a[i][j]
	rl, ru []float64
}

func randomLP(rng *rand.Rand, n, m int) *denseLP {
	p := &denseLP{n: n, m: m}
	for j := 0; j < n; j++ {
		p.c = append(p.c, float64(rng.Intn(21)-10))
		p.lb = append(p.lb, 0)
		p.ub = append(p.ub, float64(1+rng.Intn(10)))
	}
	for i := 0; i < m; i++ {
		row := make([]float64, n)
		s := 0.0
		for j := range row {
			if rng.Float64() < 0.5 {
				row[j] = float64(rng.Intn(11) - 3)
				s += math.Abs(row[j])
			}
		}
		p.a = append(p.a, row)
		p.rl = append(p.rl, math.Inf(-1))
		p.ru = append(p.ru, s*rng.Float64())
	}
	return p
}

func (p *denseLP) build(ncols int) *LP {
	var start, idx []int
	var val []float64
	start = append(start, 0)
	for j := 0; j < ncols; j++ {
		for i := 0; i < p.m; i++ {
			if p.a[i][j] != 0 {
				idx = append(idx, i)
				val = append(val, p.a[i][j])
			}
		}
		start = append(start, len(idx))
	}
	return New(ncols, p.m, p.c[:ncols], p.lb[:ncols], p.ub[:ncols], start, idx, val, p.rl, p.ru)
}

func TestAddColsMatchesFresh(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for trial := 0; trial < 200; trial++ {
		n := 4 + rng.Intn(20)
		m := 2 + rng.Intn(15)
		p := randomLP(rng, n, m)
		k := 1 + rng.Intn(n-1)
		lp := p.build(n - k)
		if st := lp.Solve(); st != Optimal {
			t.Fatalf("trial %d: initial %v", trial, st)
		}
		// add remaining columns in two batches
		for b := 0; b < 2; b++ {
			lo, hi := n-k, n
			if b == 0 {
				hi = n - k/2
			} else {
				lo = n - k/2
			}
			if lo >= hi {
				continue
			}
			var cs, l, u []float64
			var ci [][]int
			var cv [][]float64
			for j := lo; j < hi; j++ {
				cs = append(cs, p.c[j])
				l = append(l, p.lb[j])
				u = append(u, p.ub[j])
				var ii []int
				var vv []float64
				for i := 0; i < m; i++ {
					if p.a[i][j] != 0 {
						ii = append(ii, i)
						vv = append(vv, p.a[i][j])
					}
				}
				ci = append(ci, ii)
				cv = append(cv, vv)
			}
			b0 := lp.GetBasis()
			lp.AddCols(cs, l, u, ci, cv)
			if b == 1 {
				lp.SetBasis(b0) // exercise basis extension
			}
			if st := lp.Solve(); st != Optimal {
				t.Fatalf("trial %d: after add %v", trial, st)
			}
		}
		fresh := p.build(n)
		if st := fresh.Solve(); st != Optimal {
			t.Fatalf("trial %d: fresh %v", trial, st)
		}
		if math.Abs(lp.Objective()-fresh.Objective()) > 1e-6*(1+math.Abs(fresh.Objective())) {
			t.Fatalf("trial %d: incremental %v fresh %v", trial, lp.Objective(), fresh.Objective())
		}
		// duals give a valid certificate: reduced costs consistent
		for j := 0; j < n; j++ {
			d := p.c[j]
			for i := 0; i < m; i++ {
				d -= p.a[i][j] * lp.Dual(i)
			}
			if math.Abs(d-lp.ReducedCost(j)) > 1e-6*(1+math.Abs(d)) {
				t.Fatalf("trial %d: reduced cost %d: %v vs %v", trial, j, lp.ReducedCost(j), d)
			}
		}
	}
}

func TestAddRowsMatchesFresh(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for trial := 0; trial < 200; trial++ {
		n := 3 + rng.Intn(15)
		m := 2 + rng.Intn(12)
		p := randomLP(rng, n, m)
		k := 1 + rng.Intn(m-1)
		sub := *p
		sub.m = m - k
		sub.a, sub.rl, sub.ru = p.a[:m-k], p.rl[:m-k], p.ru[:m-k]
		lp := sub.build(n)
		if st := lp.Solve(); st != Optimal {
			t.Fatalf("trial %d: initial %v", trial, st)
		}
		var ri [][]int
		var rv [][]float64
		for i := m - k; i < m; i++ {
			var ii []int
			var vv []float64
			for j := 0; j < n; j++ {
				if p.a[i][j] != 0 {
					ii = append(ii, j)
					vv = append(vv, p.a[i][j])
				}
			}
			ri = append(ri, ii)
			rv = append(rv, vv)
		}
		lp.AddRows(ri, rv, p.rl[m-k:], p.ru[m-k:])
		if st := lp.Solve(); st != Optimal {
			t.Fatalf("trial %d: after rows %v", trial, st)
		}
		fresh := p.build(n)
		fresh.Solve()
		if math.Abs(lp.Objective()-fresh.Objective()) > 1e-6*(1+math.Abs(fresh.Objective())) {
			t.Fatalf("trial %d: incremental %v fresh %v", trial, lp.Objective(), fresh.Objective())
		}
	}
}
