package gost_test

import (
	"math"
	"math/rand"
	"testing"

	"gost"
	"gost/internal/mip"
)

// TestWarmStartDynamic checks that the LP re-solves that follow added rows
// (lazy constraints, user cuts) or added columns (pricing) start from the
// previous basis: each re-solve must take far fewer simplex iterations than
// solving the final LP from scratch.
func TestWarmStartDynamic(t *testing.T) {
	report := func(name string, cold int) {
		d := mip.DynStats
		for _, k := range []string{"rows", "cols"} {
			if d.Resolves[k] == 0 {
				continue
			}
			avg := float64(d.Iters[k]) / float64(d.Resolves[k])
			t.Logf("%s: %d re-solves after added %s, %.1f iters on average (max %d; phase 1 %d, primal %d, dual %d); cold solve of the final LP: %d iters",
				name, d.Resolves[k], k, avg, d.MaxIters[k], d.Phase1[k], d.Primal[k], d.Dual[k], cold)
			if avg > 0.5*float64(cold) {
				t.Errorf("%s: re-solves after added %s are not warm (%.1f vs %d)", name, k, avg, cold)
			}
		}
	}

	// Column generation: cutting stock LP with many widths.
	{
		rng := rand.New(rand.NewSource(5))
		cs := &cuttingStock{W: 1000}
		for i := 0; i < 40; i++ {
			cs.widths = append(cs.widths, 50+rng.Intn(400))
			cs.demand = append(cs.demand, float64(10+rng.Intn(90)))
		}
		mip.DynStats.Reset()
		r, m := cs.solve(t, false, 0)
		cold := m.SolveRelaxation()
		if math.Abs(cold.Objective-r.Objective) > 1e-6*math.Abs(r.Objective) {
			t.Fatalf("cutting stock: %v vs cold %v", r.Objective, cold.Objective)
		}
		report("cutting stock (pricing)", cold.Iterations)
	}

	// Row generation: TSP with lazy subtour constraints and user cuts.
	{
		rng := rand.New(rand.NewSource(8))
		n := 70
		px, py := make([]float64, n), make([]float64, n)
		for i := range px {
			px[i], py[i] = rng.Float64()*1000, rng.Float64()*1000
		}
		d := make([][]float64, n)
		for i := range d {
			d[i] = make([]float64, n)
			for j := range d[i] {
				d[i][j] = math.Round(math.Hypot(px[i]-px[j], py[i]-py[j]))
			}
		}
		tp := newTSP(d)
		p := gost.DefaultParams()
		add := func(ctx *gost.ConstrContext, thr float64) {
			comps := tp.components(func(k int) float64 { return ctx.Value(tp.x[k]) }, thr)
			if len(comps) > 1 {
				for _, c := range comps {
					ctx.AddConstr(tp.subtourCut(c), gost.GreaterEq, 2, "subtour")
				}
			}
		}
		p.LazyConstraints = func(ctx *gost.ConstrContext) { add(ctx, 0.5) }
		mip.DynStats.Reset()
		r, _ := tp.m.Solve(p)
		if r.Status != gost.StatusOptimal {
			t.Fatalf("tsp: %v", r.Status)
		}
		cold := tp.m.SolveRelaxation()
		report("TSP (lazy constraints only)", cold.Iterations)

		tp = newTSP(d)
		p.UserCuts = func(ctx *gost.ConstrContext) { add(ctx, 1e-6) }
		mip.DynStats.Reset()
		r2, _ := tp.m.Solve(p)
		if r2.Status != gost.StatusOptimal || math.Abs(r2.Objective-r.Objective) > 1e-6 {
			t.Fatalf("tsp with cuts: %v %v vs %v", r2.Status, r2.Objective, r.Objective)
		}
		cold = tp.m.SolveRelaxation()
		report("TSP (lazy + user cuts)", cold.Iterations)
	}
}
