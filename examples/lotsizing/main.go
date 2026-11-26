// Multi-item capacitated lot sizing (CLSP): when, and in which batch sizes,
// to produce several items on one shared resource.
//
// Producing an item in a period requires a setup (fixed cost and setup
// time); stock carried from one period to the next costs a holding cost.
// Large batches save setups but build up stock; the shared capacity of each
// period limits the batches. This is the classic batching trade-off of
// production planning (Trigeiro, Thomas and McClain, 1989).
//
// Model (items i, periods t = 0..T−1, initial stock 0):
//
//	min  Σ_i Σ_t (sc_i·y_it + h_i·s_it)                  (setup + holding)
//	s.t. s_i,t−1 + x_it − s_it = d_it        for each i, t   (stock balance)
//	     x_it ≤ M_it·y_it                    for each i, t   (setup forcing)
//	     Σ_i (x_it + st_i·y_it) ≤ C_t        for each t      (capacity)
//	     x, s ≥ 0, y ∈ {0, 1}
//
// x_it: quantity of item i produced in period t (the batch size);
// s_it: stock of item i at the end of period t;
// y_it: 1 if item i is set up in period t.
// M_it = min(C_t − st_i, remaining demand of i from t on) is the smallest
// valid "big M", which keeps the LP relaxation as tight as possible.
//
// The LP relaxation of this model is weak: y_it can take the small value
// x_it/M_it, so setups look almost free. The classic remedy is the family
// of (l,S) inequalities of Barany, Van Roy and Wolsey (1984), valid for
// each item i, each period l and each subset S of periods 0..l:
//
//	Σ_{t ∈ S} x_it ≤ Σ_{t ∈ S} d_i,t..l · y_it + s_il
//
// where d_i,t..l is the demand of item i from t to l. Meaning: what is
// produced in the periods of S either serves demand up to l — and then at
// most d_i,t..l can be produced in period t, and only if set up — or is
// still in stock at the end of l. There are exponentially many subsets S,
// but the most violated one for given i and l is found greedily: put t in S
// exactly when x_it > d_i,t..l · y_it. They are added as user cuts.
//
// The model is solved twice, without and with these cuts. The output
// shows the LP relaxation, the bound reached at the end of the root node
// (after the solver's own cuts, and the (l,S) cuts in the second run) and
// the size of the search tree. Note that GoST disables presolve when a
// callback is given, so the second run also differs in that respect.
//
//	go run ./examples/lotsizing
package main

import (
	"fmt"
	"math"
	"math/rand"

	"gost"
)

func main() {
	// Fixed seed: the same instance on every run.
	rng := rand.New(rand.NewSource(5))
	const nI, nT = 6, 8 // items and periods

	// --- Data ------------------------------------------------------------------
	// d[i][t]: demand; sc[i]: setup cost; st[i]: setup time (capacity
	// used by a setup); h[i]: holding cost per unit and period.
	d := make([][]float64, nI)
	sc, st, h := make([]float64, nI), make([]float64, nI), make([]float64, nI)
	load := 0.0 // total demand + one setup per item and period, for C
	for i := 0; i < nI; i++ {
		d[i] = make([]float64, nT)
		for t := range d[i] {
			// no demand in the first two periods (the plan starts with
			// empty stocks), and none in some later periods
			if t >= 2 && rng.Intn(5) > 0 {
				d[i][t] = float64(20 + rng.Intn(60))
			}
			load += d[i][t]
		}
		sc[i] = float64(200 + rng.Intn(600))
		st[i] = float64(10 + rng.Intn(20))
		h[i] = float64(1 + rng.Intn(3))
		load += st[i] * nT / 3 // about one setup every three periods
	}
	// C[t]: capacity of period t, about 10% above the average need.
	C := make([]float64, nT)
	for t := range C {
		C[t] = math.Round(1.1 * load / nT)
	}
	// dsum(i, a, b): demand of item i from period a to period b (inclusive).
	dsum := func(i, a, b int) float64 {
		v := 0.0
		for t := a; t <= b && t < nT; t++ {
			v += d[i][t]
		}
		return v
	}

	// build creates the model; withCuts adds the (l,S) separation callback.
	build := func(withCuts bool) (*gost.Model, gost.Params, *int) {
		m := gost.NewModel("lot-sizing")
		x := make([][]gost.Var, nI) // production (batch size)
		s := make([][]gost.Var, nI) // end-of-period stock
		y := make([][]gost.Var, nI) // setup indicator
		for i := 0; i < nI; i++ {
			x[i], s[i], y[i] = make([]gost.Var, nT), make([]gost.Var, nT), make([]gost.Var, nT)
			for t := 0; t < nT; t++ {
				x[i][t] = m.AddVar(0, gost.Inf, 0, gost.Continuous, fmt.Sprintf("x%d_%d", i, t))
				s[i][t] = m.AddVar(0, gost.Inf, h[i], gost.Continuous, fmt.Sprintf("s%d_%d", i, t))
				y[i][t] = m.AddVar(0, 1, sc[i], gost.Binary, fmt.Sprintf("y%d_%d", i, t))
			}
		}
		for i := 0; i < nI; i++ {
			for t := 0; t < nT; t++ {
				// balance_i_t: s_i,t−1 + x_it − s_it = d_it.
				e := gost.NewExpr().Add(1, x[i][t]).Add(-1, s[i][t])
				if t > 0 {
					e.Add(1, s[i][t-1])
				}
				m.AddConstr(e, gost.Equal, d[i][t], fmt.Sprintf("balance%d_%d", i, t))
				// setup_i_t: x_it ≤ M_it·y_it with the tightest valid M.
				M := math.Min(C[t]-st[i], dsum(i, t, nT-1))
				m.AddConstr(gost.NewExpr().Add(1, x[i][t]).Add(-M, y[i][t]), gost.LessEq, 0, fmt.Sprintf("setup%d_%d", i, t))
			}
		}
		for t := 0; t < nT; t++ {
			// capacity_t: production plus setup times fit in period t.
			e := gost.NewExpr()
			for i := 0; i < nI; i++ {
				e.Add(1, x[i][t]).Add(st[i], y[i][t])
			}
			m.AddConstr(e, gost.LessEq, C[t], fmt.Sprintf("capacity%d", t))
		}

		p := gost.DefaultParams()
		cuts := new(int) // number of (l,S) cuts added
		if withCuts {
			// UserCuts: called on fractional LP solutions. For each item i
			// and period l, build the most violated (l,S) inequality and add
			// it if it is violated.
			p.UserCuts = func(ctx *gost.ConstrContext) {
				for i := 0; i < nI; i++ {
					for l := 0; l < nT; l++ {
						lhs := 0.0 // Σ_{t∈S} (x_it − d_i,t..l·y_it), to compare with s_il
						var S []int
						for t := 0; t <= l; t++ {
							if v := ctx.Value(x[i][t]) - dsum(i, t, l)*ctx.Value(y[i][t]); v > 1e-9 {
								lhs += v
								S = append(S, t)
							}
						}
						if lhs > ctx.Value(s[i][l])+1e-4 {
							// Σ_{t∈S} x_it − Σ_{t∈S} d_i,t..l·y_it − s_il ≤ 0
							e := gost.NewExpr().Add(-1, s[i][l])
							for _, t := range S {
								e.Add(1, x[i][t]).Add(-dsum(i, t, l), y[i][t])
							}
							ctx.AddConstr(e, gost.LessEq, 0, fmt.Sprintf("lS_%d_%d", i, l))
							*cuts++
						}
					}
				}
			}
		}
		return m, p, cuts
	}

	var objs []float64
	for _, withCuts := range []bool{false, true} {
		m, p, cuts := build(withCuts)
		// LP relaxation, before any branching or cut, for comparison.
		lp := m.SolveRelaxation()
		res, err := m.Solve(p)
		if err != nil {
			panic(err)
		}
		label := "without (l,S) cuts"
		if withCuts {
			label = "with (l,S) cuts   "
		}
		fmt.Printf("%s: %s, cost %.0f | LP relaxation %.0f, root bound after cuts %.0f | %d (l,S) cuts, %d nodes, %v\n",
			label, res.Status, res.Objective, lp.Objective, res.RootBound, *cuts, res.Nodes, res.Time.Round(1e6))
		objs = append(objs, res.Objective)
		if withCuts {
			printPlan(m, res, nI, nT, d)
		}
	}
	if math.Abs(objs[0]-objs[1]) > 1e-6*math.Abs(objs[0]) {
		fmt.Println("ERROR: the two runs disagree")
	}
}

// printPlan prints the production plan (batch sizes) of each item, with
// the demand of each period below it, and checks the stock balances.
func printPlan(m *gost.Model, res *gost.Result, nI, nT int, d [][]float64) {
	val := func(name string) float64 {
		for j := 0; j < m.NumVars(); j++ {
			if n, _, _, _, _ := m.VarInfo(m.VarByIndex(j)); n == name {
				return res.Value(m.VarByIndex(j))
			}
		}
		return math.NaN()
	}
	fmt.Println("\nproduction plan (batch sizes per period; demand below):")
	for i := 0; i < nI; i++ {
		stock := 0.0
		ok := true
		fmt.Printf("item %d  prod:", i)
		for t := 0; t < nT; t++ {
			x := val(fmt.Sprintf("x%d_%d", i, t))
			stock += x - d[i][t]
			if stock < -1e-6 {
				ok = false // demand not met in time
			}
			if x > 1e-6 {
				fmt.Printf(" %4.0f", x)
			} else {
				fmt.Print("    .")
			}
		}
		fmt.Printf("\n         dem:")
		for t := 0; t < nT; t++ {
			fmt.Printf(" %4.0f", d[i][t])
		}
		fmt.Printf("   demand met on time: %v\n", ok)
	}
}
