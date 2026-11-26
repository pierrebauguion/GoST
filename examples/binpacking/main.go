// Bin packing: group items into as few bins (batches) of capacity C as
// possible — the basic batching problem.
//
// Model (assignment formulation, Kantorovich 1939/1960), with n items and
// at most U bins (U = number of bins used by a heuristic):
//
//	min  Σ_b y_b                                        (bins used)
//	s.t. Σ_b x_ib = 1                for each item i     (assignment)
//	     Σ_i w_i·x_ib ≤ C·y_b        for each bin b      (capacity)
//	     x_ib ≤ y_b                  for each i, b       (linking)
//	     y_b ≥ y_b+1                 for each bin b      (symmetry breaking)
//	     x_ib = 0 for b > i                              (symmetry breaking)
//	     x, y ∈ {0, 1}
//
// x_ib = 1 if item i goes into bin b, y_b = 1 if bin b is used.
//
// All bins are identical, so any solution can be rewritten in U!
// equivalent ways by renaming the bins; branch-and-bound would explore all
// of them. Two classic constraints remove most of this symmetry: bins are
// used in order (y_b ≥ y_b+1), and item i may only go into bins 0..i (with
// items sorted by decreasing size, item 0 is in bin 0, item 1 in bin 0 or
// 1, and so on).
//
// The linking constraints x_ib ≤ y_b are implied by the capacity rows for
// integer solutions, but they make the LP relaxation much stronger.
//
// Bounds compared with the optimum:
//   - L1 = ⌈Σ w_i / C⌉, the continuous lower bound;
//   - First Fit Decreasing (FFD), the classic heuristic (≤ 11/9·OPT + 6/9).
//
// See examples/cuttingstock for the column generation (Gilmore–Gomory)
// approach, better suited when many items have the same size.
//
//	go run ./examples/binpacking
package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"gost"
)

var seed = int64(10) // instance where FFD is not optimal

func main() {
	// Fixed seed: the same instance on every run.
	rng := rand.New(rand.NewSource(seed))
	const n = 30   // number of items
	const C = 100. // bin capacity

	// w[i]: size of item i, sorted by decreasing size (needed by the
	// symmetry breaking "item i only in bins 0..i" and by FFD).
	w := make([]float64, n)
	for i := range w {
		w[i] = float64(10 + rng.Intn(45))
	}
	sort.Sort(sort.Reverse(sort.Float64Slice(w)))
	total := 0.0
	for _, v := range w {
		total += v
	}

	// --- First Fit Decreasing ---------------------------------------------------
	// Items in decreasing order, each into the first bin where it fits.
	var load []float64 // load[b]: content of bin b
	for _, v := range w {
		placed := false
		for b := range load {
			if load[b]+v <= C {
				load[b] += v
				placed = true
				break
			}
		}
		if !placed {
			load = append(load, v)
		}
	}
	U := len(load)                  // number of bins of the model
	L1 := int(math.Ceil(total / C)) // continuous lower bound
	fmt.Printf("%d items, total size %.0f, capacity %.0f: lower bound L1 = %d, FFD = %d bins\n", n, total, C, L1, U)

	// --- Model -----------------------------------------------------------------
	m := gost.NewModel("bin-packing")
	// y[b]: 1 if bin b is used (cost 1 per bin).
	y := make([]gost.Var, U)
	for b := range y {
		y[b] = m.AddVar(0, 1, 1, gost.Binary, fmt.Sprintf("y%d", b))
	}
	// x[i][b]: 1 if item i goes into bin b. Only bins b ≤ i exist for item i
	// (symmetry breaking), the others are simply not created.
	x := make([][]gost.Var, n)
	for i := range x {
		x[i] = make([]gost.Var, min(i+1, U))
		for b := range x[i] {
			x[i][b] = m.AddVar(0, 1, 0, gost.Binary, fmt.Sprintf("x%d_%d", i, b))
		}
	}
	// assign_i: each item goes into exactly one bin.
	for i := range x {
		e := gost.NewExpr()
		for b := range x[i] {
			e.Add(1, x[i][b])
		}
		m.AddConstr(e, gost.Equal, 1, fmt.Sprintf("assign%d", i))
	}
	for b := 0; b < U; b++ {
		// capacity_b: what bin b contains fits, and nothing goes into an
		// unused bin.
		e := gost.NewExpr().Add(-C, y[b])
		for i := range x {
			if b < len(x[i]) {
				e.Add(w[i], x[i][b])
				// link_i_b: x_ib ≤ y_b (strengthens the LP).
				m.AddConstr(gost.NewExpr().Add(1, x[i][b]).Add(-1, y[b]), gost.LessEq, 0, fmt.Sprintf("link%d_%d", i, b))
			}
		}
		m.AddConstr(e, gost.LessEq, 0, fmt.Sprintf("capacity%d", b))
		// order_b: bins are used in order (symmetry breaking).
		if b+1 < U {
			m.AddConstr(gost.NewExpr().Add(1, y[b]).Add(-1, y[b+1]), gost.GreaterEq, 0, fmt.Sprintf("order%d", b))
		}
	}
	// The objective Σ y_b comes from the costs given in AddVar (default
	// sense: minimise).

	// --- Solve -----------------------------------------------------------------
	res, err := m.Solve(gost.DefaultParams())
	if err != nil {
		panic(err)
	}
	fmt.Printf("optimal packing: %s, %.0f bins (%d nodes, %v)\n\n", res.Status, res.Objective, res.Nodes, res.Time)

	// --- Independent check and printout -----------------------------------------
	content := make([][]int, U) // content[b]: items in bin b
	for i := range x {
		cnt := 0
		for b := range x[i] {
			if res.Value(x[i][b]) > 0.5 {
				content[b] = append(content[b], i)
				cnt++
			}
		}
		if cnt != 1 {
			fmt.Printf("ERROR: item %d is in %d bins\n", i, cnt)
		}
	}
	used := 0
	for b, items := range content {
		if len(items) == 0 {
			continue
		}
		used++
		sum := 0.0
		fmt.Printf("bin %2d:", b)
		for _, i := range items {
			sum += w[i]
			fmt.Printf(" %2.0f", w[i])
		}
		status := ""
		if sum > C {
			status = "  ERROR: over capacity"
		}
		fmt.Printf("   (load %3.0f/%.0f)%s\n", sum, C, status)
	}
	fmt.Printf("\nindependent check: %d bins used\n", used)
}
