// Cutting stock by column generation (Gilmore–Gomory).
//
// Raw rolls of width 100 must be cut to meet a demand for strips of several
// widths. A "pattern" says how many strips of each width are cut from one
// roll. There are far too many possible patterns to write them all, so they
// are generated on the fly.
//
// Master problem:
//
//	min  Σ_p z_p                                   (number of rolls)
//	s.t. Σ_p a_ip · z_p ≥ demand_i   for each width i
//	     z_p ≥ 0 (continuous in step 1, integer in step 2)
//
// where z_p is the number of rolls cut with pattern p and a_ip the number
// of strips of width i in that pattern.
//
// The master starts with one trivial pattern per width. The Pricing
// callback solves a knapsack on the dual values and adds the best pattern
// while its reduced cost is negative.
//
//	go run ./examples/cuttingstock
//
// Step 1 solves the LP relaxation by column generation (pure LP).
// Step 2 solves the integer problem by price-and-branch (pricing at the
// root only, then branch-and-bound on the generated patterns).
// Set priceDepth to a large value for full branch-and-price.
package main

import (
	"fmt"

	"gost"
)

// roll: width of a raw roll.
const roll = 100

var (
	// widths: widths of the requested strips.
	widths = []int{45, 36, 31, 14}
	// demand: number of strips requested for each width (same index as
	// widths).
	demand = []float64{97, 610, 395, 211}
)

// bestPattern solves the pricing subproblem
//
//	max Σ_i y_i · a_i   s.t.  Σ_i w_i · a_i ≤ roll,  a_i ∈ ℕ
//
// (unbounded integer knapsack, by dynamic programming). y holds the dual
// values of the demand constraints; the result is the pattern a (number of
// strips of each width) with the largest dual value.
func bestPattern(y []float64) []int {
	// best[c]: best dual value achievable with capacity c.
	best := make([]float64, roll+1)
	// item[c]: width added last to reach best[c] (-1 if capacity c gives
	// nothing more than c-1).
	item := make([]int, roll+1)
	for c := 1; c <= roll; c++ {
		best[c], item[c] = best[c-1], -1
		for i, w := range widths {
			if w <= c && best[c-w]+y[i] > best[c]+1e-9 {
				best[c], item[c] = best[c-w]+y[i], i
			}
		}
	}
	// Rebuild the pattern by walking back the choices from roll.
	pat := make([]int, len(widths)) // pat[i]: number of strips of width i
	for c := roll; c > 0; {
		if i := item[c]; i >= 0 {
			pat[i]++
			c -= widths[i]
		} else {
			c--
		}
	}
	return pat
}

// solve builds and solves the master problem.
// typ: type of the pattern variables (Continuous for the LP relaxation,
// Integer for the integer solution).
// priceDepth: maximum tree depth at which pricing is called
// (0 = root only).
func solve(typ gost.VarType, priceDepth int) {
	m := gost.NewModel("cutting-stock")

	// --- Demand constraints --------------------------------------------------
	// dem[i]: constraint "strips of width i produced ≥ demand[i]".
	// They are created empty (no variable): the columns added afterwards
	// insert their coefficients into them.
	dem := make([]gost.Constr, len(widths))
	for i := range widths {
		dem[i] = m.AddConstr(gost.NewExpr(), gost.GreaterEq, demand[i], fmt.Sprintf("width%d", widths[i]))
	}

	// patterns[j]: pattern of the variable with index j (for printing).
	var patterns [][]int

	// --- Initial variables: trivial patterns ---------------------------------
	// One pattern per width, containing only that width as many times as
	// it fits. Each variable z is the number of rolls cut this way; cost 1
	// (one roll). AddVarWithColumn creates the variable together with its
	// coefficients in the constraints.
	for i, w := range widths {
		pat := make([]int, len(widths))
		pat[i] = roll / w
		patterns = append(patterns, pat)
		m.AddVarWithColumn(0, gost.Inf, 1, typ, "trivial", gost.NewColumn().Add(float64(pat[i]), dem[i]))
	}

	// --- Column generation ---------------------------------------------------
	p := gost.DefaultParams()
	// Pricing is called at every LP optimum of the master (root, then
	// nodes). If it adds variables, the LP is solved again; it stops when
	// no variable is added any more.
	p.Pricing = func(ctx *gost.PricingContext) {
		// No pricing below priceDepth: branching then only uses the
		// patterns generated so far.
		if ctx.Depth() > priceDepth {
			return
		}
		// y[i]: dual value of demand constraint i.
		y := make([]float64, len(dem))
		for i, c := range dem {
			y[i] = ctx.Dual(c)
		}
		// pat: pattern with the largest dual value.
		pat := bestPattern(y)
		// col: coefficients of the new pattern in the demand constraints
		// (a_i strips of width i).
		col := gost.NewColumn()
		for i, a := range pat {
			if a > 0 {
				col.Add(float64(a), dem[i])
			}
		}
		// Reduced cost = 1 (one roll) − Σ a_i · y_i. If negative, the
		// pattern improves the LP: add it as a new variable.
		if ctx.ReducedCost(1, col) < -1e-9 {
			patterns = append(patterns, pat)
			ctx.AddVar(0, gost.Inf, 1, typ, "pattern", col)
		}
	}

	// --- Solve and print -----------------------------------------------------
	res, err := m.Solve(p)
	if err != nil {
		panic(err)
	}
	// Generated patterns = total variables − trivial patterns.
	fmt.Printf("%s: %.2f rolls, %d patterns generated\n", res.Status, res.Objective, m.NumVars()-len(widths))
	// Patterns used: the variable with index j corresponds to patterns[j].
	for j, pat := range patterns {
		if v := res.Value(m.VarByIndex(j)); v > 1e-6 {
			fmt.Printf("  %6.2f x pattern", v)
			for i, a := range pat {
				if a > 0 {
					fmt.Printf("  %d x %d", a, widths[i])
				}
			}
			fmt.Println()
		}
	}
	// Dual values are only defined for a continuous problem (LP).
	if typ == gost.Continuous {
		for i, c := range dem {
			fmt.Printf("  dual of width %d: %.4f\n", widths[i], res.Dual(c))
		}
	}
}

func main() {
	fmt.Println("LP relaxation by column generation:")
	solve(gost.Continuous, 0)
	fmt.Println("\nInteger solution (price-and-branch):")
	solve(gost.Integer, 0)
}
