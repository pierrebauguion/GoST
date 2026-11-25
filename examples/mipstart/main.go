// MIP start: giving the solver a feasible (not necessarily optimal)
// solution to begin with, on a capacitated facility location problem.
//
// Facilities can be opened at a fixed cost; each customer's demand must be
// served by open facilities without exceeding their capacity; serving a
// customer from a facility costs its demand times the distance.
//
// Model (F = facilities, C = customers):
//
//	min  Σ_i f_i · y_i + Σ_i Σ_j c_ij · x_ij
//	s.t. Σ_i x_ij = 1                    for each customer j   (demand)
//	     Σ_j d_j · x_ij ≤ u_i · y_i       for each facility i   (capacity)
//	     x_ij ≤ y_i                       for each i, j         (linking)
//	     y_i ∈ {0, 1}, x_ij ∈ [0, 1]
//
// where y_i = 1 if facility i is open and x_ij is the fraction of customer
// j's demand served by facility i.
//
// The program solves the model four times:
//
//  1. without a start;
//  2. with a complete start built by a greedy heuristic (accepted as the
//     first incumbent);
//  3. with a partial start (only the y_i of the greedy solution): the
//     solver completes it (repair);
//  4. with an infeasible start (every facility closed): the solver finds
//     the feasible solution closest to it (repair).
//
// The optimum is the same in every case: a start only helps the search.
// On this small instance the solver finds the optimum quickly by itself,
// so the start mainly illustrates the API; on harder models a good start
// gives early pruning and a reference point for improvement heuristics.
//
//	go run ./examples/mipstart
package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"gost"
)

func main() {
	// Fixed seed: the same instance on every run.
	rng := rand.New(rand.NewSource(11))
	const nF, nC = 25, 80 // numbers of facilities and customers

	// fx, fy / cx, cy: coordinates of facilities and customers.
	fx, fy := make([]float64, nF), make([]float64, nF)
	cx, cy := make([]float64, nC), make([]float64, nC)
	for i := range fx {
		fx[i], fy[i] = rng.Float64()*100, rng.Float64()*100
	}
	for j := range cx {
		cx[j], cy[j] = rng.Float64()*100, rng.Float64()*100
	}
	// d[j]: demand of customer j; total: sum of demands.
	d := make([]float64, nC)
	total := 0.0
	for j := range d {
		d[j] = float64(5 + rng.Intn(30))
		total += d[j]
	}
	// fixed[i]: opening cost of facility i; capa[i]: its capacity, between
	// 3 and 5 times the average demand per facility (total / nF).
	fixed, capa := make([]float64, nF), make([]float64, nF)
	for i := range fixed {
		fixed[i] = float64(300 + rng.Intn(500))
		capa[i] = math.Round(total / nF * (3 + 2*rng.Float64()))
	}
	// cost(i, j): cost of serving all of customer j from facility i.
	cost := func(i, j int) float64 { return d[j] * math.Hypot(fx[i]-cx[j], fy[i]-cy[j]) / 10 }

	// --- Model -----------------------------------------------------------------
	m := gost.NewModel("facility-location")
	// y[i]: 1 if facility i is open (binary, opening cost in the objective).
	y := make([]gost.Var, nF)
	for i := range y {
		y[i] = m.AddVar(0, 1, fixed[i], gost.Binary, fmt.Sprintf("y%d", i))
	}
	// x[i][j]: fraction of customer j served by facility i (continuous).
	x := make([][]gost.Var, nF)
	for i := range x {
		x[i] = make([]gost.Var, nC)
		for j := range x[i] {
			x[i][j] = m.AddVar(0, 1, cost(i, j), gost.Continuous, fmt.Sprintf("x%d_%d", i, j))
		}
	}
	// demand_j: customer j is fully served.
	for j := 0; j < nC; j++ {
		e := gost.NewExpr()
		for i := 0; i < nF; i++ {
			e.Add(1, x[i][j])
		}
		m.AddConstr(e, gost.Equal, 1, fmt.Sprintf("demand%d", j))
	}
	// capacity_i: what facility i serves fits in its capacity, and nothing
	// is served from a closed facility.
	for i := 0; i < nF; i++ {
		e := gost.NewExpr().Add(-capa[i], y[i])
		for j := 0; j < nC; j++ {
			e.Add(d[j], x[i][j])
		}
		m.AddConstr(e, gost.LessEq, 0, fmt.Sprintf("capacity%d", i))
	}
	// link_ij: x_ij ≤ y_i (redundant with the capacity rows for integer
	// solutions, but it makes the LP relaxation much tighter).
	for i := 0; i < nF; i++ {
		for j := 0; j < nC; j++ {
			m.AddConstr(gost.NewExpr().Add(1, x[i][j]).Add(-1, y[i]), gost.LessEq, 0, fmt.Sprintf("link%d_%d", i, j))
		}
	}

	// --- Greedy heuristic: a feasible start --------------------------------------
	// Customers are taken by decreasing demand and each one is assigned to
	// the facility with the lowest cost (opening cost included if it is
	// still closed) that has enough remaining capacity. Customers are not
	// split, so the start is feasible but usually not optimal.
	open := make([]bool, nF)
	left := append([]float64(nil), capa...) // remaining capacities
	assign := make([]int, nC)               // assign[j]: facility of customer j
	order := make([]int, nC)
	for j := range order {
		order[j] = j
	}
	sort.Slice(order, func(a, b int) bool { return d[order[a]] > d[order[b]] })
	greedyCost := 0.0
	for _, j := range order {
		best, bestCost := -1, math.Inf(1)
		for i := 0; i < nF; i++ {
			if left[i] < d[j] {
				continue
			}
			c := cost(i, j)
			if !open[i] {
				c += fixed[i]
			}
			if c < bestCost {
				best, bestCost = i, c
			}
		}
		open[best] = true
		left[best] -= d[j]
		assign[j] = best
		greedyCost += bestCost
	}
	fmt.Printf("greedy heuristic: cost %.2f\n\n", greedyCost)

	// solve runs the solver and prints what happened to the start.
	solve := func(label string) {
		r, err := m.Solve(gost.DefaultParams())
		if err != nil {
			panic(err)
		}
		fmt.Printf("%-24s %s, cost %.2f, %5d nodes, %v\n", label+":", r.Status, r.Objective, r.Nodes, r.Time.Round(1e6))
		if r.StartStatus != gost.StartNone {
			fmt.Printf("%-24s start %s %s\n", "", r.StartStatus, r.StartInfo)
		}
	}

	// 1. No start.
	solve("no start")

	// 2. Complete start: every variable gets the greedy value.
	for i := 0; i < nF; i++ {
		m.SetStart(y[i], b2f(open[i]))
		for j := 0; j < nC; j++ {
			m.SetStart(x[i][j], b2f(assign[j] == i))
		}
	}
	solve("complete greedy start")

	// 3. Partial start: only the open/closed decisions; the solver computes
	// the best assignment for these facilities, then continues.
	m.ClearStart()
	for i := 0; i < nF; i++ {
		m.SetStart(y[i], b2f(open[i]))
	}
	solve("partial start (y only)")

	// 4. Infeasible start: every facility closed. No assignment exists, so
	// the solver looks for the feasible solution with the fewest changes.
	m.ClearStart()
	for i := 0; i < nF; i++ {
		m.SetStart(y[i], 0)
	}
	solve("infeasible start")
}

// b2f converts a boolean to 0/1.
func b2f(b bool) float64 {
	if b {
		return 1
	}
	return 0
}
