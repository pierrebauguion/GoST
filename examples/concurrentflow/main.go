// Maximum concurrent flow by column generation (path formulation).
//
// A directed network has arc capacities and a set of commodities, each with
// a source, a sink and a demand. We look for the largest fraction λ such
// that λ times every demand can be routed simultaneously without exceeding
// any capacity.
//
// Path formulation (P_k = set of source-sink paths of commodity k):
//
//	max  λ
//	s.t. Σ_{p ∈ P_k} f_p − d_k · λ ≥ 0     for each commodity k   (demand)
//	     Σ_k Σ_{p ∈ P_k : a ∈ p} f_p ≤ c_a   for each arc a         (capacity)
//	     f_p ≥ 0, λ ≥ 0
//
// where f_p is the flow sent along path p. The number of paths is
// exponential, so they are generated on demand: with duals σ_k ≤ 0 of the
// demand rows and μ_a ≥ 0 of the capacity rows, the reduced cost of a path
// p of commodity k is
//
//	rc(p) = 0 − σ_k − Σ_{a ∈ p} μ_a,
//
// and p improves the LP (maximisation) iff rc(p) > 0, i.e. iff its length
// with arc weights μ_a is below −σ_k. The pricing problem is therefore a
// shortest path problem per commodity (Dijkstra, since μ_a ≥ 0).
//
// At the end, the result is cross-checked against the arc formulation
// (one flow variable per commodity and arc, flow conservation at nodes),
// which is polynomial in size and solved directly.
//
//	go run ./examples/concurrentflow
package main

import (
	"fmt"
	"math"
	"math/rand"

	"gost"
)

// arc: directed arc from node `from` to node `to` with capacity cap.
type arc struct {
	from, to int
	cap      float64
}

// commodity: demand d to be routed from node src to node dst.
type commodity struct {
	src, dst int
	d        float64
}

// grid builds a side × side grid network: one node per cell, and one arc in
// each direction between horizontally or vertically adjacent cells, with a
// random integer capacity in [5, 20]. It returns the number of nodes and
// the arcs.
func grid(side int, rng *rand.Rand) (int, []arc) {
	var arcs []arc
	id := func(r, c int) int { return r*side + c } // node index of cell (r, c)
	for r := 0; r < side; r++ {
		for c := 0; c < side; c++ {
			if c+1 < side { // horizontal neighbour
				arcs = append(arcs,
					arc{id(r, c), id(r, c+1), float64(5 + rng.Intn(16))},
					arc{id(r, c+1), id(r, c), float64(5 + rng.Intn(16))})
			}
			if r+1 < side { // vertical neighbour
				arcs = append(arcs,
					arc{id(r, c), id(r+1, c), float64(5 + rng.Intn(16))},
					arc{id(r+1, c), id(r, c), float64(5 + rng.Intn(16))})
			}
		}
	}
	return side * side, arcs
}

// shortestPath returns the arcs (indices into arcs) of a shortest path from
// src to dst for the nonnegative arc weights w, and its length (Dijkstra in
// O(n²), enough for small networks). It returns nil, +Inf if dst cannot be
// reached.
func shortestPath(n int, arcs []arc, w []float64, src, dst int) ([]int, float64) {
	// out[v]: indices of the arcs leaving node v.
	out := make([][]int, n)
	for a, e := range arcs {
		out[e.from] = append(out[e.from], a)
	}
	dist := make([]float64, n) // dist[v]: best known distance from src
	pred := make([]int, n)     // pred[v]: arc used to reach v (-1 if none)
	done := make([]bool, n)    // done[v]: distance of v is final
	for v := range dist {
		dist[v], pred[v] = math.Inf(1), -1
	}
	dist[src] = 0
	for {
		// u: closest node not yet settled.
		u := -1
		for v := 0; v < n; v++ {
			if !done[v] && !math.IsInf(dist[v], 1) && (u < 0 || dist[v] < dist[u]) {
				u = v
			}
		}
		if u < 0 || u == dst {
			break
		}
		done[u] = true
		for _, a := range out[u] {
			if v := arcs[a].to; dist[u]+w[a] < dist[v] {
				dist[v], pred[v] = dist[u]+w[a], a
			}
		}
	}
	if math.IsInf(dist[dst], 1) {
		return nil, math.Inf(1)
	}
	// Walk back the predecessor arcs from dst to src.
	var path []int
	for v := dst; v != src; v = arcs[pred[v]].from {
		path = append([]int{pred[v]}, path...)
	}
	return path, dist[dst]
}

// pathFormulation solves the problem by column generation and returns λ,
// the generated paths (arc lists), the commodity of each path and the flow
// on each path.
func pathFormulation(n int, arcs []arc, coms []commodity) (float64, [][]int, []int, []float64) {
	m := gost.NewModel("concurrent-flow-paths")

	// --- Variables (initial) -------------------------------------------------
	// lambda: fraction of every demand that is routed (the objective).
	lambda := m.AddVar(0, gost.Inf, 1, gost.Continuous, "lambda")
	m.SetObjective(gost.NewExpr().Add(1, lambda), gost.Maximize)

	// --- Constraints ---------------------------------------------------------
	// demand[k]: Σ_{p ∈ P_k} f_p − d_k·λ ≥ 0 (flow of commodity k covers
	// λ times its demand). Path variables add their coefficient later.
	demand := make([]gost.Constr, len(coms))
	for k, c := range coms {
		demand[k] = m.AddConstr(gost.NewExpr().Add(-c.d, lambda), gost.GreaterEq, 0, fmt.Sprintf("demand%d", k))
	}
	// capacity[a]: Σ_{p ∋ a} f_p ≤ c_a (total flow on arc a). Created
	// empty: paths through a insert a coefficient 1.
	capacity := make([]gost.Constr, len(arcs))
	for a, e := range arcs {
		capacity[a] = m.AddConstr(gost.NewExpr(), gost.LessEq, e.cap, fmt.Sprintf("cap%d", a))
	}

	// paths[j], pathCom[j], pathVar[j]: arcs, commodity and variable of the
	// j-th generated path.
	var paths [][]int
	var pathCom []int
	var pathVar []gost.Var

	// column returns the column of a path of commodity k: coefficient 1 in
	// the demand row of k and in the capacity row of each arc of the path.
	column := func(k int, path []int) *gost.Column {
		col := gost.NewColumn().Add(1, demand[k])
		for _, a := range path {
			col.Add(1, capacity[a])
		}
		return col
	}

	// Initial columns: one minimum-hop path per commodity (all weights 1),
	// so that λ > 0 is possible from the first LP.
	ones := make([]float64, len(arcs))
	for a := range ones {
		ones[a] = 1
	}
	for k, c := range coms {
		path, _ := shortestPath(n, arcs, ones, c.src, c.dst)
		paths, pathCom = append(paths, path), append(pathCom, k)
		// f_p: flow on the path, ≥ 0, no cost (only λ is in the objective).
		pathVar = append(pathVar, m.AddVarWithColumn(0, gost.Inf, 0, gost.Continuous, fmt.Sprintf("f%d", len(paths)-1), column(k, path)))
	}

	// --- Column generation ---------------------------------------------------
	p := gost.DefaultParams()
	// Pricing is called at every LP optimum; the LP is solved again as long
	// as it adds paths.
	p.Pricing = func(ctx *gost.PricingContext) {
		// w[a] = μ_a: dual of the capacity row of arc a (≥ 0 for a ≤ row in
		// a maximisation; clamped to absorb tiny negative round-off).
		w := make([]float64, len(arcs))
		for a := range arcs {
			w[a] = math.Max(ctx.Dual(capacity[a]), 0)
		}
		for k, c := range coms {
			// Shortest path for commodity k under the weights μ.
			path, length := shortestPath(n, arcs, w, c.src, c.dst)
			// −σ_k: threshold below which a path has a positive reduced
			// cost (σ_k ≤ 0 is the dual of the demand row).
			if length < -ctx.Dual(demand[k])-1e-9 {
				col := column(k, path)
				// rc = 0 − σ_k − Σ μ_a > 0 (also checked by ReducedCost).
				if ctx.ReducedCost(0, col) > 1e-9 {
					paths, pathCom = append(paths, path), append(pathCom, k)
					pathVar = append(pathVar, ctx.AddVar(0, gost.Inf, 0, gost.Continuous, fmt.Sprintf("f%d", len(paths)-1), col))
				}
			}
		}
	}

	res, err := m.Solve(p)
	if err != nil {
		panic(err)
	}
	// flow[j]: optimal flow on path j.
	flow := make([]float64, len(pathVar))
	for j, v := range pathVar {
		flow[j] = res.Value(v)
	}
	return res.Objective, paths, pathCom, flow
}

// arcFormulation solves the same problem with one variable per commodity
// and arc (x_ka = flow of commodity k on arc a) and flow conservation:
//
//	max  λ
//	s.t. Σ_{a ∈ out(v)} x_ka − Σ_{a ∈ in(v)} x_ka = d_k·λ  (v = src_k),
//	     −d_k·λ (v = dst_k), 0 otherwise                   for each k, v
//	     Σ_k x_ka ≤ c_a                                     for each arc a
//	     x ≥ 0, λ ≥ 0
//
// It is used only to cross-check the column generation result.
func arcFormulation(n int, arcs []arc, coms []commodity) float64 {
	m := gost.NewModel("concurrent-flow-arcs")
	// lambda: routed fraction of every demand.
	lambda := m.AddVar(0, gost.Inf, 1, gost.Continuous, "lambda")
	m.SetObjective(gost.NewExpr().Add(1, lambda), gost.Maximize)
	// x[k][a]: flow of commodity k on arc a.
	x := make([][]gost.Var, len(coms))
	for k := range coms {
		x[k] = make([]gost.Var, len(arcs))
		for a := range arcs {
			x[k][a] = m.AddVar(0, gost.Inf, 0, gost.Continuous, fmt.Sprintf("x_%d_%d", k, a))
		}
	}
	// Flow conservation of each commodity at each node: outflow − inflow
	// − (net supply)·λ = 0.
	for k, c := range coms {
		for v := 0; v < n; v++ {
			e := gost.NewExpr()
			for a, ar := range arcs {
				if ar.from == v {
					e.Add(1, x[k][a])
				}
				if ar.to == v {
					e.Add(-1, x[k][a])
				}
			}
			if v == c.src {
				e.Add(-c.d, lambda)
			} else if v == c.dst {
				e.Add(c.d, lambda)
			}
			m.AddConstr(e, gost.Equal, 0, fmt.Sprintf("flow_%d_%d", k, v))
		}
	}
	// Capacity of each arc, shared by all commodities.
	for a, ar := range arcs {
		e := gost.NewExpr()
		for k := range coms {
			e.Add(1, x[k][a])
		}
		m.AddConstr(e, gost.LessEq, ar.cap, fmt.Sprintf("cap%d", a))
	}
	res, err := m.Solve(gost.DefaultParams())
	if err != nil {
		panic(err)
	}
	return res.Objective
}

func main() {
	// Fixed seed: the same instance on every run.
	rng := rand.New(rand.NewSource(7))

	// n, arcs: 5 × 5 grid network (25 nodes, 80 arcs).
	n, arcs := grid(5, rng)

	// coms: 12 commodities between random distinct nodes, demand in [5, 15].
	var coms []commodity
	for len(coms) < 12 {
		s, t := rng.Intn(n), rng.Intn(n)
		if s != t {
			coms = append(coms, commodity{s, t, float64(5 + rng.Intn(11))})
		}
	}

	lambda, paths, pathCom, flow := pathFormulation(n, arcs, coms)
	fmt.Printf("path formulation: lambda = %.6f, %d paths (%d generated)\n",
		lambda, len(paths), len(paths)-len(coms))

	// Flow decomposition: paths with positive flow, per commodity.
	for k, c := range coms {
		fmt.Printf("commodity %2d: %2d -> %2d, demand %2.0f, routed %.3f\n", k, c.src, c.dst, c.d, lambda*c.d)
		for j, path := range paths {
			if pathCom[j] != k || flow[j] < 1e-7 {
				continue
			}
			// Print the path as its sequence of nodes.
			nodes := []int{arcs[path[0]].from}
			for _, a := range path {
				nodes = append(nodes, arcs[a].to)
			}
			fmt.Printf("    %7.3f on %v\n", flow[j], nodes)
		}
	}

	// Cross-check with the arc formulation.
	ref := arcFormulation(n, arcs, coms)
	fmt.Printf("arc formulation:  lambda = %.6f (difference %.1e)\n", ref, math.Abs(ref-lambda))
}
