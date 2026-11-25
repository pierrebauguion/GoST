// Travelling salesman with lazy subtour elimination constraints.
//
// Model (complete undirected graph on n cities):
//
//	min  Σ_e d_e · x_e                           (tour length)
//	s.t. Σ_{e ∈ δ(v)} x_e = 2   for each city v      (degree)
//	     Σ_{e ∈ δ(S)} x_e ≥ 2   for each subset S    (subtours)
//	     x_e ∈ {0, 1}
//
// where x_e = 1 if edge e is in the tour and δ(S) is the set of edges with
// exactly one end in S.
//
// The initial model only contains the degree constraints. Subtour
// elimination constraints are exponential in number, so they are generated
// on demand:
//
//   - LazyConstraints checks every candidate integer solution and adds
//     Σ_{e ∈ δ(S)} x_e ≥ 2 for each subtour S;
//
//   - UserCuts does the same on fractional LP solutions (connected
//     components of the support graph) to strengthen the relaxation.
//
//     go run ./examples/tsp          # 60 random cities
//     go run ./examples/tsp 100
package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"

	"gost"
)

// edge: undirected edge between cities i and j (i < j).
type edge struct{ i, j int }

func main() {
	// n: number of cities (1st argument, 60 by default).
	n := 60
	if len(os.Args) > 1 {
		if v, err := strconv.Atoi(os.Args[1]); err == nil {
			n = v
		}
	}
	// px, py: city coordinates, drawn in a 1000 × 1000 square (fixed
	// seed: same instance on every run).
	rng := rand.New(rand.NewSource(1))
	px, py := make([]float64, n), make([]float64, n)
	for i := range px {
		px[i], py[i] = rng.Float64()*1000, rng.Float64()*1000
	}
	// dist: rounded Euclidean distance between cities i and j.
	dist := func(i, j int) float64 { return math.Round(math.Hypot(px[i]-px[j], py[i]-py[j])) }

	m := gost.NewModel("tsp")

	// --- Variables -----------------------------------------------------------
	// edges[k]: k-th edge of the complete graph.
	// x[k]: binary variable of edge edges[k] (1 if it is in the tour), with
	// a cost equal to its length.
	var edges []edge
	var x []gost.Var
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			edges = append(edges, edge{i, j})
			x = append(x, m.AddVar(0, 1, dist(i, j), gost.Binary, fmt.Sprintf("x_%d_%d", i, j)))
		}
	}

	// --- Degree constraints --------------------------------------------------
	// degree_v: exactly two tour edges touch city v (the tour enters by one
	// edge and leaves by another).
	for v := 0; v < n; v++ {
		e := gost.NewExpr()
		for k, ed := range edges {
			if ed.i == v || ed.j == v {
				e.Add(1, x[k])
			}
		}
		m.AddConstr(e, gost.Equal, 2, fmt.Sprintf("degree_%d", v))
	}

	// --- Helper functions ----------------------------------------------------

	// components returns the connected components of the graph made of the
	// edges k with val(k) > thr (union-find). More than one component means
	// the solution contains subtours.
	components := func(val func(k int) float64, thr float64) [][]int {
		// parent[v]: parent of v in the union-find forest.
		parent := make([]int, n)
		for i := range parent {
			parent[i] = i
		}
		// find returns the representative of a's component (with path
		// compression).
		var find func(int) int
		find = func(a int) int {
			for parent[a] != a { //if parent of a is not himself
				parent[a] = parent[parent[a]] //then  it will be the parent of parent of a (direct propagation)
				a = parent[a]                 //shift to parent of a (indirect propagation)
			}
			return a
		}
		// Merge the two ends of every selected edge.
		for k, e := range edges {
			if val(k) > thr {
				parent[find(e.i)] = find(e.j)
			}
		}
		// groups: cities grouped by component representative.
		groups := map[int][]int{}
		for v := 0; v < n; v++ {
			groups[find(v)] = append(groups[find(v)], v)
		}
		var out [][]int
		for _, g := range groups {
			out = append(out, g)
		}
		return out
	}

	// subtour returns the expression Σ_{e ∈ δ(S)} x_e for the city set
	// set = S: sum of the edges with exactly one end in S.
	subtour := func(set []int) *gost.Expr {
		// in[v]: true if city v belongs to S.
		in := make([]bool, n)
		for _, v := range set {
			in[v] = true
		}
		e := gost.NewExpr()
		for k, ed := range edges {
			if in[ed.i] != in[ed.j] {
				e.Add(1, x[k])
			}
		}
		return e
	}

	// --- Subtour constraints generated on demand -----------------------------
	// lazyAdded, cutsAdded: counters for the final summary.
	lazyAdded, cutsAdded := 0, 0
	p := gost.DefaultParams()
	p.Log = os.Stdout // branch-and-cut log

	// LazyConstraints: called on every candidate integer solution. If the
	// selected edges (x > 0.5) form several cycles, add
	// Σ_{e ∈ δ(S)} x_e ≥ 2 for each cycle S; the candidate is rejected.
	p.LazyConstraints = func(ctx *gost.ConstrContext) {
		comps := components(func(k int) float64 { return ctx.Value(x[k]) }, 0.5)
		if len(comps) > 1 {
			for _, c := range comps {
				ctx.AddConstr(subtour(c), gost.GreaterEq, 2, "subtour")
				lazyAdded++
			}
		}
	}

	// UserCuts: called on fractional LP solutions. If the graph of edges
	// with nonzero value is disconnected, each component S yields a
	// violated cut Σ_{e ∈ δ(S)} x_e ≥ 2 (its left-hand side is 0), valid for
	// every tour. These cuts tighten the lower bound.
	p.UserCuts = func(ctx *gost.ConstrContext) {
		comps := components(func(k int) float64 { return ctx.Value(x[k]) }, 1e-6)
		if len(comps) > 1 {
			for _, c := range comps {
				ctx.AddConstr(subtour(c), gost.GreaterEq, 2, "subtour-cut")
				cutsAdded++
			}
		}
	}

	// --- Solve ---------------------------------------------------------------
	res, err := m.Solve(p)
	if err != nil {
		panic(err)
	}
	fmt.Printf("\n%d cities: %s, tour length %.0f (%d lazy constraints, %d user cuts, %d nodes, %v)\n",
		n, res.Status, res.Objective, lazyAdded, cutsAdded, res.Nodes, res.Time)

	// --- Print the tour ------------------------------------------------------
	// next[v]: the two neighbours of city v in the tour.
	next := make([][]int, n)
	for k, e := range edges {
		if res.Value(x[k]) > 0.5 {
			next[e.i] = append(next[e.i], e.j)
			next[e.j] = append(next[e.j], e.i)
		}
	}
	// Walk from city 0: at each city, take the neighbour we did not come
	// from (prev).
	tour := []int{0}
	prev, cur := -1, 0
	for len(tour) < n {
		nx := next[cur][0]
		if nx == prev {
			nx = next[cur][1]
		}
		tour = append(tour, nx)
		prev, cur = cur, nx
	}
	fmt.Println("tour:", tour)
}
