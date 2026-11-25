// Steiner TSP on a sparse directed graph, solved with lazy constraints and
// user cuts.
//
// The road network is a directed graph that is not complete (some streets
// are one-way, most pairs of nodes are not adjacent). Only a subset R of
// "required" nodes must be visited; the other nodes may be used to pass
// through, and any node or arc may be traversed several times. We look for
// the shortest closed walk that visits every required node.
//
// Model (A = arcs, δ⁺(S)/δ⁻(S) = arcs leaving/entering the node set S):
//
//	min  Σ_a c_a · x_a                                     (walk length)
//	s.t. Σ_{a ∈ δ⁻(v)} x_a = Σ_{a ∈ δ⁺(v)} x_a   for each node v      (conservation)
//	     Σ_{a ∈ δ⁻(v)} x_a ≥ 1                   for each v ∈ R       (visit)
//	     Σ_{a ∈ δ⁺(S)} x_a ≥ 1     for each S with S ∩ R ≠ ∅, R ⊄ S    (connectivity)
//	     x_a ∈ {0, 1, ..., |R|}
//
// x_a is the number of times arc a is traversed (an integer, not a binary:
// repeated passages are allowed). A multigraph in which every node has as
// many entering as leaving arcs and which is connected is a closed walk
// (Euler), hence the formulation. An optimal walk is made of |R| shortest
// paths, so no arc is used more than |R| times.
//
// The connectivity constraints are exponential in number and generated on
// demand:
//
//   - LazyConstraints: on each integer candidate, the arcs used split into
//     connected pieces; a piece that contains some but not all required
//     nodes yields a violated constraint;
//
//   - UserCuts: on fractional LP solutions, a max-flow computation from a
//     fixed required node r to each other required node t (capacities x_a)
//     finds a cut S of value < 1 when one exists.
//
// The result is cross-checked by an independent method: shortest-path
// distances between required nodes (Floyd–Warshall), then an exact
// asymmetric TSP on the required nodes only (Held–Karp dynamic program).
//
//	go run ./examples/steinertsp
package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"

	"gost"
)

// arc: directed arc from node `from` to node `to`, of length cost.
type arc struct {
	from, to int
	cost     float64
}

// network builds a random sparse directed graph on n nodes placed in a
// 1000 × 1000 square. Each node gets arcs towards its 3 nearest neighbours
// (so many arcs exist in one direction only), plus a directed ring visiting
// the nodes in random order, which makes the graph strongly connected.
// Arc lengths are rounded Euclidean distances. It returns the arcs and the
// node coordinates.
func network(n int, rng *rand.Rand) ([]arc, []float64, []float64) {
	px, py := make([]float64, n), make([]float64, n) // node coordinates
	for i := range px {
		px[i], py[i] = rng.Float64()*1000, rng.Float64()*1000
	}
	dist := func(i, j int) float64 { return math.Round(math.Hypot(px[i]-px[j], py[i]-py[j])) }
	seen := map[[2]int]bool{} // arcs already created (no duplicates)
	var arcs []arc
	add := func(i, j int) {
		if i != j && !seen[[2]int{i, j}] {
			seen[[2]int{i, j}] = true
			arcs = append(arcs, arc{i, j, dist(i, j)})
		}
	}
	for i := 0; i < n; i++ {
		// order: other nodes sorted by distance from i.
		order := make([]int, 0, n-1)
		for j := 0; j < n; j++ {
			if j != i {
				order = append(order, j)
			}
		}
		sort.Slice(order, func(a, b int) bool { return dist(i, order[a]) < dist(i, order[b]) })
		for _, j := range order[:3] {
			add(i, j)
		}
	}
	perm := rng.Perm(n) // ring order
	for k := range perm {
		add(perm[k], perm[(k+1)%n])
	}
	return arcs, px, py
}

// components returns the connected pieces (ignoring arc directions) of the
// graph made of the arcs a with val(a) > thr, as lists of nodes. Nodes
// touched by no such arc form singleton pieces.
func components(n int, arcs []arc, val func(a int) float64, thr float64) [][]int {
	parent := make([]int, n) // union-find forest
	for i := range parent {
		parent[i] = i
	}
	// find returns the representative of v's piece (path halving).
	find := func(v int) int {
		for parent[v] != v {
			parent[v] = parent[parent[v]]
			v = parent[v]
		}
		return v
	}
	for a, e := range arcs {
		if val(a) > thr {
			parent[find(e.from)] = find(e.to)
		}
	}
	groups := map[int][]int{} // nodes by representative
	for v := 0; v < n; v++ {
		groups[find(v)] = append(groups[find(v)], v)
	}
	var out [][]int
	for _, g := range groups {
		out = append(out, g)
	}
	return out
}

// minCut computes a maximum flow from s to t with arc capacities capa
// (Edmonds–Karp, stopped as soon as the flow reaches limit) and returns the
// flow value and the source side S of a minimum cut (nodes reachable from s
// in the residual graph). If the flow reaches limit, S is meaningless.
func minCut(n int, arcs []arc, capa []float64, s, t int, limit float64) (float64, []bool) {
	// Residual graph: arc 2a is arc a (capacity capa[a]), arc 2a+1 its
	// reverse (capacity 0). res[e] is the residual capacity of e.
	res := make([]float64, 2*len(arcs))
	head := make([]int, 2*len(arcs)) // head[e]: end node of residual arc e
	out := make([][]int, n)          // out[v]: residual arcs leaving v
	for a, e := range arcs {
		res[2*a], head[2*a] = capa[a], e.to
		res[2*a+1], head[2*a+1] = 0, e.from
		out[e.from] = append(out[e.from], 2*a)
		out[e.to] = append(out[e.to], 2*a+1)
	}
	flow := 0.0
	for {
		// Breadth-first search for a shortest augmenting path.
		pred := make([]int, n) // pred[v]: residual arc used to reach v
		for v := range pred {
			pred[v] = -1
		}
		reached := make([]bool, n)
		reached[s] = true
		queue := []int{s}
		for len(queue) > 0 && !reached[t] {
			u := queue[0]
			queue = queue[1:]
			for _, e := range out[u] {
				if v := head[e]; !reached[v] && res[e] > 1e-9 {
					reached[v], pred[v] = true, e
					queue = append(queue, v)
				}
			}
		}
		if !reached[t] || flow >= limit {
			return flow, reached // reached = source side of a minimum cut
		}
		// Bottleneck of the path, then push it.
		push := math.Inf(1)
		for v := t; v != s; v = head[pred[v]^1] {
			push = math.Min(push, res[pred[v]])
		}
		for v := t; v != s; v = head[pred[v]^1] {
			res[pred[v]] -= push
			res[pred[v]^1] += push
		}
		flow += push
	}
}

// eulerWalk turns the arc multiplicities count into a closed walk starting
// at node start (Hierholzer's algorithm) and returns the node sequence.
func eulerWalk(n int, arcs []arc, count []int, start int) []int {
	out := make([][]int, n) // out[v]: unused copies of arcs leaving v
	for a, e := range arcs {
		for c := 0; c < count[a]; c++ {
			out[e.from] = append(out[e.from], a)
		}
	}
	stack := []int{start}
	var walk []int
	for len(stack) > 0 {
		v := stack[len(stack)-1]
		if len(out[v]) > 0 {
			a := out[v][len(out[v])-1]
			out[v] = out[v][:len(out[v])-1]
			stack = append(stack, arcs[a].to)
		} else {
			walk = append(walk, v)
			stack = stack[:len(stack)-1]
		}
	}
	// Nodes are popped in reverse order of the walk.
	for i, j := 0, len(walk)-1; i < j; i, j = i+1, j-1 {
		walk[i], walk[j] = walk[j], walk[i]
	}
	return walk
}

// heldKarp is the cross-check: it computes all shortest-path distances
// (Floyd–Warshall), then the shortest closed tour through the required
// nodes by dynamic programming over subsets (Held–Karp), and returns its
// length.
func heldKarp(n int, arcs []arc, req []int) float64 {
	// d[i][j]: shortest path length from i to j in the network.
	d := make([][]float64, n)
	for i := range d {
		d[i] = make([]float64, n)
		for j := range d[i] {
			if i != j {
				d[i][j] = math.Inf(1)
			}
		}
	}
	for _, e := range arcs {
		d[e.from][e.to] = math.Min(d[e.from][e.to], e.cost)
	}
	for k := 0; k < n; k++ {
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				d[i][j] = math.Min(d[i][j], d[i][k]+d[k][j])
			}
		}
	}
	// f[S][j]: shortest path that starts at req[0], visits the required
	// nodes req[1..] in subset S (bit i-1 for req[i]) and ends at req[j].
	r := len(req)
	full := 1 << (r - 1)
	f := make([][]float64, full)
	for S := range f {
		f[S] = make([]float64, r)
		for j := range f[S] {
			f[S][j] = math.Inf(1)
		}
	}
	for j := 1; j < r; j++ {
		f[1<<(j-1)][j] = d[req[0]][req[j]]
	}
	for S := 1; S < full; S++ {
		for j := 1; j < r; j++ {
			if S&(1<<(j-1)) == 0 || math.IsInf(f[S][j], 1) {
				continue
			}
			for k := 1; k < r; k++ {
				if S&(1<<(k-1)) == 0 {
					T := S | 1<<(k-1)
					f[T][k] = math.Min(f[T][k], f[S][j]+d[req[j]][req[k]])
				}
			}
		}
	}
	best := math.Inf(1)
	for j := 1; j < r; j++ {
		best = math.Min(best, f[full-1][j]+d[req[j]][req[0]])
	}
	return best
}

func main() {
	// Fixed seed: the same instance on every run.
	rng := rand.New(rand.NewSource(3))

	// n: number of nodes; arcs: the sparse directed network.
	n := 80
	arcs, _, _ := network(n, rng)

	// req: the required nodes (16 distinct nodes drawn at random);
	// isReq[v] tells whether node v is required.
	req := rng.Perm(n)[:16]
	sort.Ints(req)
	isReq := make([]bool, n)
	for _, v := range req {
		isReq[v] = true
	}

	m := gost.NewModel("steiner-tsp")

	// --- Variables -----------------------------------------------------------
	// x[a]: number of traversals of arc a, integer in [0, |R|], cost = its
	// length.
	x := make([]gost.Var, len(arcs))
	for a, e := range arcs {
		x[a] = m.AddVar(0, float64(len(req)), e.cost, gost.Integer, fmt.Sprintf("x_%d_%d", e.from, e.to))
	}

	// --- Constraints ---------------------------------------------------------
	// in[v], out[v]: arcs entering / leaving node v.
	in, out := make([][]int, n), make([][]int, n)
	for a, e := range arcs {
		out[e.from] = append(out[e.from], a)
		in[e.to] = append(in[e.to], a)
	}
	for v := 0; v < n; v++ {
		// conservation_v: the walk leaves v as many times as it enters it.
		e := gost.NewExpr()
		for _, a := range in[v] {
			e.Add(1, x[a])
		}
		for _, a := range out[v] {
			e.Add(-1, x[a])
		}
		m.AddConstr(e, gost.Equal, 0, fmt.Sprintf("conservation_%d", v))
		// visit_v: a required node is entered at least once.
		if isReq[v] {
			e := gost.NewExpr()
			for _, a := range in[v] {
				e.Add(1, x[a])
			}
			m.AddConstr(e, gost.GreaterEq, 1, fmt.Sprintf("visit_%d", v))
		}
	}

	// cutExpr returns Σ_{a ∈ δ⁺(S)} x_a for the node set S (inS[v] = v ∈ S).
	cutExpr := func(inS []bool) *gost.Expr {
		e := gost.NewExpr()
		for a, ar := range arcs {
			if inS[ar.from] && !inS[ar.to] {
				e.Add(1, x[a])
			}
		}
		return e
	}

	// --- Connectivity constraints generated on demand ------------------------
	lazyAdded, cutsAdded := 0, 0 // counters for the summary
	p := gost.DefaultParams()
	p.Log = os.Stdout

	// LazyConstraints: on an integer candidate, each connected piece of the
	// arcs used (x ≥ 1) that holds some but not all required nodes is a set
	// S that the walk never leaves: add Σ_{a ∈ δ⁺(S)} x_a ≥ 1.
	p.LazyConstraints = func(ctx *gost.ConstrContext) {
		pieces := components(n, arcs, func(a int) float64 { return ctx.Value(x[a]) }, 0.5)
		for _, piece := range pieces {
			inS := make([]bool, n)
			nreq := 0 // required nodes in this piece
			for _, v := range piece {
				inS[v] = true
				if isReq[v] {
					nreq++
				}
			}
			if nreq > 0 && nreq < len(req) {
				ctx.AddConstr(cutExpr(inS), gost.GreaterEq, 1, "connect")
				lazyAdded++
			}
		}
	}

	// UserCuts: on a fractional LP solution, for each required node t ≠ r
	// (r = req[0]), a maximum flow from r to t below 1 gives a set S (r ∈ S,
	// t ∉ S) whose leaving arcs carry less than 1: that inequality is
	// violated and added as a cut.
	p.UserCuts = func(ctx *gost.ConstrContext) {
		capa := make([]float64, len(arcs)) // capa[a]: LP value of x_a
		for a := range arcs {
			capa[a] = math.Max(ctx.Value(x[a]), 0)
		}
		for _, t := range req[1:] {
			flow, inS := minCut(n, arcs, capa, req[0], t, 1)
			if flow < 1-1e-6 {
				ctx.AddConstr(cutExpr(inS), gost.GreaterEq, 1, "connect-cut")
				cutsAdded++
			}
		}
	}

	// --- Solve ---------------------------------------------------------------
	res, err := m.Solve(p)
	if err != nil {
		panic(err)
	}
	fmt.Printf("\n%d nodes, %d arcs, %d required: %s, walk length %.0f (%d lazy constraints, %d user cuts, %d nodes, %v)\n",
		n, len(arcs), len(req), res.Status, res.Objective, lazyAdded, cutsAdded, res.Nodes, res.Time)

	// --- Print the walk ------------------------------------------------------
	// count[a]: number of traversals of arc a in the optimal solution.
	count := make([]int, len(arcs))
	for a := range arcs {
		count[a] = int(math.Round(res.Value(x[a])))
	}
	walk := eulerWalk(n, arcs, count, req[0])
	// visits[v]: number of times v appears in the walk (start counted once).
	visits := make([]int, n)
	for _, v := range walk[1:] {
		visits[v]++
	}
	fmt.Print("walk:")
	for _, v := range walk {
		if isReq[v] {
			fmt.Printf(" [%d]", v) // required node
		} else {
			fmt.Printf(" %d", v) // node used only to pass through
		}
	}
	fmt.Println()
	fmt.Println("required nodes are in brackets; nodes visited more than once:")
	for v := 0; v < n; v++ {
		if visits[v] > 1 {
			fmt.Printf("  node %d: %d times\n", v, visits[v])
		}
	}

	// --- Cross-check ---------------------------------------------------------
	ref := heldKarp(n, arcs, req)
	fmt.Printf("shortest paths + Held–Karp: %.0f (difference %.0f)\n", ref, math.Abs(ref-res.Objective))
}
