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

// printWalk prints a walk, with required nodes in brackets.
func printWalk(label string, walk []int, isReq []bool) {
	fmt.Print(label)
	for _, v := range walk {
		if isReq[v] {
			fmt.Printf(" [%d]", v) // required node
		} else {
			fmt.Printf(" %d", v) // node used only to pass through
		}
	}
	fmt.Println()
}

// heldKarp is the cross-check. It solves the same problem by a completely
// different method and returns the length of the optimal walk and the walk
// itself (sequence of network nodes, starting and ending at req[0]).
//
// Step 1: reduction to a classical TSP on the required nodes only.
//
// Since nodes and arcs may be reused, an optimal walk goes from one required
// node to the next along a shortest path of the network. So once the
// shortest-path distances d[i][j] are known, the problem becomes: in which
// order should the required nodes be visited? That is an ordinary
// asymmetric TSP on the |R| required nodes, with distances d (a complete
// "virtual" graph, even though the network itself is sparse).
//
// Step 2: exact TSP by dynamic programming over subsets (Held–Karp, 1962).
//
// Instead of trying the (|R|−1)! orders, Held–Karp notices that the best
// way to finish a tour only depends on WHICH required nodes have already
// been visited and WHERE we currently stand, not on the order in which they
// were visited. It therefore stores, for every subset S of required nodes
// and every node j of S, the length of the best path that
//
//	starts at a fixed node req[0], visits exactly the nodes of S, ends at j.
//
// The tour is a cycle, so fixing its start at req[0] loses nothing. Cost:
// 2^(|R|−1) subsets × |R| end nodes × |R| extensions, i.e. about 7 million
// steps for |R| = 16 (instead of 15! ≈ 1.3·10¹² orders); memory 2^(|R|−1)·|R|
// numbers. Beyond ~25 required nodes it becomes too large, which is why the
// example itself uses the branch-and-cut formulation.
//
// Step 3: recovering the walk. The dynamic program only computes lengths;
// to rebuild the tour we also record, for every state (S, k), which node j
// it was reached from (parent), then walk these records backwards from the
// best final state. Each leg between two consecutive required nodes is
// finally expanded into the actual network nodes of its shortest path,
// using the "next hop" table filled by Floyd–Warshall.
func heldKarp(n int, arcs []arc, req []int) (float64, []int) {
	// --- Shortest paths between all pairs (Floyd–Warshall) -------------------
	// d[i][j]: length of the shortest path from i to j in the network
	// (+Inf if j cannot be reached from i). Start with the direct arcs.
	// next[i][j]: first node after i on that shortest path (-1 if none),
	// so that the path can be rebuilt hop by hop: i, next[i][j],
	// next[next[i][j]][j], ..., j.
	d := make([][]float64, n)
	next := make([][]int, n)
	for i := range d {
		d[i] = make([]float64, n)
		next[i] = make([]int, n)
		for j := range d[i] {
			next[i][j] = -1
			if i != j {
				d[i][j] = math.Inf(1)
			}
		}
	}
	for _, e := range arcs {
		if e.cost < d[e.from][e.to] {
			d[e.from][e.to] = e.cost
			next[e.from][e.to] = e.to // direct arc: the next hop is the end
		}
	}
	// After iteration k, d[i][j] is the shortest path from i to j whose
	// intermediate nodes are all among 0..k: going through k improves it
	// iff d[i][k] + d[k][j] is shorter. After the last k, all nodes are
	// allowed as intermediates, so d holds the true shortest distances.
	for k := 0; k < n; k++ {
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				if d[i][k]+d[k][j] < d[i][j] {
					d[i][j] = d[i][k] + d[k][j]
					// the path i → j now goes through k: it starts like
					// the path i → k
					next[i][j] = next[i][k]
				}
			}
		}
	}

	// --- Held–Karp dynamic program --------------------------------------------
	// The required nodes are numbered 0..r-1 through req. Node 0 (req[0]) is
	// the start; the other r-1 nodes are encoded in a bit mask S: bit j-1
	// of S is set iff req[j] has been visited. full = 2^(r-1) is the number
	// of such subsets; full-1 (all bits set) means "everybody visited".
	r := len(req)
	full := 1 << (r - 1)

	// f[S][j] (for j in S): length of the shortest path that starts at
	// req[0], visits exactly the required nodes of S (in some order) and
	// ends at req[j]. +Inf means "not computed / impossible".
	// parent[S][j]: the required node visited just before req[j] on the
	// best path of state (S, j) (0 = the start req[0]).
	f := make([][]float64, full)
	parent := make([][]int, full)
	for S := range f {
		f[S] = make([]float64, r)
		parent[S] = make([]int, r)
		for j := range f[S] {
			f[S][j] = math.Inf(1)
		}
	}

	// Base case: subsets with a single node j — go straight from req[0] to
	// req[j].
	for j := 1; j < r; j++ {
		f[1<<(j-1)][j] = d[req[0]][req[j]]
		parent[1<<(j-1)][j] = 0 // reached directly from the start
	}

	// Recurrence ("push" form): from a path described by (S, j), extend it
	// by one not-yet-visited node k. The new path visits T = S ∪ {k}, ends
	// at k and has length f[S][j] + d(req[j], req[k]); keep the best one:
	//
	//	f[S ∪ {k}][k] = min over j ∈ S of  f[S][j] + d[req[j]][req[k]]
	//
	// T is always numerically larger than S (one more bit set), so
	// processing the subsets in increasing order of S guarantees that
	// f[S][·] is final before it is used to build larger subsets.
	for S := 1; S < full; S++ {
		for j := 1; j < r; j++ {
			// j must belong to S, and the state must be reachable.
			if S&(1<<(j-1)) == 0 || math.IsInf(f[S][j], 1) {
				continue
			}
			for k := 1; k < r; k++ {
				if S&(1<<(k-1)) == 0 { // k not visited yet
					T := S | 1<<(k-1)
					if v := f[S][j] + d[req[j]][req[k]]; v < f[T][k] {
						f[T][k] = v
						parent[T][k] = j // remember where (T, k) came from
					}
					//T (compare and write) is strictly greater than S (read)
					//S has to be final at this point, it is ok has T > S
				}
			}
		}
	}

	// Closing the tour: all required nodes visited (S = full-1), ending at
	// some j, then back to the start req[0]. The best such cycle is the
	// optimal tour, i.e. the length of the optimal walk in the network.
	best, last := math.Inf(1), -1 // last: required node visited last
	for j := 1; j < r; j++ {
		if v := f[full-1][j] + d[req[j]][req[0]]; v < best {
			best, last = v, j
		}
	}
	if last < 0 {
		return best, nil // some required node cannot be reached
	}

	// --- Recovering the order of the required nodes ---------------------------
	// Start from the final state (all visited, ending at last) and follow
	// the parent records backwards: each step removes the current node from
	// S and moves to the node it was reached from, until S is empty. The
	// nodes come out in reverse order.
	order := []int{}
	for S, j := full-1, last; S != 0; {
		order = append(order, j)
		pj := parent[S][j]
		S &^= 1 << (j - 1) // remove req[j] from S
		j = pj
	}
	// reverse, and add the start at both ends: 0, ..., 0
	for a, b := 0, len(order)-1; a < b; a, b = a+1, b-1 {
		order[a], order[b] = order[b], order[a]
	}
	order = append(append([]int{0}, order...), 0)

	// --- Expanding each leg into network nodes --------------------------------
	// Between two consecutive required nodes u and v, follow the next-hop
	// table from u until v is reached.
	walk := []int{req[0]}
	for t := 0; t+1 < len(order); t++ {
		u, v := req[order[t]], req[order[t+1]]
		for u != v {
			u = next[u][v]
			walk = append(walk, u)
		}
	}
	return best, walk
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
	printWalk("walk:", walk, isReq)
	fmt.Println("required nodes are in brackets; nodes visited more than once:")
	for v := 0; v < n; v++ {
		if visits[v] > 1 {
			fmt.Printf("  node %d: %d times\n", v, visits[v])
		}
	}

	// --- Cross-check ---------------------------------------------------------
	ref, refWalk := heldKarp(n, arcs, req)
	fmt.Printf("shortest paths + Held–Karp: %.0f (difference %.0f)\n", ref, math.Abs(ref-res.Objective))
	// The Held–Karp walk may differ from the solver's one when several
	// optimal walks exist; its length is recomputed arc by arc as a check.
	printWalk("Held–Karp walk:", refWalk, isReq)
	length := 0.0
	for t := 0; t+1 < len(refWalk); t++ {
		for _, e := range arcs {
			if e.from == refWalk[t] && e.to == refWalk[t+1] {
				length += e.cost
				break
			}
		}
	}
	fmt.Printf("Held–Karp walk length, recomputed from its arcs: %.0f\n", length)
}
