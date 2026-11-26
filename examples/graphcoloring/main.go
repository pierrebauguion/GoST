// Graph coloring: give each node a color so that adjacent nodes get
// different colors, with as few colors as possible (the chromatic number).
// Applications: timetabling (exams sharing students cannot be at the same
// time), register allocation, frequency assignment.
//
// Two classic benchmark families of the DIMACS challenge (1993), generated
// here rather than read from files:
//
//   - queen n×n graphs: one node per square of an n×n chess board, two
//     squares adjacent when a queen on one attacks the other (same row,
//     column or diagonal). queen5_5 has chromatic number 5;
//   - Mycielski graphs: triangle-free graphs whose chromatic number grows
//     by one at each Mycielski construction step (myciel3: 11 nodes, χ = 4;
//     myciel4: 23 nodes, χ = 5). They are notoriously hard for integer
//     programming because the LP relaxation barely proves χ > 2.
//
// Model (assignment formulation), with at most K colors (K = number of
// colors used by the DSatur heuristic):
//
//	min  Σ_c w_c                                       (colors used)
//	s.t. Σ_c x_vc = 1               for each node v     (one color per node)
//	     Σ_{v ∈ K} x_vc ≤ w_c       for each clique K and color c
//	     w_c ≥ w_c+1                for each color c    (symmetry breaking)
//	     x_vc = 0 for c > v                             (symmetry breaking)
//	     x, w ∈ {0, 1}
//
// x_vc = 1 if node v gets color c, w_c = 1 if color c is used. The basic
// version uses one constraint per edge (cliques of size 2). For the queen
// graphs, each row, column and diagonal of the board is a clique, and one
// constraint per clique and color is both smaller and much stronger: it
// lets the LP relaxation see that a row of n squares needs n colors.
//
//	go run ./examples/graphcoloring
//	go run ./examples/graphcoloring hard     # also queen6_6 (about 30 s)
package main

import (
	"fmt"
	"os"
	"strings"
	"time"

	"gost"
)

// graph: number of nodes and adjacency sets, plus the cliques used by the
// model (by default its edges).
type graph struct {
	name    string
	n       int
	adj     []map[int]bool
	cliques [][]int
}

func newGraph(name string, n int) *graph {
	g := &graph{name: name, n: n, adj: make([]map[int]bool, n)}
	for v := range g.adj {
		g.adj[v] = map[int]bool{}
	}
	return g
}

func (g *graph) addEdge(u, v int) {
	if u != v && !g.adj[u][v] {
		g.adj[u][v], g.adj[v][u] = true, true
		g.cliques = append(g.cliques, []int{u, v})
	}
}

func (g *graph) numEdges() int {
	e := 0
	for _, a := range g.adj {
		e += len(a)
	}
	return e / 2
}

// queen builds the queen n×n graph. If byLines, the model's cliques are the
// rows, columns and diagonals of the board instead of the edges.
func queen(n int, byLines bool) *graph {
	g := newGraph(fmt.Sprintf("queen%d_%d", n, n), n*n)
	id := func(r, c int) int { return r*n + c }
	var lines [][]int // rows, columns and diagonals (of length ≥ 2)
	for r := 0; r < n; r++ {
		var row, col []int
		for c := 0; c < n; c++ {
			row, col = append(row, id(r, c)), append(col, id(c, r))
		}
		lines = append(lines, row, col)
	}
	for k := -(n - 1); k <= n-1; k++ {
		var d1, d2 []int // r − c = k and r + c = k + n − 1
		for r := 0; r < n; r++ {
			if c := r - k; c >= 0 && c < n {
				d1 = append(d1, id(r, c))
			}
			if c := k + n - 1 - r; c >= 0 && c < n {
				d2 = append(d2, id(r, c))
			}
		}
		for _, d := range [][]int{d1, d2} {
			if len(d) >= 2 {
				lines = append(lines, d)
			}
		}
	}
	for _, l := range lines {
		for a := range l {
			for b := a + 1; b < len(l); b++ {
				g.addEdge(l[a], l[b])
			}
		}
	}
	if byLines {
		g.cliques = lines
		g.name += " (line cliques)"
	}
	return g
}

// mycielski builds the DIMACS graph mycielk: the Mycielski construction
// applied k−1 times to a single edge K2 (K2 → C5 → myciel3 with 11 nodes →
// myciel4 with 23 nodes ...). From G with nodes v_i, it adds a copy u_i of
// each node, linked to the neighbours of v_i, and a new node w linked to all
// u_i: the result is still triangle-free and needs one more color.
func mycielski(k int) *graph {
	edges := [][2]int{{0, 1}}
	n := 2
	for step := 1; step < k; step++ {
		var ne [][2]int
		for _, e := range edges {
			ne = append(ne, e, [2]int{e[0], n + e[1]}, [2]int{e[1], n + e[0]})
		}
		w := 2 * n
		for i := 0; i < n; i++ {
			ne = append(ne, [2]int{n + i, w})
		}
		edges, n = ne, 2*n+1
	}
	g := newGraph(fmt.Sprintf("myciel%d", k), n)
	for _, e := range edges {
		g.addEdge(e[0], e[1])
	}
	return g
}

// dsatur colors the graph greedily (Brélaz, 1979): repeatedly color the
// uncolored node with the most distinct colors among its neighbours (ties:
// highest degree) with the smallest color it can take.
func dsatur(g *graph) []int {
	color := make([]int, g.n)
	for v := range color {
		color[v] = -1
	}
	for done := 0; done < g.n; done++ {
		best, bestSat, bestDeg := -1, -1, -1
		for v := 0; v < g.n; v++ {
			if color[v] >= 0 {
				continue
			}
			sat := map[int]bool{}
			for u := range g.adj[v] {
				if color[u] >= 0 {
					sat[color[u]] = true
				}
			}
			if len(sat) > bestSat || (len(sat) == bestSat && len(g.adj[v]) > bestDeg) {
				best, bestSat, bestDeg = v, len(sat), len(g.adj[v])
			}
		}
		c := 0
		for {
			free := true
			for u := range g.adj[best] {
				if color[u] == c {
					free = false
				}
			}
			if free {
				break
			}
			c++
		}
		color[best] = c
	}
	return color
}

// solve builds and solves the coloring model and returns the coloring.
func solve(g *graph) []int {
	heur := dsatur(g)
	K := 0 // colors used by DSatur: an upper bound
	for _, c := range heur {
		K = max(K, c+1)
	}
	m := gost.NewModel(g.name)
	// w[c]: color c is used (cost 1).
	w := make([]gost.Var, K)
	for c := range w {
		w[c] = m.AddVar(0, 1, 1, gost.Binary, fmt.Sprintf("w%d", c))
	}
	// x[v][c] for c ≤ v: node v gets color c (symmetry breaking: node v
	// can only take one of the first v+1 colors).
	x := make([][]gost.Var, g.n)
	for v := range x {
		x[v] = make([]gost.Var, min(v+1, K))
		for c := range x[v] {
			x[v][c] = m.AddVar(0, 1, 0, gost.Binary, fmt.Sprintf("x%d_%d", v, c))
		}
	}
	for v := range x {
		// one_v: each node gets exactly one color.
		e := gost.NewExpr()
		for c := range x[v] {
			e.Add(1, x[v][c])
		}
		m.AddConstr(e, gost.Equal, 1, fmt.Sprintf("one%d", v))
	}
	for k, clq := range g.cliques {
		for c := 0; c < K; c++ {
			// clique_k_c: at most one node of the clique has color c, and
			// none if color c is not used.
			e := gost.NewExpr().Add(-1, w[c])
			cnt := 0
			for _, v := range clq {
				if c < len(x[v]) {
					e.Add(1, x[v][c])
					cnt++
				}
			}
			if cnt > 0 {
				m.AddConstr(e, gost.LessEq, 0, fmt.Sprintf("clique%d_%d", k, c))
			}
		}
	}
	for c := 0; c+1 < K; c++ {
		// order_c: colors are used in order (symmetry breaking).
		m.AddConstr(gost.NewExpr().Add(1, w[c]).Add(-1, w[c+1]), gost.GreaterEq, 0, fmt.Sprintf("order%d", c))
	}
	p := gost.DefaultParams()
	p.TimeLimit = 20 * time.Second
	lp := m.SolveRelaxation()
	res, err := m.Solve(p)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%-24s %2d nodes %3d edges %4d rows | DSatur %d | LP bound %.2f | %s: %.0f colors, %d B&B nodes, %v\n",
		g.name, g.n, g.numEdges(), m.NumConstrs(), K, lp.Objective, res.Status, res.Objective, res.Nodes, res.Time.Round(time.Millisecond))

	color := make([]int, g.n)
	for v := range x {
		for c := range x[v] {
			if res.Value(x[v][c]) > 0.5 {
				color[v] = c
			}
		}
	}
	// Independent check: no edge joins two nodes of the same color.
	for u := range g.adj {
		for v := range g.adj[u] {
			if color[u] == color[v] {
				fmt.Fprintf(os.Stderr, "ERROR: nodes %d and %d share color %d\n", u, v, color[u])
			}
		}
	}
	return color
}

func main() {
	solve(mycielski(3))
	solve(mycielski(4))
	solve(queen(5, false))
	color := solve(queen(5, true))
	if len(os.Args) > 1 && os.Args[1] == "hard" {
		// queen6_6: chromatic number 7 but largest clique 6, so the LP
		// bound is 6 and the branch-and-bound must prove that 6 colors are
		// impossible (about 15 to 20 s each).
		solve(queen(6, false))
		solve(queen(6, true))
	}

	// The 5-coloring of the queen graph: no two squares of the same color
	// attack each other, i.e. each color class is a set of 5 non-attacking
	// queens.
	fmt.Println("\nqueen5_5 coloring (each digit is a color):")
	for r := 0; r < 5; r++ {
		var b strings.Builder
		for c := 0; c < 5; c++ {
			fmt.Fprintf(&b, " %d", color[r*5+c])
		}
		fmt.Println(b.String())
	}
}
