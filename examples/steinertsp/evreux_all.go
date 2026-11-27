package main

// Whole-warehouse evaluation of real pick lists (Evreux, EU-FRA-002).
//
//	go run ./examples/steinertsp evreux-all [picklists.txt]
//
// For every pick list of the data file (one week by default), compare the
// walk the picker actually did (aisles in the recorded order) with the
// shortest walk that visits the same aisles.
//
// Warehouse model (generic, built from the floor plan of evreux_plan.go):
//
//   - the floor is a grid of 0.5 m cells; a picker walks between free
//     cells in 8 directions (no corner cutting), at the Euclidean cost;
//
//   - racks are obstacles: the strip between two parallel aisles of the
//     same zone that face each other (closer than rackGap), and a band of
//     rackDepth on the outer side of an aisle without such a neighbour;
//
//   - an aisle is one-way (evreux_direction.go) and cannot be left half
//     way: its cells are blocked for free walking, and the only way through
//     is the arc entry port → exit port (ports sit just outside both ends),
//     whose cost is the aisle length. Picking in an aisle = traversing it.
//     Aisles not on the list may be used as shortcuts, in their direction;
//
//   - the walk starts and ends at a cart zone (depot): the one nearest to
//     the first aisle of the list, for both the recorded and the optimal
//     walk.
//
// Distances: Dijkstra on the grid from every exit port and every depot.
// Then, for a list with aisles a_1..a_k, the cost of going to aisle j
// (from the exit of aisle i, or from the depot) and traversing it is
//
//	c(i, j) = dist(exit_i, entry_j) + L_j
//
// Since dist is a shortest-path metric that includes every aisle arc,
// visiting each aisle once is enough (no revisit can help), and the optimal
// walk is an asymmetric TSP on {depot, a_1..a_k}: Held–Karp when k ≤ 15,
// otherwise a GoST MIP (assignment + subtour constraints generated lazily
// and as user cuts, recorded order given as MIP start).
//
// Outputs evreux_all_results.csv (one row per list) and a summary.

import (
	"bufio"
	"container/heap"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"gost"
)

// Grid and geometry parameters (meters).
const (
	gridRes    = 0.5   // cell size
	gridX0     = -4.0  // grid origin
	gridY0     = -4.0  //
	gridX1     = 304.0 // grid upper corner
	gridY1     = 110.0 //
	rackGap    = 6.0   // two parallel aisles closer than this face each other across a rack
	rackDepth  = 1.2   // depth of the rack on the outer side of an aisle
	portOffset = 0.75  // ports sit this far outside the aisle ends
	hkMaxAisle = 15    // Held–Karp up to this many aisles, MIP beyond
	mipLimit   = 30 * time.Second
)

// whAisle is an aisle of the warehouse model.
type whAisle struct {
	code, zone     string
	px, py, qx, qy float64 // walking direction: from p to q
	cost           float64 // cost of the traversal arc entry → exit (0 for point aisles)
	entry, exit    int     // grid cells of the ports
}

// wEdge is an extra arc of the grid graph (aisle traversal, port links).
type wEdge struct {
	to   int
	cost float64
}

// warehouse is the grid graph.
type warehouse struct {
	nx, ny    int
	blocked   []bool          // blocked[cell]: not walkable freely
	extra     map[int][]wEdge // arcs added on top of the grid moves
	aisles    []whAisle
	index     map[string]int // aisle code → index in aisles
	depots    []int          // cells of the cart zones
	depotXY   [][2]float64
	junctions int // cross aisles carved at junctions of aisle pieces
	teleports int // ports linked to the corridor through an obstacle (diagnostic)
}

// cell returns the grid cell containing (x, y).
func (w *warehouse) cell(x, y float64) int {
	ix := int(math.Round((x - gridX0) / gridRes))
	iy := int(math.Round((y - gridY0) / gridRes))
	ix = max(0, min(w.nx-1, ix))
	iy = max(0, min(w.ny-1, iy))
	return iy*w.nx + ix
}

// xy returns the center of a cell.
func (w *warehouse) xy(c int) (float64, float64) {
	return gridX0 + float64(c%w.nx)*gridRes, gridY0 + float64(c/w.nx)*gridRes
}

// blockRect blocks the cells whose center lies in the rectangle.
func (w *warehouse) blockRect(x1, y1, x2, y2 float64) {
	if x1 > x2 {
		x1, x2 = x2, x1
	}
	if y1 > y2 {
		y1, y2 = y2, y1
	}
	for c := range w.blocked {
		if x, y := w.xy(c); x >= x1-1e-9 && x <= x2+1e-9 && y >= y1-1e-9 && y <= y2+1e-9 {
			w.blocked[c] = true
		}
	}
}

// buildWarehouse builds the grid model from evreuxPlan, evreuxDirection
// and evreuxCartZones.
func buildWarehouse() *warehouse {
	w := &warehouse{
		nx:    int((gridX1-gridX0)/gridRes) + 1,
		ny:    int((gridY1-gridY0)/gridRes) + 1,
		extra: map[int][]wEdge{},
		index: map[string]int{},
	}
	w.blocked = make([]bool, w.nx*w.ny)

	// --- Aisles, oriented by their traversal direction -----------------------
	for _, a := range evreuxPlan {
		px, py, qx, qy := a.sx, a.sy, a.ex, a.ey
		if evreuxDirection[a.code] == -1 {
			px, py, qx, qy = qx, qy, px, py
		}
		w.index[a.code] = len(w.aisles)
		w.aisles = append(w.aisles, whAisle{code: a.code, zone: a.zone, px: px, py: py, qx: qx, qy: qy})
	}

	// --- Obstacles: aisle lines and racks ------------------------------------
	// geometry of an aisle seen along its axis: horizontal or not, position
	// across (c), extent along [lo, hi].
	type axis struct {
		hor       bool
		c, lo, hi float64
		length    float64
		point     bool
	}
	ax := make([]axis, len(w.aisles))
	for i, a := range w.aisles {
		dx, dy := math.Abs(a.qx-a.px), math.Abs(a.qy-a.py)
		g := axis{hor: dx >= dy, length: math.Hypot(dx, dy), point: math.Hypot(dx, dy) < 0.5}
		if g.hor {
			g.c, g.lo, g.hi = (a.py+a.qy)/2, math.Min(a.px, a.qx), math.Max(a.px, a.qx)
		} else {
			g.c, g.lo, g.hi = (a.px+a.qx)/2, math.Min(a.py, a.qy), math.Max(a.py, a.qy)
		}
		ax[i] = g
	}
	// rect blocks a rectangle given in aisle coordinates (along, across).
	rect := func(hor bool, lo, hi, c1, c2 float64) {
		if hor {
			w.blockRect(lo, c1, hi, c2)
		} else {
			w.blockRect(c1, lo, c2, hi)
		}
	}
	for i, g := range ax {
		if g.point {
			continue
		}
		rect(g.hor, g.lo, g.hi, g.c, g.c) // the aisle itself
		for _, side := range []float64{-1, 1} {
			// Nearest parallel aisle of the same zone on this side, facing
			// this one over at least half of the shorter length.
			width := math.Inf(1)
			for j, h := range ax {
				if j == i || h.point || h.hor != g.hor || w.aisles[j].zone != w.aisles[i].zone {
					continue
				}
				gap := (h.c - g.c) * side
				overlap := math.Min(g.hi, h.hi) - math.Max(g.lo, h.lo)
				if gap > 0.1 && gap <= rackGap && overlap >= 0.5*math.Min(g.length, h.length) {
					width = math.Min(width, gap)
				}
			}
			if math.IsInf(width, 1) {
				width = rackDepth // no facing aisle: rack on the outer side only
			}
			rect(g.hor, g.lo, g.hi, g.c, g.c+side*width)
		}
	}

	// --- Cross aisles at junctions -------------------------------------------
	// Collinear aisle pieces drawn end to end (e.g. 2P1_2P2_A..D) are
	// separated by a cross aisle in the building; the plan does not draw it,
	// so carve a one-cell corridor across the block at the junction, on
	// both sides, until it meets a free cell.
	for i, g := range ax {
		for j, h := range ax {
			if i == j || g.point || h.point || g.hor != h.hor || math.Abs(g.c-h.c) > 0.1 || math.Abs(h.lo-g.hi) > 1 {
				continue
			}
			along := (g.hi + h.lo) / 2
			for _, side := range []float64{-1, 1} {
				for d := gridRes; d <= 40; d += gridRes {
					x, y := along, g.c+side*d
					if !g.hor {
						x, y = g.c+side*d, along
					}
					c := w.cell(x, y)
					if !w.blocked[c] {
						break
					}
					w.blocked[c] = false
				}
			}
			x, y := along, g.c
			if !g.hor {
				x, y = g.c, along
			}
			w.junctions++
			_ = w.cell(x, y) // the junction cell itself stays blocked: the aisle line is one-way
		}
	}

	// --- Ports and traversal arcs --------------------------------------------
	var special []int // ports and depots, to be linked to the corridors
	for i := range w.aisles {
		a := &w.aisles[i]
		if ax[i].point {
			a.entry = w.cell(a.px, a.py)
			a.exit = a.entry
		} else {
			L := ax[i].length
			ux, uy := (a.qx-a.px)/L, (a.qy-a.py)/L
			a.entry = w.cell(a.px-ux*portOffset, a.py-uy*portOffset)
			a.exit = w.cell(a.qx+ux*portOffset, a.qy+uy*portOffset)
			a.cost = L + 2*portOffset
			w.extra[a.entry] = append(w.extra[a.entry], wEdge{a.exit, a.cost})
		}
		special = append(special, a.entry, a.exit)
	}
	for _, z := range evreuxCartZones {
		x, y := (z[0]+z[2])/2, (z[1]+z[3])/2
		w.depots = append(w.depots, w.cell(x, y))
		w.depotXY = append(w.depotXY, [2]float64{x, y})
		special = append(special, w.cell(x, y))
	}
	for _, c := range special {
		w.blocked[c] = false
	}
	// Ports close to each other (touching aisle pieces) are linked directly.
	for i, c := range special {
		x1, y1 := w.xy(c)
		for _, d := range special[i+1:] {
			x2, y2 := w.xy(d)
			if dd := math.Hypot(x2-x1, y2-y1); d != c && dd <= 2 {
				w.extra[c] = append(w.extra[c], wEdge{d, dd})
				w.extra[d] = append(w.extra[d], wEdge{c, dd})
			}
		}
	}
	// A special cell walled in by obstacles is linked to the nearest free
	// cell (both ways, Euclidean cost): data glitches, e.g. aisle pieces
	// drawn end to end with no cross aisle between them.
	for _, c := range special {
		if w.freeNeighbours(c) > 0 {
			continue
		}
		x1, y1 := w.xy(c)
		best, bd := -1, math.Inf(1)
		for d := range w.blocked {
			if !w.blocked[d] && d != c && w.freeNeighbours(d) > 0 {
				x2, y2 := w.xy(d)
				if dd := math.Hypot(x2-x1, y2-y1); dd < bd {
					best, bd = d, dd
				}
			}
		}
		if best >= 0 {
			w.extra[c] = append(w.extra[c], wEdge{best, bd})
			w.extra[best] = append(w.extra[best], wEdge{c, bd})
			w.teleports++
		}
	}
	return w
}

// freeNeighbours counts the free 4-neighbours of a cell.
func (w *warehouse) freeNeighbours(c int) int {
	n := 0
	ix, iy := c%w.nx, c/w.nx
	for _, d := range [][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}} {
		jx, jy := ix+d[0], iy+d[1]
		if jx >= 0 && jx < w.nx && jy >= 0 && jy < w.ny && !w.blocked[jy*w.nx+jx] {
			n++
		}
	}
	return n
}

// pq is a binary heap of (cell, distance) for Dijkstra.
type pqItem struct {
	c int
	d float64
}
type pq []pqItem

func (h pq) Len() int            { return len(h) }
func (h pq) Less(i, j int) bool  { return h[i].d < h[j].d }
func (h pq) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *pq) Push(x interface{}) { *h = append(*h, x.(pqItem)) }
func (h *pq) Pop() interface{} {
	old := *h
	it := old[len(old)-1]
	*h = old[:len(old)-1]
	return it
}

// dijkstra returns the walking distance from cell s to every cell: grid
// moves between free cells (diagonals only when both side cells are free)
// plus the extra arcs.
func (w *warehouse) dijkstra(s int) []float64 {
	dist, _ := w.dijkstraPred(s)
	return dist
}

// dijkstraPred is dijkstra that also returns, for every cell, the cell it
// is reached from on a shortest path (-1 for s and unreachable cells).
func (w *warehouse) dijkstraPred(s int) ([]float64, []int32) {
	dist := make([]float64, len(w.blocked))
	pred := make([]int32, len(w.blocked))
	for i := range pred {
		pred[i] = -1
	}
	for i := range dist {
		dist[i] = math.Inf(1)
	}
	dist[s] = 0
	h := &pq{{s, 0}}
	free := func(x, y int) bool { return x >= 0 && x < w.nx && y >= 0 && y < w.ny && !w.blocked[y*w.nx+x] }
	moves := [][3]float64{{1, 0, 1}, {-1, 0, 1}, {0, 1, 1}, {0, -1, 1},
		{1, 1, math.Sqrt2}, {1, -1, math.Sqrt2}, {-1, 1, math.Sqrt2}, {-1, -1, math.Sqrt2}}
	for h.Len() > 0 {
		it := heap.Pop(h).(pqItem)
		if it.d > dist[it.c] {
			continue
		}
		relax := func(v int, d float64) {
			if d < dist[v] {
				dist[v] = d
				pred[v] = int32(it.c)
				heap.Push(h, pqItem{v, d})
			}
		}
		for _, e := range w.extra[it.c] {
			relax(e.to, it.d+e.cost)
		}
		if w.blocked[it.c] {
			continue // inside an obstacle: only the extra arcs lead out
		}
		x, y := it.c%w.nx, it.c/w.nx
		for _, m := range moves {
			dx, dy := int(m[0]), int(m[1])
			if !free(x+dx, y+dy) || (dx != 0 && dy != 0 && (!free(x+dx, y) || !free(x, y+dy))) {
				continue
			}
			relax((y+dy)*w.nx+x+dx, it.d+m[2]*gridRes)
		}
	}
	return dist, pred
}

// pickRecord is a pick list of the data file.
type pickRecord struct {
	id, start      string
	picks, seconds int
	visits         []string // aisle codes in the recorded order
}

// readPickLists parses the data file (format in its header).
func readPickLists(path string) ([]pickRecord, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var dict []string
	var out []pickRecord
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if line == "" || line[0] == '#' {
			continue
		}
		fs := strings.Fields(line)
		if fs[0] == "aisles" {
			dict = fs[1:]
			continue
		}
		if len(fs) < 5 {
			continue
		}
		r := pickRecord{id: fs[0], start: fs[3]}
		r.picks, _ = strconv.Atoi(fs[1])
		r.seconds, _ = strconv.Atoi(fs[2])
		for _, s := range strings.Split(fs[4], ".") {
			k, _ := strconv.Atoi(s)
			r.visits = append(r.visits, dict[k])
		}
		out = append(out, r)
	}
	return out, sc.Err()
}

// listResult is the evaluation of one pick list.
type listResult struct {
	rec                  pickRecord
	status               string // ok, nocoord, unreachable
	zone                 string
	distinct, revisits   int
	depot                int
	recorded, firstOrder float64 // recorded walk; same order without revisits
	optimal, bound       float64
	method               string // single, hk, mip
	optOrder             []string
}

// atspHeldKarp solves the ATSP on the cost matrix c (node 0 = start) by
// dynamic programming (see heldKarp in main.go); returns the length and
// the order of the nodes 1..n-1.
func atspHeldKarp(c [][]float64) (float64, []int) {
	r := len(c)
	if r == 1 {
		return 0, nil
	}
	full := 1 << (r - 1)
	f := make([]float64, full*r) // f[S*r+j]
	parent := make([]int8, full*r)
	for i := range f {
		f[i] = math.Inf(1)
	}
	for j := 1; j < r; j++ {
		f[(1<<(j-1))*r+j] = c[0][j]
	}
	for S := 1; S < full; S++ {
		for j := 1; j < r; j++ {
			fs := f[S*r+j]
			if S&(1<<(j-1)) == 0 || math.IsInf(fs, 1) {
				continue
			}
			for k := 1; k < r; k++ {
				if S&(1<<(k-1)) == 0 {
					T := S | 1<<(k-1)
					if v := fs + c[j][k]; v < f[T*r+k] {
						f[T*r+k] = v
						parent[T*r+k] = int8(j)
					}
				}
			}
		}
	}
	best, last := math.Inf(1), -1
	for j := 1; j < r; j++ {
		if v := f[(full-1)*r+j] + c[j][0]; v < best {
			best, last = v, j
		}
	}
	if last < 0 {
		return best, nil
	}
	order := make([]int, r-1)
	S := full - 1
	for i := r - 2; i >= 0; i-- {
		order[i] = last
		p := int(parent[S*r+last])
		S &^= 1 << (last - 1)
		last = p
	}
	return best, order
}

// atspMIP solves the ATSP on c (node 0 = start) with GoST: binary arc
// variables, one arc in and one arc out of every node, and the subtour
// elimination constraints Σ_{δ⁺(S)} x ≥ 1 generated on demand (lazily on
// integer candidates, as user cuts on fractional LP solutions). start is a
// feasible order (MIP start). Returns objective, bound and order.
func atspMIP(c [][]float64, start []int) (float64, float64, []int) {
	n := len(c)
	m := gost.NewModel("atsp")
	var arcs []arc
	var x []gost.Var
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i != j && !math.IsInf(c[i][j], 1) {
				arcs = append(arcs, arc{i, j, c[i][j]})
				x = append(x, m.AddVar(0, 1, c[i][j], gost.Binary, fmt.Sprintf("x_%d_%d", i, j)))
			}
		}
	}
	id := map[[2]int]int{} // (i, j) → arc index
	for a, e := range arcs {
		id[[2]int{e.from, e.to}] = a
	}
	for v := 0; v < n; v++ {
		in, out := gost.NewExpr(), gost.NewExpr()
		for a, e := range arcs {
			if e.to == v {
				in.Add(1, x[a])
			}
			if e.from == v {
				out.Add(1, x[a])
			}
		}
		m.AddConstr(in, gost.Equal, 1, fmt.Sprintf("in_%d", v))
		m.AddConstr(out, gost.Equal, 1, fmt.Sprintf("out_%d", v))
	}
	// No 2-cycles (the smallest subtours), given up front.
	for a, e := range arcs {
		if b, ok := id[[2]int{e.to, e.from}]; ok && e.from < e.to && n > 2 {
			m.AddConstr(gost.NewExpr().Add(1, x[a]).Add(1, x[b]), gost.LessEq, 1, "no2cycle")
		}
	}
	cut := func(inS []bool) *gost.Expr {
		e := gost.NewExpr()
		for a, ar := range arcs {
			if inS[ar.from] && !inS[ar.to] {
				e.Add(1, x[a])
			}
		}
		return e
	}
	// MIP start: the given order (arcs of the tour at 1, the others at 0).
	startX := make([]float64, len(arcs))
	prev := 0
	for _, v := range append(append([]int{}, start...), 0) {
		if a, ok := id[[2]int{prev, v}]; ok {
			startX[a] = 1
		}
		prev = v
	}
	for a := range arcs {
		m.SetStart(x[a], startX[a])
	}

	p := gost.DefaultParams()
	p.TimeLimit = mipLimit
	p.LazyConstraints = func(ctx *gost.ConstrContext) {
		pieces := components(n, arcs, func(a int) float64 { return ctx.Value(x[a]) }, 0.5)
		if len(pieces) == 1 {
			return
		}
		for _, piece := range pieces {
			inS := make([]bool, n)
			for _, v := range piece {
				inS[v] = true
			}
			ctx.AddConstr(cut(inS), gost.GreaterEq, 1, "subtour")
		}
	}
	p.UserCuts = func(ctx *gost.ConstrContext) {
		capa := make([]float64, len(arcs))
		for a := range arcs {
			capa[a] = math.Max(ctx.Value(x[a]), 0)
		}
		added := 0
		for t := 1; t < n && added < 20; t++ {
			if flow, inS := minCut(n, arcs, capa, 0, t, 1); flow < 1-1e-6 {
				ctx.AddConstr(cut(inS), gost.GreaterEq, 1, "subtour-cut")
				added++
			}
		}
	}
	res, err := m.Solve(p)
	if err != nil {
		panic(err)
	}
	// Follow the successors from node 0.
	succ := make([]int, n)
	for a, e := range arcs {
		if res.Value(x[a]) > 0.5 {
			succ[e.from] = e.to
		}
	}
	var order []int
	for v := succ[0]; v != 0 && len(order) < n; v = succ[v] {
		order = append(order, v)
	}
	return res.Objective, res.Bound, order
}

// evaluate computes the recorded and the optimal walks of one list. dExit
// and dDepot give the walking distance from an aisle exit / a depot to
// every cell.
func evaluate(w *warehouse, r pickRecord, dExit, dDepot [][]float64) listResult {
	res := listResult{rec: r, status: "ok"}
	// Aisles of the list, in first-visit order.
	var seq []int
	for _, code := range r.visits {
		i, ok := w.index[code]
		if !ok {
			res.status = "nocoord"
			return res
		}
		seq = append(seq, i)
	}
	pos := map[int]int{} // aisle → node number (1..k)
	var nodes []int
	for _, i := range seq {
		if _, ok := pos[i]; !ok {
			pos[i] = len(nodes) + 1
			nodes = append(nodes, i)
		}
	}
	res.distinct = len(nodes)
	res.revisits = len(seq) - len(nodes)
	// Zone: the most frequent one among the aisles.
	cnt := map[string]int{}
	for _, i := range nodes {
		cnt[w.aisles[i].zone]++
	}
	for z, k := range cnt {
		if k > cnt[res.zone] || (k == cnt[res.zone] && z < res.zone) {
			res.zone = z
		}
	}
	// Depot: the cart zone nearest to the first aisle.
	a0 := w.aisles[seq[0]]
	bd := math.Inf(1)
	for d, p := range w.depotXY {
		if dd := math.Hypot(p[0]-a0.px, p[1]-a0.py); dd < bd {
			bd, res.depot = dd, d
		}
	}
	// Cost matrix on {depot (0), aisles (1..k)}.
	k := len(nodes)
	c := make([][]float64, k+1)
	for u := range c {
		c[u] = make([]float64, k+1)
		for v := range c[u] {
			switch {
			case u == v:
				c[u][v] = 0
			case u == 0:
				c[u][v] = dDepot[res.depot][w.aisles[nodes[v-1]].entry] + w.aisles[nodes[v-1]].cost
			case v == 0:
				c[u][v] = dExit[nodes[u-1]][w.depots[res.depot]]
			default:
				c[u][v] = dExit[nodes[u-1]][w.aisles[nodes[v-1]].entry] + w.aisles[nodes[v-1]].cost
			}
			if math.IsInf(c[u][v], 1) {
				res.status = "unreachable"
				return res
			}
		}
	}
	// Recorded walk (with revisits) and the same order without revisits.
	walk := func(order []int) float64 {
		t, prev := 0.0, 0
		for _, v := range order {
			t += c[prev][v]
			prev = v
		}
		return t + c[prev][0]
	}
	var recOrder, first []int
	for _, i := range seq {
		recOrder = append(recOrder, pos[i])
	}
	for v := 1; v <= k; v++ {
		first = append(first, v)
	}
	res.recorded, res.firstOrder = walk(recOrder), walk(first)
	// Optimal walk.
	var order []int
	switch {
	case k == 1:
		res.method, order, res.optimal = "single", first, res.firstOrder
		res.bound = res.optimal
	case k <= hkMaxAisle:
		res.method = "hk"
		res.optimal, order = atspHeldKarp(c)
		res.bound = res.optimal
	default:
		res.method = "mip"
		res.optimal, res.bound, order = atspMIP(c, first)
		if l := walk(order); len(order) != k || math.Abs(l-res.optimal) > 1e-6 {
			// Should not happen; fall back on the recorded order.
			res.optimal, order = res.firstOrder, first
		}
	}
	for _, v := range order {
		res.optOrder = append(res.optOrder, w.aisles[nodes[v-1]].code)
	}
	return res
}

// evreuxAll runs the whole evaluation.
func evreuxAll(path string) {
	t0 := time.Now()
	w := buildWarehouse()
	fmt.Printf("grid %d × %d cells (%.1f m), %d aisles, %d cross aisles carved, %d ports linked through an obstacle\n",
		w.nx, w.ny, gridRes, len(w.aisles), w.junctions, w.teleports)

	if path := os.Getenv("EVREUX_GRID_DUMP"); path != "" { // debug: obstacle map
		g, _ := os.Create(path)
		fmt.Fprintf(g, "%d %d\n", w.nx, w.ny)
		for c, b := range w.blocked {
			v := 0
			if b {
				v = 1
			}
			fmt.Fprint(g, v)
			if c%w.nx == w.nx-1 {
				fmt.Fprintln(g)
			}
		}
		for _, a := range w.aisles {
			fmt.Fprintln(g, "P", a.entry, a.exit)
		}
		for c, es := range w.extra {
			for _, e := range es {
				fmt.Fprintln(g, "E", c, e.to, e.cost)
			}
		}
		g.Close()
	}

	// Distances from every aisle exit and every depot.
	dExit := make([][]float64, len(w.aisles))
	dDepot := make([][]float64, len(w.depots))
	var wg sync.WaitGroup
	jobs := make(chan int)
	for g := 0; g < 2; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				if j < len(w.aisles) {
					dExit[j] = w.dijkstra(w.aisles[j].exit)
				} else {
					dDepot[j-len(w.aisles)] = w.dijkstra(w.depots[j-len(w.aisles)])
				}
			}
		}()
	}
	for j := 0; j < len(w.aisles)+len(w.depots); j++ {
		jobs <- j
	}
	close(jobs)
	wg.Wait()
	fmt.Printf("distances computed in %v\n", time.Since(t0).Round(time.Millisecond))

	// Aisles never reached from depot 0 (diagnostic).
	unreached := 0
	for _, a := range w.aisles {
		if math.IsInf(dDepot[0][a.entry], 1) {
			unreached++
		}
	}
	fmt.Printf("%d aisle entries unreachable from the first cart zone\n", unreached)

	// Check on the two lists of evreux.go (3E block model: 388.9 → 367.9 m and 148.5 m).
	for _, pl := range []struct {
		name  string
		nodes []int
	}{{"pil_db6yrpjfms75r10r", pickList}, {"pil_db6yk5ctmh22dmmp", pickListGeeli}} {
		r := pickRecord{id: pl.name}
		for _, v := range pl.nodes {
			r.visits = append(r.visits, evreuxNodeCode[v])
		}
		e := evaluate(w, r, dExit, dDepot)
		fmt.Printf("check %s: recorded %.1f m, optimal %.1f m (%s)\n", pl.name, e.recorded, e.optimal, e.status)
	}

	lists, err := readPickLists(path)
	if err != nil {
		panic(err)
	}
	results := make([]listResult, len(lists))
	jobs = make(chan int)
	var mu sync.Mutex
	done := 0
	for g := 0; g < 2; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				results[i] = evaluate(w, lists[i], dExit, dDepot)
				mu.Lock()
				done++
				if done%1000 == 0 {
					fmt.Printf("  %d / %d lists (%v)\n", done, len(lists), time.Since(t0).Round(time.Second))
				}
				mu.Unlock()
			}
		}()
	}
	for i := range lists {
		jobs <- i
	}
	close(jobs)
	wg.Wait()

	// --- CSV -------------------------------------------------------------------
	f, err := os.Create("evreux_all_results.csv")
	if err != nil {
		panic(err)
	}
	bw := bufio.NewWriter(f)
	fmt.Fprintln(bw, "pick_list,start,picks,seconds,status,zone,visits,distinct_aisles,revisits,depot,recorded_m,no_revisit_m,optimal_m,bound_m,saving_m,saving_pct,method,recorded_order,optimal_order")
	for _, r := range results {
		saving, pct := 0.0, 0.0
		if r.status == "ok" {
			saving = r.recorded - r.optimal
			pct = 100 * saving / r.recorded
		}
		fmt.Fprintf(bw, "%s,%s,%d,%d,%s,%s,%d,%d,%d,%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.2f,%s,%s,%s\n",
			r.rec.id, r.rec.start, r.rec.picks, r.rec.seconds, r.status, r.zone, len(r.rec.visits),
			r.distinct, r.revisits, r.depot, r.recorded, r.firstOrder, r.optimal, r.bound, saving, pct,
			r.method, strings.Join(r.rec.visits, " "), strings.Join(r.optOrder, " "))
	}
	bw.Flush()
	f.Close()

	// --- Summary ---------------------------------------------------------------
	var rec, opt, nore float64
	var n, nOpt, nMIP, nGap int
	var pcts []float64
	count := map[string]int{}
	for _, r := range results {
		count[r.status]++
		if r.status != "ok" {
			continue
		}
		n++
		rec += r.recorded
		opt += r.optimal
		nore += r.firstOrder
		pcts = append(pcts, 100*(r.recorded-r.optimal)/r.recorded)
		if r.recorded-r.optimal < 0.05 {
			nOpt++
		}
		if r.method == "mip" {
			nMIP++
			if r.optimal-r.bound > 1e-4*r.optimal+1e-6 {
				nGap++
			}
		}
	}
	sort.Float64s(pcts)
	fmt.Printf("\n%d lists: %v\n", len(results), count)
	fmt.Printf("evaluated: %d, recorded %.1f km, without revisits %.1f km, optimal %.1f km: saving %.1f km (%.1f %%)\n",
		n, rec/1000, nore/1000, opt/1000, (rec-opt)/1000, 100*(rec-opt)/rec)
	fmt.Printf("already optimal: %d (%.1f %%), median saving %.1f %%, p90 %.1f %%; MIP solves %d (%d not proven optimal)\n",
		nOpt, 100*float64(nOpt)/float64(n), pcts[len(pcts)/2], pcts[len(pcts)*9/10], nMIP, nGap)
	// --- Floor plan with the most improved lists ----------------------------------
	if err := writeAllPlanHTML("evreux_all_plan.html", w, pickShowcase(results)); err != nil {
		panic(err)
	}
	fmt.Printf("results in evreux_all_results.csv and evreux_all_plan.html, total time %v\n", time.Since(t0).Round(time.Second))
}
