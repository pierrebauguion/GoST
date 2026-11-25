package gost_test

import (
	"math"
	"math/rand"
	"sort"
	"testing"

	"gost"
)

// ---------------------------------------------------------------------------
// Cutting stock by column generation (Gilmore–Gomory).

type cuttingStock struct {
	W      int
	widths []int
	demand []float64
}

// knapsack maximizes sum y_i a_i s.t. sum w_i a_i <= W, a_i integer >= 0
// (unbounded knapsack by DP) and returns the pattern.
func (cs *cuttingStock) knapsack(y []float64) (float64, []int) {
	W := cs.W
	best := make([]float64, W+1)
	choice := make([]int, W+1)
	for c := range choice {
		choice[c] = -1
	}
	for c := 1; c <= W; c++ {
		best[c] = best[c-1]
		choice[c] = -2 // waste one unit
		for i, w := range cs.widths {
			if w <= c && best[c-w]+y[i] > best[c]+1e-12 {
				best[c] = best[c-w] + y[i]
				choice[c] = i
			}
		}
	}
	pat := make([]int, len(cs.widths))
	for c := W; c > 0; {
		switch i := choice[c]; i {
		case -2:
			c--
		default:
			pat[i]++
			c -= cs.widths[i]
		}
	}
	return best[W], pat
}

// solve builds the master with one homogeneous pattern per width and
// prices columns while depthLimit allows. Returns the result and the model.
func (cs *cuttingStock) solve(t *testing.T, integer bool, priceDepth int) (*gost.Result, *gost.Model) {
	m := gost.NewModel("cutstock")
	typ := gost.Continuous
	if integer {
		typ = gost.Integer
	}
	var dem []gost.Constr
	for i := range cs.widths {
		dem = append(dem, m.AddConstr(gost.NewExpr(), gost.GreaterEq, cs.demand[i], "demand"))
	}
	for i, w := range cs.widths {
		m.AddVarWithColumn(0, gost.Inf, 1, typ, "init", gost.NewColumn().Add(float64(cs.W/w), dem[i]))
	}
	p := gost.DefaultParams()
	p.RelGap = 1e-9
	p.Pricing = func(ctx *gost.PricingContext) {
		if ctx.Depth() > priceDepth {
			return
		}
		y := make([]float64, len(dem))
		for i, c := range dem {
			y[i] = ctx.Dual(c)
		}
		val, pat := cs.knapsack(y)
		if val <= 1+1e-9 {
			return // no column with negative reduced cost
		}
		col := gost.NewColumn()
		for i, a := range pat {
			if a > 0 {
				col.Add(float64(a), dem[i])
			}
		}
		if rc := ctx.ReducedCost(1, col); rc > -1e-9 {
			t.Fatalf("pricing reduced cost %v should be negative", rc)
		}
		ctx.AddVar(0, gost.Inf, 1, typ, "pattern", col)
	}
	r, err := m.Solve(p)
	if err != nil {
		t.Fatal(err)
	}
	return r, m
}

// enumerateLP solves the full LP with all maximal patterns.
func (cs *cuttingStock) enumerateLP(t *testing.T) float64 {
	var pats [][]int
	var rec func(i, rem int, cur []int)
	rec = func(i, rem int, cur []int) {
		if i == len(cs.widths) {
			// keep maximal patterns only
			for _, w := range cs.widths {
				if w <= rem {
					return
				}
			}
			pats = append(pats, append([]int(nil), cur...))
			return
		}
		for a := rem / cs.widths[i]; a >= 0; a-- {
			cur[i] = a
			rec(i+1, rem-a*cs.widths[i], cur)
		}
		cur[i] = 0
	}
	rec(0, cs.W, make([]int, len(cs.widths)))
	m := gost.NewModel("full")
	var dem []gost.Constr
	for i := range cs.widths {
		dem = append(dem, m.AddConstr(gost.NewExpr(), gost.GreaterEq, cs.demand[i], ""))
	}
	for _, pt := range pats {
		col := gost.NewColumn()
		for i, a := range pt {
			if a > 0 {
				col.Add(float64(a), dem[i])
			}
		}
		m.AddVarWithColumn(0, gost.Inf, 1, gost.Continuous, "", col)
	}
	r, _ := m.Solve(gost.DefaultParams())
	if r.Status != gost.StatusOptimal {
		t.Fatalf("full LP: %v", r.Status)
	}
	return r.Objective
}

func TestColumnGenerationCuttingStock(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	generated := 0
	defer func() {
		if generated == 0 {
			t.Errorf("no column was ever generated")
		}
		t.Logf("%d columns generated", generated)
	}()
	for trial := 0; trial < 10; trial++ {
		cs := &cuttingStock{W: 100 + rng.Intn(50)}
		k := 3 + rng.Intn(5)
		for i := 0; i < k; i++ {
			cs.widths = append(cs.widths, 15+rng.Intn(50))
			cs.demand = append(cs.demand, float64(5+rng.Intn(40)))
		}
		want := cs.enumerateLP(t)
		r, m := cs.solve(t, false, 0)
		if r.Status != gost.StatusOptimal || math.Abs(r.Objective-want) > 1e-6*math.Max(1, want) {
			t.Fatalf("trial %d: CG LP %v %v, full LP %v", trial, r.Status, r.Objective, want)
		}
		generated += m.NumVars() - k
		// LP duals are exposed for pure LPs.
		for i := 0; i < m.NumConstrs(); i++ {
			if d := r.Dual(m.ConstrByIndex(i)); math.IsNaN(d) || d < -1e-9 {
				t.Fatalf("trial %d: bad dual %v", trial, d)
			}
		}
		// Branch-and-price (pricing at all depths) and price-and-branch
		// (root only) give integer solutions >= ceil(LP).
		for _, depth := range []int{0, 1000} {
			ri, mi := cs.solve(t, true, depth)
			if ri.Status != gost.StatusOptimal {
				t.Fatalf("trial %d depth %d: %v", trial, depth, ri.Status)
			}
			if ri.Objective < math.Ceil(want-1e-6)-1e-6 {
				t.Fatalf("trial %d: integer %v below LP bound %v", trial, ri.Objective, want)
			}
			// demands satisfied
			for i := 0; i < mi.NumConstrs(); i++ {
				_, lo, _, vars, coefs := mi.ConstrInfo(mi.ConstrByIndex(i))
				s := 0.0
				for q, v := range vars {
					s += coefs[q] * ri.Value(v)
				}
				if s < lo-1e-6 {
					t.Fatalf("trial %d: demand %d not met", trial, i)
				}
			}
		}
	}
}

// ---------------------------------------------------------------------------
// TSP with lazy subtour elimination constraints.

func heldKarp(d [][]float64) float64 {
	n := len(d)
	full := 1 << n
	dp := make([][]float64, full)
	for s := range dp {
		dp[s] = make([]float64, n)
		for j := range dp[s] {
			dp[s][j] = math.Inf(1)
		}
	}
	dp[1][0] = 0
	for s := 1; s < full; s++ {
		if s&1 == 0 {
			continue
		}
		for j := 0; j < n; j++ {
			if dp[s][j] == math.Inf(1) {
				continue
			}
			for k := 1; k < n; k++ {
				if s&(1<<k) == 0 {
					ns := s | 1<<k
					if v := dp[s][j] + d[j][k]; v < dp[ns][k] {
						dp[ns][k] = v
					}
				}
			}
		}
	}
	best := math.Inf(1)
	for j := 1; j < n; j++ {
		best = math.Min(best, dp[full-1][j]+d[j][0])
	}
	return best
}

type tsp struct {
	n     int
	edges [][2]int
	x     []gost.Var
	m     *gost.Model
}

func newTSP(d [][]float64) *tsp {
	n := len(d)
	t := &tsp{n: n, m: gost.NewModel("tsp")}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			t.edges = append(t.edges, [2]int{i, j})
			t.x = append(t.x, t.m.AddVar(0, 1, d[i][j], gost.Binary, ""))
		}
	}
	for v := 0; v < n; v++ {
		e := gost.NewExpr()
		for k, ed := range t.edges {
			if ed[0] == v || ed[1] == v {
				e.Add(1, t.x[k])
			}
		}
		t.m.AddConstr(e, gost.Equal, 2, "deg")
	}
	return t
}

// components of the graph with edges whose value exceeds thr.
func (t *tsp) components(val func(k int) float64, thr float64) [][]int {
	parent := make([]int, t.n)
	for i := range parent {
		parent[i] = i
	}
	var find func(int) int
	find = func(a int) int {
		for parent[a] != a {
			parent[a] = parent[parent[a]]
			a = parent[a]
		}
		return a
	}
	for k, e := range t.edges {
		if val(k) > thr {
			parent[find(e[0])] = find(e[1])
		}
	}
	groups := map[int][]int{}
	for v := 0; v < t.n; v++ {
		r := find(v)
		groups[r] = append(groups[r], v)
	}
	var out [][]int
	for _, g := range groups {
		out = append(out, g)
	}
	sort.Slice(out, func(a, b int) bool { return out[a][0] < out[b][0] })
	return out
}

func (t *tsp) subtourCut(set []int) *gost.Expr {
	in := make([]bool, t.n)
	for _, v := range set {
		in[v] = true
	}
	e := gost.NewExpr()
	for k, ed := range t.edges {
		if in[ed[0]] != in[ed[1]] {
			e.Add(1, t.x[k])
		}
	}
	return e
}

func TestLazySubtourTSP(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	for trial := 0; trial < 8; trial++ {
		n := 6 + rng.Intn(10)
		px, py := make([]float64, n), make([]float64, n)
		for i := range px {
			px[i], py[i] = rng.Float64()*100, rng.Float64()*100
		}
		d := make([][]float64, n)
		for i := range d {
			d[i] = make([]float64, n)
			for j := range d[i] {
				d[i][j] = math.Round(math.Hypot(px[i]-px[j], py[i]-py[j]))
			}
		}
		want := heldKarp(d)
		for _, withCuts := range []bool{false, true} {
			tp := newTSP(d)
			lazyCalls := 0
			p := gost.DefaultParams()
			p.RelGap = 1e-9
			p.LazyConstraints = func(ctx *gost.ConstrContext) {
				lazyCalls++
				comps := tp.components(func(k int) float64 { return ctx.Value(tp.x[k]) }, 0.5)
				if len(comps) > 1 {
					for _, c := range comps {
						ctx.AddConstr(tp.subtourCut(c), gost.GreaterEq, 2, "subtour")
					}
				}
			}
			if withCuts {
				p.UserCuts = func(ctx *gost.ConstrContext) {
					comps := tp.components(func(k int) float64 { return ctx.Value(tp.x[k]) }, 1e-6)
					if len(comps) > 1 {
						for _, c := range comps {
							ctx.AddConstr(tp.subtourCut(c), gost.GreaterEq, 2, "subtour-cut")
						}
					}
				}
			}
			r, _ := tp.m.Solve(p)
			if r.Status != gost.StatusOptimal || math.Abs(r.Objective-want) > 1e-6 {
				t.Fatalf("trial %d (cuts=%v): got %v %v, want %v", trial, withCuts, r.Status, r.Objective, want)
			}
			// the returned tour is a single cycle
			if comps := tp.components(func(k int) float64 { return r.Value(tp.x[k]) }, 0.5); len(comps) != 1 {
				t.Fatalf("trial %d: solution has %d subtours", trial, len(comps))
			}
			t.Logf("trial %d n=%d cuts=%v obj=%v nodes=%d lazy calls=%d constrs=%d", trial, n, withCuts, r.Objective, r.Nodes, lazyCalls, tp.m.NumConstrs())
			if lazyCalls == 0 {
				t.Fatalf("trial %d: lazy callback never called", trial)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// LP row generation with lazy constraints and LP duals.

func TestLPDualsAndRowGeneration(t *testing.T) {
	// max 3x + 2y  s.t. x + y <= 4, x + 3y <= 6, x <= 3
	m := gost.NewModel("lp")
	x := m.AddVar(0, gost.Inf, 3, gost.Continuous, "x")
	y := m.AddVar(0, gost.Inf, 2, gost.Continuous, "y")
	c1 := m.AddConstr(gost.NewExpr().Add(1, x).Add(1, y), gost.LessEq, 4, "c1")
	c2 := m.AddConstr(gost.NewExpr().Add(1, x).Add(3, y), gost.LessEq, 6, "c2")
	m.SetSense(gost.Maximize)
	// x <= 3 is only added lazily.
	p := gost.DefaultParams()
	added := false
	p.LazyConstraints = func(ctx *gost.ConstrContext) {
		if ctx.Value(x) > 3+1e-9 {
			ctx.AddConstr(gost.NewExpr().Add(1, x), gost.LessEq, 3, "c3")
			added = true
		}
	}
	r, _ := m.Solve(p)
	if !added || r.Status != gost.StatusOptimal || math.Abs(r.Objective-11) > 1e-9 {
		t.Fatalf("got %v %v (added %v), want 11", r.Status, r.Objective, added)
	}
	// optimum x=3, y=1: duals c1=2, c2=0, c3=1
	if math.Abs(r.Dual(c1)-2) > 1e-9 || math.Abs(r.Dual(c2)) > 1e-9 || math.Abs(r.Dual(m.ConstrByIndex(2))-1) > 1e-9 {
		t.Fatalf("duals %v %v %v", r.Dual(c1), r.Dual(c2), r.Dual(m.ConstrByIndex(2)))
	}
	if math.Abs(r.ReducedCost(x)) > 1e-9 || math.Abs(r.ReducedCost(y)) > 1e-9 {
		t.Fatalf("reduced costs %v %v", r.ReducedCost(x), r.ReducedCost(y))
	}
}

func TestLazyTSPLarger(t *testing.T) {
	if testing.Short() {
		t.Skip()
	}
	rng := rand.New(rand.NewSource(21))
	for _, n := range []int{30, 50} {
		px, py := make([]float64, n), make([]float64, n)
		for i := range px {
			px[i], py[i] = rng.Float64()*1000, rng.Float64()*1000
		}
		d := make([][]float64, n)
		for i := range d {
			d[i] = make([]float64, n)
			for j := range d[i] {
				d[i][j] = math.Round(math.Hypot(px[i]-px[j], py[i]-py[j]))
			}
		}
		var objs []float64
		for _, withCuts := range []bool{false, true} {
			tp := newTSP(d)
			p := gost.DefaultParams()
			p.RelGap = 1e-9
			p.LazyConstraints = func(ctx *gost.ConstrContext) {
				comps := tp.components(func(k int) float64 { return ctx.Value(tp.x[k]) }, 0.5)
				if len(comps) > 1 {
					for _, c := range comps {
						ctx.AddConstr(tp.subtourCut(c), gost.GreaterEq, 2, "subtour")
					}
				}
			}
			if withCuts {
				p.UserCuts = func(ctx *gost.ConstrContext) {
					comps := tp.components(func(k int) float64 { return ctx.Value(tp.x[k]) }, 1e-6)
					if len(comps) > 1 {
						for _, c := range comps {
							ctx.AddConstr(tp.subtourCut(c), gost.GreaterEq, 2, "subtour-cut")
						}
					}
				}
			}
			r, _ := tp.m.Solve(p)
			if r.Status != gost.StatusOptimal {
				t.Fatalf("n=%d: %v", n, r.Status)
			}
			if comps := tp.components(func(k int) float64 { return r.Value(tp.x[k]) }, 0.5); len(comps) != 1 {
				t.Fatalf("n=%d: %d subtours", n, len(comps))
			}
			t.Logf("n=%d cuts=%v obj=%v nodes=%d constrs=%d time=%v", n, withCuts, r.Objective, r.Nodes, tp.m.NumConstrs(), r.Time)
			objs = append(objs, r.Objective)
		}
		if math.Abs(objs[0]-objs[1]) > 1e-6 {
			t.Fatalf("n=%d: lazy-only %v vs with cuts %v", n, objs[0], objs[1])
		}
	}
}
