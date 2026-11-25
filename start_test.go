package gost_test

import (
	"math"
	"math/rand"
	"testing"

	"gost"
	"gost/mps"
)

// TestMIPStartMIPLIB solves MIPLIB instances, then solves them again with
// (a) the optimal solution as a complete start, (b) a partial start (half
// of the integer values), (c) an infeasible start (a few integer values
// changed). The optimum must not change, and the complete start must be
// accepted, the partial one repaired.
func TestMIPStartMIPLIB(t *testing.T) {
	if testing.Short() {
		t.Skip()
	}
	for _, name := range []string{"p0033", "p0201", "lseu", "egout", "fiber", "vpm1", "misc03", "mod008", "khb05250", "dcmulti"} {
		m, err := mps.ReadFile("testdata/" + name + ".mps.gz")
		if err != nil {
			t.Fatal(err)
		}
		p := gost.DefaultParams()
		p.RelGap = 1e-9
		ref, _ := m.Solve(p)
		if ref.Status != gost.StatusOptimal {
			t.Fatalf("%s: %v", name, ref.Status)
		}
		x := ref.Values()
		var ints []int
		for j := range x {
			if _, _, _, _, typ := m.VarInfo(m.VarByIndex(j)); typ != gost.Continuous {
				ints = append(ints, j)
			}
		}
		rng := rand.New(rand.NewSource(1))
		check := func(kind string, want ...gost.StartStatus) {
			r, _ := m.Solve(p)
			ok := false
			for _, w := range want {
				ok = ok || r.StartStatus == w
			}
			if !ok {
				t.Errorf("%s %s: start status %v (%s), want %v", name, kind, r.StartStatus, r.StartInfo, want)
			}
			if r.Status != gost.StatusOptimal || math.Abs(r.Objective-ref.Objective) > 1e-6*math.Max(1, math.Abs(ref.Objective)) {
				t.Errorf("%s %s: %v %v, want optimal %v", name, kind, r.Status, r.Objective, ref.Objective)
			}
			t.Logf("%-8s %-10s start %-8v nodes %6d (without start %6d) %v", name, kind, r.StartStatus, r.Nodes, ref.Nodes, r.StartInfo)
		}

		// (a) complete feasible start
		m.SetStartValues(x)
		check("complete", gost.StartAccepted)

		// (b) partial start: half of the integer variables
		m.ClearStart()
		for _, j := range ints {
			if rng.Intn(2) == 0 {
				m.SetStart(m.VarByIndex(j), x[j])
			}
		}
		check("partial", gost.StartRepaired)

		// (c) infeasible start: three integer values moved by one
		y := append([]float64(nil), x...)
		for k := 0; k < 3 && len(ints) > 0; k++ {
			j := ints[rng.Intn(len(ints))]
			_, lo, up, _, _ := m.VarInfo(m.VarByIndex(j))
			if y[j]+1 <= up {
				y[j]++
			} else if y[j]-1 >= lo {
				y[j]--
			}
		}
		m.SetStartValues(y)
		check("infeasible", gost.StartAccepted, gost.StartRepaired, gost.StartRejected)
		m.ClearStart()
	}
}

// TestMIPStartKnapsack checks the API on a small model: a feasible but
// suboptimal start, a partial start and a start with a wrong length.
func TestMIPStartKnapsack(t *testing.T) {
	m := gost.NewModel("knap")
	w := []float64{12, 7, 11, 8, 9}
	v := []float64{24, 13, 23, 15, 16}
	var x []gost.Var
	e := gost.NewExpr()
	obj := gost.NewExpr()
	for i := range w {
		x = append(x, m.AddVar(0, 1, 0, gost.Binary, ""))
		e.Add(w[i], x[i])
		obj.Add(v[i], x[i])
	}
	m.AddConstr(e, gost.LessEq, 26, "cap")
	m.SetObjective(obj, gost.Maximize)

	// feasible, suboptimal: items 1 and 3 (value 28)
	m.SetStartValues([]float64{0, 1, 0, 1, 0})
	r, _ := m.Solve(gost.DefaultParams())
	if r.StartStatus != gost.StartAccepted || r.Objective != 51 {
		t.Fatalf("suboptimal start: %v %v %v", r.StartStatus, r.Objective, r.StartInfo)
	}
	// partial: only item 4 given (taken)
	m.ClearStart()
	m.SetStart(x[4], 1)
	r, _ = m.Solve(gost.DefaultParams())
	if r.StartStatus != gost.StartRepaired || r.Objective != 51 {
		t.Fatalf("partial start: %v %v %v", r.StartStatus, r.Objective, r.StartInfo)
	}
	// infeasible (all items taken): no completion with these values, so
	// the closest feasible solution is used
	m.SetStartValues([]float64{1, 1, 1, 1, 1})
	r, _ = m.Solve(gost.DefaultParams())
	if r.StartStatus != gost.StartRepaired || r.Objective != 51 {
		t.Fatalf("infeasible start: %v %v %v", r.StartStatus, r.Objective, r.StartInfo)
	}
	// round trip through WriteSolution / ReadStart
	m.ClearStart()
	if n, err := m.ReadStart(m.WriteSolution(r)); err != nil || n != len(w) {
		t.Fatalf("ReadStart: %v %v", n, err)
	}
	r2, _ := m.Solve(gost.DefaultParams())
	if r2.StartStatus != gost.StartAccepted || r2.Objective != 51 {
		t.Fatalf("round trip: %v %v", r2.StartStatus, r2.Objective)
	}
}

// TestMIPStartLazy gives a start made of two subtours to a TSP solved with
// lazy subtour constraints: the start must be refused by the lazy
// callback, and the optimum found anyway. A real tour is accepted.
func TestMIPStartLazy(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	n := 12
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
	tp := newTSP(d)
	p := gost.DefaultParams()
	p.LazyConstraints = func(ctx *gost.ConstrContext) {
		comps := tp.components(func(k int) float64 { return ctx.Value(tp.x[k]) }, 0.5)
		if len(comps) > 1 {
			for _, c := range comps {
				ctx.AddConstr(tp.subtourCut(c), gost.GreaterEq, 2, "subtour")
			}
		}
	}
	// edge index of (i, j)
	edge := func(i, j int) int {
		if i > j {
			i, j = j, i
		}
		for k, e := range tp.edges {
			if e[0] == i && e[1] == j {
				return k
			}
		}
		return -1
	}
	cycleStart := func(cycles ...[]int) []float64 {
		s := make([]float64, len(tp.x))
		for _, c := range cycles {
			for k := range c {
				s[edge(c[k], c[(k+1)%len(c)])] = 1
			}
		}
		return s
	}
	// two subtours: 0..5 and 6..11
	tp.m.SetStartValues(cycleStart([]int{0, 1, 2, 3, 4, 5}, []int{6, 7, 8, 9, 10, 11}))
	r, _ := tp.m.Solve(p)
	if r.StartStatus != gost.StartRejected {
		t.Fatalf("subtour start: %v (%s)", r.StartStatus, r.StartInfo)
	}
	want := heldKarp(d)
	if math.Abs(r.Objective-want) > 1e-6 {
		t.Fatalf("subtour start: obj %v, want %v", r.Objective, want)
	}
	// a real tour (identity order)
	tp.m.SetStartValues(cycleStart([]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}))
	r, _ = tp.m.Solve(p)
	if r.StartStatus != gost.StartAccepted || math.Abs(r.Objective-want) > 1e-6 {
		t.Fatalf("tour start: %v %v, want accepted %v", r.StartStatus, r.Objective, want)
	}
}

// TestMIPStartCutoffPath forces the path where the start is kept outside
// the solver and only used as a cutoff (as when presolve reductions make it
// unrepresentable in the reduced problem).
func TestMIPStartCutoffPath(t *testing.T) {
	t.Setenv("GOST_STARTCUTOFF", "1")
	for _, name := range []string{"p0201", "enigma", "mod008"} {
		m, err := mps.ReadFile("testdata/" + name + ".mps.gz")
		if err != nil {
			t.Fatal(err)
		}
		ref, _ := m.Solve(gost.DefaultParams())
		m.SetStartValues(ref.Values())
		r, _ := m.Solve(gost.DefaultParams())
		if r.Status != gost.StatusOptimal || r.StartStatus != gost.StartAccepted || !r.HasSolution() ||
			math.Abs(r.Objective-ref.Objective) > 1e-6*math.Max(1, math.Abs(ref.Objective)) {
			t.Fatalf("%s: %v %v %v, want %v", name, r.Status, r.StartStatus, r.Objective, ref.Objective)
		}
	}
}
