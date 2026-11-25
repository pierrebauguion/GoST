package gost_test

import (
	"bytes"
	"math"
	"math/rand"
	"path/filepath"
	"testing"
	"time"

	"gost"
	"gost/mps"
)

func TestKnapsack(t *testing.T) {
	m := gost.NewModel("knapsack")
	w := []float64{12, 2, 1, 1, 4}
	v := []float64{4, 2, 1, 2, 10}
	e := gost.NewExpr()
	obj := gost.NewExpr()
	var xs []gost.Var
	for k := range w {
		x := m.AddVar(0, 1, 0, gost.Binary, "")
		xs = append(xs, x)
		e.Add(w[k], x)
		obj.Add(v[k], x)
	}
	m.AddConstr(e, gost.LessEq, 15, "cap")
	m.SetObjective(obj, gost.Maximize)
	r, err := m.Solve(gost.DefaultParams())
	if err != nil {
		t.Fatal(err)
	}
	if r.Status != gost.StatusOptimal || math.Abs(r.Objective-15) > 1e-6 {
		t.Fatalf("got %v %v, want optimal 15", r.Status, r.Objective)
	}
	if r.Value(xs[0]) > 0.5 {
		t.Fatalf("item 0 should not be selected")
	}
}

func TestInfeasibleAndUnbounded(t *testing.T) {
	m := gost.NewModel("inf")
	x := m.AddVar(0, 10, 1, gost.Integer, "x")
	y := m.AddVar(0, 10, 1, gost.Integer, "y")
	m.AddConstr(gost.NewExpr().Add(2, x).Add(2, y), gost.Equal, 3, "odd")
	r, _ := m.Solve(gost.DefaultParams())
	if r.Status != gost.StatusInfeasible {
		t.Fatalf("expected infeasible, got %v", r.Status)
	}

	m2 := gost.NewModel("unb")
	a := m2.AddVar(0, gost.Inf, -1, gost.Integer, "a")
	b := m2.AddVar(0, gost.Inf, 0, gost.Continuous, "b")
	m2.AddConstr(gost.NewExpr().Add(1, a).Add(-1, b), gost.LessEq, 4, "c")
	r2, _ := m2.Solve(gost.DefaultParams())
	if r2.Status != gost.StatusUnbounded {
		t.Fatalf("expected unbounded, got %v", r2.Status)
	}
}

// bruteForce enumerates all integer points of a small pure-integer model.
func bruteForce(n int, lo, up []int, obj []float64, rows [][]float64, rl, ru []float64) (float64, bool) {
	best := math.Inf(1)
	found := false
	x := make([]int, n)
	copy(x, lo)
	for {
		ok := true
		for i, r := range rows {
			a := 0.0
			for j := 0; j < n; j++ {
				a += r[j] * float64(x[j])
			}
			if a < rl[i]-1e-9 || a > ru[i]+1e-9 {
				ok = false
				break
			}
		}
		if ok {
			o := 0.0
			for j := 0; j < n; j++ {
				o += obj[j] * float64(x[j])
			}
			if o < best {
				best = o
			}
			found = true
		}
		k := 0
		for k < n {
			x[k]++
			if x[k] <= up[k] {
				break
			}
			x[k] = lo[k]
			k++
		}
		if k == n {
			break
		}
	}
	return best, found
}

func TestRandomIntegerVsBruteForce(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	nfeas := 0
	defer func() { t.Logf("%d feasible instances checked", nfeas) }()
	for trial := 0; trial < 1500; trial++ {
		n := 2 + rng.Intn(7)
		mrows := 1 + rng.Intn(6)
		lo := make([]int, n)
		up := make([]int, n)
		obj := make([]float64, n)
		m := gost.NewModel("rand")
		var vars []gost.Var
		for j := 0; j < n; j++ {
			lo[j] = rng.Intn(3) - 1
			up[j] = lo[j] + rng.Intn(5)
			obj[j] = float64(rng.Intn(21) - 10)
			vars = append(vars, m.AddVar(float64(lo[j]), float64(up[j]), obj[j], gost.Integer, ""))
		}
		rows := make([][]float64, mrows)
		rl := make([]float64, mrows)
		ru := make([]float64, mrows)
		for i := 0; i < mrows; i++ {
			rows[i] = make([]float64, n)
			e := gost.NewExpr()
			for j := 0; j < n; j++ {
				if rng.Float64() < 0.7 {
					rows[i][j] = float64(rng.Intn(11) - 5)
					e.Add(rows[i][j], vars[j])
				}
			}
			c := float64(rng.Intn(11) - 3)
			switch rng.Intn(3) {
			case 0:
				rl[i], ru[i] = math.Inf(-1), c
			case 1:
				rl[i], ru[i] = c, math.Inf(1)
			default:
				rl[i], ru[i] = c, c+float64(rng.Intn(3))
			}
			m.AddRange(e, rl[i], ru[i], "")
		}
		want, feas := bruteForce(n, lo, up, obj, rows, rl, ru)
		p := gost.DefaultParams()
		p.RelGap = 1e-9
		r, _ := m.Solve(p)
		if !feas {
			if r.Status != gost.StatusInfeasible {
				t.Fatalf("trial %d: expected infeasible, got %v obj %v", trial, r.Status, r.Objective)
			}
			continue
		}
		nfeas++
		if r.Status != gost.StatusOptimal || math.Abs(r.Objective-want) > 1e-6 {
			t.Fatalf("trial %d: got %v %v, want %v", trial, r.Status, r.Objective, want)
		}
	}
}

func TestRandomMixedVsSelf(t *testing.T) {
	// Mixed problems: solve with all features on and with presolve/cuts/
	// heuristics off; both must agree.
	rng := rand.New(rand.NewSource(11))
	for trial := 0; trial < 200; trial++ {
		n := 5 + rng.Intn(30)
		mr := 3 + rng.Intn(20)
		m := gost.NewModel("mixed")
		var vars []gost.Var
		for j := 0; j < n; j++ {
			typ := gost.Continuous
			if rng.Float64() < 0.6 {
				typ = gost.Integer
			}
			vars = append(vars, m.AddVar(0, float64(1+rng.Intn(10)), float64(rng.Intn(21)-10), typ, ""))
		}
		for i := 0; i < mr; i++ {
			e := gost.NewExpr()
			s := 0.0
			for j := 0; j < n; j++ {
				if rng.Float64() < 0.4 {
					c := float64(rng.Intn(15)-4) + rng.Float64()
					e.Add(c, vars[j])
					s += math.Abs(c)
				}
			}
			m.AddConstr(e, gost.LessEq, s*rng.Float64()*3, "")
		}
		p1 := gost.DefaultParams()
		p1.RelGap = 1e-9
		r1, _ := m.Solve(p1)
		p2 := p1
		p2.Presolve, p2.Cuts, p2.Heuristics, p2.StrongBranchCands = false, false, false, 0
		r2, _ := m.Solve(p2)
		if r1.Status != r2.Status {
			t.Fatalf("trial %d: status %v vs %v", trial, r1.Status, r2.Status)
		}
		if r1.Status == gost.StatusOptimal && math.Abs(r1.Objective-r2.Objective) > 1e-6*math.Max(1, math.Abs(r1.Objective)) {
			t.Fatalf("trial %d: obj %v vs %v", trial, r1.Objective, r2.Objective)
		}
	}
}

func TestMPSRoundTrip(t *testing.T) {
	m, err := mps.ReadFile(filepath.Join("testdata", "p0033.mps.gz"))
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := mps.Write(&buf, m); err != nil {
		t.Fatal(err)
	}
	m2, err := mps.Read(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if m2.NumVars() != m.NumVars() || m2.NumConstrs() != m.NumConstrs() {
		t.Fatalf("size mismatch")
	}
	r, _ := m2.Solve(gost.DefaultParams())
	if math.Abs(r.Objective-3089) > 1e-6 {
		t.Fatalf("roundtrip objective %v", r.Objective)
	}
}

var mipOpt = map[string]float64{
	"p0033": 3089, "p0201": 7615, "p0282": 258411, "p0548": 8691, "lseu": 1120,
	"egout": 568.1007, "enigma": 0, "flugpl": 1201500, "gt2": 21166, "khb05250": 106940226,
	"misc03": 3360, "mod008": 307, "rgn": 82.1999992, "stein27": 18, "vpm1": 20,
	"bell5": 8966406.49152, "dcmulti": 188182, "air03": 340160, "p2756": 3124,
	"fiber": 405935.18, "blend2": 7.598985, "qnet1": 16029.692681, "gesa2": 25779856.372,
}

func TestMIPLIB(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode")
	}
	for name, want := range mipOpt {
		m, err := mps.ReadFile(filepath.Join("testdata", name+".mps.gz"))
		if err != nil {
			t.Fatal(err)
		}
		p := gost.DefaultParams()
		p.TimeLimit = 60 * time.Second
		r, _ := m.Solve(p)
		rel := math.Abs(r.Objective-want) / math.Max(1, math.Abs(want))
		t.Logf("%-9s %-8s obj=%.8g want=%.8g nodes=%d time=%.2fs", name, r.Status, r.Objective, want, r.Nodes, r.Time.Seconds())
		if r.Status != gost.StatusOptimal || rel > 1e-4 {
			t.Errorf("%s: got %v %v want %v", name, r.Status, r.Objective, want)
		}
	}
}

// Without a time limit the solver must be deterministic: same nodes and
// LP iterations on repeated solves.
func TestDeterminism(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode")
	}
	for _, name := range []string{"fiber", "p0201", "vpm1"} {
		var nodes, iters []int
		for rep := 0; rep < 2; rep++ {
			m, err := mps.ReadFile(filepath.Join("testdata", name+".mps.gz"))
			if err != nil {
				t.Fatal(err)
			}
			r, _ := m.Solve(gost.DefaultParams())
			nodes = append(nodes, r.Nodes)
			iters = append(iters, r.LPIters)
		}
		if nodes[0] != nodes[1] || iters[0] != iters[1] {
			t.Errorf("%s not deterministic: nodes %v iters %v", name, nodes, iters)
		}
	}
}
