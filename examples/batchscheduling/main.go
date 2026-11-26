// Scheduling a batch processing machine (e.g. a burn-in oven in
// semiconductor testing, a heat-treatment furnace, an autoclave).
//
// Jobs of different sizes s_j and processing times p_j are loaded together
// into a machine of capacity B. All jobs of a batch start and end together,
// so a batch lasts as long as its longest job. We minimise the makespan,
// i.e. the sum of the batch durations (Uzsoy, 1994).
//
// Two formulations are compared.
//
// 1. Naive formulation: batches b = 0..n−1, P_b = duration of batch b.
//
//	min  Σ_b P_b
//	s.t. Σ_b x_jb = 1                for each job j       (assignment)
//	     Σ_j s_j·x_jb ≤ B            for each batch b     (capacity)
//	     P_b ≥ p_j·x_jb              for each j, b        (duration = max)
//	     x ∈ {0, 1}, P ≥ 0
//
// Its LP relaxation is very weak (x_jb can be fractional so that P_b stays
// small) and all batches are interchangeable (symmetry).
//
// 2. "Representative" formulation: jobs are sorted by decreasing
// processing time, and each batch is named after its longest job, which is
// then the job with the smallest index in the batch. Variable x_jb (b ≤ j)
// means "job j is in the batch represented by job b", x_bb = 1 means "job
// b opens a batch", whose duration is then exactly p_b:
//
//	min  Σ_b p_b·x_bb
//	s.t. Σ_{b ≤ j} x_jb = 1           for each job j       (assignment)
//	     Σ_{j ≥ b} s_j·x_jb ≤ B·x_bb  for each b           (capacity)
//	     x_jb ≤ x_bb                 for each b < j       (needs an open batch)
//	     x ∈ {0, 1}
//
// No duration variable, no max to linearise, and no symmetry left: each
// batch has a unique name. The LP bound is much stronger.
//
// The heuristic "First Fit Longest Processing Time" (FFLPT) is shown for
// comparison, and the optimal schedule is checked independently.
//
//	go run ./examples/batchscheduling
package main

import (
	"fmt"
	"math/rand"
	"sort"
	"time"

	"gost"
)

// job: size s (capacity used in the machine) and processing time p.
type job struct{ s, p float64 }

func main() {
	// Fixed seed: the same instance on every run.
	rng := rand.New(rand.NewSource(3))
	const n = 25   // number of jobs
	const B = 20.0 // machine capacity

	// jobs, sorted by decreasing processing time (needed by the
	// representative formulation and by FFLPT).
	jobs := make([]job, n)
	for j := range jobs {
		jobs[j] = job{s: float64(1 + rng.Intn(10)), p: float64(1 + rng.Intn(20))}
	}
	sort.SliceStable(jobs, func(a, b int) bool { return jobs[a].p > jobs[b].p })

	// --- Heuristic: First Fit Longest Processing Time --------------------------
	// Jobs in decreasing processing time, each into the first batch where it
	// fits. The first job of a batch is its longest one.
	var loads, durs []float64
	for _, jb := range jobs {
		placed := false
		for b := range loads {
			if loads[b]+jb.s <= B {
				loads[b] += jb.s
				placed = true
				break
			}
		}
		if !placed {
			loads = append(loads, jb.s)
			durs = append(durs, jb.p)
		}
	}
	ffltp := 0.0
	for _, v := range durs {
		ffltp += v
	}
	fmt.Printf("%d jobs, machine capacity %.0f: FFLPT heuristic makespan %.0f (%d batches)\n\n", n, B, ffltp, len(durs))

	// --- 1. Naive formulation ---------------------------------------------------
	{
		m := gost.NewModel("batching-naive")
		// x[j][b]: job j in batch b; P[b]: duration of batch b.
		x := make([][]gost.Var, n)
		P := make([]gost.Var, n)
		for b := range P {
			P[b] = m.AddVar(0, gost.Inf, 1, gost.Continuous, fmt.Sprintf("P%d", b))
		}
		for j := range x {
			x[j] = make([]gost.Var, n)
			for b := range x[j] {
				x[j][b] = m.AddVar(0, 1, 0, gost.Binary, fmt.Sprintf("x%d_%d", j, b))
			}
		}
		for j := 0; j < n; j++ {
			// assign_j: every job in exactly one batch.
			e := gost.NewExpr()
			for b := 0; b < n; b++ {
				e.Add(1, x[j][b])
			}
			m.AddConstr(e, gost.Equal, 1, fmt.Sprintf("assign%d", j))
		}
		for b := 0; b < n; b++ {
			// capacity_b: the jobs of batch b fit in the machine.
			e := gost.NewExpr()
			for j := 0; j < n; j++ {
				e.Add(jobs[j].s, x[j][b])
				// dur_j_b: P_b ≥ p_j·x_jb (the batch lasts as long as its
				// longest job).
				m.AddConstr(gost.NewExpr().Add(1, P[b]).Add(-jobs[j].p, x[j][b]), gost.GreaterEq, 0, fmt.Sprintf("dur%d_%d", j, b))
			}
			m.AddConstr(e, gost.LessEq, B, fmt.Sprintf("capacity%d", b))
		}
		p := gost.DefaultParams()
		p.TimeLimit = 5 * time.Second // this formulation is hopeless: stop it
		lp := m.SolveRelaxation()
		res, _ := m.Solve(p)
		fmt.Printf("naive formulation:          %-10s makespan %.0f, bound %.1f (LP relaxation %.1f), %d nodes, %v\n",
			res.Status, res.Objective, res.Bound, lp.Objective, res.Nodes, res.Time.Round(1e6))
	}

	// --- 2. Representative formulation -----------------------------------------
	m := gost.NewModel("batching-representative")
	// x[j][b] for b ≤ j: job j in the batch represented by job b. The
	// diagonal x[b][b] opens batch b and carries its duration p_b.
	x := make([][]gost.Var, n)
	for j := range x {
		x[j] = make([]gost.Var, j+1)
		for b := 0; b <= j; b++ {
			cost := 0.0
			if b == j {
				cost = jobs[b].p
			}
			x[j][b] = m.AddVar(0, 1, cost, gost.Binary, fmt.Sprintf("x%d_%d", j, b))
		}
	}
	for j := 0; j < n; j++ {
		// assign_j: job j is in exactly one batch, represented by a job
		// that is at least as long (b ≤ j).
		e := gost.NewExpr()
		for b := 0; b <= j; b++ {
			e.Add(1, x[j][b])
		}
		m.AddConstr(e, gost.Equal, 1, fmt.Sprintf("assign%d", j))
	}
	for b := 0; b < n; b++ {
		// capacity_b: the jobs of batch b fit, and only if batch b is open.
		e := gost.NewExpr().Add(-B, x[b][b])
		for j := b; j < n; j++ {
			e.Add(jobs[j].s, x[j][b])
			if j > b {
				// open_j_b: a job can only join an open batch.
				m.AddConstr(gost.NewExpr().Add(1, x[j][b]).Add(-1, x[b][b]), gost.LessEq, 0, fmt.Sprintf("open%d_%d", j, b))
			}
		}
		m.AddConstr(e, gost.LessEq, 0, fmt.Sprintf("capacity%d", b))
	}
	lp := m.SolveRelaxation()
	res, err := m.Solve(gost.DefaultParams())
	if err != nil {
		panic(err)
	}
	fmt.Printf("representative formulation: %-10s makespan %.0f, bound %.1f (LP relaxation %.1f), %d nodes, %v\n\n",
		res.Status, res.Objective, res.Bound, lp.Objective, res.Nodes, res.Time.Round(1e6))

	// --- Independent check and schedule -----------------------------------------
	t := 0.0 // current time: batches run one after another
	seen := make([]int, n)
	for b := 0; b < n; b++ {
		if res.Value(x[b][b]) < 0.5 {
			continue
		}
		var members []int
		size, dur := 0.0, 0.0
		for j := b; j < n; j++ {
			if res.Value(x[j][b]) > 0.5 {
				members = append(members, j)
				seen[j]++
				size += jobs[j].s
				dur = max(dur, jobs[j].p)
			}
		}
		flag := ""
		if size > B {
			flag = "  ERROR: over capacity"
		}
		fmt.Printf("batch [%3.0f, %3.0f)  size %2.0f/%.0f  jobs (size×time):", t, t+dur, size, B)
		for _, j := range members {
			fmt.Printf(" %.0f×%.0f", jobs[j].s, jobs[j].p)
		}
		fmt.Println(flag)
		t += dur
	}
	for j, c := range seen {
		if c != 1 {
			fmt.Printf("ERROR: job %d scheduled %d times\n", j, c)
		}
	}
	fmt.Printf("independent check: makespan %.0f\n", t)
}
