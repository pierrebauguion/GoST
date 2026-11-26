// Job shop scheduling: the classic ft06 instance of Fisher and Thompson
// (1963), 6 jobs × 6 machines, whose optimal makespan is known to be 55.
//
// Each job is a sequence of operations; operation k of job j must run on
// machine m_jk for p_jk time units, after operation k−1 of the same job has
// finished. A machine processes one operation at a time. We minimise the
// makespan (completion time of the last operation).
//
// Disjunctive model (Manne, 1960):
//
//	min  Cmax
//	s.t. s_j,k+1 ≥ s_jk + p_jk                        (job precedence)
//	     Cmax ≥ s_j,last + p_j,last                   (makespan)
//	     for two operations (j,k), (j',k') on the same machine:
//	       s_jk   ≥ s_j'k' + p_j'k' − M·z            (j' before j  if z = 0)
//	       s_j'k' ≥ s_jk   + p_jk   − M·(1 − z)      (j before j'  if z = 1)
//	     s ≥ 0, z ∈ {0, 1}
//
// s_jk is the start time of operation k of job j, and z decides the order of
// each pair of operations sharing a machine (a "disjunction"). M is a big
// number: any valid upper bound on the makespan (here the sum of all
// processing times) makes the inactive constraint redundant.
//
// The schedule is then checked independently and printed as a text Gantt
// chart.
//
//	go run ./examples/jobshop
package main

import (
	"fmt"
	"os"
	"strings"

	"gost"
)

// ft06[j]: the operations of job j in order, as (machine, processing time).
var ft06 = [][][2]int{
	{{2, 1}, {0, 3}, {1, 6}, {3, 7}, {5, 3}, {4, 6}},
	{{1, 8}, {2, 5}, {4, 10}, {5, 10}, {0, 10}, {3, 4}},
	{{2, 5}, {3, 4}, {5, 8}, {0, 9}, {1, 1}, {4, 7}},
	{{1, 5}, {0, 5}, {2, 5}, {3, 3}, {4, 8}, {5, 9}},
	{{2, 9}, {1, 3}, {4, 5}, {5, 4}, {0, 3}, {3, 1}},
	{{1, 3}, {3, 3}, {5, 9}, {0, 10}, {4, 4}, {2, 1}},
}

// knownOptimum: optimal makespan of ft06 reported in the literature.
const knownOptimum = 55

// op identifies operation k of job j.
type op struct{ j, k int }

func main() {
	nJobs, nMach := len(ft06), 6

	// bigM: sum of all processing times, an upper bound on any makespan
	// without idle time, hence large enough for the disjunctions.
	bigM := 0.0
	for _, job := range ft06 {
		for _, o := range job {
			bigM += float64(o[1])
		}
	}

	m := gost.NewModel("jobshop-ft06")

	// --- Variables -------------------------------------------------------------
	// s[j][k]: start time of operation k of job j (continuous, ≥ 0).
	s := make([][]gost.Var, nJobs)
	for j, job := range ft06 {
		s[j] = make([]gost.Var, len(job))
		for k := range job {
			s[j][k] = m.AddVar(0, bigM, 0, gost.Continuous, fmt.Sprintf("s_%d_%d", j, k))
		}
	}
	// cmax: makespan, the only term of the objective.
	cmax := m.AddVar(0, bigM, 1, gost.Continuous, "Cmax")

	// --- Precedence inside each job and makespan definition --------------------
	for j, job := range ft06 {
		for k := 0; k+1 < len(job); k++ {
			// prec_j_k: operation k+1 starts after operation k ends.
			m.AddConstr(gost.NewExpr().Add(1, s[j][k+1]).Add(-1, s[j][k]), gost.GreaterEq, float64(job[k][1]),
				fmt.Sprintf("prec_%d_%d", j, k))
		}
		last := len(job) - 1
		// end_j: the makespan is at least the end of the last operation.
		m.AddConstr(gost.NewExpr().Add(1, cmax).Add(-1, s[j][last]), gost.GreaterEq, float64(job[last][1]),
			fmt.Sprintf("end_%d", j))
	}

	// --- Disjunctions on each machine -----------------------------------------
	// onMach[mc]: the operations that use machine mc.
	onMach := make([][]op, nMach)
	for j, job := range ft06 {
		for k, o := range job {
			onMach[o[0]] = append(onMach[o[0]], op{j, k})
		}
	}
	nDisj := 0
	for mc, ops := range onMach {
		for a := 0; a < len(ops); a++ {
			for b := a + 1; b < len(ops); b++ {
				oa, ob := ops[a], ops[b]
				pa, pb := float64(ft06[oa.j][oa.k][1]), float64(ft06[ob.j][ob.k][1])
				// z = 1: oa before ob; z = 0: ob before oa.
				z := m.AddVar(0, 1, 0, gost.Binary, fmt.Sprintf("z_m%d_%d_%d", mc, oa.j, ob.j))
				// s_a ≥ s_b + p_b − M·z   (active when z = 0)
				m.AddConstr(gost.NewExpr().Add(1, s[oa.j][oa.k]).Add(-1, s[ob.j][ob.k]).Add(bigM, z),
					gost.GreaterEq, pb, fmt.Sprintf("disjA_m%d_%d_%d", mc, oa.j, ob.j))
				// s_b ≥ s_a + p_a − M·(1 − z)   (active when z = 1)
				m.AddConstr(gost.NewExpr().Add(1, s[ob.j][ob.k]).Add(-1, s[oa.j][oa.k]).Add(-bigM, z),
					gost.GreaterEq, pa-bigM, fmt.Sprintf("disjB_m%d_%d_%d", mc, oa.j, ob.j))
				nDisj++
			}
		}
	}
	m.SetObjective(gost.NewExpr().Add(1, cmax), gost.Minimize)

	// --- Solve -----------------------------------------------------------------
	p := gost.DefaultParams()
	p.Log = os.Stdout
	res, err := m.Solve(p)
	if err != nil {
		panic(err)
	}
	fmt.Printf("\nft06: %s, makespan %.0f (known optimum %d), %d disjunctions, %d nodes, %v\n",
		res.Status, res.Objective, knownOptimum, nDisj, res.Nodes, res.Time)

	// --- Independent check of the schedule --------------------------------------
	// start[j][k]: rounded start times (all data are integers).
	start := make([][]int, nJobs)
	makespan := 0
	ok := true
	for j, job := range ft06 {
		start[j] = make([]int, len(job))
		for k := range job {
			start[j][k] = int(res.Value(s[j][k]) + 0.5)
		}
		for k := 0; k+1 < len(job); k++ {
			if start[j][k+1] < start[j][k]+job[k][1] {
				ok = false
			}
		}
		makespan = max(makespan, start[j][len(job)-1]+job[len(job)-1][1])
	}
	for _, ops := range onMach {
		for a := 0; a < len(ops); a++ {
			for b := a + 1; b < len(ops); b++ {
				sa, ea := start[ops[a].j][ops[a].k], start[ops[a].j][ops[a].k]+ft06[ops[a].j][ops[a].k][1]
				sb, eb := start[ops[b].j][ops[b].k], start[ops[b].j][ops[b].k]+ft06[ops[b].j][ops[b].k][1]
				if sa < eb && sb < ea { // the two intervals overlap
					ok = false
				}
			}
		}
	}
	fmt.Printf("independent check: valid schedule = %v, makespan %d\n\n", ok, makespan)

	// --- Text Gantt chart: one line per machine, one character per time unit ---
	for mc := 0; mc < nMach; mc++ {
		line := []byte(strings.Repeat(".", makespan))
		for _, o := range onMach[mc] {
			for t := start[o.j][o.k]; t < start[o.j][o.k]+ft06[o.j][o.k][1]; t++ {
				line[t] = byte('0' + o.j) // the job number
			}
		}
		fmt.Printf("machine %d |%s|\n", mc, line)
	}
}
