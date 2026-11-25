package gost

import (
	"fmt"
	"os"
	"time"

	"gost/internal/simplex"
)

// LPResult is the result of solving the LP relaxation.
type LPResult struct {
	Status     string
	Objective  float64
	X          []float64
	Iterations int
	Time       time.Duration
}

// SolveRelaxation solves the continuous relaxation of the model (no
// presolve) with the simplex method.
func (m *Model) SolveRelaxation(timeLimit ...time.Duration) LPResult {
	t0 := time.Now()
	p := m.toProblem()
	lp := simplex.New(p.NumCols, p.NumRows, p.Obj, p.Lo, p.Up, p.ColStart, p.RowIdx, p.Val, p.RowLo, p.RowUp)
	lp.Debug = os.Getenv("GOST_LPDEBUG") != ""
	if len(timeLimit) > 0 && timeLimit[0] > 0 {
		lp.Deadline = t0.Add(timeLimit[0])
	}
	st := lp.Solve()
	if os.Getenv("GOST_LPSTATS") != "" {
		fmt.Fprintf(os.Stderr, "stats: trouble %d, updfail %d, polish %d, phase1 %d, dual %d calls/%d it, primal %d calls/%d it, perturb %d\n",
			lp.StatTrouble, lp.StatUpdFail, lp.StatPolishIter, lp.StatPhase1Iter, lp.StatDualCalls, lp.StatDualIter, lp.StatPrimalCalls, lp.StatPrimalIter, lp.StatPerturb)
	}
	x := make([]float64, p.NumCols)
	lp.Primal(x)
	if os.Getenv("GOST_LPSTATS") != "" {
		b, r, _ := p.MaxViolation(x)
		fmt.Fprintf(os.Stderr, "violation: bound %g row %g\n", b, r)
	}
	obj := lp.Objective() + p.Offset
	if m.sense == Maximize {
		obj = -obj
	}
	return LPResult{Status: st.String(), Objective: obj, X: x, Iterations: lp.TotalIter, Time: time.Since(t0)}
}
