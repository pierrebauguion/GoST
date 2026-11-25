package mip

import (
	"fmt"
	"math"

	"gost/internal/problem"
)

// StartStatus describes what happened to a user-supplied MIP start.
type StartStatus int

const (
	StartNone     StartStatus = iota // no start supplied
	StartAccepted                    // complete and feasible: used as is
	StartRepaired                    // partial or infeasible: completed by a sub-MIP
	StartRejected                    // infeasible and not repairable
)

func (s StartStatus) String() string {
	return [...]string{"none", "accepted", "repaired", "rejected"}[s]
}

// startIntTol is the distance to the nearest integer accepted for integer
// variables of a start (they are then rounded).
const startIntTol = 1e-5

// startRepairNodes bounds the effort of the repair sub-MIP.
const startRepairNodes = 1000

// prepareStart checks the user start (original space, NaN = unspecified)
// and, if it is partial or infeasible, tries to complete/repair it: the
// specified integer variables are fixed to their rounded values and the
// rest of the problem is solved as a node-limited sub-MIP. It returns a
// feasible solution of p (nil if none), its objective and a status with a
// human-readable reason.
func prepareStart(p *problem.Problem, opt Options) ([]float64, float64, StartStatus, string) {
	start := opt.Start
	if len(start) != p.NumCols {
		return nil, 0, StartRejected, fmt.Sprintf("start has %d values, model has %d variables", len(start), p.NumCols)
	}
	x := make([]float64, p.NumCols)
	given, givenInt := 0, 0
	complete := true
	for j, v := range start {
		if math.IsNaN(v) {
			complete = false
			continue
		}
		given++
		if p.Integer[j] {
			givenInt++
			if r := math.Round(v); math.Abs(v-r) <= startIntTol {
				v = r
			}
		}
		x[j] = v
	}
	reason := ""
	if complete {
		b, r, in := p.MaxViolation(x)
		if feasibleWithin(p, x, opt.FeasTol, opt.IntTol) {
			return x, p.ObjValue(x), StartAccepted, ""
		}
		reason = fmt.Sprintf("infeasible (bound violation %.2g, row violation %.2g, integrality %.2g)", b, r, in)
	} else {
		reason = fmt.Sprintf("partial (%d of %d values given)", given, p.NumCols)
	}
	if opt.Hooks != nil && opt.Hooks.Price != nil {
		return nil, 0, StartRejected, reason + "; no repair with column generation"
	}

	// Repair: fix the specified integer variables, solve the rest.
	q := p.Clone()
	for j, v := range start {
		if math.IsNaN(v) || !p.Integer[j] {
			continue
		}
		r := math.Min(math.Max(math.Round(v), q.Lo[j]), q.Up[j])
		q.Lo[j], q.Up[j] = r, r
	}
	o := DefaultOptions()
	o.noSubMIP = true
	o.noRestart = true
	o.NodeLimit = startRepairNodes
	o.MIRCuts = opt.MIRCuts
	o.FeasTol, o.IntTol = opt.FeasTol, opt.IntTol
	if opt.TimeLimit > 0 {
		o.TimeLimit = opt.TimeLimit / 10
	}
	r := Solve(q, o)
	if r.X != nil {
		return r.X, r.Objective, StartRepaired, fmt.Sprintf("%s; completed with %d integer variables fixed", reason, givenInt)
	}
	stage1 := r.Status

	// Second repair stage: find the feasible solution closest to the start
	// (minimum total change of the given integer values), ignoring the
	// objective. Binary-like variables at a bound get a linear distance;
	// other integers get a deviation column d_j ≥ |x_j − s_j|.
	q = p.Clone()
	n := p.NumCols
	for j := range q.Obj {
		q.Obj[j] = 0
	}
	q.Offset = 0
	var rows []RowSpec
	for j, v := range start {
		if math.IsNaN(v) || !p.Integer[j] {
			continue
		}
		sv := math.Min(math.Max(math.Round(v), p.Lo[j]), p.Up[j])
		switch {
		case sv == p.Lo[j]:
			q.Obj[j] += 1 // distance x_j − s_j
			q.Offset -= sv
		case sv == p.Up[j]:
			q.Obj[j] -= 1 // distance s_j − x_j
			q.Offset += sv
		default:
			d := q.NumCols
			q.Obj = append(q.Obj, 1)
			q.Lo = append(q.Lo, 0)
			q.Up = append(q.Up, math.Inf(1))
			q.Integer = append(q.Integer, false)
			q.ColStart = append(q.ColStart, len(q.RowIdx))
			q.NumCols++
			rows = append(rows,
				RowSpec{Idx: []int{j, d}, Val: []float64{1, -1}, Lo: math.Inf(-1), Up: sv},
				RowSpec{Idx: []int{j, d}, Val: []float64{1, 1}, Lo: sv, Up: math.Inf(1)})
		}
	}
	appendRows(q, rows)
	r = Solve(q, o)
	if r.X == nil {
		return nil, 0, StartRejected, fmt.Sprintf("%s; repair found no solution (fixing the %d given integer values: %s; closest solution: %s)", reason, givenInt, stage1, r.Status)
	}
	x = r.X[:n]
	changed := 0
	for j, v := range start {
		if !math.IsNaN(v) && p.Integer[j] && math.Abs(x[j]-math.Round(v)) > 0.5 {
			changed++
		}
	}
	return x, p.ObjValue(x), StartRepaired, fmt.Sprintf("%s; closest feasible solution changes %d of %d given integer values", reason, changed, givenInt)
}

// feasibleWithin checks x against p with the solver's acceptance rule:
// bounds within tol, rows within tol·max(1, |activity|), integer variables
// within intTol of an integer.
func feasibleWithin(p *problem.Problem, x []float64, tol, intTol float64) bool {
	act := make([]float64, p.NumRows)
	for j := 0; j < p.NumCols; j++ {
		v := x[j]
		if v < p.Lo[j]-tol || v > p.Up[j]+tol || math.IsNaN(v) {
			return false
		}
		if p.Integer[j] && math.Abs(v-math.Round(v)) > intTol {
			return false
		}
		for k := p.ColStart[j]; k < p.ColStart[j+1]; k++ {
			act[p.RowIdx[k]] += p.Val[k] * v
		}
	}
	for i, a := range act {
		sc := tol * math.Max(1, math.Abs(a))
		if a < p.RowLo[i]-sc || a > p.RowUp[i]+sc {
			return false
		}
	}
	return true
}
