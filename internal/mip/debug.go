package mip

import (
	"fmt"
	"gost/internal/problem"
	"math"
	"os"
)

// debugCheckCut reports if a cut is violated by the debug solution.
func (s *solver) debugCheckCut(c cut, kind string) {
	if s.debugSol == nil {
		return
	}
	act := 0.0
	for k, j := range c.idx {
		act += c.val[k] * s.debugSol[j]
	}
	if act < c.rhs-1e-6*math.Max(1, math.Abs(c.rhs)) {
		fmt.Fprintf(os.Stderr, "DEBUG: invalid %s cut: act %.9g < rhs %.9g (nnz %d)\n", kind, act, c.rhs, len(c.idx))
	}
}

// debugInBounds reports whether the debug solution lies within lo/up.
func (s *solver) debugInBounds(lo, up []float64) bool {
	if s.debugSol == nil {
		return false
	}
	for j, v := range s.debugSol {
		if v < lo[j]-1e-6 || v > up[j]+1e-6 {
			return false
		}
	}
	return true
}

func (s *solver) debugf(format string, a ...any) {
	if s.debugSol != nil {
		fmt.Fprintf(os.Stderr, "DEBUG: "+format+"\n", a...)
	}
}

// debugCheckLPRows reports LP rows violated by the debug solution.
func (s *solver) debugCheckLPRows() {
	if s.debugSol == nil {
		return
	}
	for i := 0; i < s.lp.NumRows(); i++ {
		idx, val := s.lp.RowCoefs(i)
		a := 0.0
		for k, j := range idx {
			a += val[k] * s.debugSol[j]
		}
		l, u := s.lp.VarBounds(s.n + i)
		if a < l-1e-6*math.Max(1, math.Abs(l)) || a > u+1e-6*math.Max(1, math.Abs(u)) {
			fmt.Fprintf(os.Stderr, "DEBUG: LP row %d (orig m=%d) violated: %g not in [%g,%g]\n", i, s.m, a, l, u)
		}
	}
	for j := 0; j < s.n; j++ {
		l, u := s.lp.VarBounds(j)
		if s.debugSol[j] < l-1e-6 || s.debugSol[j] > u+1e-6 {
			fmt.Fprintf(os.Stderr, "DEBUG: LP col %d bound violated: %g not in [%g,%g]\n", j, s.debugSol[j], l, u)
		}
	}
}

// debugWriteLP writes the current LP (with current bounds) as free MPS.
func (s *solver) debugWriteLP(path string) {
	f, err := os.Create(path)
	if err != nil {
		return
	}
	defer f.Close()
	mm := s.lp.NumRows()
	fmt.Fprintf(f, "NAME DEBUGLP\nROWS\n N OBJ\n")
	type e struct {
		i int
		v float64
	}
	cols := make([][]e, s.n)
	for i := 0; i < mm; i++ {
		l, u := s.lp.VarBounds(s.n + i)
		t := "G"
		if math.IsInf(l, -1) {
			t = "L"
		}
		if l == u {
			t = "E"
		}
		_ = u
		fmt.Fprintf(f, " %s R%d\n", t, i)
		idx, val := s.lp.RowCoefs(i)
		for k, j := range idx {
			cols[j] = append(cols[j], e{i, val[k]})
		}
	}
	fmt.Fprintf(f, "COLUMNS\n")
	for j := 0; j < s.n; j++ {
		fmt.Fprintf(f, " C%d OBJ %.17g\n", j, s.p.Obj[j])
		for _, x := range cols[j] {
			fmt.Fprintf(f, " C%d R%d %.17g\n", j, x.i, x.v)
		}
	}
	fmt.Fprintf(f, "RHS\n")
	for i := 0; i < mm; i++ {
		l, u := s.lp.VarBounds(s.n + i)
		if !math.IsInf(l, -1) {
			fmt.Fprintf(f, " RHS R%d %.17g\n", i, l)
		} else {
			fmt.Fprintf(f, " RHS R%d %.17g\n", i, u)
		}
	}
	fmt.Fprintf(f, "RANGES\n")
	for i := 0; i < mm; i++ {
		l, u := s.lp.VarBounds(s.n + i)
		if !math.IsInf(u, 1) && !math.IsInf(l, -1) && l != u {
			fmt.Fprintf(f, " RNG R%d %.17g\n", i, u-l)
		}
	}
	fmt.Fprintf(f, "BOUNDS\n")
	for j := 0; j < s.n; j++ {
		l, u := s.lp.VarBounds(j)
		if math.IsInf(l, -1) {
			fmt.Fprintf(f, " MI BND C%d\n", j)
		} else {
			fmt.Fprintf(f, " LO BND C%d %.17g\n", j, l)
		}
		if !math.IsInf(u, 1) {
			fmt.Fprintf(f, " UP BND C%d %.17g\n", j, u)
		}
	}
	fmt.Fprintf(f, "ENDATA\n")
}

// debugWriteProblem writes a problem in free MPS (development aid).
func debugWriteProblem(q *problem.Problem, path string) {
	f, err := os.Create(path)
	if err != nil {
		return
	}
	defer f.Close()
	fmt.Fprintf(f, "NAME DBG\nROWS\n N OBJ\n")
	for i := 0; i < q.NumRows; i++ {
		t := "E"
		switch {
		case q.RowLo[i] == q.RowUp[i]:
		case math.IsInf(q.RowLo[i], -1):
			t = "L"
		default:
			t = "G"
		}
		fmt.Fprintf(f, " %s R%d\n", t, i)
	}
	fmt.Fprintf(f, "COLUMNS\n")
	inInt := false
	for j := 0; j < q.NumCols; j++ {
		if q.Integer[j] != inInt {
			if q.Integer[j] {
				fmt.Fprintf(f, " M1 'MARKER' 'INTORG'\n")
			} else {
				fmt.Fprintf(f, " M2 'MARKER' 'INTEND'\n")
			}
			inInt = q.Integer[j]
		}
		fmt.Fprintf(f, " C%d OBJ %.17g\n", j, q.Obj[j])
		for k := q.ColStart[j]; k < q.ColStart[j+1]; k++ {
			fmt.Fprintf(f, " C%d R%d %.17g\n", j, q.RowIdx[k], q.Val[k])
		}
	}
	if inInt {
		fmt.Fprintf(f, " M2 'MARKER' 'INTEND'\n")
	}
	fmt.Fprintf(f, "RHS\n")
	for i := 0; i < q.NumRows; i++ {
		v := q.RowLo[i]
		if math.IsInf(v, -1) {
			v = q.RowUp[i]
		}
		fmt.Fprintf(f, " RHS R%d %.17g\n", i, v)
	}
	fmt.Fprintf(f, "RANGES\n")
	for i := 0; i < q.NumRows; i++ {
		if !math.IsInf(q.RowLo[i], -1) && !math.IsInf(q.RowUp[i], 1) && q.RowLo[i] != q.RowUp[i] {
			fmt.Fprintf(f, " RNG R%d %.17g\n", i, q.RowUp[i]-q.RowLo[i])
		}
	}
	fmt.Fprintf(f, "BOUNDS\n")
	for j := 0; j < q.NumCols; j++ {
		if math.IsInf(q.Lo[j], -1) {
			fmt.Fprintf(f, " MI BND C%d\n", j)
		} else {
			fmt.Fprintf(f, " LO BND C%d %.17g\n", j, q.Lo[j])
		}
		if !math.IsInf(q.Up[j], 1) {
			fmt.Fprintf(f, " UP BND C%d %.17g\n", j, q.Up[j])
		} else if q.Integer[j] {
			fmt.Fprintf(f, " PL BND C%d\n", j)
		}
	}
	fmt.Fprintf(f, "ENDATA\n")
}

// traceHash prints a hash of the LP basis (determinism debugging).
func (s *solver) traceHash(label string) {
	if os.Getenv("GOST_DIVETRACE") == "" || s.lp == nil {
		return
	}
	b := s.lp.GetBasis()
	h := 0
	for k, st := range b.Status {
		h = h*31 + int(st)*(k+1)
	}
	fmt.Fprintf(os.Stderr, "TRACE %s basis %d rows %d obj %.12g iters %d\n", label, h, s.lp.NumRows(), s.lp.Objective(), s.lp.TotalIter)
}
