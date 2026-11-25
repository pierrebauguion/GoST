package mip

import (
	"math"

	"gost/internal/problem"
	"gost/internal/simplex"
)

// RowSpec is a constraint lo <= sum val*x[idx] <= up (problem space).
type RowSpec struct {
	Idx    []int
	Val    []float64
	Lo, Up float64
}

// ColSpec is a new variable (problem space, minimization costs).
type ColSpec struct {
	Obj     float64
	Lo, Up  float64
	Integer bool
	Rows    []int
	Vals    []float64
}

// PriceInfo is passed to the pricing hook.
type PriceInfo struct {
	Duals []float64 // dual value of each problem row (minimization sense)
	X     []float64 // current LP solution
	Obj   float64   // LP objective (without offset)
	Depth int
	Node  int
}

// Hooks are user callbacks (problem space; presolve is disabled when any
// hook is set so indices match the caller's model).
type Hooks struct {
	// Lazy is called with every candidate solution that satisfies the known
	// rows. Returning rows rejects the candidate and adds them globally.
	Lazy func(x []float64) []RowSpec
	// Cut is called with fractional LP solutions; returned rows are added
	// globally (they must be valid for all integer solutions).
	Cut func(x []float64, depth, node int) []RowSpec
	// Price is called after each LP solve; returned columns are added and
	// the LP re-solved until no column is returned. The LP bound is only
	// used for pruning once pricing has converged.
	Price func(info PriceInfo) []ColSpec
}

func (h *Hooks) any() bool { return h != nil && (h.Lazy != nil || h.Cut != nil || h.Price != nil) }

func (s *solver) pricing() bool { return s.opt.Hooks != nil && s.opt.Hooks.Price != nil }

// addRows appends rows to the problem and the LP.
func (s *solver) addRows(rows []RowSpec) {
	if len(rows) == 0 {
		return
	}
	p := s.p
	idx := make([][]int, len(rows))
	val := make([][]float64, len(rows))
	rl := make([]float64, len(rows))
	ru := make([]float64, len(rows))
	base := s.lp.NumRows()
	for k, r := range rows {
		idx[k], val[k], rl[k], ru[k] = r.Idx, r.Val, r.Lo, r.Up
		p.RowLo = append(p.RowLo, r.Lo)
		p.RowUp = append(p.RowUp, r.Up)
		s.rowCol = append(s.rowCol, r.Idx...)
		s.rowVal = append(s.rowVal, r.Val...)
		s.rowStart = append(s.rowStart, len(s.rowCol))
		s.lpRowOf = append(s.lpRowOf, base+k)
		s.lpRowIsCut = append(s.lpRowIsCut, false)
	}
	s.m += len(rows)
	p.NumRows = s.m
	s.rebuildCSC()
	s.lp.AddRows(idx, val, rl, ru)
	s.dynPending = "rows"
	s.computeLocks()
	s.lpChanged = true
	s.lastBasis = nil
}

// addCols appends variables to the problem and the LP.
func (s *solver) addCols(cols []ColSpec) {
	if len(cols) == 0 {
		return
	}
	p := s.p
	cost := make([]float64, len(cols))
	lb := make([]float64, len(cols))
	ub := make([]float64, len(cols))
	idx := make([][]int, len(cols))
	val := make([][]float64, len(cols))
	for k, c := range cols {
		j := s.n + k
		cost[k], lb[k], ub[k] = c.Obj, c.Lo, c.Up
		// LP rows of the problem rows
		for t, i := range c.Rows {
			idx[k] = append(idx[k], s.lpRowOf[i])
			val[k] = append(val[k], c.Vals[t])
		}
		p.Obj = append(p.Obj, c.Obj)
		p.Lo = append(p.Lo, c.Lo)
		p.Up = append(p.Up, c.Up)
		p.Integer = append(p.Integer, c.Integer)
		p.RowIdx = append(p.RowIdx, c.Rows...)
		p.Val = append(p.Val, c.Vals...)
		p.ColStart = append(p.ColStart, len(p.RowIdx))
		s.lo0 = append(s.lo0, c.Lo)
		s.up0 = append(s.up0, c.Up)
		s.lo = append(s.lo, c.Lo)
		s.up = append(s.up, c.Up)
		s.lpLo = append(s.lpLo, c.Lo)
		s.lpUp = append(s.lpUp, c.Up)
		s.x = append(s.x, 0)
		s.pcSumDown = append(s.pcSumDown, 0)
		s.pcSumUp = append(s.pcSumUp, 0)
		s.pcCntDown = append(s.pcCntDown, 0)
		s.pcCntUp = append(s.pcCntUp, 0)
		if c.Integer {
			s.intVars = append(s.intVars, j)
		}
		if s.hasIncumbent {
			v := math.Max(c.Lo, math.Min(0, c.Up))
			s.incumbent = append(s.incumbent, v)
		}
	}
	s.n += len(cols)
	p.NumCols = s.n
	s.isInt = p.Integer
	s.debugSol = nil
	s.rebuildCSR()
	s.lp.AddCols(cost, lb, ub, idx, val)
	s.dynPending = "cols"
	s.computeLocks()
	s.detectObjIntegral()
	s.lpChanged = true
	s.lastBasis = nil
	s.numPricedCols += len(cols)
}

// rebuildCSC recomputes the column-wise problem matrix from the rows.
func (s *solver) rebuildCSC() {
	p := s.p
	n := s.n
	cnt := make([]int, n+1)
	for _, j := range s.rowCol {
		cnt[j+1]++
	}
	for j := 0; j < n; j++ {
		cnt[j+1] += cnt[j]
	}
	p.ColStart = append(p.ColStart[:0], cnt...)
	nnz := len(s.rowCol)
	p.RowIdx = make([]int, nnz)
	p.Val = make([]float64, nnz)
	next := append([]int(nil), cnt[:n]...)
	for i := 0; i < s.m; i++ {
		for k := s.rowStart[i]; k < s.rowStart[i+1]; k++ {
			j := s.rowCol[k]
			p.RowIdx[next[j]] = i
			p.Val[next[j]] = s.rowVal[k]
			next[j]++
		}
	}
}

// rebuildCSR recomputes the row-wise copy from the columns.
func (s *solver) rebuildCSR() {
	s.rowStart, s.rowCol, s.rowVal = s.p.RowWise()
}

// problemDuals returns the duals of the problem rows.
func (s *solver) problemDuals() []float64 {
	y := make([]float64, s.m)
	for i := range y {
		y[i] = s.lp.Dual(s.lpRowOf[i])
	}
	return y
}

// priceLoop calls the pricing hook until no column is returned. It returns
// the final LP status.
func (s *solver) priceLoop(st simplex.Status, depth int) simplex.Status {
	if !s.pricing() {
		return st
	}
	for round := 0; st == simplex.Optimal && round < 100000; round++ {
		if s.timeUp() {
			return simplex.TimeLimit
		}
		s.lp.Primal(s.x)
		cols := s.opt.Hooks.Price(PriceInfo{
			Duals: s.problemDuals(), X: append([]float64(nil), s.x...),
			Obj: s.lp.Objective(), Depth: depth, Node: s.nodes,
		})
		if len(cols) == 0 {
			break
		}
		s.addCols(cols)
		st = s.dynSolve()
	}
	return st
}

// appendRows adds rows to a problem (rebuilding its CSC matrix).
func appendRows(q *problem.Problem, rows []RowSpec) {
	if len(rows) == 0 {
		return
	}
	rs, rc, rv := q.RowWise()
	for _, r := range rows {
		rc = append(rc, r.Idx...)
		rv = append(rv, r.Val...)
		rs = append(rs, len(rc))
		q.RowLo = append(q.RowLo, r.Lo)
		q.RowUp = append(q.RowUp, r.Up)
	}
	q.NumRows += len(rows)
	n := q.NumCols
	cnt := make([]int, n+1)
	for _, j := range rc {
		cnt[j+1]++
	}
	for j := 0; j < n; j++ {
		cnt[j+1] += cnt[j]
	}
	q.ColStart = cnt
	q.RowIdx = make([]int, len(rc))
	q.Val = make([]float64, len(rc))
	next := append([]int(nil), cnt[:n]...)
	for i := 0; i < q.NumRows; i++ {
		for k := rs[i]; k < rs[i+1]; k++ {
			j := rc[k]
			q.RowIdx[next[j]] = i
			q.Val[next[j]] = rv[k]
			next[j]++
		}
	}
}

// dynSolve re-solves the LP after rows or columns were added, recording
// warm-start statistics (DynStats).
func (s *solver) dynSolve() simplex.Status {
	lp := s.lp
	p1, pr, du := lp.StatPhase1Iter, lp.StatPrimalIter, lp.StatDualIter
	st := lp.Solve()
	if s.dynPending != "" {
		DynStats.record(s.dynPending, lp.Iter, lp.StatPhase1Iter-p1, lp.StatPrimalIter-pr, lp.StatDualIter-du)
		s.dynPending = ""
	}
	return st
}

// DynStats accumulates, for the LP re-solves that follow added rows or
// columns, the number of re-solves and of simplex iterations by phase.
// Debug/test helper (not thread-safe).
var DynStats dynStats

type dynStats struct {
	Resolves, Iters, MaxIters, Phase1, Primal, Dual map[string]int
}

func (d *dynStats) Reset() {
	*d = dynStats{map[string]int{}, map[string]int{}, map[string]int{}, map[string]int{}, map[string]int{}, map[string]int{}}
}

func (d *dynStats) record(kind string, it, p1, pr, du int) {
	if d.Resolves == nil {
		d.Reset()
	}
	d.Resolves[kind]++
	d.Iters[kind] += it
	d.MaxIters[kind] = max(d.MaxIters[kind], it)
	d.Phase1[kind] += p1
	d.Primal[kind] += pr
	d.Dual[kind] += du
}
