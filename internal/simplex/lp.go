// Package simplex implements a bounded revised simplex method (dual and
// primal) on problems of the form
//
//	min  c^T x
//	s.t. rl <= A x <= ru
//	     l  <= x   <= u
//
// Internally each row i gets a logical variable s_i with A x - s = 0 and
// rl_i <= s_i <= ru_i. Variables 0..n-1 are structural, n..n+m-1 logical.
//
// The problem is scaled internally (geometric + equilibration, powers of
// two); all public accessors use the original (unscaled) units.
package simplex

import (
	"math"
	"math/rand"
	"time"

	"gost/internal/lu"
)

// Variable status values.
const (
	Basic   int8 = 0
	AtLower int8 = 1
	AtUpper int8 = 2
	AtZero  int8 = 3 // free nonbasic at zero
)

// Status of a solve.
type Status int

const (
	Unknown Status = iota
	Optimal
	Infeasible
	Unbounded
	IterLimit
	TimeLimit
	Cutoff
	Numerical
)

func (s Status) String() string {
	switch s {
	case Optimal:
		return "optimal"
	case Infeasible:
		return "infeasible"
	case Unbounded:
		return "unbounded"
	case IterLimit:
		return "iteration limit"
	case TimeLimit:
		return "time limit"
	case Cutoff:
		return "cutoff"
	case Numerical:
		return "numerical trouble"
	}
	return "unknown"
}

var inf = math.Inf(1)

// LP is a linear program together with its simplex state.
type LP struct {
	n, m int

	colIdx [][]int
	colVal [][]float64
	rowIdx [][]int
	rowVal [][]float64

	cost     []float64 // working (possibly perturbed) scaled costs
	costOrig []float64 // scaled costs
	lb, ub   []float64 // scaled bounds
	sc       []float64 // x_orig = sc * x_scaled
	cscale   float64   // objective scale: costs are multiplied by cscale

	status []int8
	basis  []int
	pos    []int
	x, d   []float64
	y      []float64
	w      []float64 // dual steepest-edge weights (per basis position)

	fac        *lu.Factor
	needFactor bool
	perturbed  bool
	didPerturb bool
	wValid     bool

	// Parameters.
	PrimalTol    float64
	DualTol      float64
	PivotTol     float64
	MaxIter      int       // iteration limit per Solve (0 = unlimited)
	Deadline     time.Time // zero = none
	ObjCutoff    float64   // stop dual simplex when objective exceeds (orig units)
	Perturb      bool      // allow cost perturbation in the dual simplex
	RefactorFreq int
	Debug        bool

	Iter      int // iterations of the last Solve
	TotalIter int
	// Statistics.
	StatPrimalIter, StatDualIter, StatPrimalCalls, StatDualCalls, StatPerturb int
	StatPhase1Iter, StatPolishIter, StatSolves, StatTrouble, StatUpdFail      int
	PolishTol                                                                 float64 // final dual tolerance for polishing (0 = off)
	warm                                                                      bool    // last Solve ended optimal from this basis (warm start)
	DualPhase1                                                                bool    // use dual phase 1 instead of primal when not dual feasible

	rng *rand.Rand

	// Work arrays.
	rho               []float64
	alphaRow          []float64
	col               []float64
	col2              []float64
	tau               []float64
	rowNZ             []int
	candJ             []int
	colNZ             []int // nonzero positions of the last ftranCol result
	pinf              []float64
	pinfEpoch, xEpoch int
	prMark            []bool
	flatOK            bool
	cscStart          []int
	cscRow            []int
	cscVal            []float64
	csrStart          []int
	csrCol            []int
	csrVal            []float64
	rhoNZ             []int
	nnzA              int
	candR             []float64
}

// New builds an LP. The constraint matrix is given column-wise (CSC):
// column j has entries rowIdx[colStart[j]:colStart[j+1]].
func New(n, m int, cost, lb, ub []float64, colStart, rowIdx []int, val []float64, rowLb, rowUb []float64) *LP {
	lp := &LP{
		n: n, m: m,
		PrimalTol:    1e-7,
		DualTol:      1e-7,
		PivotTol:     1e-7,
		ObjCutoff:    inf,
		Perturb:      true,
		DualPhase1:   true,
		PolishTol:    1e-9,
		RefactorFreq: int(envFloatSimplex("GOST_REFREQ", 100)),
		rng:          rand.New(rand.NewSource(12345)),
	}
	lp.colIdx = make([][]int, n)
	lp.colVal = make([][]float64, n)
	lp.rowIdx = make([][]int, m)
	lp.rowVal = make([][]float64, m)
	for j := 0; j < n; j++ {
		for p := colStart[j]; p < colStart[j+1]; p++ {
			if val[p] == 0 {
				continue
			}
			i := rowIdx[p]
			lp.colIdx[j] = append(lp.colIdx[j], i)
			lp.colVal[j] = append(lp.colVal[j], val[p])
		}
	}
	N := n + m
	lp.cost = make([]float64, N)
	lp.costOrig = make([]float64, N)
	lp.lb = make([]float64, N)
	lp.ub = make([]float64, N)
	lp.sc = make([]float64, N)
	copy(lp.costOrig, cost)
	copy(lp.lb, lb)
	copy(lp.ub, ub)
	copy(lp.lb[n:], rowLb)
	copy(lp.ub[n:], rowUb)
	for j := range lp.sc {
		lp.sc[j] = 1
	}
	lp.scale()
	lp.buildRows()
	copy(lp.cost, lp.costOrig)
	lp.initState()
	return lp
}

// NumCols returns the number of structural variables.
func (lp *LP) NumCols() int { return lp.n }

// NumRows returns the number of rows.
func (lp *LP) NumRows() int { return lp.m }

func (lp *LP) buildRows() {
	for i := range lp.rowIdx {
		lp.rowIdx[i] = lp.rowIdx[i][:0]
		lp.rowVal[i] = lp.rowVal[i][:0]
	}
	for j := 0; j < lp.n; j++ {
		for p, i := range lp.colIdx[j] {
			lp.rowIdx[i] = append(lp.rowIdx[i], j)
			lp.rowVal[i] = append(lp.rowVal[i], lp.colVal[j][p])
		}
	}
}

func pow2(v float64) float64 {
	if v == 0 || math.IsInf(v, 0) || math.IsNaN(v) {
		return 1
	}
	return math.Exp2(math.Round(math.Log2(v)))
}

// scale computes geometric-mean scaling followed by column equilibration and
// applies it to the matrix, costs and bounds.
func (lp *LP) scale() {
	n, m := lp.n, lp.m
	rs := make([]float64, m)
	cs := make([]float64, n)
	for i := range rs {
		rs[i] = 1
	}
	for j := range cs {
		cs[j] = 1
	}
	rmin := make([]float64, m)
	rmax := make([]float64, m)
	for pass := 0; pass < 6; pass++ {
		for i := 0; i < m; i++ {
			rmin[i], rmax[i] = inf, 0
		}
		for j := 0; j < n; j++ {
			for p, i := range lp.colIdx[j] {
				a := math.Abs(lp.colVal[j][p]) * cs[j]
				rmin[i] = math.Min(rmin[i], a)
				rmax[i] = math.Max(rmax[i], a)
			}
		}
		for i := 0; i < m; i++ {
			if rmax[i] > 0 {
				rs[i] = 1 / math.Sqrt(rmin[i]*rmax[i])
			}
		}
		for j := 0; j < n; j++ {
			cmin, cmax := inf, 0.0
			for p, i := range lp.colIdx[j] {
				a := math.Abs(lp.colVal[j][p]) * rs[i]
				cmin = math.Min(cmin, a)
				cmax = math.Max(cmax, a)
			}
			if cmax > 0 {
				cs[j] = 1 / math.Sqrt(cmin*cmax)
			}
		}
	}
	// Equilibrate columns so max |a_ij| = 1.
	for j := 0; j < n; j++ {
		cmax := 0.0
		for p, i := range lp.colIdx[j] {
			cmax = math.Max(cmax, math.Abs(lp.colVal[j][p])*rs[i])
		}
		if cmax > 0 {
			cs[j] = 1 / cmax
		}
	}
	for i := range rs {
		rs[i] = pow2(rs[i])
	}
	for j := range cs {
		cs[j] = pow2(cs[j])
	}
	for j := 0; j < n; j++ {
		for p, i := range lp.colIdx[j] {
			lp.colVal[j][p] *= rs[i] * cs[j]
		}
		lp.sc[j] = cs[j]
	}
	for i := 0; i < m; i++ {
		lp.sc[n+i] = 1 / rs[i]
	}
	for j := 0; j < n+m; j++ {
		lp.costOrig[j] *= lp.sc[j]
		lp.lb[j] /= lp.sc[j]
		lp.ub[j] /= lp.sc[j]
	}
	// Objective scaling: largest |c_j| becomes <= 1 so that the dual
	// feasibility tolerance is relative to the cost magnitude.
	cmax := 0.0
	for j := 0; j < n; j++ {
		cmax = math.Max(cmax, math.Abs(lp.costOrig[j]))
	}
	lp.cscale = 1
	// Only extreme cost magnitudes are scaled: shrinking costs loosens the
	// effective dual tolerance and degrades optimality (80bau3b, greenbea).
	if cmax > 1e8 {
		lp.cscale = pow2(1 / cmax)
		if lp.cscale*cmax > 1 {
			lp.cscale /= 2
		}
		for j := 0; j < n; j++ {
			lp.costOrig[j] *= lp.cscale
		}
	}
}

func (lp *LP) initState() {
	N := lp.n + lp.m
	lp.status = make([]int8, N)
	lp.pos = make([]int, N)
	lp.x = make([]float64, N)
	lp.d = make([]float64, N)
	lp.y = make([]float64, lp.m)
	lp.basis = make([]int, lp.m)
	lp.w = make([]float64, lp.m)
	for j := 0; j < lp.n; j++ {
		lp.status[j] = lp.defaultNonbasic(j)
		lp.pos[j] = -1
	}
	for i := 0; i < lp.m; i++ {
		j := lp.n + i
		lp.status[j] = Basic
		lp.pos[j] = i
		lp.basis[i] = j
		lp.w[i] = 1
	}
	lp.wValid = true
	lp.needFactor = true
	lp.fac = lu.New(lp.m)
	lp.allocWork()
}

func (lp *LP) allocWork() {
	N := lp.n + lp.m
	lp.prMark = nil
	lp.flatOK = false
	lp.rho = make([]float64, lp.m)
	lp.col = make([]float64, lp.m)
	lp.col2 = make([]float64, lp.m)
	lp.tau = make([]float64, lp.m)
	lp.alphaRow = make([]float64, N)
}

func (lp *LP) defaultNonbasic(j int) int8 {
	l, u := lp.lb[j], lp.ub[j]
	switch {
	case !math.IsInf(l, -1) && !math.IsInf(u, 1):
		if math.Abs(l) <= math.Abs(u) {
			return AtLower
		}
		return AtUpper
	case !math.IsInf(l, -1):
		return AtLower
	case !math.IsInf(u, 1):
		return AtUpper
	}
	return AtZero
}

// nonbasicValue returns the value of nonbasic variable j given its status.
func (lp *LP) nonbasicValue(j int) float64 {
	switch lp.status[j] {
	case AtLower:
		return lp.lb[j]
	case AtUpper:
		return lp.ub[j]
	}
	return 0
}

// ---------------------------------------------------------------------------
// Problem modification.

// SetColBounds changes the bounds of structural variable j (original units).
func (lp *LP) SetColBounds(j int, l, u float64) {
	lp.setBounds(j, l/lp.sc[j], u/lp.sc[j])
}

// SetRowBounds changes the bounds of row i (original units).
func (lp *LP) SetRowBounds(i int, l, u float64) {
	j := lp.n + i
	lp.setBounds(j, l/lp.sc[j], u/lp.sc[j])
}

func (lp *LP) setBounds(j int, l, u float64) {
	lp.lb[j], lp.ub[j] = l, u
	if lp.status[j] == Basic {
		return
	}
	st := lp.status[j]
	switch {
	case st == AtLower && math.IsInf(l, -1), st == AtUpper && math.IsInf(u, 1), st == AtZero:
		lp.status[j] = lp.defaultNonbasic(j)
	}
	lp.x[j] = lp.nonbasicValue(j)
}

// ColBounds returns the bounds of structural j in original units.
func (lp *LP) ColBounds(j int) (float64, float64) {
	return lp.lb[j] * lp.sc[j], lp.ub[j] * lp.sc[j]
}

// SetObjective replaces the cost of structural variable j.
func (lp *LP) SetObjective(j int, c float64) {
	lp.costOrig[j] = c * lp.sc[j] * lp.cscale
	lp.cost[j] = lp.costOrig[j]
}

// AddRows appends rows rl <= sum a x <= ru. Each row is given as sparse
// (idx, val) over structural variables. The new logicals become basic so the
// current basis stays valid.
func (lp *LP) AddRows(idx [][]int, val [][]float64, rl, ru []float64) {
	k := len(idx)
	n, m0 := lp.n, lp.m
	for r := 0; r < k; r++ {
		i := m0 + r
		// Scaled coefficients and row scale (power of two, max |a| ~ 1).
		amax := 0.0
		sv := make([]float64, len(val[r]))
		for p, j := range idx[r] {
			sv[p] = val[r][p] * lp.sc[j]
			amax = math.Max(amax, math.Abs(sv[p]))
		}
		rsc := 1.0
		if amax > 0 {
			rsc = pow2(1 / amax)
		}
		ri := make([]int, 0, len(idx[r]))
		rv := make([]float64, 0, len(idx[r]))
		for p, j := range idx[r] {
			v := sv[p] * rsc
			if v == 0 {
				continue
			}
			ri = append(ri, j)
			rv = append(rv, v)
			lp.colIdx[j] = append(lp.colIdx[j], i)
			lp.colVal[j] = append(lp.colVal[j], v)
		}
		lp.rowIdx = append(lp.rowIdx, ri)
		lp.rowVal = append(lp.rowVal, rv)
		s := 1 / rsc
		lp.sc = append(lp.sc, s)
		lp.costOrig = append(lp.costOrig, 0)
		lp.cost = append(lp.cost, 0)
		lp.lb = append(lp.lb, rl[r]/s)
		lp.ub = append(lp.ub, ru[r]/s)
		j := n + i
		lp.status = append(lp.status, Basic)
		lp.pos = append(lp.pos, i)
		lp.basis = append(lp.basis, j)
		lp.w = append(lp.w, 1)
		// value of new logical = row activity
		act := 0.0
		for p, jj := range ri {
			act += rv[p] * lp.x[jj]
		}
		lp.x = append(lp.x, act)
		lp.d = append(lp.d, 0)
		lp.y = append(lp.y, 0)
	}
	lp.m += k
	lp.needFactor = true
	lp.allocWork()
}

// RemoveRows deletes the rows whose indices are flagged in del (len m).
// If a deleted row's logical is nonbasic, some basic structural must leave;
// the basis is repaired by the next factorization.
func (lp *LP) RemoveRows(del []bool) {
	n, m := lp.n, lp.m
	newIdx := make([]int, m)
	k := 0
	for i := 0; i < m; i++ {
		if del[i] {
			newIdx[i] = -1
		} else {
			newIdx[i] = k
			k++
		}
	}
	if k == m {
		return
	}
	for j := 0; j < n; j++ {
		q := 0
		for p, i := range lp.colIdx[j] {
			if ni := newIdx[i]; ni >= 0 {
				lp.colIdx[j][q] = ni
				lp.colVal[j][q] = lp.colVal[j][p]
				q++
			}
		}
		lp.colIdx[j] = lp.colIdx[j][:q]
		lp.colVal[j] = lp.colVal[j][:q]
	}
	keepVar := func(arrF []float64) []float64 {
		out := arrF[:n]
		for i := 0; i < m; i++ {
			if newIdx[i] >= 0 {
				out = append(out, arrF[n+i])
			}
		}
		return out
	}
	lp.sc = keepVar(lp.sc)
	lp.cost = keepVar(lp.cost)
	lp.costOrig = keepVar(lp.costOrig)
	lp.lb = keepVar(lp.lb)
	lp.ub = keepVar(lp.ub)
	lp.x = keepVar(lp.x)
	lp.d = keepVar(lp.d)
	st := lp.status[:n]
	for i := 0; i < m; i++ {
		if newIdx[i] >= 0 {
			st = append(st, lp.status[n+i])
		}
	}
	lp.status = st
	ri := lp.rowIdx[:0]
	rv := lp.rowVal[:0]
	for i := 0; i < m; i++ {
		if newIdx[i] >= 0 {
			ri = append(ri, lp.rowIdx[i])
			rv = append(rv, lp.rowVal[i])
		}
	}
	lp.rowIdx, lp.rowVal = ri, rv
	lp.m = k
	lp.rebuildBasisFromStatus()
	lp.allocWork()
}

// rebuildBasisFromStatus reconstructs basis/pos from status, fixing the
// number of basic variables to m.
func (lp *LP) rebuildBasisFromStatus() {
	N := lp.n + lp.m
	lp.pos = make([]int, N)
	lp.basis = lp.basis[:0]
	for j := 0; j < N; j++ {
		lp.pos[j] = -1
		if lp.status[j] == Basic {
			if len(lp.basis) < lp.m {
				lp.pos[j] = len(lp.basis)
				lp.basis = append(lp.basis, j)
			} else {
				lp.status[j] = lp.defaultNonbasic(j)
				lp.x[j] = lp.nonbasicValue(j)
			}
		}
	}
	// Too few basics: add logicals.
	for i := 0; i < lp.m && len(lp.basis) < lp.m; i++ {
		j := lp.n + i
		if lp.status[j] != Basic {
			lp.status[j] = Basic
			lp.pos[j] = len(lp.basis)
			lp.basis = append(lp.basis, j)
		}
	}
	for j := 0; j < N; j++ {
		if lp.status[j] != Basic {
			lp.x[j] = lp.nonbasicValue(j)
		}
	}
	lp.w = make([]float64, lp.m)
	for i := range lp.w {
		lp.w[i] = 1
	}
	lp.wValid = true // unit initial weights (cheap); exact ones are costly
	lp.y = make([]float64, lp.m)
	lp.needFactor = true
}

// ---------------------------------------------------------------------------
// Basis access.

// Basis is a warm-start snapshot of variable statuses (len n+m).
type Basis struct {
	Status []int8
	N, M   int // dimensions when the basis was taken
}

// GetBasis returns a copy of the current basis.
func (lp *LP) GetBasis() Basis {
	return Basis{Status: append([]int8(nil), lp.status...), N: lp.n, M: lp.m}
}

// SetBasis installs a basis. A basis taken before columns and/or rows were
// appended is extended: new columns are nonbasic at a bound, new logicals
// basic.
func (lp *LP) SetBasis(b Basis) {
	if b.N == 0 && b.M == 0 && len(b.Status) == lp.n+lp.m {
		b.N, b.M = lp.n, lp.m
	}
	if len(b.Status) != b.N+b.M || b.N > lp.n || b.M > lp.m {
		return
	}
	if b.N == lp.n && b.M == lp.m && !lp.needFactor {
		same := true
		for j, st := range b.Status {
			if lp.status[j] != st {
				same = false
				break
			}
		}
		if same {
			return // already installed: keep the factorization
		}
	}
	for j := 0; j < lp.n; j++ {
		if j < b.N {
			lp.status[j] = b.Status[j]
		} else {
			lp.status[j] = lp.defaultNonbasic(j)
		}
	}
	for i := 0; i < lp.m; i++ {
		if i < b.M {
			lp.status[lp.n+i] = b.Status[b.N+i]
		} else {
			lp.status[lp.n+i] = Basic
		}
	}
	for j := range lp.status {
		if lp.status[j] == AtLower && math.IsInf(lp.lb[j], -1) ||
			lp.status[j] == AtUpper && math.IsInf(lp.ub[j], 1) {
			lp.status[j] = lp.defaultNonbasic(j)
		}
	}
	lp.rebuildBasisFromStatus()
}

// AddCols appends structural columns (original units). Column k has cost
// cost[k], bounds [lb[k], ub[k]] and entries (rows idx[k], values val[k]).
// The new columns are nonbasic; the current basis and factorization stay
// valid, so the next Solve warm starts (typically with the primal simplex
// when a new column has a favourable reduced cost).
func (lp *LP) AddCols(cost, lb, ub []float64, idx [][]int, val [][]float64) {
	k := len(cost)
	if k == 0 {
		return
	}
	n, m := lp.n, lp.m
	insert := func(a []float64, vals []float64) []float64 {
		out := make([]float64, 0, len(a)+k)
		out = append(out, a[:n]...)
		out = append(out, vals...)
		return append(out, a[n:]...)
	}
	nc := make([]float64, k)
	nl := make([]float64, k)
	nu := make([]float64, k)
	nsc := make([]float64, k)
	newIdx := make([][]int, k)
	newVal := make([][]float64, k)
	for t := 0; t < k; t++ {
		amax := 0.0
		for p, i := range idx[t] {
			amax = math.Max(amax, math.Abs(val[t][p]/lp.sc[n+i]))
		}
		cs := 1.0
		if amax > 0 {
			cs = pow2(1 / amax)
		}
		nsc[t] = cs
		nc[t] = cost[t] * cs * lp.cscale
		nl[t] = lb[t] / cs
		nu[t] = ub[t] / cs
		for p, i := range idx[t] {
			v := val[t][p] / lp.sc[n+i] * cs
			if v == 0 {
				continue
			}
			newIdx[t] = append(newIdx[t], i)
			newVal[t] = append(newVal[t], v)
		}
	}
	lp.costOrig = insert(lp.costOrig, nc)
	lp.cost = insert(lp.cost, nc)
	lp.lb = insert(lp.lb, nl)
	lp.ub = insert(lp.ub, nu)
	lp.sc = insert(lp.sc, nsc)
	lp.x = insert(lp.x, make([]float64, k))
	lp.d = insert(lp.d, make([]float64, k))
	st := make([]int8, 0, n+k+m)
	st = append(st, lp.status[:n]...)
	st = append(st, make([]int8, k)...)
	st = append(st, lp.status[n:]...)
	lp.status = st
	for r := range lp.basis {
		if lp.basis[r] >= n {
			lp.basis[r] += k
		}
	}
	lp.colIdx = append(lp.colIdx, newIdx...)
	lp.colVal = append(lp.colVal, newVal...)
	lp.n = n + k
	for t := 0; t < k; t++ {
		j := n + t
		for p, i := range newIdx[t] {
			lp.rowIdx[i] = append(lp.rowIdx[i], j)
			lp.rowVal[i] = append(lp.rowVal[i], newVal[t][p])
		}
		lp.status[j] = lp.defaultNonbasic(j)
		lp.x[j] = lp.nonbasicValue(j)
	}
	lp.pos = make([]int, lp.n+m)
	for j := range lp.pos {
		lp.pos[j] = -1
	}
	for r, j := range lp.basis {
		lp.pos[j] = r
	}
	lp.allocWork()
}

// ---------------------------------------------------------------------------
// Solution access (original units).

// Objective returns c^T x for the current point using unperturbed costs.
func (lp *LP) Objective() float64 {
	s := 0.0
	for j := 0; j < lp.n; j++ {
		s += lp.costOrig[j] * lp.x[j]
	}
	return s / lp.cscale
}

// X returns the value of structural variable j.
func (lp *LP) X(j int) float64 { return lp.x[j] * lp.sc[j] }

// Primal copies structural values into dst (len >= n).
func (lp *LP) Primal(dst []float64) {
	for j := 0; j < lp.n; j++ {
		dst[j] = lp.x[j] * lp.sc[j]
	}
}

// RowActivity returns the activity of row i.
func (lp *LP) RowActivity(i int) float64 { return lp.x[lp.n+i] * lp.sc[lp.n+i] }

// ReducedCost returns the reduced cost of structural j.
func (lp *LP) ReducedCost(j int) float64 { return lp.d[j] / lp.sc[j] / lp.cscale }

// Dual returns the dual value of row i.
func (lp *LP) Dual(i int) float64 { return lp.d[lp.n+i] / lp.sc[lp.n+i] / lp.cscale }

// VarStatus returns the status of variable j (structural or n+i logical).
func (lp *LP) VarStatus(j int) int8 { return lp.status[j] }

// ColScale returns the internal scale factor of variable j.
func (lp *LP) varScale(j int) float64 { return lp.sc[j] }
