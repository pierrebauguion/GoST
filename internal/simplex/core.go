package simplex

import (
	"math"
	"os"
	"strconv"
)

// column returns the column of variable j (structural or logical).
var minusOne = []float64{-1}

func (lp *LP) column(j int, buf []int) ([]int, []float64) {
	if j < lp.n {
		return lp.colIdx[j], lp.colVal[j]
	}
	buf[0] = j - lp.n
	return buf[:1], minusOne
}

// factorize computes a fresh LU of the basis, repairing singularities by
// substituting logical columns. Returns false on persistent failure.
func (lp *LP) factorize() bool {
	buf := make([]int, 1)
	for attempt := 0; attempt < 5; attempt++ {
		sing := lp.fac.Factorize(lp.m, func(k int) ([]int, []float64) {
			idx, val := lp.column(lp.basis[k], buf)
			if lp.basis[k] >= lp.n {
				// copy since buf is shared
				return []int{idx[0]}, val
			}
			return idx, val
		})
		if sing == nil {
			lp.needFactor = false
			return true
		}
		for t, k := range sing.Positions {
			out := lp.basis[k]
			in := lp.n + sing.Rows[t]
			lp.status[out] = lp.defaultNonbasic(out)
			lp.pos[out] = -1
			lp.x[out] = lp.nonbasicValue(out)
			lp.status[in] = Basic
			lp.pos[in] = k
			lp.basis[k] = in
			lp.w[k] = 1
		}
	}
	return false
}

// recomputePinf recomputes the squared primal infeasibility of every basic
// position (used by the dual pricing).
func (lp *LP) recomputePinf() {
	m := lp.m
	if cap(lp.pinf) < m {
		lp.pinf = make([]float64, m)
	}
	lp.pinf = lp.pinf[:m]
	for k := 0; k < m; k++ {
		lp.updPinf(k)
	}
	lp.pinfEpoch = lp.xEpoch
}

// updPinf refreshes the infeasibility of basis position k.
func (lp *LP) updPinf(k int) {
	j := lp.basis[k]
	v := lp.x[j]
	var inf float64
	if v < lp.lb[j]-lp.PrimalTol {
		inf = lp.lb[j] - v
	} else if v > lp.ub[j]+lp.PrimalTol {
		inf = v - lp.ub[j]
	}
	lp.pinf[k] = inf * inf
}

// computePrimal recomputes x_B from the nonbasic values.
func (lp *LP) computePrimal() {
	lp.xEpoch++
	m := lp.m
	rhs := lp.col
	for i := range rhs[:m] {
		rhs[i] = 0
	}
	N := lp.n + m
	for j := 0; j < N; j++ {
		if lp.status[j] == Basic {
			continue
		}
		v := lp.nonbasicValue(j)
		lp.x[j] = v
		if v == 0 {
			continue
		}
		if j < lp.n {
			for p, i := range lp.colIdx[j] {
				rhs[i] -= lp.colVal[j][p] * v
			}
		} else {
			rhs[j-lp.n] += v
		}
	}
	lp.fac.FTRAN(rhs)
	for k := 0; k < m; k++ {
		lp.x[lp.basis[k]] = rhs[k]
	}
}

// computeDual recomputes y and the reduced costs with the working costs.
func (lp *LP) computeDual() {
	lp.computeDualWith(lp.cost)
}

func (lp *LP) computeDualWith(c []float64) {
	m := lp.m
	y := lp.y
	for k := 0; k < m; k++ {
		y[k] = c[lp.basis[k]]
	}
	lp.fac.BTRAN(y)
	for j := 0; j < lp.n; j++ {
		if lp.status[j] == Basic {
			lp.d[j] = 0
			continue
		}
		s := c[j]
		for p, i := range lp.colIdx[j] {
			s -= lp.colVal[j][p] * y[i]
		}
		lp.d[j] = s
	}
	for i := 0; i < m; i++ {
		j := lp.n + i
		if lp.status[j] == Basic {
			lp.d[j] = 0
		} else {
			lp.d[j] = c[j] + y[i]
		}
	}
}

// primalInfeas returns the infeasibility of basic variable j (>0 if
// violated): negative distance below lower is returned as positive.
func (lp *LP) infeasOf(j int) float64 {
	v := lp.x[j]
	if v < lp.lb[j]-lp.PrimalTol {
		return lp.lb[j] - v
	}
	if v > lp.ub[j]+lp.PrimalTol {
		return v - lp.ub[j]
	}
	return 0
}

// sumPrimalInfeas returns the number and sum of basic infeasibilities.
func (lp *LP) sumPrimalInfeas() (int, float64) {
	cnt, sum := 0, 0.0
	for _, j := range lp.basis {
		if v := lp.infeasOf(j); v > 0 {
			cnt++
			sum += v
		}
	}
	return cnt, sum
}

// dualInfeasible reports whether nonbasic j has a reduced cost of the wrong
// sign (beyond tolerance).
func (lp *LP) dualInfeasible(j int) bool {
	d := lp.d[j]
	switch lp.status[j] {
	case AtLower:
		if lp.lb[j] == lp.ub[j] {
			return false
		}
		return d < -lp.DualTol
	case AtUpper:
		if lp.lb[j] == lp.ub[j] {
			return false
		}
		return d > lp.DualTol
	case AtZero:
		return math.Abs(d) > lp.DualTol
	}
	return false
}

// makeDualFeasible flips boxed nonbasic variables with wrong-sign reduced
// costs. It returns the number of remaining dual infeasibilities (non-boxed)
// and whether any flip occurred.
func (lp *LP) makeDualFeasible() (int, bool) {
	return lp.makeDualFeasibleShift(0)
}

// makeDualFeasibleShift is makeDualFeasible that additionally removes
// remaining dual infeasibilities of magnitude <= maxShift by shifting costs
// (the shift is undone by removePerturbation).
func (lp *LP) makeDualFeasibleShift(maxShift float64) (int, bool) {
	N := lp.n + lp.m
	bad := 0
	flipped := false
	for j := 0; j < N; j++ {
		if lp.status[j] == Basic || !lp.dualInfeasible(j) {
			continue
		}
		boxed := !math.IsInf(lp.lb[j], -1) && !math.IsInf(lp.ub[j], 1)
		if boxed {
			if lp.status[j] == AtLower {
				lp.status[j] = AtUpper
			} else {
				lp.status[j] = AtLower
			}
			lp.x[j] = lp.nonbasicValue(j)
			flipped = true
			continue
		}
		// One-sided bounds: move to the finite bound with the right sign if
		// possible (e.g. AtZero free var with d>0 but finite lb).
		if lp.d[j] > 0 && !math.IsInf(lp.lb[j], -1) {
			lp.status[j] = AtLower
			lp.x[j] = lp.lb[j]
			flipped = true
			continue
		}
		if lp.d[j] < 0 && !math.IsInf(lp.ub[j], 1) {
			lp.status[j] = AtUpper
			lp.x[j] = lp.ub[j]
			flipped = true
			continue
		}
		if math.Abs(lp.d[j]) <= maxShift*(1+math.Abs(lp.costOrig[j])) {
			lp.cost[j] -= lp.d[j]
			lp.d[j] = 0
			lp.perturbed = true
			continue
		}
		bad++
	}
	return bad, flipped
}

// perturbCosts applies a small random cost perturbation in the direction
// that preserves dual feasibility.
func (lp *LP) perturbCosts() { lp.perturbCostsScaled(1) }

// PerturbMode selects the perturbation magnitude rule (for experiments).
var PerturbMode = int(envFloatSimplex("GOST_PERTURB", 1))

// perturbBase returns the base cost perturbation for a largest cost cmax.
// Like HiGHS, large cost ranges are damped (fourth root) so the
// perturbation stays small relative to the costs; a large perturbation
// leaves many dual infeasibilities to clean up with the primal simplex.
func (lp *LP) perturbBase(cmax float64) float64 {
	if PerturbMode == 0 {
		return 5e-7 * math.Max(1, math.Min(cmax, 1e3))
	}
	c := cmax
	switch {
	case c > 100:
		c = math.Sqrt(math.Sqrt(c))
	case c > 1:
		c = math.Sqrt(c)
	}
	return 5e-7 * math.Max(1, c)
}

// perturbCostsScaled perturbs nonbasic costs by base*scale magnitudes.
func (lp *LP) perturbCostsScaled(scale float64) {
	N := lp.n + lp.m
	cmax := 0.0
	for j := 0; j < lp.n; j++ {
		cmax = math.Max(cmax, math.Abs(lp.costOrig[j]))
	}
	base := scale * lp.perturbBase(cmax)
	for j := 0; j < N; j++ {
		if lp.lb[j] == lp.ub[j] {
			continue
		}
		xi := base * (1 + math.Abs(lp.costOrig[j])) * (1 + lp.rng.Float64())
		if j >= lp.n {
			xi *= 0.1
		}
		// Only nonbasic costs are perturbed so that y is unchanged and the
		// reduced costs move exactly by xi in the dual feasible direction.
		switch lp.status[j] {
		case AtLower:
			lp.cost[j] += xi
			lp.d[j] += xi
		case AtUpper:
			lp.cost[j] -= xi
			lp.d[j] -= xi
		}
	}
	lp.perturbed = true
	lp.didPerturb = true
	lp.StatPerturb++
}

func (lp *LP) removePerturbation() {
	copy(lp.cost, lp.costOrig)
	lp.perturbed = false
	lp.didPerturb = false
}

// computeDSEWeights computes exact dual steepest-edge weights ||e_r B^-1||^2.
func (lp *LP) computeDSEWeights() {
	m := lp.m
	allLogical := true
	for k := 0; k < m; k++ {
		if lp.basis[k] < lp.n {
			allLogical = false
			break
		}
	}
	if allLogical || m > 5000 {
		for k := range lp.w[:m] {
			lp.w[k] = 1
		}
		lp.wValid = true
		return
	}
	r := lp.rho
	for k := 0; k < m; k++ {
		for i := range r[:m] {
			r[i] = 0
		}
		r[k] = 1
		lp.fac.BTRAN(r)
		s := 0.0
		for _, v := range r[:m] {
			s += v * v
		}
		lp.w[k] = s
	}
	lp.wValid = true
}

// refactor refactorizes and recomputes primal and dual values.
var FillStats struct{ Refactors, L, U, Eta, M int }

func (lp *LP) refactor() bool {
	l, u, e := lp.fac.FillIn()
	FillStats.Refactors++
	FillStats.L += l
	FillStats.U += u
	FillStats.Eta += e
	FillStats.M += lp.m
	if !lp.factorize() {
		return false
	}
	lp.computePrimal()
	lp.computeDual()
	return true
}

// pivotRow computes alpha_j = rho^T a_j for all nonbasic j into alphaRow and
// returns the list of nonbasic indices with nonzero alpha in candJ.
func (lp *LP) pivotRow(rho []float64) []int {
	n, m := lp.n, lp.m
	ar := lp.alphaRow
	lp.candJ = lp.candJ[:0]
	if len(lp.prMark) < n || !lp.flatOK {
		lp.buildFlat()
	}
	// Choose row-wise or column-wise computation by estimated work.
	lp.rhoNZ = lp.rhoNZ[:0]
	rowWork := 0
	rs := lp.csrStart
	for i := 0; i < m; i++ {
		if rho[i] != 0 {
			lp.rhoNZ = append(lp.rhoNZ, i)
			rowWork += rs[i+1] - rs[i]
		}
	}
	if rowWork < lp.nnzA/2+n/4 {
		mark := lp.prMark
		cols, vals := lp.csrCol, lp.csrVal
		for _, i := range lp.rhoNZ {
			ri := rho[i]
			for p := rs[i]; p < rs[i+1]; p++ {
				j := cols[p]
				if !mark[j] {
					mark[j] = true
					lp.candJ = append(lp.candJ, j)
				}
				ar[j] += ri * vals[p]
			}
		}
		q := 0
		for _, j := range lp.candJ {
			mark[j] = false
			if ar[j] != 0 && lp.status[j] != Basic {
				lp.candJ[q] = j
				q++
			} else {
				ar[j] = 0
			}
		}
		lp.candJ = lp.candJ[:q]
	} else {
		cs, rows, vals := lp.cscStart, lp.cscRow, lp.cscVal
		st := lp.status
		for j := 0; j < n; j++ {
			if st[j] == Basic {
				continue
			}
			s := 0.0
			for p := cs[j]; p < cs[j+1]; p++ {
				s += rho[rows[p]] * vals[p]
			}
			if s != 0 {
				ar[j] = s
				lp.candJ = append(lp.candJ, j)
			}
		}
	}
	for _, i := range lp.rhoNZ {
		j := n + i
		if lp.status[j] != Basic {
			ar[j] = -rho[i]
			lp.candJ = append(lp.candJ, j)
		}
	}
	return lp.candJ
}

// buildFlat builds contiguous CSC/CSR copies of the scaled matrix for the
// pivot row computation.
func (lp *LP) buildFlat() {
	n, m := lp.n, lp.m
	lp.prMark = make([]bool, n)
	nnz := 0
	for j := 0; j < n; j++ {
		nnz += len(lp.colIdx[j])
	}
	lp.nnzA = nnz
	lp.cscStart = append(lp.cscStart[:0], 0)
	lp.cscRow = lp.cscRow[:0]
	lp.cscVal = lp.cscVal[:0]
	for j := 0; j < n; j++ {
		lp.cscRow = append(lp.cscRow, lp.colIdx[j]...)
		lp.cscVal = append(lp.cscVal, lp.colVal[j]...)
		lp.cscStart = append(lp.cscStart, len(lp.cscRow))
	}
	lp.csrStart = append(lp.csrStart[:0], 0)
	lp.csrCol = lp.csrCol[:0]
	lp.csrVal = lp.csrVal[:0]
	for i := 0; i < m; i++ {
		lp.csrCol = append(lp.csrCol, lp.rowIdx[i]...)
		lp.csrVal = append(lp.csrVal, lp.rowVal[i]...)
		lp.csrStart = append(lp.csrStart, len(lp.csrCol))
	}
	lp.flatOK = true
}

func (lp *LP) clearPivotRow(cands []int) {
	for _, j := range cands {
		lp.alphaRow[j] = 0
	}
}

// ftranCol computes B^-1 a_j into out (len m).
func (lp *LP) ftranCol(j int, out []float64) {
	for i := range out[:lp.m] {
		out[i] = 0
	}
	if j < lp.n {
		for p, i := range lp.colIdx[j] {
			out[i] = lp.colVal[j][p]
		}
	} else {
		out[j-lp.n] = -1
	}
	lp.colNZ = lp.fac.FTRANCol(out)
}

// TableauRow computes the row of B^-1 A for the basic variable at basis
// position r, in original units: x_B[r] + sum_j alpha_j x_j = beta, for
// nonbasic j (structural j < n, logical n+i). It returns the indices and
// coefficients of nonbasic variables.
func (lp *LP) TableauRow(r int) ([]int, []float64) {
	rho := lp.rho
	for i := range rho[:lp.m] {
		rho[i] = 0
	}
	rho[r] = 1
	lp.fac.BTRAN(rho)
	cands := lp.pivotRow(rho)
	sb := lp.sc[lp.basis[r]]
	idx := make([]int, 0, len(cands))
	val := make([]float64, 0, len(cands))
	for _, j := range cands {
		a := lp.alphaRow[j]
		if math.Abs(a) > 1e-12 {
			idx = append(idx, j)
			val = append(val, a*sb/lp.sc[j])
		}
	}
	lp.clearPivotRow(cands)
	return idx, val
}

// Refresh takes a fresh factorization of the current basis and recomputes
// primal and dual values (e.g. before deriving cuts from tableau rows).
func (lp *LP) Refresh() bool {
	if lp.fac.NumUpdates() == 0 && !lp.needFactor {
		return true
	}
	return lp.refactor()
}

// BasisVar returns the variable at basis position r.
func (lp *LP) BasisVar(r int) int { return lp.basis[r] }

// VarValue returns the value of any variable (structural or logical) in
// original units.
func (lp *LP) VarValue(j int) float64 { return lp.x[j] * lp.sc[j] }

// VarBounds returns bounds of any variable in original units.
func (lp *LP) VarBounds(j int) (float64, float64) {
	return lp.lb[j] * lp.sc[j], lp.ub[j] * lp.sc[j]
}

// RowCoefs returns the original-unit coefficients of row i.
func (lp *LP) RowCoefs(i int) ([]int, []float64) {
	s := lp.sc[lp.n+i]
	idx := lp.rowIdx[i]
	val := make([]float64, len(idx))
	for p, j := range idx {
		val[p] = lp.rowVal[i][p] * s / lp.sc[j]
	}
	return idx, val
}

func envFloatSimplex(name string, def float64) float64 {
	if v, err := strconv.ParseFloat(os.Getenv(name), 64); err == nil {
		return v
	}
	return def
}
