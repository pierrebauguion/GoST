package simplex

import (
	"math"
	"time"
)

// primalSimplex runs the bounded primal simplex with a composite phase 1
// (minimizing the sum of infeasibilities) followed by phase 2.
func (lp *LP) primalSimplex() Status {
	m := lp.m
	N := lp.n + m
	cP := make([]float64, N)
	// Devex reference weights.
	devex := make([]float64, N)
	for j := range devex {
		devex[j] = 1
	}
	phase := 0
	numTrouble := 0
	needDual := true

	for {
		if lp.MaxIter > 0 && lp.Iter >= lp.MaxIter {
			return IterLimit
		}
		if lp.Iter&63 == 0 && !lp.Deadline.IsZero() && time.Now().After(lp.Deadline) {
			return TimeLimit
		}
		if lp.fac.NeedRefactor(lp.RefactorFreq) {
			if !lp.factorize() {
				return Numerical
			}
			lp.computePrimal()
			needDual = true
		}
		// Determine phase.
		ninf, _ := lp.sumPrimalInfeas()
		newPhase := 2
		if ninf > 0 {
			newPhase = 1
		}
		if newPhase != phase {
			phase = newPhase
			needDual = true
		}
		if phase == 1 {
			// Phase-1 costs change with the infeasibility pattern.
			for j := range cP {
				cP[j] = 0
			}
			for _, j := range lp.basis {
				if lp.x[j] < lp.lb[j]-lp.PrimalTol {
					cP[j] = -1
				} else if lp.x[j] > lp.ub[j]+lp.PrimalTol {
					cP[j] = 1
				}
			}
			lp.computeDualWith(cP)
		} else if needDual {
			lp.computeDual()
		}
		needDual = false

		// --- Pricing (Devex-weighted Dantzig).
		q := -1
		best := 0.0
		var dir float64
		for j := 0; j < N; j++ {
			st := lp.status[j]
			if st == Basic || lp.lb[j] == lp.ub[j] {
				continue
			}
			dj := lp.d[j]
			var ok bool
			var dd float64
			switch st {
			case AtLower:
				ok, dd = dj < -lp.DualTol, 1
			case AtUpper:
				ok, dd = dj > lp.DualTol, -1
			case AtZero:
				ok = math.Abs(dj) > lp.DualTol
				if dj > 0 {
					dd = -1
				} else {
					dd = 1
				}
			}
			if !ok {
				continue
			}
			score := dj * dj / devex[j]
			if score > best {
				best, q, dir = score, j, dd
			}
		}
		if q < 0 {
			if phase == 1 {
				if lp.Debug {
					n, sum := lp.sumPrimalInfeas()
					println("primal phase1 stuck: ninf", n, "sum", sum)
				}
				return Infeasible
			}
			return Optimal
		}

		// --- FTRAN.
		acol := lp.col
		lp.ftranCol(q, acol)

		// --- Ratio test (Harris two-pass).
		flipDist := inf
		if !math.IsInf(lp.lb[q], -1) && !math.IsInf(lp.ub[q], 1) {
			flipDist = lp.ub[q] - lp.lb[q]
		}
		thMax := flipDist
		tol := lp.PrimalTol
		for _, k := range lp.colNZ {
			a := acol[k]
			if math.Abs(a) < lp.PivotTol {
				continue
			}
			j := lp.basis[k]
			rate := -dir * a
			bnd, ok := lp.primalBound(j, rate, phase)
			if !ok {
				continue
			}
			t := (math.Abs(lp.x[j]-bnd) + tol) / math.Abs(rate)
			if t < thMax {
				thMax = t
			}
		}
		r := -1
		ra := 0.0
		rRatio := inf
		for _, k := range lp.colNZ {
			a := acol[k]
			if math.Abs(a) < lp.PivotTol {
				continue
			}
			j := lp.basis[k]
			rate := -dir * a
			bnd, ok := lp.primalBound(j, rate, phase)
			if !ok {
				continue
			}
			t := (lp.x[j] - bnd) / (-rate)
			if t < 0 {
				t = 0
			}
			if t <= thMax && math.Abs(a) > ra {
				ra, r, rRatio = math.Abs(a), k, t
			}
		}
		if r < 0 && math.IsInf(flipDist, 1) {
			if phase == 1 {
				numTrouble++
				if numTrouble > 5 {
					return Numerical
				}
				if !lp.refactor() {
					return Numerical
				}
				continue
			}
			return Unbounded
		}
		if r < 0 || flipDist <= rRatio {
			// Bound flip of the entering variable.
			t := flipDist
			for _, k := range lp.colNZ {
				lp.x[lp.basis[k]] -= dir * t * acol[k]
			}
			if lp.status[q] == AtLower {
				lp.status[q] = AtUpper
			} else {
				lp.status[q] = AtLower
			}
			lp.x[q] = lp.nonbasicValue(q)
			lp.Iter++
			continue
		}
		t := rRatio
		p := lp.basis[r]
		rate := -dir * acol[r]
		bnd, _ := lp.primalBound(p, rate, phase)
		for _, k := range lp.colNZ {
			lp.x[lp.basis[k]] -= dir * t * acol[k]
		}
		lp.x[q] += dir * t
		lp.x[p] = bnd

		// Dual update (phase 2 incremental) and Devex weights via pivot row.
		rho := lp.rho
		for i := range rho[:m] {
			rho[i] = 0
		}
		rho[r] = 1
		lp.fac.BTRAN(rho)
		cj := lp.pivotRow(rho)
		arq := acol[r]
		thetaD := lp.d[q] / arq
		wq := devex[q]
		for _, j := range cj {
			if j == q {
				continue
			}
			a := lp.alphaRow[j]
			lp.d[j] -= thetaD * a
			ratio := a / arq
			if w := ratio * ratio * wq; w > devex[j] {
				devex[j] = w
			}
		}
		lp.clearPivotRow(cj)
		devex[p] = math.Max(wq/(arq*arq), 1)
		lp.d[p] = -thetaD
		lp.d[q] = 0

		// Basis change.
		lp.basis[r] = q
		lp.pos[q] = r
		lp.pos[p] = -1
		lp.status[q] = Basic
		if bnd == lp.ub[p] && lp.lb[p] != lp.ub[p] {
			lp.status[p] = AtUpper
		} else {
			lp.status[p] = AtLower
		}
		if math.IsInf(bnd, 0) {
			lp.status[p] = AtZero
			lp.x[p] = 0
		}
		lp.wValid = false
		lp.Iter++
		if !lp.fac.Update(r, acol) {
			if !lp.factorize() {
				return Numerical
			}
			lp.computePrimal()
			needDual = true
		}
		// Reset Devex framework occasionally.
		if lp.Iter%1000 == 0 {
			for j := range devex {
				devex[j] = 1
			}
		}
	}
}

// primalBound returns the bound that basic variable j moves towards when
// changing at the given rate, and whether the move is limited.
func (lp *LP) primalBound(j int, rate float64, phase int) (float64, bool) {
	x := lp.x[j]
	tol := lp.PrimalTol
	if rate < 0 {
		if phase == 1 {
			if x < lp.lb[j]-tol {
				return 0, false
			}
			if x > lp.ub[j]+tol {
				return lp.ub[j], true
			}
		}
		if math.IsInf(lp.lb[j], -1) {
			return 0, false
		}
		return lp.lb[j], true
	}
	if phase == 1 {
		if x > lp.ub[j]+tol {
			return 0, false
		}
		if x < lp.lb[j]-tol {
			return lp.lb[j], true
		}
	}
	if math.IsInf(lp.ub[j], 1) {
		return 0, false
	}
	return lp.ub[j], true
}
