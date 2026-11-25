package simplex

import "math"

// dualPhase1 tries to find a dual feasible basis with the "auxiliary
// problem" approach (Fourer; Koberstein's thesis): the dual simplex is run on
// the problem with artificial boxes
//
//	free            -> [-1000, 1000]
//	[l, +inf)       -> [0, 1]
//	(-inf, u]       -> [-1, 0]
//	boxed or fixed  -> [0, 0]
//
// whose optimal basis is dual feasible for the original bounds when the
// auxiliary optimum has no remaining dual infeasibility. Returns true on
// success; the true bounds are always restored.
func (lp *LP) dualPhase1() bool {
	N := lp.n + lp.m
	lb := append([]float64(nil), lp.lb...)
	ub := append([]float64(nil), lp.ub...)
	for j := 0; j < N; j++ {
		l, u := lb[j], ub[j]
		switch {
		case math.IsInf(l, -1) && math.IsInf(u, 1):
			lp.lb[j], lp.ub[j] = -1000, 1000
		case math.IsInf(u, 1):
			lp.lb[j], lp.ub[j] = 0, 1
		case math.IsInf(l, -1):
			lp.lb[j], lp.ub[j] = -1, 0
		default:
			lp.lb[j], lp.ub[j] = 0, 0
		}
		if lp.status[j] != Basic {
			if lp.d[j] >= 0 {
				lp.status[j] = AtLower
			} else {
				lp.status[j] = AtUpper
			}
		}
	}
	cutoff := lp.ObjCutoff
	lp.ObjCutoff = inf
	lp.computePrimal()
	st := lp.dualSimplex()
	if lp.Debug {
		cnt, mx, mj := 0, 0.0, -1
		for j := 0; j < N; j++ {
			if lp.status[j] != Basic && lp.dualInfeasible(j) {
				cnt++
				if math.Abs(lp.d[j]) > mx {
					mx, mj = math.Abs(lp.d[j]), j
				}
			}
		}
		mxs, ms := 0.0, -1
		for j := 0; j < N; j++ {
			if d := math.Abs(lp.cost[j] - lp.costOrig[j]); d > mxs {
				mxs, ms = d, j
			}
		}
		println("phase1 aux end: obj", lp.Objective(), "aux dual infeas", cnt, mx, "perturbed", lp.perturbed, "max shift", mxs, "at", ms, "c", lp.costOrig[max(ms, 0)])
		if mj >= 0 {
			println("  worst", mj, "n", lp.n, "status", lp.status[mj], "aux", lp.lb[mj], lp.ub[mj], "orig", lb[mj], ub[mj], "d", lp.d[mj])
		}
	}
	lp.ObjCutoff = cutoff
	copy(lp.lb, lb)
	copy(lp.ub, ub)
	// The cost perturbation is kept for phase 2 (it is removed, with a
	// primal cleanup if needed, once phase 2 is optimal).
	if st != Optimal && st != Unknown {
		lp.restoreNonbasicStatus()
		return false
	}
	if !lp.refactor() {
		return false
	}
	lp.restoreNonbasicStatus()
	if lp.Debug {
		mx := 0.0
		cnt := 0
		for j := 0; j < N; j++ {
			if lp.status[j] != Basic && lp.dualInfeasible(j) {
				cnt++
				mx = math.Max(mx, math.Abs(lp.d[j]))
			}
		}
		println("phase1: status", st.String(), "dual infeasible", cnt, "max", mx)
	}
	// Remaining tiny infeasibilities (from perturbation) are removed by
	// cost shifting; the final cleanup handles them.
	cmax := 0.0
	for j := 0; j < lp.n; j++ {
		cmax = math.Max(cmax, math.Abs(lp.costOrig[j]))
	}
	bad, _ := lp.makeDualFeasibleShift(10 * lp.perturbBase(cmax))
	lp.computePrimal()
	return bad == 0
}

// restoreNonbasicStatus sets valid statuses for the true bounds, choosing the
// bound that matches the sign of the reduced cost when possible.
func (lp *LP) restoreNonbasicStatus() {
	N := lp.n + lp.m
	for j := 0; j < N; j++ {
		if lp.status[j] == Basic {
			continue
		}
		l, u := lp.lb[j], lp.ub[j]
		hasL, hasU := !math.IsInf(l, -1), !math.IsInf(u, 1)
		d := lp.d[j]
		switch {
		case d > 0 && hasL, d >= 0 && hasL && !hasU:
			lp.status[j] = AtLower
		case d < 0 && hasU, d <= 0 && hasU && !hasL:
			lp.status[j] = AtUpper
		case hasL:
			lp.status[j] = AtLower
		case hasU:
			lp.status[j] = AtUpper
		default:
			lp.status[j] = AtZero
		}
		lp.x[j] = lp.nonbasicValue(j)
	}
}
