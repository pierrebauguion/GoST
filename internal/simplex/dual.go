package simplex

import (
	"math"
	"sort"
	"time"
)

type ratioCand struct {
	j     int
	ratio float64
	abs   float64 // |alpha_j|
}

// dualSimplex runs the bounded dual simplex from a dual feasible basis.
// It returns Optimal (primal feasible), Infeasible, Cutoff, IterLimit,
// TimeLimit, Numerical, or Unknown when dual feasibility is lost and the
// caller should switch algorithms.
func (lp *LP) dualSimplex() Status {
	m := lp.m
	if !lp.wValid {
		lp.computeDSEWeights()
	}
	lp.pinfEpoch = -1 // x may have changed outside the dual simplex
	cands := make([]ratioCand, 0, 64)
	flips := make([]int, 0, 16)
	degenerate := 0
	cutoffScaled := lp.ObjCutoff
	numTrouble := 0
	lastObjW := math.Inf(-1)
	stalls, perturbLevel := 0, 0
	if lp.didPerturb {
		perturbLevel = 1
	}

	for {
		if lp.MaxIter > 0 && lp.Iter >= lp.MaxIter {
			return IterLimit
		}
		if lp.Iter&63 == 0 && !lp.Deadline.IsZero() && time.Now().After(lp.Deadline) {
			return TimeLimit
		}
		if lp.fac.NeedRefactor(lp.RefactorFreq) {
			var dOld []float64
			if lp.Debug {
				dOld = append([]float64(nil), lp.d...)
			}
			if !lp.refactor() {
				return Numerical
			}
			if lp.Debug {
				mx, mj := 0.0, -1
				for j := range dOld {
					if lp.status[j] != Basic && math.Abs(dOld[j]-lp.d[j]) > mx {
						mx, mj = math.Abs(dOld[j]-lp.d[j]), j
					}
				}
				if mx > 1e-6 {
					println("dual refactor: max d drift", mx, "var", mj, "status", lp.status[mj], "old", dOld[mj], "new", lp.d[mj])
				}
			}
			if bad, fl := lp.makeDualFeasibleShift(math.Inf(1)); bad > 0 {
				return Unknown
			} else if fl {
				lp.computePrimal()
			}
		}
		// Cutoff (only meaningful without perturbation).
		if !lp.perturbed && cutoffScaled < inf && lp.Iter&7 == 0 {
			if lp.Objective() > cutoffScaled {
				return Cutoff
			}
		}

		// Stall detection on the working (perturbed) dual objective.
		if lp.Iter%50 == 0 && lp.Perturb {
			ow := 0.0
			for j, c := range lp.cost {
				ow += c * lp.x[j]
			}
			// Stall: no progress over the best value seen (catches both
			// flat stretches and long cycles where the objective oscillates).
			if ow <= lastObjW+1e-9*(1+math.Abs(ow)) {
				stalls++
				if stalls >= 4 && perturbLevel < 4 {
					perturbLevel++
					lp.perturbCostsScaled(math.Pow(10, float64(perturbLevel-1)))
					stalls = 0
					lastObjW = math.Inf(-1)
				}
			} else {
				stalls = 0
				lastObjW = ow
			}
		}
		if lp.Debug && lp.Iter%500 == 0 {
			ni, si := lp.sumPrimalInfeas()
			println("dual it", lp.Iter, "obj", lp.Objective(), "pinf", ni, si, "perturbed", lp.perturbed, "degen", degenerate)
		}
		// --- Pricing: choose leaving row (DSE).
		if lp.pinfEpoch != lp.xEpoch || len(lp.pinf) != m {
			lp.recomputePinf()
		}
		r := -1
		best := 0.0
		pinf, w := lp.pinf[:m], lp.w[:m]
		for k, v := range pinf {
			if v > best*w[k] {
				best, r = v/w[k], k
			}
		}
		if r < 0 {
			return Optimal
		}
		p := lp.basis[r]
		var dir float64 // +1: leaves to upper, -1: leaves to lower
		var delta float64
		if lp.x[p] > lp.ub[p] {
			dir = 1
			delta = lp.x[p] - lp.ub[p]
		} else {
			dir = -1
			delta = lp.lb[p] - lp.x[p]
		}

		// --- BTRAN: rho = e_r B^-1.
		rho := lp.rho
		for i := range rho[:m] {
			rho[i] = 0
		}
		rho[r] = 1
		lp.fac.BTRAN(rho)
		wr := 0.0
		for _, v := range rho[:m] {
			wr += v * v
		}
		lp.w[r] = wr

		// --- Pivot row.
		cj := lp.pivotRow(rho)

		// --- Ratio test (BFRT + Harris).
		cands = cands[:0]
		for _, j := range cj {
			a := lp.alphaRow[j]
			v := dir * a
			if math.Abs(a) < lp.PivotTol {
				continue
			}
			if lp.lb[j] == lp.ub[j] {
				continue // fixed variables never block
			}
			dj := lp.d[j]
			switch lp.status[j] {
			case AtLower:
				if v > 0 {
					cands = append(cands, ratioCand{j, math.Max(dj, 0) / v, math.Abs(a)})
				}
			case AtUpper:
				if v < 0 {
					cands = append(cands, ratioCand{j, math.Min(dj, 0) / v, math.Abs(a)})
				}
			case AtZero:
				cands = append(cands, ratioCand{j, 0, math.Abs(a) * 1e3}) // prefer free vars
			}
		}
		if len(cands) == 0 {
			if lp.Debug {
				println("dual: no candidates; row", r, "var", p, "x", lp.x[p], "lb", lp.lb[p], "ub", lp.ub[p], "ncj", len(cj))
			}
			lp.clearPivotRow(cj)
			return Infeasible
		}
		// BFRT: pass breakpoints in increasing ratio order using repeated
		// selection (cheap when few flips happen; sort if many).
		slope := delta
		flips = flips[:0]
		start := 0
		sorted := false
		for start < len(cands) {
			if !sorted {
				if start >= 32 {
					rest := cands[start:]
					sort.Slice(rest, func(a, b int) bool { return rest[a].ratio < rest[b].ratio })
					sorted = true
				} else {
					mi := start
					for k := start + 1; k < len(cands); k++ {
						if cands[k].ratio < cands[mi].ratio {
							mi = k
						}
					}
					cands[start], cands[mi] = cands[mi], cands[start]
				}
			}
			c := cands[start]
			j := c.j
			if lp.status[j] != AtZero && !math.IsInf(lp.lb[j], -1) && !math.IsInf(lp.ub[j], 1) {
				ns := slope - c.abs*(lp.ub[j]-lp.lb[j])
				if ns > lp.PrimalTol {
					slope = ns
					flips = append(flips, j)
					start++
					continue
				}
			}
			break
		}
		if start >= len(cands) {
			if lp.Debug {
				println("dual: all flipped; slope", slope, "delta", delta, "ncands", len(cands))
			}
			// All breakpoints passed: dual unbounded => primal infeasible.
			lp.clearPivotRow(cj)
			return Infeasible
		}
		// Harris pass on remaining candidates.
		thMax := inf
		for _, c := range cands[start:] {
			absd := math.Abs(lp.d[c.j])
			t := (absd + lp.DualTol) / (c.abs)
			if lp.status[c.j] == AtZero {
				t = (absd + lp.DualTol) / (c.abs * 1e-3)
			}
			if t < thMax {
				thMax = t
			}
		}
		q := -1
		qa := 0.0
		for _, c := range cands[start:] {
			if c.ratio > thMax {
				if sorted {
					break
				}
				continue
			}
			if c.abs > qa {
				qa, q = c.abs, c.j
			}
		}
		if q < 0 {
			q = cands[start].j
		}
		// Candidates between start and q's position with ratio <= ratio_q are
		// not flipped; they become slightly dual infeasible (Harris).

		alphaQ := lp.alphaRow[q]

		// --- FTRAN entering column.
		acol := lp.col
		lp.ftranCol(q, acol)
		if math.Abs(acol[r]-alphaQ) > 1e-6*(1+math.Abs(alphaQ)) || math.Abs(acol[r]) < 1e-11 {
			lp.clearPivotRow(cj)
			numTrouble++
			lp.StatTrouble++
			if numTrouble > 5 {
				return Numerical
			}
			if !lp.refactor() {
				return Numerical
			}
			if bad, fl := lp.makeDualFeasibleShift(math.Inf(1)); bad > 0 {
				return Unknown
			} else if fl {
				lp.computePrimal()
			}
			continue
		}
		arq := acol[r]

		// --- Dual update. If d_q has a slightly wrong sign (Harris), shift
		// its cost so the dual step is exactly zero for it.
		if (lp.status[q] == AtLower && lp.d[q] < 0) || (lp.status[q] == AtUpper && lp.d[q] > 0) || (lp.status[q] == AtZero && lp.d[q] != 0) {
			lp.cost[q] -= lp.d[q]
			lp.d[q] = 0
			lp.perturbed = true
		}
		thetaD := lp.d[q] / alphaQ
		if math.Abs(thetaD) < 1e-12 {
			degenerate++
		} else {
			degenerate = 0
		}
		for _, j := range cj {
			if lp.status[j] != Basic {
				lp.d[j] -= thetaD * lp.alphaRow[j]
			}
		}
		lp.d[q] = 0
		lp.d[p] = -thetaD
		lp.clearPivotRow(cj)

		// --- Bound flips.
		if len(flips) > 0 {
			b := lp.col2
			for i := range b[:m] {
				b[i] = 0
			}
			for _, j := range flips {
				var dx float64
				if lp.status[j] == AtLower {
					lp.status[j] = AtUpper
					dx = lp.ub[j] - lp.lb[j]
				} else {
					lp.status[j] = AtLower
					dx = lp.lb[j] - lp.ub[j]
				}
				lp.x[j] = lp.nonbasicValue(j)
				if j < lp.n {
					for pp, i := range lp.colIdx[j] {
						b[i] += lp.colVal[j][pp] * dx
					}
				} else {
					b[j-lp.n] -= dx
				}
			}
			for _, k := range lp.fac.FTRANNZ(b) {
				lp.x[lp.basis[k]] -= b[k]
				lp.updPinf(k)
			}
		}

		// --- Primal update.
		var bound float64
		if dir > 0 {
			bound = lp.ub[p]
		} else {
			bound = lp.lb[p]
		}
		thetaP := (lp.x[p] - bound) / arq
		colNZ := lp.colNZ
		for _, k := range colNZ {
			lp.x[lp.basis[k]] -= thetaP * acol[k]
		}
		defer0 := colNZ
		lp.x[q] += thetaP
		lp.x[p] = bound

		// --- DSE weight update.
		tau := lp.tau
		copy(tau[:m], rho[:m])
		lp.fac.FTRAN(tau)
		for _, k := range colNZ {
			if k == r {
				continue
			}
			kap := acol[k] / arq
			wk := lp.w[k] + kap*(kap*wr-2*tau[k])
			if wk < 1e-4 {
				wk = 1e-4
			}
			lp.w[k] = wk
		}
		lp.w[r] = math.Max(wr/(arq*arq), 1e-4)

		// --- Basis change.
		lp.basis[r] = q
		lp.pos[q] = r
		lp.pos[p] = -1
		lp.status[q] = Basic
		for _, k := range defer0 {
			lp.updPinf(k)
		}
		lp.updPinf(r)
		if dir > 0 {
			lp.status[p] = AtUpper
		} else {
			lp.status[p] = AtLower
		}
		if lp.lb[p] == lp.ub[p] {
			lp.status[p] = AtLower
		}
		lp.Iter++
		if !lp.fac.Update(r, acol) {
			lp.StatUpdFail++
			if !lp.refactor() {
				return Numerical
			}
		}

		// Degeneracy: perturb once if stalling.
		if degenerate > 20 && lp.Perturb && !lp.didPerturb {
			lp.perturbCosts()
			perturbLevel = 1
			degenerate = 0
		}
	}
}
