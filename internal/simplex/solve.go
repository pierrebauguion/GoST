package simplex

import (
	"math"
	"os"
)

// Solve optimizes the LP from the current basis (warm start). It chooses the
// dual simplex when the basis is dual feasible (after bound flips) and the
// primal simplex otherwise, then removes any cost perturbation and cleans up.
func (lp *LP) Solve() Status {
	lp.Iter = 0
	defer func() { lp.TotalIter += lp.Iter }()

	if lp.needFactor {
		if !lp.factorize() {
			return Numerical
		}
	}
	lp.removePerturbation()
	lp.computePrimal()
	lp.computeDual()
	lp.StatSolves++

	var st Status
	triedPhase1 := false
	cleanup := false // primal feasible point: avoid bound flips that break it
	// Warm start from a primal feasible but dual infeasible basis (typically
	// after columns were added by pricing, or after a cost change): continue
	// with the primal simplex instead of a dual phase 1, which would give up
	// the primal feasibility of the current point.
	if lp.warm && !PrimalWarmOff {
		if n, _ := lp.sumPrimalInfeas(); n == 0 {
			if b, o := lp.dualInfeasCounts(); o > 0 && b == 0 {
				cleanup = true
			}
		}
	}
	for round := 0; round < 20; round++ {
		var bad int
		var flipped bool
		if cleanup {
			cleanup = false
			if _, other := lp.dualInfeasCounts(); other > 0 {
				bad = other // primal simplex phase 2 from the feasible point
				triedPhase1 = true
			} else {
				bad, flipped = lp.makeDualFeasible()
			}
		} else {
			bad, flipped = lp.makeDualFeasible()
		}
		if flipped {
			lp.computePrimal()
		}
		if bad > 0 && !triedPhase1 && lp.DualPhase1 {
			triedPhase1 = true
			it0 := lp.Iter
			ok := lp.dualPhase1()
			lp.StatPhase1Iter += lp.Iter - it0
			if ok {
				bad = 0
			} else {
				lp.computeDual()
				bad, _ = lp.makeDualFeasible()
				lp.computePrimal()
			}
		}
		it0 := lp.Iter
		if bad == 0 {
			lp.StatDualCalls++
			st = lp.dualSimplex()
			lp.StatDualIter += lp.Iter - it0
			if lp.Debug {
				ni, si := lp.sumPrimalInfeas()
				println("dual call: round", round, "status", st.String(), "iters", lp.Iter-it0, "perturbed", lp.perturbed, "pinf", ni, si)
			}
		} else {
			lp.StatPrimalCalls++
			ninf, sinf := lp.sumPrimalInfeas()
			st = lp.primalSimplex()
			lp.StatPrimalIter += lp.Iter - it0
			if lp.Debug {
				println("primal call: round", round, "perturbed", lp.perturbed, "dual infeas", bad, "primal infeas", ninf, sinf, "iters", lp.Iter-it0, "status", st.String())
			}
		}
		switch st {
		case Unknown:
			// lost dual feasibility: loop picks the right algorithm
			lp.computeDual()
			continue
		case Optimal:
			if !lp.perturbed {
				// Verify with recomputed primal and dual values; a fresh
				// factorization is only taken when many updates accumulated.
				if lp.fac.NumUpdates() > VerifyUpd {
					if !lp.refactor() {
						return Numerical
					}
				} else {
					lp.computePrimal()
					lp.computeDual()
					if VerifyDebug {
						o1 := lp.Objective()
						n1, s1 := lp.sumPrimalInfeas()
						b1, ot1 := lp.dualInfeasCounts()
						lp.refactor()
						o2 := lp.Objective()
						n2, s2 := lp.sumPrimalInfeas()
						b2, ot2 := lp.dualInfeasCounts()
						if math.Abs(o1-o2) > 1e-7*(1+math.Abs(o2)) || n1 != n2 || b1+ot1 != b2+ot2 {
							println("verify mismatch: upd", lp.fac.NumUpdates(), "obj", o1, o2, "pinf", n1, s1, n2, s2, "dinf", b1+ot1, b2+ot2)
						}
					}
				}
				if n, _ := lp.sumPrimalInfeas(); n > 0 {
					continue
				}
				if boxed, other := lp.dualInfeasCounts(); boxed+other > 0 {
					cleanup = true
					continue
				}
				lp.polish()
				lp.warm = true
				return Optimal
			}
			lp.removePerturbation()
			if !lp.refactor() {
				return Numerical
			}
			cleanup = true
			continue
		case Infeasible:
			if lp.perturbed {
				lp.removePerturbation()
				lp.computeDual()
			}
			// Primal infeasibility does not depend on costs; confirm via
			// a fresh factorization once.
			if round == 0 {
				if !lp.refactor() {
					return Numerical
				}
				if n, _ := lp.sumPrimalInfeas(); n == 0 {
					continue
				}
			}
			return Infeasible
		case Cutoff:
			return Cutoff
		default:
			if lp.perturbed {
				lp.removePerturbation()
				lp.computeDual()
			}
			return st
		}
	}
	return Numerical
}

// SolveFromScratch resets to the slack basis and solves.
func (lp *LP) SolveFromScratch() Status {
	lp.initStateKeepProblem()
	return lp.Solve()
}

func (lp *LP) initStateKeepProblem() {
	lp.warm = false
	N := lp.n + lp.m
	for j := 0; j < N; j++ {
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
}

// polish removes remaining small dual infeasibilities (between PolishTol and
// DualTol) with a few primal simplex iterations from the optimal, primal
// feasible basis. This matters on badly scaled models where the scaled
// tolerance leaves a visible objective error.
func (lp *LP) polish() {
	if lp.PolishTol <= 0 || lp.PolishTol >= lp.DualTol {
		return
	}
	cnt := 0
	N := lp.n + lp.m
	for j := 0; j < N; j++ {
		if lp.status[j] == Basic || lp.lb[j] == lp.ub[j] {
			continue
		}
		d := lp.d[j]
		if (lp.status[j] == AtLower && d < -lp.PolishTol) || (lp.status[j] == AtUpper && d > lp.PolishTol) ||
			(lp.status[j] == AtZero && math.Abs(d) > lp.PolishTol) {
			cnt++
		}
	}
	if cnt == 0 {
		return
	}
	saveTol, saveMax := lp.DualTol, lp.MaxIter
	lp.DualTol = lp.PolishTol
	limit := lp.Iter + 1000 + lp.Iter/5
	if saveMax == 0 || limit < saveMax {
		lp.MaxIter = limit
	}
	it0 := lp.Iter
	lp.primalSimplex()
	lp.StatPolishIter += lp.Iter - it0
	lp.DualTol, lp.MaxIter = saveTol, saveMax
	if lp.refactor() {
		if n, _ := lp.sumPrimalInfeas(); n > 0 {
			// should not happen; primal simplex phase 2 keeps feasibility
			lp.primalSimplex()
		}
	}
}

// dualInfeasCounts counts nonbasic dual infeasibilities of boxed variables
// (fixable by a bound flip) and of the others.
func (lp *LP) dualInfeasCounts() (boxed, other int) {
	N := lp.n + lp.m
	for j := 0; j < N; j++ {
		if lp.status[j] == Basic || !lp.dualInfeasible(j) {
			continue
		}
		if !math.IsInf(lp.lb[j], -1) && !math.IsInf(lp.ub[j], 1) {
			boxed++
		} else {
			other++
		}
	}
	return
}

// VerifyUpd is the number of updates above which the optimality check uses
// a fresh factorization.
var VerifyUpd = int(envFloatSimplex("GOST_VERIFYUPD", 25))

// VerifyDebug compares the cheap verification with a refactorization.
var VerifyDebug = envFloatSimplex("GOST_VERIFYDEBUG", 0) != 0

// PrimalWarmOff disables the primal simplex on primal feasible warm starts
// (GOST_PRIMALWARM=0).
var PrimalWarmOff = os.Getenv("GOST_PRIMALWARM") == "0"
