package mip

import (
	"math"
	"os"

	"gost/internal/simplex"
)

// simpleRounding rounds fractional integers in a lock-free direction.
func (s *solver) simpleRounding() bool {
	xr := append([]float64(nil), s.x...)
	for _, j := range s.intVars {
		v := xr[j]
		if isIntegral(v, s.opt.IntTol) {
			xr[j] = math.Round(v)
			continue
		}
		switch {
		case s.downLocks[j] == 0 && math.Floor(v) >= s.lo0[j]:
			xr[j] = math.Floor(v)
		case s.upLocks[j] == 0 && math.Ceil(v) <= s.up0[j]:
			xr[j] = math.Ceil(v)
		default:
			return false
		}
	}
	return s.newIncumbent(xr, s.objOf(xr), "rounding")
}

// dive runs a guided/fractional diving heuristic from the current LP
// solution with propagation and limited backtracking. The LP is restored to
// the node state (s.lo, s.up and the current basis) afterwards.
func (s *solver) dive(depth int, rounds int) {
	if os.Getenv("GOST_NODIVE") != "" {
		return
	}
	saved := s.lp.GetBasis()
	dlo := make([]float64, s.n)
	dup := make([]float64, s.n)
	baseX := append([]float64(nil), s.x...)
	iterBudget := 2*s.lp.TotalIter/(s.nodes+1) + 2000
	for round := 0; round < rounds; round++ {
		copy(dlo, s.lo)
		copy(dup, s.up)
		copy(s.x, baseX)
		if round > 0 {
			s.lp.SetBasis(saved)
		}
		used := 0
		for step := 0; step < 4*len(s.intVars)+10; step++ {
			if s.timeUp() || used > iterBudget {
				break
			}
			// pick variable
			bestJ := -1
			bestF := math.Inf(1)
			var bestUp bool
			for _, j := range s.intVars {
				v := s.x[j]
				if isIntegral(v, s.opt.IntTol) {
					continue
				}
				f := v - math.Floor(v)
				var up bool
				var dist float64
				switch {
				case round == 0 && s.hasIncumbent:
					up = s.incumbent[j] > v
					dist = math.Abs(s.incumbent[j] - v)
				case round == 1 || (round == 0 && !s.hasIncumbent):
					// coefficient diving: direction with fewer locks
					if s.downLocks[j] < s.upLocks[j] {
						up = false
						dist = f + float64(s.downLocks[j])
					} else if s.upLocks[j] < s.downLocks[j] {
						up = true
						dist = 1 - f + float64(s.upLocks[j])
					} else {
						up = f > 0.5
						dist = math.Min(f, 1-f) + float64(s.upLocks[j])
					}
				default:
					up = f > 0.5
					dist = math.Min(f, 1-f)
				}
				if dist < bestF {
					bestF, bestJ, bestUp = dist, j, up
				}
			}
			if bestJ < 0 {
				// integral LP solution
				xs := append([]float64(nil), s.x...)
				for _, j := range s.intVars {
					xs[j] = math.Round(xs[j])
				}
				s.newIncumbent(xs, s.objOf(xs), "diving")
				break
			}
			j := bestJ
			v := s.x[j]
			ok := false
			for attempt := 0; attempt < 2 && !ok; attempt++ {
				ol, ou := dlo[j], dup[j]
				tlo := append([]float64(nil), dlo...)
				tup := append([]float64(nil), dup...)
				if bestUp {
					tlo[j] = math.Ceil(v)
				} else {
					tup[j] = math.Floor(v)
				}
				_ = ol
				_ = ou
				if s.propagate(tlo, tup, []int{j}) {
					s.syncLPBounds(tlo, tup)
					s.lp.MaxIter = iterBudget - used
					s.lp.ObjCutoff = s.cutoffValue()
					st := s.lp.Solve()
					used += s.lp.Iter
					s.lp.MaxIter = 0
					if st == simplex.Optimal {
						copy(dlo, tlo)
						copy(dup, tup)
						ok = true
						break
					}
					if st != simplex.Infeasible && st != simplex.Cutoff {
						break
					}
				}
				bestUp = !bestUp
			}
			if !ok {
				if s.opt.heurDebug {
					println("dive round", round, "failed at step", step, "iters used", used)
				}
				break
			}
			s.lp.Primal(s.x)
			if s.simpleRounding() {
				// keep diving for better solutions
			}
		}
	}
	// Restore node LP.
	s.lp.SetBasis(saved)
	s.syncLPBounds(s.lo, s.up)
	s.lp.ObjCutoff = s.cutoffValue()
	s.lp.Solve()
	s.lp.Primal(s.x)
}
