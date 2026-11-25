package mip

import (
	"math"
	"os"
	"sort"

	"gost/internal/simplex"
)

type brAction int

const (
	brOK brAction = iota
	brInfeasible
	brResolve
)

func (s *solver) updatePseudocost(nd *node, obj float64) {
	j := nd.brVar
	if j < 0 || math.IsInf(obj, 0) {
		return
	}
	gain := math.Max(obj-nd.parentO, 0)
	if nd.brUp {
		s.addPseudocost(j, true, gain, 1-nd.brFrac)
	} else {
		s.addPseudocost(j, false, gain, nd.brFrac)
	}
}

func (s *solver) addPseudocost(j int, up bool, gain, f float64) {
	if f <= 1e-6 {
		return
	}
	v := gain / f
	if up {
		if s.pcCntUp[j] > 0 {
			s.pcAvgSum[1] -= s.pcSumUp[j] / float64(s.pcCntUp[j])
		} else {
			s.pcAvgCnt[1]++
		}
		s.pcSumUp[j] += v
		s.pcCntUp[j]++
		s.pcAvgSum[1] += s.pcSumUp[j] / float64(s.pcCntUp[j])
	} else {
		if s.pcCntDown[j] > 0 {
			s.pcAvgSum[0] -= s.pcSumDown[j] / float64(s.pcCntDown[j])
		} else {
			s.pcAvgCnt[0]++
		}
		s.pcSumDown[j] += v
		s.pcCntDown[j]++
		s.pcAvgSum[0] += s.pcSumDown[j] / float64(s.pcCntDown[j])
	}
}

// pseudocost returns the per-unit pseudocost of j in the given direction,
// falling back to the average over initialized variables.
func (s *solver) pseudocost(j int, up bool) float64 {
	if up {
		if s.pcCntUp[j] > 0 {
			return s.pcSumUp[j] / float64(s.pcCntUp[j])
		}
	} else if s.pcCntDown[j] > 0 {
		return s.pcSumDown[j] / float64(s.pcCntDown[j])
	}
	d := 0
	if up {
		d = 1
	}
	if s.pcAvgCnt[d] == 0 {
		return 1
	}
	return math.Max(s.pcAvgSum[d]/float64(s.pcAvgCnt[d]), 0)
}

func score(d, u float64) float64 {
	return math.Max(d, 1e-6) * math.Max(u, 1e-6)
}

type brCand struct {
	j     int
	score float64
}

// selectBranchVar implements reliability branching: pseudocost scores with
// strong branching on unreliable candidates.
func (s *solver) selectBranchVar(nd *node, frac []int) (int, brAction) {
	obj := s.lp.Objective()
	cands := make([]brCand, len(frac))
	for k, j := range frac {
		f := s.x[j] - math.Floor(s.x[j])
		cands[k] = brCand{j, score(s.pseudocost(j, false)*f, s.pseudocost(j, true)*(1-f))}
	}
	sort.Slice(cands, func(a, b int) bool { return cands[a].score > cands[b].score })

	best := cands[0].j
	bestScore := cands[0].score
	if s.opt.StrongBranchCands <= 0 {
		return best, brOK
	}
	// Strong-branching effort quota: at most ~50% of the other LP work.
	other := s.lp.TotalIter - s.sbIters
	if s.sbIters > other/2+5000 && os.Getenv("GOST_NOSBQUOTA") == "" {
		return best, brOK
	}
	rel := s.opt.ReliabilityThreshold
	// Strong branching budget depends on depth.
	maxSB := s.opt.StrongBranchCands
	if nd.depth > 10 {
		maxSB = max(2, maxSB/2)
	}
	saved := s.lp.GetBasis()
	didSB := false
	noImprove := 0
	sbDone := 0
	for _, c := range cands {
		if sbDone >= maxSB || noImprove >= 4 {
			break
		}
		j := c.j
		if min(s.pcCntDown[j], s.pcCntUp[j]) >= rel {
			continue
		}
		if s.timeUp() {
			break
		}
		sbDone++
		didSB = true
		v := s.x[j]
		f := v - math.Floor(v)
		dObj, dInf := s.strongBranch(j, s.lo[j], math.Floor(v), &saved)
		uObj, uInf := s.strongBranch(j, math.Ceil(v), s.up[j], &saved)
		if dInf && uInf {
			s.lp.SetBasis(saved)
			return -1, brInfeasible
		}
		if dInf || uInf {
			if dInf {
				s.lo[j] = math.Ceil(v)
				s.sbChanges = append(s.sbChanges, bndChg{j, math.Ceil(v), math.Inf(1)})
			} else {
				s.up[j] = math.Floor(v)
				s.sbChanges = append(s.sbChanges, bndChg{j, math.Inf(-1), math.Floor(v)})
			}
			s.lp.SetBasis(saved)
			return -1, brResolve
		}
		gd := math.Max(dObj-obj, 0)
		gu := math.Max(uObj-obj, 0)
		s.addPseudocost(j, false, gd, f)
		s.addPseudocost(j, true, gu, 1-f)
		sc := score(gd, gu)
		if sc > bestScore {
			bestScore = sc
			best = j
			noImprove = 0
		} else {
			noImprove++
		}
	}
	if didSB {
		s.lp.SetBasis(saved)
		// Re-solve quickly to restore the factorization and primal values
		// consistent with the node (warm start: 0 iterations expected).
		s.syncLPBounds(s.lo, s.up)
		s.lp.Solve()
	}
	return best, brOK
}

// strongBranch solves the LP with bounds [l,u] on j using a limited number of
// dual simplex iterations. Returns the objective and whether infeasible.
func (s *solver) strongBranch(j int, l, u float64, saved *simplex.Basis) (float64, bool) {
	oldL, oldU := s.lpLo[j], s.lpUp[j]
	s.lp.SetColBounds(j, l, u)
	s.lp.MaxIter = s.opt.StrongBranchIters
	s.lp.ObjCutoff = s.cutoffValue()
	st := s.lp.Solve()
	obj := s.lp.Objective()
	s.sbIters += s.lp.Iter
	s.lp.MaxIter = 0
	s.lp.SetColBounds(j, oldL, oldU)
	s.lp.SetBasis(*saved)
	switch st {
	case simplex.Infeasible, simplex.Cutoff:
		return math.Inf(1), true
	case simplex.Optimal:
		// An integral strong-branching solution may improve the incumbent.
		return obj, false
	}
	return obj, false
}
