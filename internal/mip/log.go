package mip

import (
	"fmt"
	"math"
	"time"
)

func fmtVal(v float64) string {
	if math.IsInf(v, 0) {
		return "-"
	}
	return fmt.Sprintf("%.6g", v)
}

func (s *solver) logHeader() {
	s.opt.logf("\n%8s %8s %14s %14s %8s %6s %8s\n", "Nodes", "Left", "BestInteger", "BestBound", "Gap", "Sols", "Time")
}

func (s *solver) logLine(src string, sol bool) {
	if s.opt.Log == nil {
		return
	}
	mark := " "
	if sol {
		mark = "*"
		if src != "" && src != "node" {
			mark = "H"
		}
	}
	inc := s.incObj + s.p.Offset
	bnd := s.globalBound + s.p.Offset
	gap := "-"
	if s.hasIncumbent && !math.IsInf(bnd, 0) {
		gap = fmt.Sprintf("%.2f%%", 100*relGap(inc, bnd))
	}
	if !s.hasIncumbent {
		inc = math.Inf(1)
	}
	s.opt.logf("%s%7d %8d %14s %14s %8s %6d %7.1fs\n", mark, s.nodes, s.open.Len(), fmtVal(inc), fmtVal(bnd), gap,
		s.numHeurSols, time.Since(s.t0).Seconds())
}

func (s *solver) logSummary(r *Result) {
	if treeStats {
		s.opt.logf("depth histogram: %v\nbranch vars: %v\n", tsDepth, tsBranchVar)
	}
	s.opt.logf("\nStatus: %s\n", r.Status)
	if r.X != nil {
		s.opt.logf("Objective: %.10g\n", r.Objective)
	}
	s.opt.logf("Best bound: %.10g, gap: %.4f%%\n", r.Bound, 100*r.Gap)
	if s.lp != nil {
		s.opt.logf("LP stats: dual %d calls/%d iters, primal %d calls/%d iters, perturbations %d\n",
			s.lp.StatDualCalls, s.lp.StatDualIter, s.lp.StatPrimalCalls, s.lp.StatPrimalIter, s.lp.StatPerturb)
	}
	s.opt.logf("Nodes: %d, LP iterations: %d, cuts: %d, time: %.2fs\n", r.Nodes, r.LPIters, r.Cuts, r.Time.Seconds())
}
