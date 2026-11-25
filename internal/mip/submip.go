package mip

import (
	"math"
	"os"
	"time"
)

var subMIPCount int

// subMIP solves the problem restricted to bounds [lo, up] with a node and
// time limit and installs any improving solution. Returns true if an
// improving solution was found.
func (s *solver) subMIP(name string, lo, up []float64, nodeLimit int, timeLimit time.Duration) bool {
	if s.opt.noSubMIP || os.Getenv("GOST_NOSUBMIP") != "" {
		return false
	}
	// Effort is bounded by the node limit (deterministic); a time limit is
	// only imposed when the caller has one.
	timeLimit = 0
	if s.opt.TimeLimit > 0 {
		rem := s.opt.TimeLimit - time.Since(s.t0)
		if rem < 100*time.Millisecond {
			return false
		}
		timeLimit = rem / 2
	}
	q := s.p.Clone()
	copy(q.Lo, lo)
	copy(q.Up, up)
	o := DefaultOptions()
	o.noSubMIP = true
	o.noRestart = true
	o.NodeLimit = nodeLimit
	o.TimeLimit = timeLimit
	o.CutRounds = 5
	o.MIRCuts = s.opt.MIRCuts
	o.StrongBranchCands = 4
	o.RelGap = 1e-4
	if s.hasIncumbent {
		o.Cutoff = s.incObj + s.p.Offset - math.Max(1e-6, 1e-9*math.Abs(s.incObj))
	}
	t1 := time.Now()
	subMIPCount++
	if path := os.Getenv("GOST_DUMPSUBMIP"); path != "" && subMIPCount == 3 {
		debugWriteProblem(q, path)
	}
	r := Solve(q, o)
	if s.opt.heurDebug {
		println("submip", name, "status", r.Status.String(), "nodes", r.Nodes, "iters", r.LPIters, "obj", r.Objective, "found", r.X != nil, "dt", time.Since(t1).String())
	}
	if r.X == nil {
		return false
	}
	return s.newIncumbent(r.X, s.objOf(r.X), name)
}

// rens runs Relaxation Enforced Neighborhood Search around the current LP
// solution s.x (integral variables fixed, fractional ones restricted to
// their floor/ceil).
func (s *solver) rens() bool {
	lo := append([]float64(nil), s.lo0...)
	up := append([]float64(nil), s.up0...)
	fixed := 0
	for _, j := range s.intVars {
		v := s.x[j]
		if isIntegral(v, s.opt.IntTol) {
			r := math.Round(v)
			lo[j], up[j] = r, r
			fixed++
		} else {
			lo[j] = math.Max(lo[j], math.Floor(v))
			up[j] = math.Min(up[j], math.Ceil(v))
		}
	}
	if s.opt.heurDebug {
		println("RENS fixed", fixed, "of", len(s.intVars))
	}
	if float64(fixed) < 0.5*float64(len(s.intVars)) {
		return false
	}
	return s.subMIP("RENS", lo, up, 1000, 10*time.Second)
}

// rins runs Relaxation Induced Neighborhood Search: variables whose LP value
// agrees with the incumbent are fixed.
func (s *solver) rins() bool {
	if !s.hasIncumbent {
		return false
	}
	lo := append([]float64(nil), s.lo0...)
	up := append([]float64(nil), s.up0...)
	fixed := 0
	for _, j := range s.intVars {
		if math.Abs(s.incumbent[j]-s.x[j]) < 1e-6 {
			lo[j], up[j] = s.incumbent[j], s.incumbent[j]
			fixed++
		}
	}
	if float64(fixed) < 0.3*float64(len(s.intVars)) || fixed == len(s.intVars) {
		return false
	}
	return s.subMIP("RINS", lo, up, 500, 5*time.Second)
}
