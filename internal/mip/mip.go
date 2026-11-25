// Package mip implements a branch-and-cut solver for mixed-integer linear
// programs on top of the simplex package.
package mip

import (
	"fmt"
	"io"
	"math"
	"os"
	"strings"
	"time"

	"gost/internal/presolve"
	"gost/internal/problem"
)

// Status of a MIP solve.
type Status int

const (
	StatusUnknown Status = iota
	StatusOptimal
	StatusInfeasible
	StatusUnbounded
	StatusTimeLimit
	StatusNodeLimit
	StatusNumerical
)

func (s Status) String() string {
	return [...]string{"unknown", "optimal", "infeasible", "unbounded", "time limit", "node limit", "numerical trouble"}[s]
}

// Options controls the solver.
type Options struct {
	TimeLimit  time.Duration // 0 = none
	NodeLimit  int           // 0 = none
	RelGap     float64       // relative MIP gap tolerance
	AbsGap     float64       // absolute MIP gap tolerance
	IntTol     float64       // integrality tolerance
	FeasTol    float64       // primal feasibility tolerance
	Presolve   bool
	Cuts       bool
	CutRounds  int
	MIRCuts    bool
	Heuristics bool
	Log        io.Writer // nil = silent
	LogEvery   time.Duration
	// Strong branching / reliability.
	ReliabilityThreshold int
	StrongBranchIters    int
	StrongBranchCands    int
	// Initial incumbent cutoff (in minimization units including offset).
	Cutoff float64
	// DebugSolution is a known feasible solution (original space) used to
	// detect invalid reductions/cuts during development.
	DebugSolution []float64

	// Hooks are user callbacks (lazy constraints, user cuts, pricing).
	Hooks *Hooks

	// Start is a MIP start in the original space (NaN = value not given):
	// a complete feasible solution becomes the first incumbent; a partial
	// or infeasible one is repaired by a sub-MIP first.
	Start []float64

	noSubMIP  bool // set for sub-MIPs to avoid recursion
	noRestart bool
	heurDebug bool
}

// DefaultOptions returns sensible defaults.
func DefaultOptions() Options {
	return Options{
		RelGap: 1e-4, AbsGap: 1e-6, IntTol: 1e-6, FeasTol: 1e-6,
		Presolve: true, Cuts: true, CutRounds: 20, Heuristics: true, MIRCuts: os.Getenv("GOST_NOMIR") == "",
		LogEvery:             time.Second,
		ReliabilityThreshold: 4, StrongBranchIters: 60, StrongBranchCands: 10,
		Cutoff: math.Inf(1),
	}
}

// Result of a MIP solve (minimization form, original space).
type Result struct {
	Status    Status
	X         []float64 // best solution (nil if none)
	Objective float64   // objective of X (incl. offset)
	Bound     float64   // global lower bound
	Gap       float64   // relative gap
	Nodes     int
	LPIters   int
	Time      time.Duration
	RootBound float64
	Cuts      int
	// For pure LPs solved to optimality: duals of rows and reduced costs.
	Duals        []float64
	ReducedCosts []float64
	// What happened to Options.Start.
	StartStatus StartStatus
	StartInfo   string
}

func (o *Options) logf(format string, a ...any) {
	if o.Log != nil {
		fmt.Fprintf(o.Log, format, a...)
	}
}

// Solve solves the problem.
func Solve(p *problem.Problem, opt Options) *Result {
	t0 := time.Now()
	res := &Result{Objective: math.Inf(1), Bound: math.Inf(-1)}
	opt.logf("GoST MIP solver: %d rows, %d columns, %d nonzeros, %d integer\n",
		p.NumRows, p.NumCols, len(p.Val), countTrue(p.Integer))
	q := p
	var pre *presolve.Result
	if opt.Hooks.any() {
		// Indices must match the caller's model; dual reductions would be
		// invalid with constraints/columns unknown to presolve.
		opt.Presolve = false
		if opt.Hooks.Price != nil {
			opt.Cuts = false
			opt.StrongBranchCands = 0
		}
	}
	if countTrue(p.Integer) == 0 {
		// Pure LP: solve the full problem so duals are available.
		opt.Presolve = false
	}
	// MIP start: check it (or repair it) in the original space.
	var ext []float64 // feasible start kept outside the solver (see below)
	extObj := math.Inf(1)
	if opt.Start != nil {
		var st StartStatus
		var info string
		ext, extObj, st, info = prepareStart(p, opt)
		res.StartStatus, res.StartInfo = st, info
		switch st {
		case StartAccepted:
			opt.logf("MIP start accepted: objective %.10g\n", extObj)
		case StartRepaired:
			opt.logf("MIP start repaired: objective %.10g (%s)\n", extObj, info)
		case StartRejected:
			opt.logf("MIP start rejected: %s\n", info)
		}
	}
	if opt.Presolve {
		presolve.DebugSolution = opt.DebugSolution
		pre = presolve.Presolve(p)
		presolve.DebugSolution = nil
		switch pre.Status {
		case presolve.Infeasible:
			opt.logf("Presolve: problem is infeasible\n")
			res.Status = StatusInfeasible
			res.Time = time.Since(t0)
			return res
		case presolve.Unbounded:
			opt.logf("Presolve: problem is unbounded (or infeasible)\n")
			res.Status = StatusUnbounded
			res.Time = time.Since(t0)
			return res
		}
		q = pre.Reduced
		opt.logf("Presolve: removed %d rows, %d columns (%d aggregated); tightened %d bounds, %d coefficients\n",
			pre.RowsRemoved, pre.ColsRemoved, pre.Aggregated, pre.BoundsTightened, pre.CoefsTightened)
		opt.logf("Reduced problem: %d rows, %d columns, %d nonzeros\n", q.NumRows, q.NumCols, len(q.Val))
		if os.Getenv("GOST_SHOWINT") != "" {
			for j := 0; j < q.NumCols; j++ {
				if q.Integer[j] {
					opt.logf("  int col %d [%g,%g] obj %g nnz %d\n", j, q.Lo[j], q.Up[j], q.Obj[j], q.ColStart[j+1]-q.ColStart[j])
				}
			}
		}
	}
	opt.heurDebug = os.Getenv("GOST_HEURDEBUG") != ""
	// The start becomes the first incumbent of the solver when it is
	// feasible in the solver's problem (always without presolve; after
	// presolve, dual reductions may have cut it off). Otherwise it is kept
	// aside and only its objective is passed as a cutoff.
	var startX []float64
	if ext != nil {
		xs := ext
		if pre != nil {
			xs = make([]float64, q.NumCols)
			for k, j := range pre.ColMap {
				xs[k] = ext[j]
			}
		}
		if feasibleWithin(q, xs, opt.FeasTol, opt.IntTol) && os.Getenv("GOST_STARTCUTOFF") == "" {
			startX = xs
		} else {
			opt.Cutoff = math.Min(opt.Cutoff, extObj)
			opt.logf("MIP start not representable in the presolved problem: used as cutoff\n")
		}
	}
	s := newSolver(q, opt, t0)
	s.startX = startX
	if opt.DebugSolution != nil {
		if pre != nil {
			s.debugSol = make([]float64, q.NumCols)
			for k, j := range pre.ColMap {
				s.debugSol[k] = opt.DebugSolution[j]
			}
		} else {
			s.debugSol = opt.DebugSolution
		}
		if pre != nil {
			xb := pre.Postsolve(s.debugSol)
			nd := 0
			for j := range xb {
				if math.Abs(xb[j]-opt.DebugSolution[j]) > 1e-6 {
					nd++
				}
			}
			b, r, in := p.MaxViolation(xb)
			s.debugf("presolve: %d fixed cols differ from debug sol; orig obj %g, reduced obj+off %g, postsolved obj %g viol %g %g %g",
				nd, p.ObjValue(opt.DebugSolution), q.ObjValue(s.debugSol), p.ObjValue(xb), b, r, in)
			bq, rq, iq := q.MaxViolation(s.debugSol)
			s.debugf("debug sol violation in reduced problem: bound %g row %g int %g", bq, rq, iq)
		}
	}
	s.run()
	if s.startRejected {
		res.StartStatus = StartRejected
		res.StartInfo = strings.TrimPrefix(res.StartInfo+"; refused by lazy constraints or solver tolerances", "; ")
		ext = nil
	}
	res.Status = s.status
	res.Nodes = s.nodes
	if s.lp != nil {
		res.LPIters = s.lp.TotalIter + s.extraIters
	}
	res.RootBound = s.rootBound + q.Offset
	res.Cuts = s.numCuts
	if s.hasIncumbent {
		x := s.incumbent
		if pre != nil {
			x = pre.Postsolve(x)
		}
		res.X = x
		res.Objective = p.ObjValue(x)
	}
	if s.lp != nil && len(s.intVars) == 0 && pre == nil && s.status == StatusOptimal {
		res.Duals = s.problemDuals()
		res.ReducedCosts = make([]float64, s.n)
		for j := range res.ReducedCosts {
			res.ReducedCosts[j] = s.lp.ReducedCost(j)
		}
	}
	res.Bound = s.globalBound + q.Offset
	if s.status == StatusOptimal && s.hasIncumbent {
		res.Bound = math.Min(res.Bound, res.Objective)
	}
	if s.status == StatusInfeasible {
		res.Bound = math.Inf(1)
	}
	// Start kept outside the solver: it is the answer unless the solver
	// found something better ("infeasible" then means nothing better).
	if ext != nil && startX == nil && (res.X == nil || res.Objective > extObj) {
		res.X, res.Objective = ext, extObj
		switch res.Status {
		case StatusInfeasible, StatusOptimal:
			res.Status = StatusOptimal
			res.Bound = extObj
		default:
			res.Bound = math.Min(res.Bound, extObj)
		}
	}
	res.Gap = relGap(res.Objective, res.Bound)
	res.Time = time.Since(t0)
	s.logSummary(res)
	return res
}

func relGap(primal, dual float64) float64 {
	if math.IsInf(primal, 0) || math.IsInf(dual, 0) {
		return math.Inf(1)
	}
	if primal == dual {
		return 0
	}
	return math.Abs(primal-dual) / math.Max(math.Abs(primal), 1e-10)
}

func countTrue(b []bool) int {
	c := 0
	for _, v := range b {
		if v {
			c++
		}
	}
	return c
}
