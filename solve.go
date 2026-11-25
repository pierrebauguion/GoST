package gost

import (
	"io"
	"math"
	"time"

	"gost/internal/mip"
)

// Status is the outcome of a solve.
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

// Params are solver parameters.
type Params struct {
	TimeLimit  time.Duration // 0 = no limit
	NodeLimit  int           // 0 = no limit
	RelGap     float64       // relative MIP gap (default 1e-4)
	AbsGap     float64       // absolute MIP gap (default 1e-6)
	IntTol     float64       // integrality tolerance (default 1e-6)
	FeasTol    float64       // feasibility tolerance (default 1e-6)
	Presolve   bool
	Cuts       bool
	CutRounds  int
	Heuristics bool
	// Strong branching: number of candidates (0 disables) and iteration cap.
	StrongBranchCands int
	StrongBranchIters int
	// Reliability threshold for pseudocosts.
	Reliability int
	// Log receives the solver log (nil = silent).
	Log io.Writer
	// LazyConstraints is called with every candidate integer solution that
	// satisfies the current constraints. Adding constraints through the
	// context rejects the candidate (row generation, e.g. TSP subtours).
	// For pure LPs it is called with each optimal LP solution.
	LazyConstraints func(*ConstrContext)
	// UserCuts is called with fractional LP solutions (root and nodes); the
	// constraints it adds must be valid for every integer solution.
	UserCuts func(*ConstrContext)
	// Pricing is called after each LP solve to generate columns (column
	// generation / branch-and-price). Pruning uses the LP bound only once
	// pricing has converged; nodes whose LP is infeasible are pruned, so
	// include feasibility (artificial) columns in the master problem.
	Pricing func(*PricingContext)
	// When any callback is set, presolve is disabled; with Pricing, the
	// built-in cuts, strong branching and propagation are disabled too.

	// DebugSolution (development aid): a known feasible solution indexed by
	// variable; the solver reports cuts/reductions that cut it off.
	DebugSolution []float64
}

// DefaultParams returns the default parameters.
func DefaultParams() Params {
	o := mip.DefaultOptions()
	return Params{
		RelGap: o.RelGap, AbsGap: o.AbsGap, IntTol: o.IntTol, FeasTol: o.FeasTol,
		Presolve: true, Cuts: true, CutRounds: o.CutRounds, Heuristics: true,
		StrongBranchCands: o.StrongBranchCands, StrongBranchIters: o.StrongBranchIters,
		Reliability: o.ReliabilityThreshold,
	}
}

// Result holds the outcome of Solve.
type Result struct {
	Status    Status
	Objective float64 // objective of the best solution (model sense)
	Bound     float64 // best bound (model sense)
	Gap       float64 // relative gap
	Nodes     int
	LPIters   int
	Cuts      int
	Time      time.Duration
	// StartStatus and StartInfo report what happened to the MIP start.
	StartStatus StartStatus
	StartInfo   string
	x           []float64
	duals       []float64
	redCosts    []float64
}

// Dual returns the dual value of constraint c for pure LP models (model
// sense: d_j = obj_j - sum_i a_ij * Dual(c_i)); NaN if unavailable.
func (r *Result) Dual(c Constr) float64 {
	if c.idx >= len(r.duals) {
		return math.NaN()
	}
	return r.duals[c.idx]
}

// ReducedCost returns the reduced cost of v for pure LP models (model
// sense); NaN if unavailable.
func (r *Result) ReducedCost(v Var) float64 {
	if v.idx >= len(r.redCosts) {
		return math.NaN()
	}
	return r.redCosts[v.idx]
}

// HasSolution reports whether a feasible solution is available.
func (r *Result) HasSolution() bool { return r.x != nil }

// Value returns the value of v in the best solution (NaN if none).
func (r *Result) Value(v Var) float64 {
	if r.x == nil {
		return math.NaN()
	}
	return r.x[v.idx]
}

// Values returns a copy of the solution vector.
func (r *Result) Values() []float64 { return append([]float64(nil), r.x...) }

// Solve optimizes the model.
func (m *Model) Solve(p Params) (*Result, error) {
	prob := m.toProblem()
	o := mip.DefaultOptions()
	o.TimeLimit = p.TimeLimit
	o.NodeLimit = p.NodeLimit
	if p.RelGap > 0 {
		o.RelGap = p.RelGap
	}
	if p.AbsGap > 0 {
		o.AbsGap = p.AbsGap
	}
	if p.IntTol > 0 {
		o.IntTol = p.IntTol
	}
	if p.FeasTol > 0 {
		o.FeasTol = p.FeasTol
	}
	o.Presolve = p.Presolve
	o.Cuts = p.Cuts
	o.CutRounds = p.CutRounds
	o.Heuristics = p.Heuristics
	o.StrongBranchCands = p.StrongBranchCands
	o.StrongBranchIters = p.StrongBranchIters
	o.ReliabilityThreshold = p.Reliability
	o.Log = p.Log
	o.DebugSolution = p.DebugSolution
	o.Hooks = m.hooks(p)
	o.Start = m.startVector()
	r := mip.Solve(prob, o)
	res := &Result{
		Status: Status(r.Status), Objective: r.Objective, Bound: r.Bound, Gap: r.Gap,
		Nodes: r.Nodes, LPIters: r.LPIters, Cuts: r.Cuts, Time: r.Time, x: r.X,
		StartStatus: StartStatus(r.StartStatus), StartInfo: r.StartInfo,
	}
	sign := 1.0
	if m.sense == Maximize {
		sign = -1
		res.Objective, res.Bound = -res.Objective, -res.Bound
	}
	for _, y := range r.Duals {
		res.duals = append(res.duals, sign*y)
	}
	for _, d := range r.ReducedCosts {
		res.redCosts = append(res.redCosts, sign*d)
	}
	return res, nil
}
