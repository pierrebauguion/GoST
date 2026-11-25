package mip

import (
	"container/heap"
	"math"
	"os"
	"time"

	"gost/internal/problem"
	"gost/internal/simplex"
)

type bndChg struct {
	j      int
	lo, up float64
}

type node struct {
	bound    float64 // lower bound (LP objective of parent or own)
	estimate float64
	depth    int
	changes  []bndChg
	basis    *simplex.Basis
	// Branching info for pseudocost update.
	brVar   int
	brUp    bool
	brFrac  float64
	parentO float64
	idx     int // heap index
}

type solver struct {
	p   *problem.Problem
	opt Options
	t0  time.Time

	n, m int
	// Row-wise copy of the original rows.
	rowStart []int
	rowCol   []int
	rowVal   []float64

	lp *simplex.LP

	lo0, up0 []float64 // global bounds
	lo, up   []float64 // current local bounds
	lpLo     []float64 // bounds currently set in the LP
	lpUp     []float64

	isInt   []bool
	intVars []int

	downLocks, upLocks []int

	// Incumbent (LP objective units, without offset).
	hasIncumbent bool
	incumbent    []float64
	incObj       float64
	objIntegral  bool

	rootBound   float64
	globalBound float64
	numCuts     int

	// Pseudocosts.
	pcSumDown, pcSumUp []float64
	pcCntDown, pcCntUp []int
	pcAvgSum           [2]float64
	pcAvgCnt           [2]int
	lastBasis          *simplex.Basis
	dynPending         string    // "rows"/"cols" added since the last LP solve
	startX             []float64 // MIP start (solver space), installed in run
	startRejected      bool      // the start was refused when installed
	sbIters            int

	open    nodeHeap
	nodes   int
	status  Status
	lastLog time.Time
	x       []float64

	numHeurSols int
	sbChanges   []bndChg
	inQueue     []bool
	queue       []int
	debugSol    []float64

	// Dynamic rows/columns.
	lpRowOf        []int  // LP row of each problem row
	lpRowIsCut     []bool // per LP row
	lpChanged      bool   // rows/columns were added since last check
	numPricedCols  int
	curDepth       int
	maxRowLenCache int
	extraIters     int
	objIdx         []int
	userCutRounds  int
}

func newSolver(p *problem.Problem, opt Options, t0 time.Time) *solver {
	s := &solver{p: p, opt: opt, t0: t0, n: p.NumCols, m: p.NumRows}
	s.rowStart, s.rowCol, s.rowVal = p.RowWise()
	s.lo0 = append([]float64(nil), p.Lo...)
	s.up0 = append([]float64(nil), p.Up...)
	s.lo = make([]float64, s.n)
	s.up = make([]float64, s.n)
	s.isInt = p.Integer
	for j := 0; j < s.n; j++ {
		if p.Integer[j] {
			s.intVars = append(s.intVars, j)
		}
	}
	s.x = make([]float64, s.n)
	s.pcSumDown = make([]float64, s.n)
	s.pcSumUp = make([]float64, s.n)
	s.pcCntDown = make([]int, s.n)
	s.pcCntUp = make([]int, s.n)
	s.computeLocks()
	s.detectObjIntegral()
	s.incObj = math.Inf(1)
	if !math.IsInf(opt.Cutoff, 1) {
		s.incObj = opt.Cutoff - p.Offset
	}
	s.globalBound = math.Inf(-1)
	return s
}

func (s *solver) computeLocks() {
	s.downLocks = make([]int, s.n)
	s.upLocks = make([]int, s.n)
	for i := 0; i < s.m; i++ {
		hasL := !math.IsInf(s.p.RowLo[i], -1)
		hasU := !math.IsInf(s.p.RowUp[i], 1)
		for k := s.rowStart[i]; k < s.rowStart[i+1]; k++ {
			j, v := s.rowCol[k], s.rowVal[k]
			if (v > 0 && hasL) || (v < 0 && hasU) {
				s.downLocks[j]++
			}
			if (v > 0 && hasU) || (v < 0 && hasL) {
				s.upLocks[j]++
			}
		}
	}
}

func (s *solver) detectObjIntegral() {
	s.objIntegral = len(s.intVars) > 0
	for j := 0; j < s.n; j++ {
		c := s.p.Obj[j]
		if c == 0 {
			continue
		}
		if !s.isInt[j] || math.Abs(c-math.Round(c)) > 1e-9 {
			s.objIntegral = false
			return
		}
	}
}

func (s *solver) timeUp() bool {
	return s.opt.TimeLimit > 0 && time.Since(s.t0) > s.opt.TimeLimit
}

// cutoffValue returns the LP objective value above which nodes are pruned.
func (s *solver) cutoffValue() float64 {
	if math.IsInf(s.incObj, 1) {
		return math.Inf(1)
	}
	if s.objIntegral {
		// Next better integral value (objective is integral + Offset frac).
		off := s.p.Offset - math.Floor(s.p.Offset)
		target := math.Ceil(s.incObj+off-1e-6) - 1 - off
		return target + 1e-6
	}
	return s.incObj - math.Max(s.opt.AbsGap, 1e-9*math.Abs(s.incObj+s.p.Offset))
}

// pruneBound reports whether a node with the given bound can be pruned.
func (s *solver) pruneBound(b float64) bool {
	return b > s.cutoffValue()
}

// ---------------------------------------------------------------------------

func (s *solver) run() {
	n := s.n
	if n == 0 && !s.pricing() {
		s.hasIncumbent = s.rowsFeasible(nil)
		s.incumbent = []float64{}
		s.incObj = 0
		s.status = StatusOptimal
		if !s.hasIncumbent {
			s.status = StatusInfeasible
		}
		return
	}
	copy(s.lo, s.lo0)
	copy(s.up, s.up0)
	// Root propagation.
	if !s.propagate(s.lo, s.up, nil) {
		s.status = StatusInfeasible
		return
	}
	copy(s.lo0, s.lo)
	copy(s.up0, s.up)
	if s.debugSol != nil {
		s.debugf("debug solution feasible in reduced problem: %v, in root bounds: %v, obj %g", s.rowsFeasible(s.debugSol), s.debugInBounds(s.lo0, s.up0), s.objOf(s.debugSol))
	}

	p := s.p
	s.lp = simplex.New(n, s.m, p.Obj, s.lo0, s.up0, p.ColStart, p.RowIdx, p.Val, p.RowLo, p.RowUp)
	s.lp.Debug = os.Getenv("GOST_LPDEBUG") != ""
	s.lpLo = append([]float64(nil), s.lo0...)
	s.lpUp = append([]float64(nil), s.up0...)
	if s.opt.TimeLimit > 0 {
		s.lp.Deadline = s.t0.Add(s.opt.TimeLimit)
	}
	s.lpRowOf = make([]int, s.m)
	s.lpRowIsCut = make([]bool, s.m)
	for i := range s.lpRowOf {
		s.lpRowOf[i] = i
	}
	if s.startX != nil {
		// MIP start: first incumbent (also checked by lazy constraints).
		if !s.newIncumbent(s.startX, s.objOf(s.startX), "start") {
			s.startRejected = true
			s.opt.logf("MIP start rejected by the solver (lazy constraints or tolerances)\n")
		}
		s.startX = nil
	}

	// Root LP with pricing, lazy constraints and user cuts.
	for round := 0; ; round++ {
		st := s.solveLP(0)
		switch st {
		case simplex.Infeasible:
			s.status = StatusInfeasible
			return
		case simplex.Cutoff:
			// The root bound already reaches the incumbent (e.g. a MIP
			// start) or the cutoff: nothing better exists.
			s.status = StatusInfeasible
			if s.hasIncumbent {
				s.status = StatusOptimal
				s.globalBound = s.incObj
			}
			return
		case simplex.Unbounded:
			s.status = StatusUnbounded
			return
		case simplex.Optimal:
		case simplex.TimeLimit:
			s.status = StatusTimeLimit
			return
		default:
			s.status = StatusNumerical
			return
		}
		s.lp.Primal(s.x)
		s.lpChanged = false
		if len(s.fractionals()) == 0 {
			if s.tryIncumbentFromLP() {
				s.rootBound = s.lp.Objective()
				s.globalBound = s.incObj
				s.status = StatusOptimal
				s.logRoot()
				return
			}
			if s.lpChanged {
				continue // lazy constraints were added
			}
			if len(s.intVars) == 0 {
				// numerically borderline LP point: accept it
				s.hasIncumbent = true
				s.incumbent = append(s.incumbent[:0], s.x...)
				s.incObj = s.lp.Objective()
				s.rootBound, s.globalBound = s.incObj, s.incObj
				s.status = StatusOptimal
				s.logRoot()
				return
			}
		}
		if s.opt.Hooks != nil && s.opt.Hooks.Cut != nil && round < max(s.opt.CutRounds, 1) {
			if rows := s.opt.Hooks.Cut(append([]float64(nil), s.x...), 0, 0); len(rows) > 0 {
				s.addRows(rows)
				continue
			}
		}
		break
	}
	s.rootBound = s.lp.Objective()
	s.globalBound = s.rootBound
	s.logRoot()

	if s.opt.Heuristics {
		s.simpleRounding()
	}

	// Cutting plane loop.
	if s.opt.Cuts {
		s.cutLoop()
		if s.status != StatusUnknown {
			return
		}
		s.lp.Primal(s.x)
		if s.tryIncumbentFromLP() {
			s.status = StatusOptimal
			s.globalBound = s.incObj
			return
		}
	}
	s.rootBound = s.lp.Objective()
	s.globalBound = s.rootBound

	if s.opt.Heuristics {
		s.simpleRounding()
		s.dive(0, 3)
		if !s.gapClosed() {
			xs := append([]float64(nil), s.x...)
			s.rens()
			if s.hasIncumbent {
				s.rins()
			}
			copy(s.x, xs)
		}
	}
	s.reducedCostFixing(true)
	if s.debugSol != nil && !s.debugInBounds(s.lo0, s.up0) {
		s.debugf("root reduced cost fixing cut off debug solution")
	}
	if s.restart() {
		return
	}

	s.branchAndBound()
}

// restart re-solves the problem from scratch (presolve, root, cuts) with the
// global bounds tightened at the root when enough integer variables were
// fixed (mostly by reduced cost fixing), as HiGHS and SCIP do. Returns true
// if the restarted solve replaced the tree search.
func (s *solver) restart() bool {
	if s.opt.noRestart || s.opt.Hooks.any() || !s.hasIncumbent || len(s.intVars) == 0 || os.Getenv("GOST_NORESTART") != "" {
		return false
	}
	// propagate the fixings first
	if !s.propagate(s.lo0, s.up0, nil) {
		// the incumbent is optimal: nothing better exists
		s.status = StatusOptimal
		s.globalBound = s.incObj
		return true
	}
	fixed := 0
	for _, j := range s.intVars {
		if s.lo0[j] == s.up0[j] {
			fixed++
		}
	}
	if float64(fixed) < 0.2*float64(len(s.intVars)) {
		return false
	}
	q := s.p.Clone()
	copy(q.Lo, s.lo0)
	copy(q.Up, s.up0)
	// keep the root cuts still in the LP (globally valid)
	var cutRows []RowSpec
	for i := 0; i < s.lp.NumRows(); i++ {
		if i < len(s.lpRowIsCut) && s.lpRowIsCut[i] {
			idx, val := s.lp.RowCoefs(i)
			lo, up := s.lp.VarBounds(s.n + i)
			cutRows = append(cutRows, RowSpec{Idx: append([]int(nil), idx...), Val: val, Lo: lo, Up: up})
		}
	}
	appendRows(q, cutRows)
	if path := os.Getenv("GOST_DUMPRESTART"); path != "" {
		debugWriteProblem(q, path)
	}
	o := s.opt
	o.noRestart = true
	o.DebugSolution = nil
	o.Start = nil
	// only solutions better than the incumbent by more than the gap
	// tolerance are of interest: "infeasible" then proves optimality
	inc := s.incObj + s.p.Offset
	o.Cutoff = inc - math.Max(o.AbsGap, o.RelGap*math.Abs(inc))
	if o.TimeLimit > 0 {
		o.TimeLimit -= time.Since(s.t0)
		if o.TimeLimit <= 0 {
			return false
		}
	}
	s.opt.logf("Restart: %d of %d integer variables fixed, re-solving\n", fixed, len(s.intVars))
	r := Solve(q, o)
	s.nodes += r.Nodes
	s.extraIters += r.LPIters
	if r.X != nil {
		s.newIncumbent(r.X, s.objOf(r.X), "restart")
	}
	switch r.Status {
	case StatusOptimal, StatusInfeasible:
		// infeasible means nothing better than the cutoff exists
		s.status = StatusOptimal
		s.globalBound = s.incObj
	case StatusTimeLimit, StatusNodeLimit:
		s.status = r.Status
		if !math.IsInf(r.Bound, 0) {
			s.globalBound = math.Max(s.globalBound, math.Min(r.Bound-s.p.Offset, s.incObj))
		}
	default:
		return false
	}
	return true
}

func (s *solver) logRoot() {
	s.opt.logf("Root relaxation: objective %.10g, %d iterations, %.2fs\n",
		s.lp.Objective()+s.p.Offset, s.lp.TotalIter, time.Since(s.t0).Seconds())
	if s.numPricedCols > 0 {
		s.opt.logf("Pricing: %d columns generated\n", s.numPricedCols)
	}
}

func (s *solver) tryIncumbentFromLP() bool {
	for _, j := range s.intVars {
		if !isIntegral(s.x[j], s.opt.IntTol) {
			return false
		}
	}
	xs := append([]float64(nil), s.x...)
	for _, j := range s.intVars {
		xs[j] = math.Round(xs[j])
	}
	return s.newIncumbent(xs, s.objOf(xs), "LP")
}

func isIntegral(v, tol float64) bool { return math.Abs(v-math.Round(v)) <= tol }

func (s *solver) objOf(x []float64) float64 {
	o := 0.0
	for j, c := range s.p.Obj {
		o += c * x[j]
	}
	return o
}

// rowsFeasible checks x against original rows and bounds (global).
func (s *solver) rowsFeasible(x []float64) bool {
	tol := s.opt.FeasTol
	for j := 0; j < s.n; j++ {
		if x[j] < s.p.Lo[j]-tol || x[j] > s.p.Up[j]+tol {
			return false
		}
	}
	for i := 0; i < s.m; i++ {
		a := 0.0
		for k := s.rowStart[i]; k < s.rowStart[i+1]; k++ {
			a += s.rowVal[k] * x[s.rowCol[k]]
		}
		sc := tol * math.Max(1, math.Abs(a))
		if a < s.p.RowLo[i]-sc || a > s.p.RowUp[i]+sc {
			return false
		}
	}
	return true
}

// newIncumbent installs x if it is feasible and better. Returns true if
// accepted.
func (s *solver) newIncumbent(x []float64, obj float64, src string) bool {
	if obj >= s.incObj-1e-9 {
		return false
	}
	if !s.rowsFeasible(x) {
		return false
	}
	if s.opt.Hooks != nil && s.opt.Hooks.Lazy != nil {
		if rows := s.opt.Hooks.Lazy(append([]float64(nil), x...)); len(rows) > 0 {
			s.addRows(rows)
			return false
		}
	}
	s.hasIncumbent = true
	s.incumbent = append(s.incumbent[:0], x...)
	s.incObj = obj
	if src != "LP" && src != "node" {
		s.numHeurSols++
	}
	s.logLine(src, true)
	if s.lp != nil {
		s.lp.ObjCutoff = s.cutoffValue()
	}
	return true
}

// ---------------------------------------------------------------------------
// LP bound management.

func (s *solver) syncLPBounds(lo, up []float64) {
	for j := 0; j < s.n; j++ {
		if lo[j] != s.lpLo[j] || up[j] != s.lpUp[j] {
			s.lp.SetColBounds(j, lo[j], up[j])
			s.lpLo[j], s.lpUp[j] = lo[j], up[j]
		}
	}
}

// solveLP solves the LP with the current local bounds. Returns the simplex
// status.
func (s *solver) solveLP(iterLimit int) simplex.Status {
	s.syncLPBounds(s.lo, s.up)
	s.lp.MaxIter = iterLimit
	s.lp.ObjCutoff = s.cutoffValue()
	if s.pricing() {
		s.lp.ObjCutoff = math.Inf(1) // bound not valid before pricing converges
	}
	st := s.dynSolve()
	s.lp.MaxIter = 0
	return s.priceLoop(st, s.curDepth)
}

// reducedCostFixing tightens bounds using reduced costs of the current LP
// optimum. If global is true the global bounds are tightened.
func (s *solver) reducedCostFixing(global bool) int {
	if math.IsInf(s.incObj, 1) {
		return 0
	}
	z := s.lp.Objective()
	gap := s.cutoffValue() - z
	if gap < 0 {
		return 0
	}
	lo, up := s.lo, s.up
	if global {
		lo, up = s.lo0, s.up0
	}
	cnt := 0
	for _, j := range s.intVars {
		st := s.lp.VarStatus(j)
		d := s.lp.ReducedCost(j)
		if st == simplex.AtLower && d > 1e-7 {
			nu := lo[j] + math.Floor(gap/d+1e-9)
			if nu < up[j] {
				up[j] = nu
				cnt++
			}
		} else if st == simplex.AtUpper && d < -1e-7 {
			nl := up[j] - math.Floor(gap/(-d)+1e-9)
			if nl > lo[j] {
				lo[j] = nl
				cnt++
			}
		}
	}
	if global && cnt > 0 {
		for _, j := range s.intVars {
			s.lo[j] = math.Max(s.lo[j], s.lo0[j])
			s.up[j] = math.Min(s.up[j], s.up0[j])
		}
	}
	return cnt
}

// ---------------------------------------------------------------------------
// Branch and bound.

func (s *solver) branchAndBound() {
	root := &node{bound: s.rootBound, brVar: -1}
	b := s.lp.GetBasis()
	root.basis = &b
	heap.Init(&s.open)
	heap.Push(&s.open, root)
	s.lastLog = time.Now()
	s.logHeader()

	var cur *node
	for {
		if cur == nil {
			// Prune and pick best-bound node.
			for s.open.Len() > 0 && s.pruneBound(s.open[0].bound) {
				// heap min is prunable => all prunable
				s.open = s.open[:0]
			}
			if s.open.Len() == 0 {
				break
			}
			cur = heap.Pop(&s.open).(*node)
		}
		s.updateGlobalBound(cur)
		if s.gapClosed() {
			break
		}
		if s.timeUp() {
			heap.Push(&s.open, cur)
			s.status = StatusTimeLimit
			s.updateGlobalBound(nil)
			return
		}
		if s.opt.NodeLimit > 0 && s.nodes >= s.opt.NodeLimit {
			heap.Push(&s.open, cur)
			s.status = StatusNodeLimit
			s.updateGlobalBound(nil)
			return
		}
		down, up := s.processNode(cur)
		cur = nil
		// Plunge: continue with a child if promising.
		var next *node
		if down != nil && up != nil {
			// prefer direction with better (smaller) estimated bound
			if s.preferUp(down, up) {
				next = up
				heap.Push(&s.open, down)
			} else {
				next = down
				heap.Push(&s.open, up)
			}
		} else if down != nil {
			next = down
		} else if up != nil {
			next = up
		}
		if next != nil {
			best := next.bound
			if s.open.Len() > 0 {
				best = math.Min(best, s.open[0].bound)
			}
			if !s.hasIncumbent || next.bound-best <= 0.25*(s.incObj-best)+1e-9 {
				cur = next
			} else {
				heap.Push(&s.open, next)
			}
		}
		if time.Since(s.lastLog) > s.opt.LogEvery {
			s.lastLog = time.Now()
			s.logLine("", false)
		}
	}
	s.status = StatusOptimal
	if !s.hasIncumbent {
		s.status = StatusInfeasible
	}
	if s.hasIncumbent {
		s.globalBound = math.Max(s.globalBound, math.Min(s.incObj, s.boundOfOpen()))
		s.globalBound = math.Min(s.globalBound, s.incObj)
	}
}

func (s *solver) boundOfOpen() float64 {
	if s.open.Len() == 0 {
		return s.incObj
	}
	return s.open[0].bound
}

func (s *solver) updateGlobalBound(cur *node) {
	b := math.Inf(1)
	if s.open.Len() > 0 {
		b = s.open[0].bound
	}
	if cur != nil {
		b = math.Min(b, cur.bound)
	}
	if math.IsInf(b, 1) {
		b = s.incObj
	}
	if b > s.globalBound {
		s.globalBound = b
	}
}

func (s *solver) gapClosed() bool {
	if !s.hasIncumbent {
		return false
	}
	gap := s.incObj - s.globalBound
	if gap <= s.opt.AbsGap {
		return true
	}
	obj := s.incObj + s.p.Offset
	if gap <= s.opt.RelGap*math.Max(math.Abs(obj), 1e-10) {
		return true
	}
	return s.pruneBound(s.globalBound)
}

func (s *solver) preferUp(down, up *node) bool {
	// Go in the direction of the rounding of the LP value; ties go up.
	return up.brFrac >= 0.5 || up.bound < down.bound
}

// processNode solves a node and returns its children (nil if pruned).
var treeStats = os.Getenv("GOST_TREESTATS") != ""
var tsDepth = map[int]int{}
var tsBranchVar = map[int]int{}

func (s *solver) processNode(nd *node) (*node, *node) {
	if treeStats {
		tsDepth[nd.depth]++
		if nd.brVar >= 0 {
			tsBranchVar[nd.brVar]++
		}
	}
	s.nodes++
	s.sbChanges = s.sbChanges[:0]
	if s.pruneBound(nd.bound) {
		return nil, nil
	}
	// Set up local bounds.
	copy(s.lo, s.lo0)
	copy(s.up, s.up0)
	changed := make([]int, 0, len(nd.changes))
	for _, c := range nd.changes {
		if c.lo > s.lo[c.j] {
			s.lo[c.j] = c.lo
		}
		if c.up < s.up[c.j] {
			s.up[c.j] = c.up
		}
		if s.lo[c.j] > s.up[c.j]+1e-9 {
			return nil, nil
		}
		changed = append(changed, c.j)
	}
	debugHere := s.debugSol != nil && s.debugInBounds(s.lo, s.up)
	if !s.propagate(s.lo, s.up, changed) {
		if debugHere {
			s.debugf("node %d: propagation pruned debug node", s.nodes)
		}
		return nil, nil
	}
	if debugHere && !s.debugInBounds(s.lo, s.up) {
		s.debugf("node %d: propagation cut off debug solution", s.nodes)
	}
	if nd.basis != nil && nd.basis != s.lastBasis {
		s.lp.SetBasis(*nd.basis)
	}
	s.lastBasis = nil
	s.curDepth = nd.depth
	nodeCutRounds := 0
	for {
		s.lpChanged = false
		st := s.solveLP(0)
		switch st {
		case simplex.Optimal:
		case simplex.Infeasible, simplex.Cutoff:
			if debugHere && s.debugInBounds(s.lo, s.up) {
				s.debugf("node %d: LP %v pruned debug node (lp obj %g, debug obj %g, cutoff %g)", s.nodes, st, s.lp.Objective(), s.objOf(s.debugSol), s.cutoffValue())
			}
			if nd.brVar >= 0 && st == simplex.Cutoff {
				s.updatePseudocost(nd, s.lp.Objective())
			}
			return nil, nil
		case simplex.TimeLimit:
			return nil, nil
		default:
			// Numerical trouble: retry from scratch once.
			st = s.lp.SolveFromScratch()
			if st != simplex.Optimal {
				return nil, nil
			}
		}
		obj := s.lp.Objective()
		if nd.brVar >= 0 {
			s.updatePseudocost(nd, obj)
			nd.brVar = -1
		}
		if obj > nd.bound {
			nd.bound = obj
		}
		if s.pruneBound(obj) {
			if debugHere && s.debugInBounds(s.lo, s.up) {
				s.debugf("node %d: bound prune of debug node (lp obj %g, debug obj %g, cutoff %g)", s.nodes, obj, s.objOf(s.debugSol), s.cutoffValue())
			}
			return nil, nil
		}
		s.lp.Primal(s.x)
		frac := s.fractionals()
		if len(frac) == 0 {
			xs := append([]float64(nil), s.x...)
			for _, j := range s.intVars {
				xs[j] = math.Round(xs[j])
			}
			if !s.newIncumbent(xs, s.objOf(xs), "node") && !s.lpChanged {
				// Rounded point slightly infeasible: accept LP point value
				s.newIncumbent(s.x, obj, "node")
			}
			if s.lpChanged {
				continue // lazy constraints added: re-solve this node
			}
			return nil, nil
		}
		if s.opt.Hooks != nil && s.opt.Hooks.Cut != nil && nodeCutRounds < 3 {
			nodeCutRounds++
			if rows := s.opt.Hooks.Cut(append([]float64(nil), s.x...), nd.depth, s.nodes); len(rows) > 0 {
				s.addRows(rows)
				continue
			}
		}
		if s.opt.Heuristics {
			s.simpleRounding()
			if s.nodes%1000 == 0 && s.hasIncumbent {
				s.rins()
			}
			if s.nodes%200 == 0 && nd.depth < 60 {
				s.dive(nd.depth, 1)
				if s.pruneBound(s.lp.Objective()) {
					return nil, nil
				}
				frac = s.fractionals()
				if len(frac) == 0 {
					return nil, nil
				}
			}
		}
		if s.lpChanged {
			s.lpChanged = false
			continue // heuristics added lazy constraints
		}
		if !math.IsInf(s.incObj, 1) {
			if n := s.reducedCostFixing(false); n > 0 {
				if !s.propagate(s.lo, s.up, nil) {
					return nil, nil
				}
			}
		}
		// Branching variable selection (may tighten bounds via strong branching).
		j, action := s.selectBranchVar(nd, frac)
		switch action {
		case brInfeasible:
			return nil, nil
		case brResolve:
			continue
		}
		v := s.x[j]
		f := v - math.Floor(v)
		b := s.lp.GetBasis()
		down := &node{
			bound: obj, depth: nd.depth + 1, basis: &b,
			changes: appendChg(nd.changes, bndChg{j, math.Inf(-1), math.Floor(v)}),
			brVar:   j, brUp: false, brFrac: f, parentO: obj,
		}
		up := &node{
			bound: obj, depth: nd.depth + 1, basis: &b,
			changes: appendChg(nd.changes, bndChg{j, math.Ceil(v), math.Inf(1)}),
			brVar:   j, brUp: true, brFrac: f, parentO: obj,
		}
		// Carry local bound changes from strong branching / propagation that
		// are not implied by the global bounds? They are re-derived by
		// propagation; strong-branching tightenings are stored explicitly.
		if len(s.sbChanges) > 0 {
			down.changes = append(down.changes, s.sbChanges...)
			up.changes = append(append([]bndChg(nil), up.changes...), s.sbChanges...)
			s.sbChanges = s.sbChanges[:0]
		}
		s.lastBasis = &b
		down.estimate = obj + s.pseudocost(j, false)*f
		up.estimate = obj + s.pseudocost(j, true)*(1-f)
		return down, up
	}
}

func appendChg(base []bndChg, c bndChg) []bndChg {
	out := make([]bndChg, len(base)+1)
	copy(out, base)
	out[len(base)] = c
	return out
}

func (s *solver) fractionals() []int {
	var fr []int
	for _, j := range s.intVars {
		if !isIntegral(s.x[j], s.opt.IntTol) {
			fr = append(fr, j)
		}
	}
	return fr
}

// ---------------------------------------------------------------------------

type nodeHeap []*node

func (h nodeHeap) Len() int { return len(h) }
func (h nodeHeap) Less(a, b int) bool {
	if h[a].bound != h[b].bound {
		return h[a].bound < h[b].bound
	}
	return h[a].depth > h[b].depth
}
func (h nodeHeap) Swap(a, b int) {
	h[a], h[b] = h[b], h[a]
	h[a].idx = a
	h[b].idx = b
}
func (h *nodeHeap) Push(x any) {
	nd := x.(*node)
	nd.idx = len(*h)
	*h = append(*h, nd)
}
func (h *nodeHeap) Pop() any {
	old := *h
	k := len(old)
	nd := old[k-1]
	*h = old[:k-1]
	return nd
}
