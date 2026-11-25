package mip

import (
	"math"
	"sort"
	"time"

	"gost/internal/simplex"
)

type cut struct {
	idx []int
	val []float64
	rhs float64 // sum val*x >= rhs
	eff float64
}

// cutLoop runs rounds of Gomory mixed-integer and lifted cover cuts at the
// root node.
func (s *solver) cutLoop() {
	if s.opt.CutRounds <= 0 {
		return
	}
	startObj := s.lp.Objective()
	lastObj := startObj
	stall := 0
	var mir *cmirSep
	if s.opt.MIRCuts {
		mir = newCMIR(s)
	}
	for round := 0; round < s.opt.CutRounds; round++ {
		if s.timeUp() {
			break
		}
		s.lp.Primal(s.x)
		cuts := s.separateCovers()
		if s.opt.heurDebug {
			println("covers", len(cuts))
		}
		for _, c := range cuts {
			s.debugCheckCut(c, "cover")
		}
		if mir != nil {
			mc := mir.separate(cmirMaxCuts, time.Now().Add(30*time.Second)) // safety only: effort is bounded by counts
			if s.opt.heurDebug {
				println("cmir round", round, "cuts", len(mc), "by aggregation depth", cmirStats[0], cmirStats[1], cmirStats[2], cmirStats[3], cmirStats[4], cmirStats[5])
				cmirStats = [8]int{}
			}
			for _, c := range mc {
				s.debugCheckCut(c, "cmir")
			}
			cuts = append(cuts, mc...)
		}
		s.lp.Refresh() // accurate tableau rows
		g := s.separateGMI()
		for _, c := range g {
			s.debugCheckCut(c, "gmi")
		}
		cuts = append(cuts, g...)
		if len(cuts) == 0 {
			break
		}
		cuts = s.selectCuts(cuts)
		idx := make([][]int, len(cuts))
		val := make([][]float64, len(cuts))
		rl := make([]float64, len(cuts))
		ru := make([]float64, len(cuts))
		for k, c := range cuts {
			idx[k], val[k], rl[k], ru[k] = c.idx, c.val, c.rhs, math.Inf(1)
		}
		s.lp.AddRows(idx, val, rl, ru)
		for range cuts {
			s.lpRowIsCut = append(s.lpRowIsCut, true)
		}
		s.numCuts += len(cuts)
		st := s.solveLP(0)
		switch st {
		case simplex.Optimal:
		case simplex.Infeasible, simplex.Cutoff:
			s.debugf("cut LP status %v (round %d); debug sol in bounds: %v", st, round, s.debugInBounds(s.lo, s.up))
			s.debugCheckLPRows()
			s.debugWriteLP("/tmp/debuglp.mps")
			if s.debugSol != nil {
				st2 := s.lp.SolveFromScratch()
				s.debugf("re-solve from scratch: %v obj %g", st2, s.lp.Objective())
				s.lp.Debug = true
				s.lp.PivotTol = 1e-10
				st2 = s.lp.SolveFromScratch()
				s.debugf("re-solve from scratch pivtol 1e-10: %v obj %g", st2, s.lp.Objective())
				s.lp.PivotTol = 1e-7
			}
			if s.hasIncumbent {
				s.status = StatusOptimal
				s.globalBound = s.incObj
			} else {
				s.status = StatusInfeasible
			}
			return
		case simplex.TimeLimit:
			s.status = StatusTimeLimit
			return
		default:
			// numerical trouble: drop all cuts from this round
			del := make([]bool, s.lp.NumRows())
			for i := s.lp.NumRows() - len(cuts); i < s.lp.NumRows(); i++ {
				del[i] = true
			}
			s.removeLPRows(del)
			s.numCuts -= len(cuts)
			s.lp.Solve()
			return
		}
		obj := s.lp.Objective()
		s.opt.logf("Cut round %2d: %4d cuts, objective %.10g\n", round+1, len(cuts), obj+s.p.Offset)
		impr := obj - lastObj
		if impr < 1e-4*math.Max(1, math.Abs(obj)) && impr < 0.01*(obj-startObj+1e-9) {
			stall++
		} else {
			stall = 0
		}
		lastObj = obj
		if stall >= 2 {
			break
		}
		s.lp.Primal(s.x)
		if len(s.fractionals()) == 0 {
			break
		}
	}
	s.purgeInactiveCuts()
}

// purgeInactiveCuts removes cut rows whose logical is basic (slack).
func (s *solver) purgeInactiveCuts() {
	mm := s.lp.NumRows()
	if mm == s.m {
		s.opt.logf("Cuts: %d added, 0 kept in LP\n", s.numCuts)
		return
	}
	del := make([]bool, mm)
	cnt := 0
	for i := 0; i < mm; i++ {
		if !s.lpRowIsCut[i] {
			continue
		}
		if s.lp.VarStatus(s.n+i) == simplex.Basic {
			act := s.lp.RowActivity(i)
			lo, _ := s.lp.VarBounds(s.n + i)
			if act > lo+1e-6*math.Max(1, math.Abs(lo)) {
				del[i] = true
				cnt++
			}
		}
	}
	if cnt > 0 {
		s.removeLPRows(del)
		s.solveLP(0)
	}
	s.opt.logf("Cuts: %d added, %d kept in LP\n", s.numCuts, s.lp.NumRows()-s.m)
}

// selectCuts keeps the most efficacious cuts, skipping near-parallel ones.
func (s *solver) selectCuts(cuts []cut) []cut {
	sort.Slice(cuts, func(a, b int) bool { return cuts[a].eff > cuts[b].eff })
	maxCuts := max(20, min(500, s.m/2+20))
	var out []cut
	dense := make([]float64, s.n)
	for _, c := range cuts {
		if len(out) >= maxCuts {
			break
		}
		// parallelism check against kept cuts
		for k, j := range c.idx {
			dense[j] = c.val[k]
		}
		n1 := norm(c.val)
		parallel := false
		for _, o := range out {
			dot := 0.0
			for k, j := range o.idx {
				dot += o.val[k] * dense[j]
			}
			if dot/(n1*norm(o.val)) > 0.999 {
				parallel = true
				break
			}
		}
		for _, j := range c.idx {
			dense[j] = 0
		}
		if !parallel {
			out = append(out, c)
		}
	}
	return out
}

func norm(v []float64) float64 {
	s := 0.0
	for _, x := range v {
		s += x * x
	}
	return math.Sqrt(s)
}

// finishCut cleans a dense cut sum pi_j x_j >= rhs and checks violation.
func (s *solver) finishCut(pi []float64, rhs float64, touched []int) (cut, bool) {
	maxAbs := 0.0
	for _, j := range touched {
		maxAbs = math.Max(maxAbs, math.Abs(pi[j]))
	}
	if maxAbs < 1e-9 {
		return cut{}, false
	}
	var c cut
	minAbs := math.Inf(1)
	for _, j := range touched {
		v := pi[j]
		pi[j] = 0
		if v == 0 {
			continue
		}
		if math.Abs(v) < 1e-9*maxAbs {
			// relax: v*x_j >= v*bound giving max contribution
			if v > 0 {
				if math.IsInf(s.up0[j], 1) {
					return cut{}, false
				}
				rhs -= v * s.up0[j]
			} else {
				if math.IsInf(s.lo0[j], -1) {
					return cut{}, false
				}
				rhs -= v * s.lo0[j]
			}
			continue
		}
		c.idx = append(c.idx, j)
		c.val = append(c.val, v)
		minAbs = math.Min(minAbs, math.Abs(v))
	}
	if len(c.idx) == 0 || maxAbs/minAbs > 1e6 {
		if s.opt.heurDebug {
			println("  reject: dynamism", maxAbs/minAbs, "nnz", len(c.idx))
		}
		return cut{}, false
	}
	if len(c.idx) > max(100, s.n/2, s.maxRowLen()+10) {
		if s.opt.heurDebug {
			println("  reject: dense", len(c.idx))
		}
		return cut{}, false
	}
	act := 0.0
	for k, j := range c.idx {
		act += c.val[k] * s.x[j]
	}
	nrm := norm(c.val)
	viol := rhs - act
	if viol <= 1e-6*math.Max(1, math.Abs(rhs)) {
		return cut{}, false
	}
	c.eff = viol / nrm
	if c.eff < 1e-5 {
		return cut{}, false
	}
	c.rhs = rhs
	return c, true
}

// separateGMI derives Gomory mixed-integer cuts from tableau rows of
// fractional integer basic variables.
func (s *solver) separateGMI() []cut {
	var cuts []cut
	n := s.n
	mm := s.lp.NumRows()
	type cand struct {
		r    int
		frac float64
	}
	var cands []cand
	for r := 0; r < mm; r++ {
		j := s.lp.BasisVar(r)
		if j >= n || !s.isInt[j] {
			continue
		}
		v := s.x[j]
		f := v - math.Floor(v)
		if f < 0.005 || f > 0.995 {
			continue
		}
		cands = append(cands, cand{r, math.Abs(f - 0.5)})
	}
	sort.Slice(cands, func(a, b int) bool { return cands[a].frac < cands[b].frac })
	if len(cands) > 300 {
		cands = cands[:300]
	}
	pi := make([]float64, n)
	mark := make([]bool, n)
	deadline := time.Now().Add(30 * time.Second) // safety only
	for _, cd := range cands {
		if time.Now().After(deadline) {
			break
		}
		r := cd.r
		xb := s.x[s.lp.BasisVar(r)]
		f0 := xb - math.Floor(xb)
		idx, val := s.lp.TableauRow(r)
		var touched []int
		rhs := 1.0
		ok := true
		add := func(j int, v float64) {
			if !mark[j] {
				mark[j] = true
				touched = append(touched, j)
			}
			pi[j] += v
		}
		for k, j := range idx {
			a := val[k]
			if math.Abs(a) < 1e-12 {
				continue
			}
			st := s.lp.VarStatus(j)
			l, u := s.lp.VarBounds(j)
			if l == u {
				continue
			}
			var abar float64
			atUpper := false
			switch st {
			case simplex.AtLower:
				abar = a
			case simplex.AtUpper:
				abar = -a
				atUpper = true
			default:
				ok = false
			}
			if !ok {
				break
			}
			var g float64
			if j < n && s.isInt[j] {
				fk := abar - math.Floor(abar)
				if fk < 1e-12 || fk > 1-1e-12 {
					continue
				}
				if fk <= f0 {
					g = fk / f0
				} else {
					g = (1 - fk) / (1 - f0)
				}
			} else {
				if abar >= 0 {
					g = abar / f0
				} else {
					g = -abar / (1 - f0)
				}
			}
			if g == 0 {
				continue
			}
			// substitute t = x - l  or  t = u - x
			var coef float64
			if atUpper {
				coef = -g
				rhs -= g * u
			} else {
				coef = g
				rhs += g * l
			}
			if j < n {
				add(j, coef)
			} else {
				ri, rv := s.lp.RowCoefs(j - n)
				for p, jj := range ri {
					add(jj, coef*rv[p])
				}
			}
		}
		if !ok {
			for _, j := range touched {
				pi[j] = 0
				mark[j] = false
			}
			continue
		}
		for _, j := range touched {
			mark[j] = false
		}
		c, good := s.finishCut(pi, rhs, touched)
		if good {
			cuts = append(cuts, c)
		}
		if s.opt.heurDebug {
			println("gmi cand row", r, "f0", f0, "good", good, "nnz", len(touched))
		}
	}
	return cuts
}

// separateCovers separates lifted (extended) cover inequalities from
// original knapsack rows over binary variables.
func (s *solver) separateCovers() []cut {
	var cuts []cut
	type item struct {
		j    int
		a    float64 // positive weight
		comp bool    // complemented: x' = 1 - x
		xv   float64 // value of x' at LP point
	}
	pi := make([]float64, s.n)
	for i := 0; i < s.m; i++ {
		st, en := s.rowStart[i], s.rowStart[i+1]
		if en-st < 2 {
			continue
		}
		allBin := true
		for k := st; k < en; k++ {
			j := s.rowCol[k]
			if !s.isInt[j] || s.lo0[j] != 0 || s.up0[j] != 1 {
				allBin = false
				break
			}
		}
		if !allBin {
			continue
		}
		for side := 0; side < 2; side++ {
			var sign, b float64
			if side == 0 {
				if math.IsInf(s.p.RowUp[i], 1) {
					continue
				}
				sign, b = 1, s.p.RowUp[i]
			} else {
				if math.IsInf(s.p.RowLo[i], -1) {
					continue
				}
				sign, b = -1, -s.p.RowLo[i]
			}
			items := make([]item, 0, en-st)
			for k := st; k < en; k++ {
				j, a := s.rowCol[k], sign*s.rowVal[k]
				if a > 0 {
					items = append(items, item{j, a, false, s.x[j]})
				} else if a < 0 {
					b -= a
					items = append(items, item{j, -a, true, 1 - s.x[j]})
				}
			}
			total := 0.0
			for _, it := range items {
				total += it.a
			}
			if total <= b+1e-9 || b < 0 {
				continue
			}
			// Greedy cover: prefer items with large x' (small (1-x')/a).
			sort.Slice(items, func(p, q int) bool {
				return (1-items[p].xv)/items[p].a < (1-items[q].xv)/items[q].a
			})
			w := 0.0
			nc := 0
			for nc < len(items) && w <= b+1e-9 {
				w += items[nc].a
				nc++
			}
			if w <= b+1e-9 {
				continue
			}
			cover := append([]item(nil), items[:nc]...)
			// Make the cover minimal: drop items with the smallest x'
			// while the remaining weight still exceeds b.
			sort.Slice(cover, func(p, q int) bool { return cover[p].xv < cover[q].xv })
			for k := 0; k < len(cover); {
				if w-cover[k].a > b+1e-9 {
					w -= cover[k].a
					cover = append(cover[:k], cover[k+1:]...)
					continue
				}
				k++
			}
			r := len(cover)
			lhs := 0.0
			for _, it := range cover {
				lhs += it.xv
			}
			if lhs <= float64(r-1)-0.5 {
				continue // lifting is unlikely to give a violated cut
			}
			// Sequential up-lifting of the other items (largest x' first)
			// by dynamic programming over the cut value: W[v] is the
			// minimum weight of a set with value >= v (v capped at r).
			Wt := make([]float64, r+1)
			for v := 1; v <= r; v++ {
				Wt[v] = math.Inf(1)
			}
			addItem := func(a float64, alpha int) {
				for v := r; v >= 0; v-- {
					if math.IsInf(Wt[v], 1) {
						continue
					}
					nv := min(v+alpha, r)
					if Wt[v]+a < Wt[nv] {
						Wt[nv] = Wt[v] + a
					}
				}
				for v := r - 1; v >= 0; v-- {
					if Wt[v+1] < Wt[v] {
						Wt[v] = Wt[v+1]
					}
				}
			}
			inCover := map[int]bool{}
			for _, it := range cover {
				inCover[it.j] = true
				addItem(it.a, 1)
			}
			rest := make([]item, 0, len(items))
			for _, it := range items {
				if !inCover[it.j] {
					rest = append(rest, it)
				}
			}
			sort.Slice(rest, func(p, q int) bool { return rest[p].xv > rest[q].xv })
			type lifted struct {
				it    item
				alpha int
			}
			var lift []lifted
			for _, it := range rest {
				capa := b - it.a
				if capa < -1e-9 {
					// item alone exceeds the capacity: x' = 0 always
					continue
				}
				best := 0
				for v := r; v >= 0; v-- {
					if Wt[v] <= capa+1e-9 {
						best = v
						break
					}
				}
				alpha := r - 1 - best
				if alpha > 0 {
					lift = append(lift, lifted{it, alpha})
					addItem(it.a, alpha)
				}
			}
			// Build sum_C x' + sum alpha x' <= r-1 as ">=" form.
			rhs := -float64(r - 1)
			var touched []int
			addTerm := func(it item, coef float64) {
				if it.comp {
					pi[it.j] += coef
					rhs += coef
				} else {
					pi[it.j] -= coef
				}
				touched = append(touched, it.j)
			}
			for _, it := range cover {
				addTerm(it, 1)
			}
			for _, l := range lift {
				addTerm(l.it, float64(l.alpha))
			}
			if c, good := s.finishCut(pi, rhs, touched); good {
				cuts = append(cuts, c)
			}
		}
	}
	return cuts
}

// removeLPRows deletes LP rows (cuts only) and updates the row mappings.
func (s *solver) removeLPRows(del []bool) {
	newIdx := make([]int, len(del))
	k := 0
	for i := range del {
		if del[i] {
			newIdx[i] = -1
		} else {
			newIdx[i] = k
			k++
		}
	}
	for i, r := range s.lpRowOf {
		s.lpRowOf[i] = newIdx[r]
	}
	isCut := s.lpRowIsCut[:0]
	for i := range del {
		if !del[i] {
			isCut = append(isCut, s.lpRowIsCut[i])
		}
	}
	s.lpRowIsCut = isCut
	s.lp.RemoveRows(del)
	s.lastBasis = nil
}

// maxRowLen returns the length of the longest problem row (cached).
func (s *solver) maxRowLen() int {
	if s.maxRowLenCache == 0 {
		for i := 0; i < s.m; i++ {
			s.maxRowLenCache = max(s.maxRowLenCache, s.rowStart[i+1]-s.rowStart[i])
		}
		s.maxRowLenCache++
	}
	return s.maxRowLenCache
}
