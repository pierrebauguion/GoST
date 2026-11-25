package mip

import (
	"math"
	"os"
	"sort"
	"strconv"
	"time"
)

// Complemented mixed-integer rounding (c-MIR) cuts, following Marchand &
// Wolsey (2001): a base inequality (a problem row, or an aggregation of up
// to maxAggr rows that eliminates continuous variables strictly between
// their bounds) is transformed by
//   - substituting continuous variables by their closest simple or
//     variable bound (variable upper/lower bounds give flow-cover-like cuts),
//   - complementing integer variables to their closest bound,
//   - dividing by a scaling factor delta taken from the integer coefficients,
// and the MIR inequality of the result is mapped back to the original space.

// vbound is a variable bound y <= coef*x (upper) or y >= coef*x (lower) for a
// continuous y and a binary x.
type vbound struct {
	x    int
	coef float64
}

var cmirStats [8]int

type cmirSep struct {
	s       *solver
	n       int
	vub     []vbound // per continuous variable (x = -1 if none)
	vlb     []vbound
	colRows [][]int // rows of each column (original problem rows)
	// work
	agg      []float64 // dense aggregated row (by variable)
	aggMark  []bool
	aggList  []int
	pi       []float64
	slackTol float64
}

func newCMIR(s *solver) *cmirSep {
	n := s.n
	c := &cmirSep{s: s, n: n, slackTol: 1e-6}
	c.vub = make([]vbound, n)
	c.vlb = make([]vbound, n)
	for j := range c.vub {
		c.vub[j].x, c.vlb[j].x = -1, -1
	}
	c.colRows = make([][]int, n)
	p := s.p
	for i := 0; i < s.m; i++ {
		st, en := s.rowStart[i], s.rowStart[i+1]
		for k := st; k < en; k++ {
			c.colRows[s.rowCol[k]] = append(c.colRows[s.rowCol[k]], i)
		}
		if en-st != 2 {
			continue
		}
		// a_y y + a_x x (<= or >=) 0 with y continuous, x binary
		j0, j1 := s.rowCol[st], s.rowCol[st+1]
		a0, a1 := s.rowVal[st], s.rowVal[st+1]
		y, x, ay, ax := j0, j1, a0, a1
		if s.isInt[y] {
			y, x, ay, ax = j1, j0, a1, a0
		}
		if s.isInt[y] || !s.isInt[x] || s.lo0[x] < 0 || math.IsInf(s.up0[x], 1) || s.lo0[y] < 0 {
			continue
		}
		coef := -ax / ay
		if !math.IsInf(p.RowUp[i], 1) && p.RowUp[i] == 0 {
			// ay*y + ax*x <= 0
			if ay > 0 {
				c.vub[y] = vbound{x, coef}
			} else {
				c.vlb[y] = vbound{x, coef}
			}
		}
		if !math.IsInf(p.RowLo[i], -1) && p.RowLo[i] == 0 {
			// ay*y + ax*x >= 0
			if ay > 0 {
				c.vlb[y] = vbound{x, coef}
			} else {
				c.vub[y] = vbound{x, coef}
			}
		}
	}
	c.agg = make([]float64, n)
	c.aggMark = make([]bool, n)
	c.pi = make([]float64, n)
	return c
}

func (c *cmirSep) clearAgg() {
	for _, j := range c.aggList {
		c.agg[j] = 0
		c.aggMark[j] = false
	}
	c.aggList = c.aggList[:0]
}

func (c *cmirSep) addRow(i int, lambda float64) {
	s := c.s
	for k := s.rowStart[i]; k < s.rowStart[i+1]; k++ {
		j := s.rowCol[k]
		if !c.aggMark[j] {
			c.aggMark[j] = true
			c.aggList = append(c.aggList, j)
		}
		c.agg[j] += lambda * s.rowVal[k]
	}
}

// separate returns c-MIR cuts for the current LP solution s.x.
func (c *cmirSep) separate(maxCuts int, deadline time.Time) []cut {
	s := c.s
	p := s.p
	x := s.x
	type start struct {
		i     int
		sign  float64
		score float64
	}
	var starts []start
	for i := 0; i < s.m; i++ {
		st, en := s.rowStart[i], s.rowStart[i+1]
		if en-st < 2 {
			continue
		}
		hasInt := false
		act := 0.0
		for k := st; k < en; k++ {
			j := s.rowCol[k]
			act += s.rowVal[k] * x[j]
			if s.isInt[j] {
				hasInt = true
			}
		}
		if !hasInt && len(c.colRows) == 0 {
			continue
		}
		// prefer tight rows
		if !math.IsInf(p.RowUp[i], 1) {
			sl := p.RowUp[i] - act
			starts = append(starts, start{i, 1, sl / (1 + math.Abs(p.RowUp[i]))})
		}
		if !math.IsInf(p.RowLo[i], -1) && p.RowLo[i] != p.RowUp[i] {
			sl := act - p.RowLo[i]
			starts = append(starts, start{i, -1, sl / (1 + math.Abs(p.RowLo[i]))})
		}
	}
	sort.Slice(starts, func(a, b int) bool { return starts[a].score < starts[b].score })
	if len(starts) > 2000 {
		starts = starts[:2000]
	}
	var cuts []cut
	used := make(map[int]bool)
	for _, st := range starts {
		if len(cuts) >= maxCuts || time.Now().After(deadline) {
			break
		}
		if st.score > cmirMaxSlack {
			break // only (nearly) tight rows
		}
		c.clearAgg()
		rhs := 0.0
		// base row as "<=": sign*row <= sign*bound
		if st.sign > 0 {
			c.addRow(st.i, 1)
			rhs = p.RowUp[st.i]
		} else {
			c.addRow(st.i, -1)
			rhs = -p.RowLo[st.i]
		}
		for k := range used {
			delete(used, k)
		}
		used[st.i] = true
		for aggr := 0; aggr <= cmirMaxAggr; aggr++ {
			if cu, ok := c.tryMIR(rhs); ok {
				cuts = append(cuts, cu)
				if s.opt.heurDebug {
					cmirStats[aggr]++
				}
				break
			}
			// aggregate: eliminate the continuous variable farthest from
			// its bounds using another tight row.
			if !c.aggregate(&rhs, used) {
				break
			}
		}
	}
	c.clearAgg()
	return cuts
}

// aggregate adds a multiple of a tight row eliminating a continuous variable
// of the aggregated row that lies strictly between its bounds.
func (c *cmirSep) aggregate(rhs *float64, used map[int]bool) bool {
	s := c.s
	p := s.p
	x := s.x
	bestJ, bestD := -1, 1e-6
	for _, j := range c.aggList {
		a := c.agg[j]
		if s.isInt[j] || math.Abs(a) < 1e-9 {
			continue
		}
		d := math.Min(x[j]-s.lo0[j], s.up0[j]-x[j])
		if v := c.vub[j]; v.x >= 0 {
			d = math.Min(d, v.coef*x[v.x]-x[j])
		}
		if v := c.vlb[j]; v.x >= 0 {
			d = math.Min(d, x[j]-v.coef*x[v.x])
		}
		if d > bestD {
			bestD, bestJ = d, j
		}
	}
	if bestJ < 0 {
		return false
	}
	j := bestJ
	bestRow, bestScore := -1, math.Inf(1)
	var bestLambda, bestB float64
	for _, i := range c.colRows[j] {
		if used[i] {
			continue
		}
		if s.rowStart[i+1]-s.rowStart[i] == 2 && (c.vub[j].x >= 0 || c.vlb[j].x >= 0) {
			continue // variable bound rows are used through bound substitution
		}
		var rj float64
		act := 0.0
		for k := s.rowStart[i]; k < s.rowStart[i+1]; k++ {
			jj := s.rowCol[k]
			act += s.rowVal[k] * x[jj]
			if jj == j {
				rj = s.rowVal[k]
			}
		}
		if rj == 0 {
			continue
		}
		lambda := -c.agg[j] / rj
		var b, slack float64
		switch {
		case lambda > 0 && !math.IsInf(p.RowUp[i], 1):
			b, slack = p.RowUp[i], p.RowUp[i]-act
		case lambda < 0 && !math.IsInf(p.RowLo[i], -1):
			b, slack = p.RowLo[i], act-p.RowLo[i]
		default:
			continue
		}
		score := slack / (1 + math.Abs(b))
		if score < bestScore {
			bestScore, bestRow, bestLambda, bestB = score, i, lambda, b
		}
	}
	if bestRow < 0 || bestScore > 1e-3 {
		return false
	}
	c.addRow(bestRow, bestLambda)
	c.agg[j] = 0
	*rhs += bestLambda * bestB
	used[bestRow] = true
	return true
}

type mirTerm struct {
	j    int
	a    float64 // coefficient of the transformed variable
	comp bool    // integer: complemented (x = u - x'); continuous: see kind
	kind int8    // continuous: 0 lb, 1 ub, 2 vlb, 3 vub
	vx   int
	vc   float64
}

// tryMIR builds the best c-MIR cut from the aggregated row sum agg*x <= rhs.
func (c *cmirSep) tryMIR(rhs float64) (cut, bool) {
	s := c.s
	x := s.x
	// integer part (dense on variables) after variable-bound substitution
	intCoef := map[int]float64{}
	var conts []mirTerm
	beta := rhs
	for _, j := range c.aggList {
		a := c.agg[j]
		if math.Abs(a) < 1e-12 {
			continue
		}
		if s.isInt[j] {
			intCoef[j] += a
			continue
		}
		// continuous: choose closest bound (variable bounds preferred on ties)
		l, u := s.lo0[j], s.up0[j]
		best := -1
		bestD := math.Inf(1)
		var vx int
		var vc float64
		if !math.IsInf(l, -1) {
			best, bestD = 0, x[j]-l
		}
		if !math.IsInf(u, 1) && u-x[j] < bestD {
			best, bestD = 1, u-x[j]
		}
		if v := c.vlb[j]; v.x >= 0 && x[j]-v.coef*x[v.x] <= bestD+1e-9 {
			best, bestD, vx, vc = 2, x[j]-v.coef*x[v.x], v.x, v.coef
		}
		if v := c.vub[j]; v.x >= 0 && v.coef*x[v.x]-x[j] <= bestD+1e-9 {
			best, bestD, vx, vc = 3, v.coef*x[v.x]-x[j], v.x, v.coef
		}
		switch best {
		case 0: // x = l + s
			beta -= a * l
			conts = append(conts, mirTerm{j: j, a: a, kind: 0})
		case 1: // x = u - s
			beta -= a * u
			conts = append(conts, mirTerm{j: j, a: -a, kind: 1})
		case 2: // x = vc*vx + s
			intCoef[vx] += a * vc
			conts = append(conts, mirTerm{j: j, a: a, kind: 2, vx: vx, vc: vc})
		case 3: // x = vc*vx - s
			intCoef[vx] += a * vc
			conts = append(conts, mirTerm{j: j, a: -a, kind: 3, vx: vx, vc: vc})
		default:
			return cut{}, false // free continuous variable
		}
	}
	// complement integer variables to the closest bound
	var ints []mirTerm
	// iterate in index order: floating-point sums must not depend on map
	// iteration order (determinism)
	intKeys := make([]int, 0, len(intCoef))
	for j := range intCoef {
		intKeys = append(intKeys, j)
	}
	sort.Ints(intKeys)
	for _, j := range intKeys {
		a := intCoef[j]
		if math.Abs(a) < 1e-12 {
			continue
		}
		l, u := s.lo0[j], s.up0[j]
		switch {
		case !math.IsInf(u, 1) && (math.IsInf(l, -1) || u-x[j] < x[j]-l):
			beta -= a * u // a x = a u - a x'
			ints = append(ints, mirTerm{j: j, a: -a, comp: true})
		case !math.IsInf(l, -1):
			beta -= a * l
			ints = append(ints, mirTerm{j: j, a: a})
		default:
			return cut{}, false
		}
	}
	if len(ints) == 0 {
		return cut{}, false
	}
	sort.Slice(ints, func(p, q int) bool { return ints[p].j < ints[q].j })
	// candidate deltas: |a| of integers strictly between their bounds
	var deltas []float64
	seen := map[float64]bool{}
	for _, t := range ints {
		xv := x[t.j]
		if xv <= s.lo0[t.j]+1e-6 || xv >= s.up0[t.j]-1e-6 {
			continue
		}
		d := math.Abs(t.a)
		if d < 1e-6 || seen[d] {
			continue
		}
		seen[d] = true
		deltas = append(deltas, d)
		if len(deltas) >= 8 {
			break
		}
	}
	if len(deltas) == 0 {
		deltas = append(deltas, 1)
	}
	bestEff := 1e-4
	var bestCut cut
	found := false
	try := func(delta float64) float64 {
		cu, ok := c.buildMIR(ints, conts, beta, delta)
		if ok && cu.eff > bestEff {
			bestEff, bestCut, found = cu.eff, cu, true
			return cu.eff
		}
		return -1
	}
	bestDelta := 0.0
	for _, d := range deltas {
		if try(d) > 0 {
			bestDelta = d
		}
	}
	if bestDelta > 0 {
		base := bestDelta
		for _, f := range []float64{2, 4, 8} {
			if try(base/f) > 0 {
				bestDelta = base / f
			}
		}
		// Try complementing integers strictly between their bounds the
		// other way (closest to the upper bound first).
		type cand struct {
			k int
			d float64
		}
		var cands []cand
		for k, t := range ints {
			l, u := s.lo0[t.j], s.up0[t.j]
			if math.IsInf(l, -1) || math.IsInf(u, 1) {
				continue
			}
			xv := x[t.j]
			if xv <= l+1e-6 || xv >= u-1e-6 {
				continue
			}
			cands = append(cands, cand{k, math.Min(u-xv, xv-l)})
		}
		sort.Slice(cands, func(p, q int) bool { return cands[p].d < cands[q].d })
		for _, cd := range cands {
			t := &ints[cd.k]
			l, u := s.lo0[t.j], s.up0[t.j]
			oldBeta, oldA, oldComp := beta, t.a, t.comp
			if t.comp { // x = u - x' -> x = l + x' ; original coef a = -t.a
				a := -t.a
				beta += a*u - a*l
				t.a, t.comp = a, false
			} else { // x = l + x' -> x = u - x'
				a := t.a
				beta += a*l - a*u
				t.a, t.comp = -a, true
			}
			if c2, ok := c.buildMIR(ints, conts, beta, bestDelta); ok && c2.eff > bestEff+1e-9 {
				bestEff, bestCut, found = c2.eff, c2, true
			} else {
				beta, t.a, t.comp = oldBeta, oldA, oldComp
			}
		}
	}
	return bestCut, found
}

// buildMIR applies the MIR formula with divisor delta and maps back.
func (c *cmirSep) buildMIR(ints, conts []mirTerm, beta, delta float64) (cut, bool) {
	s := c.s
	b := beta / delta
	f0 := b - math.Floor(b)
	if f0 < 0.01 || f0 > 0.99 {
		return cut{}, false
	}
	// <= cut in transformed space:
	//   sum F(a/delta) x' + 1/(1-f0) sum_{c<0} (c/delta) s <= floor(b)
	pi := c.pi
	var touched []int
	add := func(j int, v float64) {
		if pi[j] == 0 {
			touched = append(touched, j)
		}
		pi[j] += v
		if pi[j] == 0 {
			pi[j] = 1e-300
		}
	}
	cutRhs := math.Floor(b)
	for _, t := range ints {
		ad := t.a / delta
		fa := ad - math.Floor(ad)
		g := math.Floor(ad) + math.Max(0, fa-f0)/(1-f0)
		if g == 0 {
			continue
		}
		// x' = x - l  or  x' = u - x
		if t.comp {
			add(t.j, -g)
			cutRhs -= g * s.up0[t.j]
		} else {
			add(t.j, g)
			cutRhs += g * s.lo0[t.j]
		}
	}
	for _, t := range conts {
		cd := t.a / delta
		if cd >= 0 {
			continue
		}
		g := cd / (1 - f0)
		// s in terms of x
		switch t.kind {
		case 0: // s = x - l
			add(t.j, g)
			cutRhs += g * s.lo0[t.j]
		case 1: // s = u - x
			add(t.j, -g)
			cutRhs -= g * s.up0[t.j]
		case 2: // s = x - vc*vx
			add(t.j, g)
			add(t.vx, -g*t.vc)
		case 3: // s = vc*vx - x
			add(t.j, -g)
			add(t.vx, g*t.vc)
		}
	}
	// convert "pi x <= cutRhs" to ">=" form expected by finishCut
	for _, j := range touched {
		if pi[j] == 1e-300 {
			pi[j] = 0
		}
		pi[j] = -pi[j]
	}
	return s.finishCut(pi, -cutRhs, touched)
}

var (
	cmirMaxAggr  = int(envF("GOST_CMIR_AGGR", 5))
	cmirMaxSlack = envF("GOST_CMIR_SLACK", 0.1)
	cmirMaxCuts  = int(envF("GOST_CMIR_CUTS", 200))
)

func envF(name string, def float64) float64 {
	if v, err := strconv.ParseFloat(os.Getenv(name), 64); err == nil {
		return v
	}
	return def
}
