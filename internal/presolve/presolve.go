// Package presolve implements MIP/LP presolve reductions with primal
// postsolve.
//
// Reductions: empty/singleton/redundant/forcing rows, fixed and empty
// columns, activity-based bound tightening of integer variables,
// coefficient tightening (Savelsbergh) on one-sided rows with binaries,
// and dual fixing.
package presolve

import (
	"math"
	"os"
	"strings"

	"gost/internal/problem"
)

// Status of presolve.
type Status int

const (
	Reduced Status = iota
	Infeasible
	Unbounded
)

type entry struct {
	j int
	v float64
}

// Result holds the reduced problem and the information for postsolve.
type Result struct {
	Status  Status
	Reduced *problem.Problem
	// ColMap[k] = original column of reduced column k.
	ColMap []int
	// RowMap[k] = original row of reduced row k.
	RowMap []int
	// Fixed values for removed columns (NaN for kept columns).
	fixedVal []float64
	n0       int
	ops      []postOp

	RowsRemoved, ColsRemoved, BoundsTightened, CoefsTightened, Aggregated int
}

// Postsolve maps a reduced solution back to the original space.
func (r *Result) Postsolve(xr []float64) []float64 {
	x := make([]float64, r.n0)
	copy(x, r.fixedVal)
	for k, j := range r.ColMap {
		x[j] = xr[k]
	}
	r.applyOps(x)
	return x
}

type presolver struct {
	p          *problem.Problem
	rows       [][]entry
	colRows    [][]int
	rowAlive   []bool
	colAlive   []bool
	lo, up     []float64
	rl, ru     []float64
	obj        []float64
	offset     float64
	integer    []bool
	fixedVal   []float64
	tol        float64
	res        *Result
	changed    bool
	debug      []float64
	broken     bool
	infeasible bool
}

// DebugSolution, if set, is checked after every reduction (development aid).
var DebugSolution []float64

func (ps *presolver) check(what string, i int) {
	if ps.debug == nil || ps.broken {
		return
	}
	x := ps.debug
	for j := range ps.colAlive {
		if !ps.colAlive[j] {
			continue
		}
		if x[j] < ps.lo[j]-1e-6 || x[j] > ps.up[j]+1e-6 {
			println("PRESOLVE BROKEN by", what, i, ": col", j, "val", x[j], "bounds", ps.lo[j], ps.up[j])
			ps.broken = true
			return
		}
	}
	for r := range ps.rowAlive {
		if !ps.rowAlive[r] {
			continue
		}
		a := 0.0
		for _, e := range ps.rows[r] {
			a += e.v * x[e.j]
		}
		if a < ps.rl[r]-1e-6 || a > ps.ru[r]+1e-6 {
			println("PRESOLVE BROKEN by", what, i, ": row", r, "act", a, "bounds", ps.rl[r], ps.ru[r])
			ps.broken = true
			return
		}
	}
}

const infBound = 1e30

func isInf(v float64) bool { return math.Abs(v) >= infBound }

// Presolve reduces p. The input is not modified.
func Presolve(p *problem.Problem) *Result {
	n, m := p.NumCols, p.NumRows
	ps := &presolver{
		p:        p,
		rows:     make([][]entry, m),
		colRows:  make([][]int, n),
		rowAlive: make([]bool, m),
		colAlive: make([]bool, n),
		lo:       append([]float64(nil), p.Lo...),
		up:       append([]float64(nil), p.Up...),
		rl:       append([]float64(nil), p.RowLo...),
		ru:       append([]float64(nil), p.RowUp...),
		obj:      append([]float64(nil), p.Obj...),
		offset:   p.Offset,
		integer:  p.Integer,
		fixedVal: make([]float64, n),
		tol:      1e-9,
		res:      &Result{n0: n},
	}
	for j := 0; j < n; j++ {
		ps.colAlive[j] = true
		ps.fixedVal[j] = math.NaN()
		for k := p.ColStart[j]; k < p.ColStart[j+1]; k++ {
			if p.Val[k] == 0 {
				continue
			}
			i := p.RowIdx[k]
			ps.rows[i] = append(ps.rows[i], entry{j, p.Val[k]})
			ps.colRows[j] = append(ps.colRows[j], i)
		}
		if p.Integer[j] {
			ps.lo[j] = math.Ceil(ps.lo[j] - 1e-9)
			ps.up[j] = math.Floor(ps.up[j] + 1e-9)
		}
	}
	for i := 0; i < m; i++ {
		ps.rowAlive[i] = true
	}
	ps.debug = DebugSolution
	st := ps.run()
	ps.res.Status = st
	if st == Reduced {
		ps.build()
	}
	return ps.res
}

var disabled = os.Getenv("GOST_PRESOLVE_OFF")

func off(name string) bool { return strings.Contains(disabled, name) }

func (ps *presolver) run() Status {
	for pass := 0; pass < 25; pass++ {
		ps.changed = false
		for j := range ps.colAlive {
			if !ps.colAlive[j] {
				continue
			}
			if st := ps.processCol(j); st != Reduced {
				return st
			}
			ps.check("col", j)
		}
		for i := range ps.rowAlive {
			if !ps.rowAlive[i] {
				continue
			}
			if st := ps.processRow(i); st != Reduced {
				return st
			}
			ps.check("row", i)
		}
		if off("dual") {
			if !ps.changed {
				break
			}
			continue
		}
		if st := ps.dualFixing(); st != Reduced {
			return st
		}
		if !ps.changed {
			break
		}
	}
	return Reduced
}

// activity computes min/max activity of row i with infinity counts.
func (ps *presolver) activity(i int) (minA, maxA float64, minInf, maxInf int) {
	for _, e := range ps.rows[i] {
		l, u := ps.lo[e.j], ps.up[e.j]
		if e.v > 0 {
			if isInf(l) {
				minInf++
			} else {
				minA += e.v * l
			}
			if isInf(u) {
				maxInf++
			} else {
				maxA += e.v * u
			}
		} else {
			if isInf(u) {
				minInf++
			} else {
				minA += e.v * u
			}
			if isInf(l) {
				maxInf++
			} else {
				maxA += e.v * l
			}
		}
	}
	return
}

func (ps *presolver) removeRow(i int) {
	ps.rowAlive[i] = false
	ps.res.RowsRemoved++
	ps.changed = true
}

func (ps *presolver) fixCol(j int, v float64) {
	ps.colAlive[j] = false
	ps.fixedVal[j] = v
	ps.offset += ps.obj[j] * v
	ps.res.ColsRemoved++
	ps.changed = true
	for _, i := range ps.colRows[j] {
		if !ps.rowAlive[i] {
			continue
		}
		r := ps.rows[i]
		for k, e := range r {
			if e.j == j {
				if !isInf(ps.rl[i]) {
					ps.rl[i] -= e.v * v
				}
				if !isInf(ps.ru[i]) {
					ps.ru[i] -= e.v * v
				}
				r[k] = r[len(r)-1]
				ps.rows[i] = r[:len(r)-1]
				break
			}
		}
	}
}

func (ps *presolver) setBounds(j int, l, u float64) Status {
	if ps.integer[j] {
		l = math.Ceil(l - 1e-6)
		u = math.Floor(u + 1e-6)
	}
	if l > ps.lo[j]+1e-9 {
		ps.lo[j] = l
		ps.changed = true
		ps.res.BoundsTightened++
	}
	if u < ps.up[j]-1e-9 {
		ps.up[j] = u
		ps.changed = true
		ps.res.BoundsTightened++
	}
	if ps.lo[j] > ps.up[j]+1e-6 {
		return Infeasible
	}
	if ps.lo[j] > ps.up[j] {
		ps.lo[j] = ps.up[j]
	}
	return Reduced
}

func (ps *presolver) processCol(j int) Status {
	l, u := ps.lo[j], ps.up[j]
	if l > u+1e-6 {
		return Infeasible
	}
	if u-l <= 1e-9 && !isInf(l) {
		ps.fixCol(j, l)
		return Reduced
	}
	alive := 0
	for _, i := range ps.colRows[j] {
		if ps.rowAlive[i] {
			alive++
		}
	}
	if alive == 1 && !off("aggr") {
		return ps.singletonCol(j)
	}
	if alive == 0 {
		c := ps.obj[j]
		switch {
		case c > 0:
			if isInf(l) {
				return Unbounded
			}
			ps.fixCol(j, l)
		case c < 0:
			if isInf(u) {
				return Unbounded
			}
			ps.fixCol(j, u)
		default:
			v := 0.0
			if !isInf(l) {
				v = l
			} else if !isInf(u) {
				v = u
			}
			ps.fixCol(j, v)
		}
	}
	return Reduced
}

func (ps *presolver) processRow(i int) Status {
	if len(ps.rows[i]) == 2 && ps.rl[i] == ps.ru[i] && !off("aggr") {
		if st := ps.doubleton(i); st != Reduced {
			return st
		}
		if !ps.rowAlive[i] {
			return Reduced
		}
	}
	r := ps.rows[i]
	rl, ru := ps.rl[i], ps.ru[i]
	feasTol := 1e-6 * math.Max(1, math.Max(math.Abs(rlOr0(rl)), math.Abs(rlOr0(ru))))
	if len(r) == 0 {
		if (!isInf(rl) && rl > feasTol) || (!isInf(ru) && ru < -feasTol) {
			return Infeasible
		}
		ps.removeRow(i)
		return Reduced
	}
	if isInf(rl) && isInf(ru) {
		ps.removeRow(i)
		return Reduced
	}
	if len(r) == 1 {
		e := r[0]
		var l, u float64
		if e.v > 0 {
			l, u = rl/e.v, ru/e.v
		} else {
			l, u = ru/e.v, rl/e.v
		}
		if isInf(rl) {
			if e.v > 0 {
				l = math.Inf(-1)
			} else {
				u = math.Inf(1)
			}
		}
		if isInf(ru) {
			if e.v > 0 {
				u = math.Inf(1)
			} else {
				l = math.Inf(-1)
			}
		}
		if st := ps.setBounds(e.j, l, u); st != Reduced {
			return st
		}
		ps.removeRow(i)
		return Reduced
	}
	minA, maxA, minInf, maxInf := ps.activity(i)
	if minInf == 0 && !isInf(ru) && minA > ru+feasTol {
		return Infeasible
	}
	if maxInf == 0 && !isInf(rl) && maxA < rl-feasTol {
		return Infeasible
	}
	// Redundant.
	if (isInf(rl) || (minInf == 0 && minA >= rl-1e-9)) && (isInf(ru) || (maxInf == 0 && maxA <= ru+1e-9)) {
		ps.removeRow(i)
		return Reduced
	}
	// Forcing constraints.
	if off("forcing") {
		goto tighten
	}
	if minInf == 0 && !isInf(ru) && math.Abs(minA-ru) <= 1e-9 {
		for _, e := range append([]entry(nil), r...) {
			if e.v > 0 {
				ps.fixCol(e.j, ps.lo[e.j])
			} else {
				ps.fixCol(e.j, ps.up[e.j])
			}
		}
		ps.removeRow(i)
		return Reduced
	}
	if maxInf == 0 && !isInf(rl) && math.Abs(maxA-rl) <= 1e-9 {
		for _, e := range append([]entry(nil), r...) {
			if e.v > 0 {
				ps.fixCol(e.j, ps.up[e.j])
			} else {
				ps.fixCol(e.j, ps.lo[e.j])
			}
		}
		ps.removeRow(i)
		return Reduced
	}
tighten:
	// Bound tightening (integer columns only, for LP numerics).
	for _, e := range r {
		if off("bound") {
			break
		}
		j := e.j
		if !ps.integer[j] {
			continue
		}
		l, u := ps.lo[j], ps.up[j]
		// residual activities excluding j
		var resMin, resMax float64
		var okMin, okMax bool
		if e.v > 0 {
			okMin = minInf == 0 || (minInf == 1 && isInf(l))
			okMax = maxInf == 0 || (maxInf == 1 && isInf(u))
			if okMin {
				resMin = minA
				if minInf == 0 {
					resMin -= e.v * l
				}
			}
			if okMax {
				resMax = maxA
				if maxInf == 0 {
					resMax -= e.v * u
				}
			}
		} else {
			okMin = minInf == 0 || (minInf == 1 && isInf(u))
			okMax = maxInf == 0 || (maxInf == 1 && isInf(l))
			if okMin {
				resMin = minA
				if minInf == 0 {
					resMin -= e.v * u
				}
			}
			if okMax {
				resMax = maxA
				if maxInf == 0 {
					resMax -= e.v * l
				}
			}
		}
		nl, nu := math.Inf(-1), math.Inf(1)
		if !isInf(ru) && okMin {
			// e.v * x <= ru - resMin
			if e.v > 0 {
				nu = (ru - resMin) / e.v
			} else {
				nl = (ru - resMin) / e.v
			}
		}
		if !isInf(rl) && okMax {
			if e.v > 0 {
				nl = math.Max(nl, (rl-resMax)/e.v)
			} else {
				nu = math.Min(nu, (rl-resMax)/e.v)
			}
		}
		if (nl > l+1e-6 && math.Abs(nl) < 1e9) || (nu < u-1e-6 && math.Abs(nu) < 1e9) {
			if st := ps.setBounds(j, nl, nu); st != Reduced {
				return st
			}
			minA, maxA, minInf, maxInf = ps.activity(i)
		}
	}
	// Coefficient tightening on one-sided rows with binary variables.
	if isInf(rl) != isInf(ru) && !off("coef") {
		sign := 1.0 // row: sign*a x <= b
		b := ru
		if isInf(ru) {
			sign = -1
			b = -rl
		}
		_, maxS, _, maxSInf := ps.signedActivity(i, sign)
		if maxSInf == 0 && maxS <= b+1e-9 {
			// became redundant after bound tightening
			ps.removeRow(i)
			return Reduced
		}
		if maxSInf == 0 {
			for k, e := range ps.rows[i] {
				j := e.j
				if !ps.integer[j] || ps.lo[j] != 0 || ps.up[j] != 1 {
					continue
				}
				a := sign * e.v
				if maxS <= b+1e-9 {
					break
				}
				if a > 0 && maxS-a < b-1e-9 {
					d := b - (maxS - a)
					a -= d
					b -= d
					maxS -= d
				} else if a < 0 && maxS+a < b-1e-9 {
					d := b - (maxS + a)
					a += d
				} else {
					continue
				}
				ps.rows[i][k].v = sign * a
				ps.res.CoefsTightened++
				ps.changed = true
			}
			if sign > 0 {
				ps.ru[i] = b
			} else {
				ps.rl[i] = -b
			}
		}
	}
	return Reduced
}

func (ps *presolver) signedActivity(i int, sign float64) (minA, maxA float64, minInf, maxInf int) {
	minA, maxA, minInf, maxInf = ps.activity(i)
	if sign < 0 {
		return -maxA, -minA, maxInf, minInf
	}
	return
}

func rlOr0(v float64) float64 {
	if isInf(v) {
		return 0
	}
	return v
}

// dualFixing fixes columns whose objective and locks allow moving to a bound.
func (ps *presolver) dualFixing() Status {
	for j := range ps.colAlive {
		if !ps.colAlive[j] {
			continue
		}
		downLocks, upLocks := 0, 0
		for _, i := range ps.colRows[j] {
			if !ps.rowAlive[i] {
				continue
			}
			v := ps.coef(i, j)
			if v == 0 {
				continue
			}
			hasL, hasU := !isInf(ps.rl[i]), !isInf(ps.ru[i])
			if v > 0 {
				if hasL {
					downLocks++
				}
				if hasU {
					upLocks++
				}
			} else {
				if hasL {
					upLocks++
				}
				if hasU {
					downLocks++
				}
			}
		}
		c := ps.obj[j]
		if c >= 0 && downLocks == 0 {
			if isInf(ps.lo[j]) {
				if c > 0 {
					return Unbounded
				}
				continue
			}
			ps.fixCol(j, ps.lo[j])
		} else if c <= 0 && upLocks == 0 {
			if isInf(ps.up[j]) {
				if c < 0 {
					return Unbounded
				}
				continue
			}
			ps.fixCol(j, ps.up[j])
		}
	}
	return Reduced
}

func (ps *presolver) coef(i, j int) float64 {
	for _, e := range ps.rows[i] {
		if e.j == j {
			return e.v
		}
	}
	return 0
}

func (ps *presolver) build() {
	p := ps.p
	res := ps.res
	newCol := make([]int, p.NumCols)
	for j := range newCol {
		newCol[j] = -1
		if ps.colAlive[j] {
			newCol[j] = len(res.ColMap)
			res.ColMap = append(res.ColMap, j)
		}
	}
	newRow := make([]int, p.NumRows)
	for i := range newRow {
		newRow[i] = -1
		if ps.rowAlive[i] {
			newRow[i] = len(res.RowMap)
			res.RowMap = append(res.RowMap, i)
		}
	}
	n, m := len(res.ColMap), len(res.RowMap)
	q := &problem.Problem{NumCols: n, NumRows: m, Offset: ps.offset}
	q.Obj = make([]float64, n)
	q.Lo = make([]float64, n)
	q.Up = make([]float64, n)
	q.Integer = make([]bool, n)
	q.RowLo = make([]float64, m)
	q.RowUp = make([]float64, m)
	for k, j := range res.ColMap {
		q.Obj[k] = ps.obj[j]
		q.Lo[k] = ps.lo[j]
		q.Up[k] = ps.up[j]
		q.Integer[k] = ps.integer[j]
	}
	cnt := make([]int, n+1)
	for k, i := range res.RowMap {
		q.RowLo[k] = ps.rl[i]
		q.RowUp[k] = ps.ru[i]
		if isInf(q.RowLo[k]) {
			q.RowLo[k] = math.Inf(-1)
		}
		if isInf(q.RowUp[k]) {
			q.RowUp[k] = math.Inf(1)
		}
		for _, e := range ps.rows[i] {
			cnt[newCol[e.j]+1]++
		}
	}
	for k := 0; k < n; k++ {
		cnt[k+1] += cnt[k]
	}
	q.ColStart = append([]int(nil), cnt...)
	q.RowIdx = make([]int, cnt[n])
	q.Val = make([]float64, cnt[n])
	next := append([]int(nil), cnt[:n]...)
	for k, i := range res.RowMap {
		for _, e := range ps.rows[i] {
			c := newCol[e.j]
			q.RowIdx[next[c]] = k
			q.Val[next[c]] = e.v
			next[c]++
		}
	}
	res.Reduced = q
	res.fixedVal = ps.fixedVal
}
