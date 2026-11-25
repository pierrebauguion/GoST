package presolve

import "math"

// Variable elimination reductions with primal postsolve:
//   - doubleton equations a_x x + a_y y = c: x is substituted by
//     (c - a_y y)/a_x in all rows and the objective, its bounds move to y;
//   - implied free column singletons in an equality row: the column and the
//     row are removed, the column absorbs the row;
//   - continuous slack singletons with zero cost: the column is removed and
//     the row bounds are widened.

const (
	opSubst = iota // x_j = (rhs - sum val*x[idx]) / a
	opSlack        // choose x_j in [lo,up] with rl <= sum val*x[idx] + a x_j <= ru
)

type postOp struct {
	kind   int
	j      int
	a      float64
	rhs    float64
	rl, ru float64
	lo, up float64
	idx    []int
	val    []float64
}

func (r *Result) applyOps(x []float64) {
	for k := len(r.ops) - 1; k >= 0; k-- {
		op := &r.ops[k]
		s := 0.0
		for p, j := range op.idx {
			s += op.val[p] * x[j]
		}
		switch op.kind {
		case opSubst:
			x[op.j] = (op.rhs - s) / op.a
		case opSlack:
			lo, hi := math.Inf(-1), math.Inf(1)
			if !isInf(op.rl) {
				lo = (op.rl - s) / op.a
			}
			if !isInf(op.ru) {
				hi = (op.ru - s) / op.a
			}
			if op.a < 0 {
				lo, hi = hi, lo
				if isInf(op.rl) {
					hi = math.Inf(1)
				}
				if isInf(op.ru) {
					lo = math.Inf(-1)
				}
			}
			lo = math.Max(lo, op.lo)
			hi = math.Min(hi, op.up)
			v := math.Max(lo, math.Min(0, hi))
			if math.IsInf(v, 0) {
				v = 0
			}
			x[op.j] = v
		}
	}
}

// aliveRowsOf returns the alive rows that contain column j.
func (ps *presolver) aliveRowsOf(j int, limit int) []int {
	var out []int
	for _, i := range ps.colRows[j] {
		if !ps.rowAlive[i] || ps.coef(i, j) == 0 {
			continue
		}
		dup := false
		for _, k := range out {
			if k == i {
				dup = true
				break
			}
		}
		if dup {
			continue
		}
		out = append(out, i)
		if len(out) > limit {
			break
		}
	}
	return out
}

// eliminate substitutes column x using equality row i (rhs c) in every
// other row and the objective, then removes row i and column x.
func (ps *presolver) eliminate(i, x int) {
	c := ps.rl[i]
	var ax float64
	var others []entry
	for _, e := range ps.rows[i] {
		if e.j == x {
			ax = e.v
		} else {
			others = append(others, e)
		}
	}
	// objective
	if cx := ps.obj[x]; cx != 0 {
		ps.offset += cx * c / ax
		for _, e := range others {
			ps.obj[e.j] -= cx * e.v / ax
		}
		ps.obj[x] = 0
	}
	// other rows
	for _, k := range ps.aliveRowsOf(x, math.MaxInt) {
		if k == i {
			continue
		}
		akx := 0.0
		r := ps.rows[k]
		for p, e := range r {
			if e.j == x {
				akx = e.v
				r[p] = r[len(r)-1]
				r = r[:len(r)-1]
				break
			}
		}
		shift := akx * c / ax
		if !isInf(ps.rl[k]) {
			ps.rl[k] -= shift
		}
		if !isInf(ps.ru[k]) {
			ps.ru[k] -= shift
		}
		for _, e := range others {
			d := -akx * e.v / ax
			found := false
			for p := range r {
				if r[p].j == e.j {
					r[p].v += d
					found = true
					break
				}
			}
			if !found {
				r = append(r, entry{e.j, d})
				ps.colRows[e.j] = append(ps.colRows[e.j], k)
			}
		}
		// drop cancelled entries
		q := 0
		for _, e := range r {
			if math.Abs(e.v) > 1e-12 {
				r[q] = e
				q++
			}
		}
		ps.rows[k] = r[:q]
	}
	op := postOp{kind: opSubst, j: x, a: ax, rhs: c}
	for _, e := range others {
		op.idx = append(op.idx, e.j)
		op.val = append(op.val, e.v)
	}
	ps.res.ops = append(ps.res.ops, op)
	ps.colAlive[x] = false
	ps.fixedVal[x] = math.NaN()
	ps.res.ColsRemoved++
	ps.res.Aggregated++
	ps.removeRow(i)
}

// doubleton handles an equality row with two entries.
func (ps *presolver) doubleton(i int) Status {
	r := ps.rows[i]
	if len(r) != 2 || ps.rl[i] != ps.ru[i] || isInf(ps.rl[i]) {
		return Reduced
	}
	c := ps.rl[i]
	try := func(xe, ye entry) bool {
		x, y := xe.j, ye.j
		ax, ay := xe.v, ye.v
		if math.Abs(ax) < 1e-9 || math.Abs(ay) < 1e-9 {
			return false
		}
		rr := -ay / ax // x = q + rr*y
		q := c / ax
		if math.Abs(rr) > 1e3 || math.Abs(rr) < 1e-3 {
			return false
		}
		if ps.integer[x] {
			if !ps.integer[y] || math.Abs(rr-math.Round(rr)) > 1e-9 || math.Abs(q-math.Round(q)) > 1e-9 {
				return false
			}
		}
		// transfer x bounds to y
		l, u := math.Inf(-1), math.Inf(1)
		lx, ux := ps.lo[x], ps.up[x]
		if rr > 0 {
			if !isInf(lx) {
				l = (lx - q) / rr
			}
			if !isInf(ux) {
				u = (ux - q) / rr
			}
		} else {
			if !isInf(ux) {
				l = (ux - q) / rr
			}
			if !isInf(lx) {
				u = (lx - q) / rr
			}
		}
		if st := ps.setBounds(y, l, u); st != Reduced {
			ps.infeasible = true
			return true
		}
		ps.eliminate(i, x)
		return true
	}
	a, b := r[0], r[1]
	// prefer eliminating a continuous variable
	if !ps.integer[a.j] && try(a, b) {
	} else if !ps.integer[b.j] && try(b, a) {
	} else if try(a, b) {
	} else {
		try(b, a)
	}
	if ps.infeasible {
		return Infeasible
	}
	return Reduced
}

// singletonCol handles a continuous column appearing in a single row.
func (ps *presolver) singletonCol(j int) Status {
	if ps.integer[j] {
		return Reduced
	}
	rows := ps.aliveRowsOf(j, 1)
	if len(rows) != 1 {
		return Reduced
	}
	i := rows[0]
	if len(ps.rows[i]) < 2 {
		return Reduced
	}
	a := ps.coef(i, j)
	if math.Abs(a) < 1e-9 {
		return Reduced
	}
	l, u := ps.lo[j], ps.up[j]
	rl, ru := ps.rl[i], ps.ru[i]
	// Implied free singleton in an equality row: x absorbs the row.
	if rl == ru && !isInf(rl) {
		// activity of the rest
		var minR, maxR float64
		minInf, maxInf := 0, 0
		for _, e := range ps.rows[i] {
			if e.j == j {
				continue
			}
			lo, up := ps.lo[e.j], ps.up[e.j]
			if e.v > 0 {
				if isInf(lo) {
					minInf++
				} else {
					minR += e.v * lo
				}
				if isInf(up) {
					maxInf++
				} else {
					maxR += e.v * up
				}
			} else {
				if isInf(up) {
					minInf++
				} else {
					minR += e.v * up
				}
				if isInf(lo) {
					maxInf++
				} else {
					maxR += e.v * lo
				}
			}
		}
		// a x = rl - rest
		il, iu := math.Inf(-1), math.Inf(1)
		if a > 0 {
			if maxInf == 0 {
				il = (rl - maxR) / a
			}
			if minInf == 0 {
				iu = (rl - minR) / a
			}
		} else {
			if minInf == 0 {
				il = (rl - minR) / a
			}
			if maxInf == 0 {
				iu = (rl - maxR) / a
			}
		}
		tol := 1e-9
		if (isInf(l) || il >= l-tol*(1+math.Abs(l))) && (isInf(u) || iu <= u+tol*(1+math.Abs(u))) {
			ps.eliminate(i, j)
			return Reduced
		}
	}
	// Slack singleton with zero cost: widen the row and drop the column.
	if ps.obj[j] != 0 || isInf(l) && isInf(u) {
		return Reduced
	}
	amin, amax := a*l, a*u
	if a < 0 {
		amin, amax = a*u, a*l
	}
	if isInf(l) || isInf(u) {
		if a > 0 {
			if isInf(l) {
				amin = math.Inf(-1)
			}
			if isInf(u) {
				amax = math.Inf(1)
			}
		} else {
			if isInf(u) {
				amin = math.Inf(-1)
			}
			if isInf(l) {
				amax = math.Inf(1)
			}
		}
	}
	op := postOp{kind: opSlack, j: j, a: a, rl: rl, ru: ru, lo: l, up: u}
	r := ps.rows[i]
	for p, e := range r {
		if e.j == j {
			r[p] = r[len(r)-1]
			r = r[:len(r)-1]
			break
		}
	}
	ps.rows[i] = r
	for _, e := range r {
		op.idx = append(op.idx, e.j)
		op.val = append(op.val, e.v)
	}
	if !isInf(rl) {
		if math.IsInf(amax, 1) {
			ps.rl[i] = math.Inf(-1)
		} else {
			ps.rl[i] = rl - amax
		}
	}
	if !isInf(ru) {
		if math.IsInf(amin, -1) {
			ps.ru[i] = math.Inf(1)
		} else {
			ps.ru[i] = ru - amin
		}
	}
	ps.res.ops = append(ps.res.ops, op)
	ps.colAlive[j] = false
	ps.fixedVal[j] = math.NaN()
	ps.res.ColsRemoved++
	ps.changed = true
	return Reduced
}
