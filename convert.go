package gost

import (
	"gost/internal/problem"
)

// toProblem converts the model to the internal minimization form.
func (m *Model) toProblem() *problem.Problem {
	n, nr := len(m.vars), len(m.cons)
	p := &problem.Problem{
		NumCols: n, NumRows: nr,
		Obj: make([]float64, n), Lo: make([]float64, n), Up: make([]float64, n),
		Integer: make([]bool, n),
		RowLo:   make([]float64, nr), RowUp: make([]float64, nr),
		Offset: m.objOffset,
	}
	sign := 1.0
	if m.sense == Maximize {
		sign = -1
		p.Offset = -m.objOffset
	}
	cnt := make([]int, n+1)
	for i, c := range m.cons {
		p.RowLo[i], p.RowUp[i] = c.lo, c.up
		for _, j := range c.idx {
			cnt[j+1]++
		}
	}
	for j := 0; j < n; j++ {
		cnt[j+1] += cnt[j]
	}
	p.ColStart = append([]int(nil), cnt...)
	p.RowIdx = make([]int, cnt[n])
	p.Val = make([]float64, cnt[n])
	next := append([]int(nil), cnt[:n]...)
	for i, c := range m.cons {
		for k, j := range c.idx {
			p.RowIdx[next[j]] = i
			p.Val[next[j]] = c.val[k]
			next[j]++
		}
	}
	for j, v := range m.vars {
		p.Obj[j] = sign * v.obj
		p.Lo[j], p.Up[j] = v.lo, v.up
		p.Integer[j] = v.typ != Continuous
	}
	return p
}
