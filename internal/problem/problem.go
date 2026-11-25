// Package problem defines the internal column-oriented problem
// representation shared by presolve, LP and MIP components.
package problem

import "math"

// Problem is min c^T x + Offset s.t. RowLo <= A x <= RowUp, Lo <= x <= Up,
// x_j integer for Integer[j].
type Problem struct {
	NumCols, NumRows int
	Obj              []float64
	Offset           float64
	Lo, Up           []float64
	Integer          []bool
	RowLo, RowUp     []float64
	// CSC matrix.
	ColStart []int
	RowIdx   []int
	Val      []float64
}

// Inf is +infinity.
var Inf = math.Inf(1)

// Clone returns a deep copy.
func (p *Problem) Clone() *Problem {
	q := *p
	q.Obj = append([]float64(nil), p.Obj...)
	q.Lo = append([]float64(nil), p.Lo...)
	q.Up = append([]float64(nil), p.Up...)
	q.Integer = append([]bool(nil), p.Integer...)
	q.RowLo = append([]float64(nil), p.RowLo...)
	q.RowUp = append([]float64(nil), p.RowUp...)
	q.ColStart = append([]int(nil), p.ColStart...)
	q.RowIdx = append([]int(nil), p.RowIdx...)
	q.Val = append([]float64(nil), p.Val...)
	return &q
}

// RowWise returns a CSR copy of the matrix.
func (p *Problem) RowWise() (rowStart, colIdx []int, val []float64) {
	cnt := make([]int, p.NumRows+1)
	for _, i := range p.RowIdx {
		cnt[i+1]++
	}
	for i := 0; i < p.NumRows; i++ {
		cnt[i+1] += cnt[i]
	}
	rowStart = append([]int(nil), cnt...)
	colIdx = make([]int, len(p.RowIdx))
	val = make([]float64, len(p.RowIdx))
	next := append([]int(nil), cnt[:p.NumRows]...)
	for j := 0; j < p.NumCols; j++ {
		for k := p.ColStart[j]; k < p.ColStart[j+1]; k++ {
			i := p.RowIdx[k]
			colIdx[next[i]] = j
			val[next[i]] = p.Val[k]
			next[i]++
		}
	}
	return
}

// MaxViolation returns the maximum bound/row/integrality violation of x.
func (p *Problem) MaxViolation(x []float64) (bound, row, integ float64) {
	act := make([]float64, p.NumRows)
	for j := 0; j < p.NumCols; j++ {
		v := x[j]
		bound = math.Max(bound, math.Max(p.Lo[j]-v, v-p.Up[j]))
		if p.Integer[j] {
			integ = math.Max(integ, math.Abs(v-math.Round(v)))
		}
		for k := p.ColStart[j]; k < p.ColStart[j+1]; k++ {
			act[p.RowIdx[k]] += p.Val[k] * v
		}
	}
	for i, a := range act {
		row = math.Max(row, math.Max(p.RowLo[i]-a, a-p.RowUp[i]))
	}
	return
}

// ObjValue returns c^T x + Offset.
func (p *Problem) ObjValue(x []float64) float64 {
	s := p.Offset
	for j, c := range p.Obj {
		s += c * x[j]
	}
	return s
}
