// Package lu implements a sparse LU factorization of simplex basis matrices
// with product-form (eta) updates.
//
// The factorization is a left-looking Gilbert–Peierls LU with threshold
// partial pivoting. Columns are processed in increasing order of their
// nonzero count (logical/singleton columns first); among acceptable pivot
// candidates (|x_i| >= Threshold*max|x|) the row with the smallest original
// count is chosen, which approximates Markowitz ordering for basis matrices.
//
// Notation: B has m rows (constraint indices) and m columns (basis
// positions). Pivot step t eliminates row prow[t] using basis position
// pcol[t].
package lu

import (
	"math"
	"os"
	"sort"
	"strconv"
)

// ColumnFunc returns the sparse column at basis position k.
type ColumnFunc func(k int) (idx []int, val []float64)

// Factor holds L, U and the eta file of a basis factorization.
type Factor struct {
	m int

	// Pivot sequence.
	prow []int
	pcol []int
	piv  []float64
	step []int // step[row] = pivot step of row (or -1)

	// L columns per step: entries (row, multiplier).
	lStart []int
	lIdx   []int
	lVal   []float64

	// U columns per step k: entries (step t < k, value).
	uStart []int
	uIdx   []int
	uVal   []float64

	// Cross references between the column-wise and row-wise U copies and
	// the row index of each U entry (for Forrest–Tomlin zeroing).
	uToUt, utToU, uRow []int

	// Transposed copies for sparse BTRAN.
	utStart []int // per step t: entries (step k > t, value U[t,k])
	utIdx   []int
	utVal   []float64
	ltStart []int // per row i: entries (step t, multiplier) with i in L_t
	ltIdx   []int
	ltRow   []int // prow[ltIdx]
	ltVal   []float64

	// Eta file for product-form updates.
	etaPos   []int
	etaPiv   []float64
	etaStart []int
	etaIdx   []int
	etaVal   []float64

	// Work arrays.
	work  []float64
	work2 []float64
	mark  []int
	gen   int
	smark []int
	sgen  int
	stack []int
	pstk  []int
	topo  []int

	mk mkWork
	ft ftState

	nzCol, nzAux []int
	nzOut        *[]int

	// Options.
	Threshold float64 // relative pivot threshold (default 0.1)
	AbsTol    float64 // absolute pivot tolerance (default 1e-11)
	DropTol   float64 // drop tolerance for L/U/eta entries (default 1e-14)
}

// New allocates a factor for dimension m.
func New(m int) *Factor {
	f := &Factor{Threshold: 0.1, AbsTol: 1e-11, DropTol: 1e-14}
	f.resize(m)
	return f
}

func (f *Factor) resize(m int) {
	f.m = m
	f.prow = make([]int, m)
	f.pcol = make([]int, m)
	f.piv = make([]float64, m)
	f.step = make([]int, m)
	f.work = make([]float64, m)
	f.work2 = make([]float64, m)
	f.mark = make([]int, m)
	f.gen = 0
	f.smark = make([]int, m)
	f.sgen = 0
}

// Dim returns the dimension.
func (f *Factor) Dim() int { return f.m }

// NumUpdates returns the number of eta updates since the last factorization.
func (f *Factor) NumUpdates() int {
	if f.ft.on {
		return f.ft.nUpd
	}
	return len(f.etaPos)
}

// EtaGrowth is the ratio threshold used by NeedRefactor.
var EtaGrowth = envFloat("GOST_ETAGROWTH", 2.0)

// NeedRefactor reports whether a fresh factorization is advisable: too many
// updates, or an eta file whose size exceeds that of L and U (solves would
// then cost more than refactoring).
func (f *Factor) NeedRefactor(maxUpdates int) bool {
	if f.ft.on {
		ft := &f.ft
		if ft.nUpd >= maxUpdates {
			return true
		}
		return ft.nUpd >= 10 && float64(ft.nDyn+len(ft.rIdx)) > EtaGrowth*float64(len(f.lIdx)+len(f.uIdx)+f.m)
	}
	if len(f.etaPos) >= maxUpdates {
		return true
	}
	return len(f.etaPos) >= 10 && float64(len(f.etaIdx)) > EtaGrowth*float64(len(f.lIdx)+len(f.uIdx)+f.m)
}

// Singularity describes basis positions that could not be pivoted and the
// rows that remained without a pivot. The caller should replace each
// position Positions[i] by the logical column of Rows[i] and refactor.
type Singularity struct {
	Positions []int
	Rows      []int
}

// Factorize computes the LU factorization of the m x m basis given by cols.
// It returns nil on success or a Singularity description.
func (f *Factor) Factorize(m int, cols ColumnFunc) *Singularity {
	if m != f.m {
		f.resize(m)
	}
	f.etaPos = f.etaPos[:0]
	f.etaPiv = f.etaPiv[:0]
	f.etaStart = append(f.etaStart[:0], 0)
	f.etaIdx = f.etaIdx[:0]
	f.etaVal = f.etaVal[:0]
	if dumpPath != "" {
		dumpBasis(m, cols)
	}
	if UseMarkowitz {
		for i := 0; i < m; i++ {
			f.step[i] = -1
		}
		return f.factorizeMarkowitz(m, cols)
	}

	// Row counts and column order.
	rowCount := make([]int, m)
	colCount := make([]int, m)
	for k := 0; k < m; k++ {
		idx, _ := cols(k)
		colCount[k] = len(idx)
		for _, i := range idx {
			rowCount[i]++
		}
	}
	order := make([]int, m)
	for k := range order {
		order[k] = k
	}
	sort.SliceStable(order, func(a, b int) bool { return colCount[order[a]] < colCount[order[b]] })

	for i := 0; i < m; i++ {
		f.step[i] = -1
	}
	f.lStart = append(f.lStart[:0], 0)
	f.lIdx = f.lIdx[:0]
	f.lVal = f.lVal[:0]
	f.uStart = append(f.uStart[:0], 0)
	f.uIdx = f.uIdx[:0]
	f.uVal = f.uVal[:0]

	x := f.work
	var sing *Singularity
	nstep := 0
	pattern := make([]int, 0, 64)

	for _, k := range order {
		idx, val := cols(k)
		// Scatter column.
		f.gen++
		g := f.gen
		pattern = pattern[:0]
		for p, i := range idx {
			x[i] = val[p]
			f.mark[i] = g
			pattern = append(pattern, i)
		}
		// Topological order of pivot steps reached from the pattern.
		topo := f.reach(idx)
		for _, t := range topo {
			xr := x[f.prow[t]]
			if xr == 0 {
				continue
			}
			for p := f.lStart[t]; p < f.lStart[t+1]; p++ {
				i := f.lIdx[p]
				if f.mark[i] != g {
					f.mark[i] = g
					x[i] = 0
					pattern = append(pattern, i)
				}
				x[i] -= f.lVal[p] * xr
			}
		}
		// Choose pivot among non-pivoted rows.
		maxAbs := 0.0
		for _, i := range pattern {
			if f.step[i] < 0 {
				if a := math.Abs(x[i]); a > maxAbs {
					maxAbs = a
				}
			}
		}
		if maxAbs <= f.AbsTol {
			if sing == nil {
				sing = &Singularity{}
			}
			sing.Positions = append(sing.Positions, k)
			for _, i := range pattern {
				x[i] = 0
			}
			continue
		}
		best := -1
		bestCount := math.MaxInt
		bestAbs := 0.0
		thr := f.Threshold * maxAbs
		for _, i := range pattern {
			if f.step[i] >= 0 {
				continue
			}
			a := math.Abs(x[i])
			if a < thr {
				continue
			}
			if rowCount[i] < bestCount || (rowCount[i] == bestCount && a > bestAbs) {
				best, bestCount, bestAbs = i, rowCount[i], a
			}
		}
		t := nstep
		nstep++
		pv := x[best]
		f.prow[t] = best
		f.pcol[t] = k
		f.piv[t] = pv
		f.step[best] = t
		// U column: entries in already pivoted rows.
		for _, i := range pattern {
			s := f.step[i]
			if s >= 0 && s < t && math.Abs(x[i]) > f.DropTol {
				f.uIdx = append(f.uIdx, s)
				f.uVal = append(f.uVal, x[i])
			}
		}
		f.uStart = append(f.uStart, len(f.uIdx))
		// L column: non-pivoted rows.
		for _, i := range pattern {
			if f.step[i] < 0 && math.Abs(x[i]) > f.DropTol {
				f.lIdx = append(f.lIdx, i)
				f.lVal = append(f.lVal, x[i]/pv)
			}
		}
		f.lStart = append(f.lStart, len(f.lIdx))
		for _, i := range pattern {
			x[i] = 0
		}
	}
	if sing != nil {
		for i := 0; i < m; i++ {
			if f.step[i] < 0 {
				sing.Rows = append(sing.Rows, i)
			}
		}
		return sing
	}
	f.buildTransposes()
	f.initFT()
	return nil
}

func (f *Factor) buildTransposes() {
	m := f.m
	// U^T: rows by step.
	f.utStart = resizeInts(f.utStart, m+1)
	for i := range f.utStart {
		f.utStart[i] = 0
	}
	for _, t := range f.uIdx {
		f.utStart[t+1]++
	}
	for t := 0; t < m; t++ {
		f.utStart[t+1] += f.utStart[t]
	}
	nu := len(f.uIdx)
	f.utIdx = resizeInts(f.utIdx, nu)
	f.utVal = resizeFloats(f.utVal, nu)
	f.uToUt = resizeInts(f.uToUt, nu)
	f.utToU = resizeInts(f.utToU, nu)
	f.uRow = resizeInts(f.uRow, nu)
	next := resizeInts(f.stack, m)
	copy(next, f.utStart[:m])
	for k := 0; k < m; k++ {
		for p := f.uStart[k]; p < f.uStart[k+1]; p++ {
			t := f.uIdx[p]
			q := next[t]
			f.utIdx[q] = k
			f.utVal[q] = f.uVal[p]
			f.uToUt[p] = q
			f.utToU[q] = p
			f.uRow[p] = f.prow[t]
			next[t]++
		}
	}
	// L^T: by row.
	f.ltStart = resizeInts(f.ltStart, m+1)
	for i := range f.ltStart {
		f.ltStart[i] = 0
	}
	for _, i := range f.lIdx {
		f.ltStart[i+1]++
	}
	for i := 0; i < m; i++ {
		f.ltStart[i+1] += f.ltStart[i]
	}
	nl := len(f.lIdx)
	f.ltIdx = resizeInts(f.ltIdx, nl)
	f.ltRow = resizeInts(f.ltRow, nl)
	f.ltVal = resizeFloats(f.ltVal, nl)
	copy(next, f.ltStart[:m])
	for t := 0; t < m; t++ {
		for p := f.lStart[t]; p < f.lStart[t+1]; p++ {
			i := f.lIdx[p]
			q := next[i]
			f.ltIdx[q] = t
			f.ltRow[q] = f.prow[t]
			f.ltVal[q] = f.lVal[p]
			next[i]++
		}
	}
	f.stack = next[:0]
}

func resizeInts(a []int, n int) []int {
	if cap(a) < n {
		return make([]int, n)
	}
	return a[:n]
}

func resizeFloats(a []float64, n int) []float64 {
	if cap(a) < n {
		return make([]float64, n)
	}
	return a[:n]
}

// reach returns pivot steps reachable from the rows in idx, in topological
// order (a step appears before all steps that depend on it).
func (f *Factor) reach(idx []int) []int {
	f.sgen++
	g := f.sgen
	f.topo = f.topo[:0]
	for _, i0 := range idx {
		t0 := f.step[i0]
		if t0 < 0 || f.smark[t0] == g {
			continue
		}
		f.stack = append(f.stack[:0], t0)
		f.pstk = append(f.pstk[:0], f.lStart[t0])
		f.smark[t0] = g
		for len(f.stack) > 0 {
			top := len(f.stack) - 1
			t := f.stack[top]
			p := f.pstk[top]
			pushed := false
			for ; p < f.lStart[t+1]; p++ {
				s := f.step[f.lIdx[p]]
				if s >= 0 && f.smark[s] != g {
					f.smark[s] = g
					f.pstk[top] = p + 1
					f.stack = append(f.stack, s)
					f.pstk = append(f.pstk, f.lStart[s])
					pushed = true
					break
				}
			}
			if !pushed {
				f.stack = f.stack[:top]
				f.pstk = f.pstk[:top]
				f.topo = append(f.topo, t)
			}
		}
	}
	for a, b := 0, len(f.topo)-1; a < b; a, b = a+1, b-1 {
		f.topo[a], f.topo[b] = f.topo[b], f.topo[a]
	}
	return f.topo
}

// FTRAN solves B x = b in place. b is indexed by rows on input and x by
// basis positions on output (dense vectors of length m).
func (f *Factor) FTRAN(b []float64) {
	if f.ft.on {
		f.ftranFT(b, false)
		return
	}
	m := f.m
	// L
	for t := 0; t < m; t++ {
		v := b[f.prow[t]]
		if v == 0 {
			continue
		}
		for p := f.lStart[t]; p < f.lStart[t+1]; p++ {
			b[f.lIdx[p]] -= f.lVal[p] * v
		}
	}
	// U (column-oriented back substitution); y in step space stored in work.
	y := f.work2
	for k := m - 1; k >= 0; k-- {
		v := b[f.prow[k]]
		if v == 0 {
			y[k] = 0
			continue
		}
		v /= f.piv[k]
		y[k] = v
		for p := f.uStart[k]; p < f.uStart[k+1]; p++ {
			b[f.prow[f.uIdx[p]]] -= f.uVal[p] * v
		}
	}
	for k := 0; k < m; k++ {
		b[f.pcol[k]] = y[k]
	}
	// Etas.
	for e := range f.etaPos {
		pos := f.etaPos[e]
		v := b[pos]
		if v == 0 {
			continue
		}
		v /= f.etaPiv[e]
		b[pos] = v
		for p := f.etaStart[e]; p < f.etaStart[e+1]; p++ {
			b[f.etaIdx[p]] -= f.etaVal[p] * v
		}
	}
}

// BTRAN solves B^T y = d in place. d is indexed by basis positions on input,
// y by rows on output.
func (f *Factor) BTRAN(d []float64) {
	if f.ft.on {
		f.btranFT(d)
		return
	}
	m := f.m
	// Etas in reverse.
	for e := len(f.etaPos) - 1; e >= 0; e-- {
		pos := f.etaPos[e]
		v := d[pos]
		for p := f.etaStart[e]; p < f.etaStart[e+1]; p++ {
			v -= f.etaVal[p] * d[f.etaIdx[p]]
		}
		d[pos] = v / f.etaPiv[e]
	}
	// U^T forward (scatter form, skips zeros): z[step].
	z := f.work2
	for k := 0; k < m; k++ {
		z[k] = d[f.pcol[k]]
	}
	for k := 0; k < m; k++ {
		v := z[k]
		if v == 0 {
			continue
		}
		v /= f.piv[k]
		z[k] = v
		for p := f.utStart[k]; p < f.utStart[k+1]; p++ {
			z[f.utIdx[p]] -= f.utVal[p] * v
		}
	}
	// Move to row space.
	for k := 0; k < m; k++ {
		d[f.prow[k]] = z[k]
	}
	// L^T backward (scatter form).
	for s := m - 1; s >= 0; s-- {
		r := f.prow[s]
		v := d[r]
		if v == 0 {
			continue
		}
		for p := f.ltStart[r]; p < f.ltStart[r+1]; p++ {
			d[f.ltRow[p]] -= f.ltVal[p] * v
		}
	}
}

// Update records the replacement of the column at basis position pos by a
// column whose FTRAN'd representation is alpha (dense, indexed by
// positions). Returns false if the pivot is too small.
func (f *Factor) Update(pos int, alpha []float64) bool {
	if f.ft.on {
		return f.updateFT(pos, alpha[pos])
	}
	pv := alpha[pos]
	if math.Abs(pv) < 1e-12 {
		return false
	}
	for i, v := range alpha {
		if i != pos && v != 0 && math.Abs(v) > f.DropTol {
			f.etaIdx = append(f.etaIdx, i)
			f.etaVal = append(f.etaVal, v)
		}
	}
	f.etaPos = append(f.etaPos, pos)
	f.etaPiv = append(f.etaPiv, pv)
	f.etaStart = append(f.etaStart, len(f.etaIdx))
	return true
}

// FTRANCol is FTRAN for the entering column of a basis change; with
// Forrest–Tomlin updates it saves the spike needed by the next Update. It
// returns the positions of the nonzeros of the result (valid until the next
// FTRANCol call).
func (f *Factor) FTRANCol(b []float64) []int {
	f.nzCol = f.nzCol[:0]
	if f.ft.on {
		f.nzOut = &f.nzCol
		f.ftranFT(b, true)
		f.nzOut = nil
		return f.nzCol
	}
	f.FTRAN(b)
	for i, v := range b[:f.m] {
		if v != 0 {
			f.nzCol = append(f.nzCol, i)
		}
	}
	return f.nzCol
}

// FTRANNZ is FTRAN returning the nonzero positions of the result (valid
// until the next FTRANNZ call).
func (f *Factor) FTRANNZ(b []float64) []int {
	f.nzAux = f.nzAux[:0]
	if f.ft.on {
		f.nzOut = &f.nzAux
		f.ftranFT(b, false)
		f.nzOut = nil
		return f.nzAux
	}
	f.FTRAN(b)
	for i, v := range b[:f.m] {
		if v != 0 {
			f.nzAux = append(f.nzAux, i)
		}
	}
	return f.nzAux
}

// FillIn returns the number of nonzeros in L, U and the update file.
func (f *Factor) FillIn() (l, u, eta int) {
	if f.ft.on {
		return len(f.lIdx), len(f.uIdx) + f.m, f.ft.nDyn + len(f.ft.rIdx)
	}
	return len(f.lIdx), len(f.uIdx) + f.m, len(f.etaIdx)
}

func envFloat(name string, def float64) float64 {
	if v, err := strconv.ParseFloat(os.Getenv(name), 64); err == nil {
		return v
	}
	return def
}
