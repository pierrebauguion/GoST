package lu

import (
	"math"
)

// UseMarkowitz selects the right-looking Markowitz factorization (default)
// instead of the left-looking Gilbert–Peierls one.
var UseMarkowitz = envFloat("GOST_MARKOWITZ", 1) != 0

// MarkowitzSearch is the number of candidate columns/rows examined.
var rowSearchMax = int(envFloat("GOST_ROWSEARCH", 1e9))

// DenseSwitch is the active-submatrix density above which the
// factorization finishes with a dense LU.
var DenseSwitch = envFloat("GOST_DENSE", 0.15)

var MarkowitzSearch = int(envFloat("GOST_MKSEARCH", 4))

// bucket lists of indices by count.
type buckets struct {
	head       []int
	next, prev []int
	count      []int
	in         []bool
}

func newBuckets(n, maxCount int) *buckets {
	b := &buckets{}
	b.reset(n, maxCount)
	return b
}

func (b *buckets) reset(n, maxCount int) {
	b.head = resizeInts(b.head, maxCount+2)
	b.next = resizeInts(b.next, n)
	b.prev = resizeInts(b.prev, n)
	b.count = resizeInts(b.count, n)
	if cap(b.in) < n {
		b.in = make([]bool, n)
	}
	b.in = b.in[:n]
	for i := range b.head {
		b.head[i] = -1
	}
	for i := range b.in {
		b.in[i] = false
	}
}

// mkWork holds reusable buffers of the Markowitz factorization.
type mkWork struct {
	rIdx      [][]int
	cVal      [][]float64
	cRows     [][]int
	rowActive []bool
	colActive []bool
	colCount  []int
	mark      []int
	rmark     []int
	rb, cb    buckets
	uRowStart []int
	uRowCol   []int
	uRowVal   []float64
	stepOfCol []int
	cnt       []int
	cValid    []bool
	dRows     []int
	dCols     []int
	dRp       []int
	dCp       []int
	dA        []float64
	dRnz      []int
	dCnz      []int
	dPat      []int
	cStale    []bool
	touched   []bool
	oStart    []int
	oNext     []int
	oCol      []int
	oVal      []float64
	cBestR    []int
	cMax      []float64
	cBestCost []float64
	cBestAbs  []float64
}

func resizeBools(a []bool, n int) []bool {
	if cap(a) < n {
		return make([]bool, n)
	}
	return a[:n]
}

func (b *buckets) insert(i, k int) {
	if k >= len(b.head) {
		k = len(b.head) - 1
	}
	b.count[i] = k
	b.prev[i] = -1
	b.next[i] = b.head[k]
	if b.head[k] >= 0 {
		b.prev[b.head[k]] = i
	}
	b.head[k] = i
	b.in[i] = true
}

func (b *buckets) remove(i int) {
	if !b.in[i] {
		return
	}
	k := b.count[i]
	if b.prev[i] >= 0 {
		b.next[b.prev[i]] = b.next[i]
	} else {
		b.head[k] = b.next[i]
	}
	if b.next[i] >= 0 {
		b.prev[b.next[i]] = b.prev[i]
	}
	b.in[i] = false
}

func (b *buckets) move(i, k int) {
	b.remove(i)
	b.insert(i, k)
}

// factorizeMarkowitz computes P B Q = L U with a right-looking elimination
// on the active submatrix, choosing pivots by the Markowitz criterion with
// threshold partial pivoting (column-wise), Suhl & Suhl style limited search.
//
// The active submatrix is stored column-wise with values (cRow/cVal) and
// row-wise as patterns only (rCol), so column evaluations need no searches.
func (f *Factor) factorizeMarkowitz(m int, cols ColumnFunc) *Singularity {
	w := &f.mk
	if len(w.rIdx) < m {
		w.rIdx = append(w.rIdx, make([][]int, m-len(w.rIdx))...)
		w.cVal = append(w.cVal, make([][]float64, m-len(w.cVal))...)
		w.cRows = append(w.cRows, make([][]int, m-len(w.cRows))...)
	}
	rCol, cVal, cRow := w.rIdx[:m], w.cVal[:m], w.cRows[:m]
	for i := 0; i < m; i++ {
		rCol[i] = rCol[i][:0]
	}
	// Original values by row (valid for rows never updated).
	w.oStart = resizeInts(w.oStart, m+1)
	oStart := w.oStart
	for i := range oStart {
		oStart[i] = 0
	}
	for k := 0; k < m; k++ {
		idx, val := cols(k)
		for p, i := range idx {
			if val[p] != 0 {
				oStart[i+1]++
			}
		}
	}
	for i := 0; i < m; i++ {
		oStart[i+1] += oStart[i]
	}
	w.oCol = resizeInts(w.oCol, oStart[m])
	w.oVal = resizeFloats(w.oVal, oStart[m])
	oCol, oVal := w.oCol, w.oVal
	w.oNext = append(w.oNext[:0], oStart[:m]...)
	for k := 0; k < m; k++ {
		idx, val := cols(k)
		for p, i := range idx {
			if val[p] != 0 {
				q := w.oNext[i]
				oCol[q], oVal[q] = k, val[p]
				w.oNext[i]++
			}
		}
	}
	w.touched = resizeBools(w.touched, m)
	w.cStale = resizeBools(w.cStale, m)
	w.colCount = resizeInts(w.colCount, m)
	touched, cStale, colCount := w.touched, w.cStale, w.colCount
	for i := 0; i < m; i++ {
		touched[i], cStale[i] = false, false
	}
	for k := 0; k < m; k++ {
		idx, val := cols(k)
		cr := cRow[k][:0]
		cv := cVal[k][:0]
		for p, i := range idx {
			if val[p] == 0 {
				continue
			}
			rCol[i] = append(rCol[i], k)
			cr = append(cr, i)
			cv = append(cv, val[p])
		}
		cRow[k], cVal[k] = cr, cv
	}
	w.rowActive = resizeBools(w.rowActive, m)
	w.colActive = resizeBools(w.colActive, m)
	rowActive, colActive := w.rowActive, w.colActive
	w.rb.reset(m, m)
	w.cb.reset(m, m)
	rb, cb := &w.rb, &w.cb
	for i := 0; i < m; i++ {
		rowActive[i] = true
		rb.insert(i, len(rCol[i]))
	}
	for c := 0; c < m; c++ {
		colActive[c] = true
		colCount[c] = len(cRow[c])
		cb.insert(c, colCount[c])
	}
	actNNZ := 0
	for c := 0; c < m; c++ {
		actNNZ += colCount[c]
	}
	w.mark = resizeInts(w.mark, m)
	mark := w.mark
	for i := range mark {
		mark[i] = -1
	}

	f.lStart = append(f.lStart[:0], 0)
	f.lIdx = f.lIdx[:0]
	f.lVal = f.lVal[:0]
	uRowStart := append(w.uRowStart[:0], 0)
	uRowCol := w.uRowCol[:0]
	uRowVal := w.uRowVal[:0]
	defer func() { w.uRowStart, w.uRowCol, w.uRowVal = uRowStart, uRowCol, uRowVal }()

	u := f.Threshold
	absTol := f.AbsTol

	w.cValid = resizeBools(w.cValid, m)
	w.cMax = resizeFloats(w.cMax, m)
	cValid, cMax := w.cValid, w.cMax
	for c := range cValid {
		cValid[c] = false
	}
	// compact removes rows pivoted since the last use of column c.
	compact := func(c int) {
		if !cStale[c] {
			return
		}
		cr, cv := cRow[c], cVal[c]
		n := 0
		for p, i := range cr {
			if rowActive[i] {
				cr[n], cv[n] = i, cv[p]
				n++
			}
		}
		cRow[c], cVal[c] = cr[:n], cv[:n]
		cStale[c] = false
	}
	colMax := func(c int) float64 {
		if cValid[c] {
			return cMax[c]
		}
		compact(c)
		mx := 0.0
		for _, v := range cVal[c] {
			if a := math.Abs(v); a > mx {
				mx = a
			}
		}
		cMax[c], cValid[c] = mx, true
		return mx
	}
	// removeFromRow drops column c from the pattern of row i.
	removeFromRow := func(i, c int) {
		ri := rCol[i]
		for p, cc := range ri {
			if cc == c {
				last := len(ri) - 1
				ri[p] = ri[last]
				rCol[i] = ri[:last]
				return
			}
		}
	}

	nstep := 0
	var sing *Singularity
	for nstep < m {
		if ra := m - nstep; ra >= 16 && ra <= 3000 && float64(actNNZ) > DenseSwitch*float64(ra)*float64(ra) {
			var ok bool
			nstep, uRowCol, uRowVal, uRowStart, ok = f.denseKernel(nstep, m, compact, uRowCol, uRowVal, uRowStart)
			if !ok {
				sing = &Singularity{}
				for c := 0; c < m; c++ {
					if colActive[c] {
						sing.Positions = append(sing.Positions, c)
					}
				}
				for i := 0; i < m; i++ {
					if rowActive[i] {
						sing.Rows = append(sing.Rows, i)
					}
				}
			}
			break
		}
		// ---- Pivot search.
		bestR, bestC, bestP := -1, -1, -1
		bestCost := math.Inf(1)
		bestAbs := 0.0
		searched := 0
	search:
		for k := 1; k <= m; k++ {
			kk := min(k, len(cb.head)-1)
			for c := cb.head[kk]; c >= 0; c = cb.next[c] {
				mx := colMax(c)
				compact(c)
				cr, cv := cRow[c], cVal[c]
				if mx <= absTol {
					continue
				}
				if len(cr) == 1 {
					bestR, bestC, bestP, bestCost = cr[0], c, 0, 0
					break search
				}
				n1 := float64(len(cr) - 1)
				for p, i := range cr {
					a := math.Abs(cv[p])
					if a < u*mx {
						continue
					}
					cost := float64(len(rCol[i])-1) * n1
					if cost < bestCost || (cost == bestCost && a > bestAbs) {
						bestR, bestC, bestP, bestCost, bestAbs = i, c, p, cost, a
					}
				}
				searched++
				if bestR >= 0 && (searched >= MarkowitzSearch || bestCost <= float64((k-1)*(k-1))) {
					break search
				}
			}
			// Rows of count k.
			for i := rb.head[kk]; k <= rowSearchMax && i >= 0; i = rb.next[i] {
				ki := float64(k - 1)
				for _, c := range rCol[i] {
					cost := ki * float64(colCount[c]-1)
					if cost > bestCost {
						continue
					}
					mx := colMax(c)
					compact(c)
					cr, cv := cRow[c], cVal[c]
					pi := -1
					for p, ii := range cr {
						if ii == i {
							pi = p
							break
						}
					}
					if pi < 0 {
						continue
					}
					a := math.Abs(cv[pi])
					if a <= absTol || a < u*mx {
						continue
					}
					if cost < bestCost || (cost == bestCost && a > bestAbs) {
						bestR, bestC, bestP, bestCost, bestAbs = i, c, pi, cost, a
					}
				}
				searched++
				if bestR >= 0 && (searched >= MarkowitzSearch || bestCost <= float64((k-1)*k)) {
					break search
				}
			}
			if bestR >= 0 && float64(k*k) > bestCost {
				break
			}
		}
		if bestR < 0 {
			// Remaining submatrix is numerically singular.
			sing = &Singularity{}
			for c := 0; c < m; c++ {
				if colActive[c] {
					sing.Positions = append(sing.Positions, c)
				}
			}
			for i := 0; i < m; i++ {
				if rowActive[i] {
					sing.Rows = append(sing.Rows, i)
				}
			}
			break
		}

		// ---- Elimination with pivot (r, c).
		r, c := bestR, bestC
		piv := cVal[c][bestP]
		t := nstep
		nstep++
		f.prow[t] = r
		f.pcol[t] = c
		f.piv[t] = piv
		rowActive[r] = false
		colActive[c] = false
		rb.remove(r)
		cb.remove(c)
		actNNZ -= colCount[c]
		// L column: multipliers of the other rows of column c; drop c
		// from their patterns.
		compact(c)
		lBeg := len(f.lIdx)
		for p, i := range cRow[c] {
			if i == r {
				continue
			}
			f.lIdx = append(f.lIdx, i)
			f.lVal = append(f.lVal, cVal[c][p]/piv)
			removeFromRow(i, c)
		}
		lRows, lMul := f.lIdx[lBeg:], f.lVal[lBeg:]
		for _, i := range lRows {
			touched[i] = true
		}
		// U row; row r stays in its columns until they are compacted.
		if !touched[r] {
			for q := oStart[r]; q < oStart[r+1]; q++ {
				cc := oCol[q]
				if cc != c && colActive[cc] {
					uRowCol = append(uRowCol, cc)
					uRowVal = append(uRowVal, oVal[q])
				}
			}
		} else {
			for _, cc := range rCol[r] {
				if cc == c {
					continue
				}
				cr, cv := cRow[cc], cVal[cc]
				for p, i := range cr {
					if i == r {
						uRowCol = append(uRowCol, cc)
						uRowVal = append(uRowVal, cv[p])
						break
					}
				}
			}
		}
		uRowStart = append(uRowStart, len(uRowCol))
		rowStartU := uRowStart[len(uRowStart)-2]
		for q := rowStartU; q < len(uRowCol); q++ {
			cc := uRowCol[q]
			cStale[cc] = true
			cValid[cc] = false
			colCount[cc]--
			actNNZ--
		}
		// Update each column of the pivot row.
		if len(lRows) == 0 {
			for q := rowStartU; q < len(uRowCol); q++ {
				cc := uRowCol[q]
				cb.move(cc, colCount[cc])
			}
		}
		for q := rowStartU; len(lRows) > 0 && q < len(uRowCol); q++ {
			cc := uRowCol[q]
			uv := uRowVal[q]
			compact(cc)
			cr, cv := cRow[cc], cVal[cc]
			for p, i := range cr {
				mark[i] = p
			}
			for e, i := range lRows {
				dv := -lMul[e] * uv
				if p := mark[i]; p >= 0 {
					cv[p] += dv
				} else {
					cr = append(cr, i)
					cv = append(cv, dv)
					rCol[i] = append(rCol[i], cc)
				}
			}
			// unscatter and drop tiny entries
			n := 0
			for p, i := range cr {
				mark[i] = -1
				if math.Abs(cv[p]) <= f.DropTol {
					removeFromRow(i, cc)
					continue
				}
				cr[n], cv[n] = i, cv[p]
				n++
			}
			cRow[cc], cVal[cc] = cr[:n], cv[:n]
			actNNZ += n - colCount[cc]
			colCount[cc] = n
			cb.move(cc, n)
		}
		for _, i := range lRows {
			rb.move(i, len(rCol[i]))
		}
		f.lStart = append(f.lStart, len(f.lIdx))
	}
	if sing != nil {
		return sing
	}
	// Convert U rows (by step, entries by column) to U columns by step.
	w.stepOfCol = resizeInts(w.stepOfCol, m)
	stepOfCol := w.stepOfCol
	for t := 0; t < m; t++ {
		f.step[f.prow[t]] = t
		stepOfCol[f.pcol[t]] = t
	}
	w.cnt = resizeInts(w.cnt, m+1)
	cnt := w.cnt
	for i := range cnt {
		cnt[i] = 0
	}
	for _, cc := range uRowCol {
		cnt[stepOfCol[cc]+1]++
	}
	for k := 0; k < m; k++ {
		cnt[k+1] += cnt[k]
	}
	f.uStart = append(f.uStart[:0], cnt...)
	nu := len(uRowCol)
	f.uIdx = resizeInts(f.uIdx, nu)
	f.uVal = resizeFloats(f.uVal, nu)
	next := append(w.mark[:0], cnt[:m]...)
	for t := 0; t < m; t++ {
		for q := uRowStart[t]; q < uRowStart[t+1]; q++ {
			k := stepOfCol[uRowCol[q]]
			f.uIdx[next[k]] = t
			f.uVal[next[k]] = uRowVal[q]
			next[k]++
		}
	}
	f.buildTransposes()
	f.initFT()
	return nil
}
