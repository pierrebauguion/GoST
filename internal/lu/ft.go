package lu

import "math"

// Forrest–Tomlin update.
//
// After a fresh factorization, L^{-1} B = U with U upper triangular in step
// order. Replacing the column at basis position p (column step kp) by a_q
// puts the "spike" s = R L^{-1} a_q in column kp; moving step kp to the end
// of the triangular order leaves row kp with entries left of the diagonal,
// which are eliminated by a row eta R (b_kp -= sum mu_k b_k). The FTRAN is
// then L, the row etas R in creation order, and a triangular solve with the
// updated U.
//
// U storage is "static + appended": the factor-time U (column-wise uIdx and
// row-wise utIdx) is never modified; spike columns are appended to an arena.
// Validity of an entry (row t, column k, id) where id = -1 for static
// entries and the spike id otherwise: colCur[k] == id && id > rowClear[t].

// UseFT selects Forrest–Tomlin updates (default) instead of product-form etas.
var UseFT = envFloat("GOST_FT", 1) != 0

type ftEntry struct {
	k  int32 // column step (row lists) or row step (column lists)
	id int32 // spike id (-1 static)
	v  float64
}

type ftState struct {
	on       bool
	order    []int // steps in triangular order
	pos      []int // position of each step in order
	diag     []float64
	invDiag  []float64
	colCur   []int32 // current spike id of each column step (-1 static)
	rowClear []int32 // entries with id <= rowClear[t] in row t are void
	// spike columns: arena of (row step, value) per spike id
	spStart []int
	spRow   []int32
	spVal   []float64
	// dynamic row lists: entries of spike columns, by row step
	dynRow [][]ftEntry
	nDyn   int
	// row etas (row indices i, not steps)
	rTarget []int
	rStart  []int
	rIdx    []int
	rVal    []float64

	spike     []float64 // saved L/R-transformed column (row space)
	haveSpike bool
	r         []float64 // dense work over steps
	nUpd      int
}

// initFT prepares the update structures after a fresh factorization.
func (f *Factor) initFT() {
	ft := &f.ft
	m := f.m
	ft.on = UseFT
	if !ft.on {
		return
	}
	ft.order = resizeInts(ft.order, m)
	ft.pos = resizeInts(ft.pos, m)
	ft.diag = resizeFloats(ft.diag, m)
	ft.invDiag = resizeFloats(ft.invDiag, m)
	if cap(ft.colCur) < m {
		ft.colCur = make([]int32, m)
		ft.rowClear = make([]int32, m)
	}
	ft.colCur, ft.rowClear = ft.colCur[:m], ft.rowClear[:m]
	for k := 0; k < m; k++ {
		ft.order[k] = k
		ft.pos[k] = k
		ft.diag[k] = f.piv[k]
		ft.invDiag[k] = 1 / f.piv[k]
		ft.colCur[k] = -1
		ft.rowClear[k] = -2
	}
	ft.spStart = append(ft.spStart[:0], 0)
	ft.spRow = ft.spRow[:0]
	ft.spVal = ft.spVal[:0]
	if len(ft.dynRow) < m {
		ft.dynRow = append(ft.dynRow, make([][]ftEntry, m-len(ft.dynRow))...)
	}
	ft.dynRow = ft.dynRow[:m]
	for t := range ft.dynRow {
		ft.dynRow[t] = ft.dynRow[t][:0]
	}
	ft.nDyn = 0
	ft.rTarget = ft.rTarget[:0]
	ft.rStart = append(ft.rStart[:0], 0)
	ft.rIdx = ft.rIdx[:0]
	ft.rVal = ft.rVal[:0]
	ft.spike = resizeFloats(ft.spike, m)
	ft.r = resizeFloats(ft.r, m)
	for i := range ft.r {
		ft.r[i] = 0
	}
	ft.haveSpike = false
	ft.nUpd = 0
}

// ftranFT is FTRAN with Forrest–Tomlin updates. If save, the vector after L
// and R is stored as the spike for the next Update.
func (f *Factor) ftranFT(b []float64, save bool) {
	m := f.m
	ft := &f.ft
	for t := 0; t < m; t++ {
		v := b[f.prow[t]]
		if v == 0 {
			continue
		}
		for p := f.lStart[t]; p < f.lStart[t+1]; p++ {
			b[f.lIdx[p]] -= f.lVal[p] * v
		}
	}
	for e := range ft.rTarget {
		s := 0.0
		for p := ft.rStart[e]; p < ft.rStart[e+1]; p++ {
			s += ft.rVal[p] * b[ft.rIdx[p]]
		}
		b[ft.rTarget[e]] -= s
	}
	if save {
		copy(ft.spike, b[:m])
		ft.haveSpike = true
	}
	// U back substitution in reverse triangular order, in step space.
	z := f.work2
	prow := f.prow
	for t := 0; t < m; t++ {
		z[t] = b[prow[t]]
	}
	order, invDiag, colCur := ft.order, ft.invDiag, ft.colCur
	uStart, uIdx, uVal := f.uStart, f.uIdx, f.uVal
	for p := m - 1; p >= 0; p-- {
		k := order[p]
		v := z[k]
		if v == 0 {
			continue
		}
		v *= invDiag[k]
		z[k] = v
		if id := colCur[k]; id < 0 {
			// static column: entries of cleared rows were zeroed
			for q := uStart[k]; q < uStart[k+1]; q++ {
				z[uIdx[q]] -= uVal[q] * v
			}
		} else {
			for q := ft.spStart[id]; q < ft.spStart[id+1]; q++ {
				t := ft.spRow[q]
				if id > ft.rowClear[t] {
					z[t] -= ft.spVal[q] * v
				}
			}
		}
	}
	pcol := f.pcol
	if out := f.nzOut; out != nil {
		nz := *out
		for k := 0; k < m; k++ {
			v := z[k]
			b[pcol[k]] = v
			if v != 0 {
				nz = append(nz, pcol[k])
			}
		}
		*out = nz
		return
	}
	for k := 0; k < m; k++ {
		b[pcol[k]] = z[k]
	}
}

// btranFT is BTRAN with Forrest–Tomlin updates.
func (f *Factor) btranFT(d []float64) {
	m := f.m
	ft := &f.ft
	z := f.work2
	for k := 0; k < m; k++ {
		z[k] = d[f.pcol[k]]
	}
	for p := 0; p < m; p++ {
		k := ft.order[p]
		v := z[k]
		if v == 0 {
			continue
		}
		v *= ft.invDiag[k]
		z[k] = v
		if ft.rowClear[k] < -1 {
			// static row: entries of replaced columns were zeroed
			utIdx, utVal := f.utIdx, f.utVal
			for q := f.utStart[k]; q < f.utStart[k+1]; q++ {
				z[utIdx[q]] -= utVal[q] * v
			}
		}
		rc := ft.rowClear[k]
		for _, e := range ft.dynRow[k] {
			if e.id > rc && ft.colCur[e.k] == e.id {
				z[e.k] -= e.v * v
			}
		}
	}
	for k := 0; k < m; k++ {
		d[f.prow[k]] = z[k]
	}
	for e := len(ft.rTarget) - 1; e >= 0; e-- {
		v := d[ft.rTarget[e]]
		if v == 0 {
			continue
		}
		for p := ft.rStart[e]; p < ft.rStart[e+1]; p++ {
			d[ft.rIdx[p]] -= ft.rVal[p] * v
		}
	}
	ltStart, ltRow, ltVal, prow := f.ltStart, f.ltRow, f.ltVal, f.prow
	for s := m - 1; s >= 0; s-- {
		r := prow[s]
		v := d[r]
		if v == 0 {
			continue
		}
		for p := ltStart[r]; p < ltStart[r+1]; p++ {
			d[ltRow[p]] -= ltVal[p] * v
		}
	}
}

// updateFT performs the Forrest–Tomlin update for basis position pos.
// alphaP is the pivot element (FTRAN'd entering column at pos), used for a
// stability check. Returns false if the update is unstable.
func (f *Factor) updateFT(pos int, alphaP float64) bool {
	ft := &f.ft
	if !ft.haveSpike {
		return false
	}
	ft.haveSpike = false
	m := f.m
	// column step of basis position pos
	kp := -1
	for k := 0; k < m; k++ {
		if f.pcol[k] == pos {
			kp = k
			break
		}
	}
	if kp < 0 {
		return false
	}
	oldDiag := ft.diag[kp]
	id := int32(len(ft.spStart) - 1)
	s := ft.spike

	// Row kp (before the update): entries U[kp, k] for k after kp in the
	// current order; these become the row to eliminate.
	r := ft.r
	var touched []int
	add := func(k int, v float64) {
		if r[k] == 0 {
			touched = append(touched, k)
		}
		r[k] += v
		if r[k] == 0 {
			r[k] = 1e-300 // keep it marked
		}
	}
	if ft.rowClear[kp] < -1 {
		for q := f.utStart[kp]; q < f.utStart[kp+1]; q++ {
			k := f.utIdx[q]
			if ft.colCur[k] == -1 && k != kp {
				add(k, f.utVal[q])
			}
		}
	}
	rc := ft.rowClear[kp]
	for _, e := range ft.dynRow[kp] {
		if e.id > rc && ft.colCur[e.k] == e.id && int(e.k) != kp {
			add(int(e.k), e.v)
		}
	}

	// Void the static entries of column kp (in the row-wise copy) and of
	// row kp (in the column-wise copy).
	if ft.colCur[kp] == -1 {
		for q := f.uStart[kp]; q < f.uStart[kp+1]; q++ {
			f.utVal[f.uToUt[q]] = 0
		}
	}
	if ft.rowClear[kp] < -1 {
		for q := f.utStart[kp]; q < f.utStart[kp+1]; q++ {
			f.uVal[f.utToU[q]] = 0
			f.utVal[q] = 0
		}
	}
	// Install the spike as column kp (new id).
	ft.colCur[kp] = id
	newDiag := s[f.prow[kp]]
	for t := 0; t < m; t++ {
		if t == kp {
			continue
		}
		v := s[f.prow[t]]
		if v == 0 || math.Abs(v) <= f.DropTol {
			continue
		}
		ft.spRow = append(ft.spRow, int32(t))
		ft.spVal = append(ft.spVal, v)
		ft.dynRow[t] = append(ft.dynRow[t], ftEntry{int32(kp), id, v})
		ft.nDyn++
	}
	ft.spStart = append(ft.spStart, len(ft.spRow))
	// Row kp is cleared: all its entries with id <= current are void
	// (the new spike's entry in row kp is the diagonal).
	ft.rowClear[kp] = id

	// Move kp to the end of the order.
	pk := ft.pos[kp]
	copy(ft.order[pk:], ft.order[pk+1:])
	ft.order[m-1] = kp
	for p := pk; p < m; p++ {
		ft.pos[ft.order[p]] = p
	}

	// Eliminate row kp using rows in increasing order position.
	etaStart := len(ft.rIdx)
	if len(touched) > 0 {
		minPos := m
		for _, k := range touched {
			if ft.pos[k] < minPos {
				minPos = ft.pos[k]
			}
		}
		for p := minPos; p < m-1; p++ {
			k := ft.order[p]
			v := r[k]
			if v == 0 {
				continue
			}
			r[k] = 0
			if math.Abs(v) <= 1e-280 {
				continue
			}
			mu := v / ft.diag[k]
			ft.rIdx = append(ft.rIdx, f.prow[k])
			ft.rVal = append(ft.rVal, mu)
			// row k: static part
			if ft.rowClear[k] < -1 {
				for q := f.utStart[k]; q < f.utStart[k+1]; q++ {
					k2 := f.utIdx[q]
					if ft.colCur[k2] != -1 {
						continue
					}
					if k2 == kp {
						continue
					}
					if r[k2] == 0 {
						touched = append(touched, k2)
					}
					r[k2] -= mu * f.utVal[q]
					if r[k2] == 0 {
						r[k2] = 1e-300
					}
				}
			}
			rck := ft.rowClear[k]
			for _, e := range ft.dynRow[k] {
				if e.id <= rck || ft.colCur[e.k] != e.id {
					continue
				}
				if int(e.k) == kp {
					newDiag -= mu * e.v
					continue
				}
				k2 := int(e.k)
				if r[k2] == 0 {
					touched = append(touched, k2)
				}
				r[k2] -= mu * e.v
				if r[k2] == 0 {
					r[k2] = 1e-300
				}
			}
		}
	}
	for _, k := range touched {
		r[k] = 0
	}
	if len(ft.rIdx) > etaStart {
		ft.rTarget = append(ft.rTarget, f.prow[kp])
		ft.rStart = append(ft.rStart, len(ft.rIdx))
	}
	ft.diag[kp] = newDiag
	ft.invDiag[kp] = 1 / newDiag
	ft.nUpd++
	// Stability: |newDiag| should equal |alphaP * oldDiag|.
	want := alphaP * oldDiag
	if math.Abs(newDiag) < 1e-12 || math.Abs(math.Abs(newDiag)-math.Abs(want)) > 1e-6*(1+math.Abs(want)) {
		return false
	}
	return true
}
