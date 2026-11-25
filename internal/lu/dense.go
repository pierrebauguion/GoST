package lu

import "math"

// denseKernel finishes the Markowitz factorization on the remaining active
// submatrix with a dense LU (threshold partial pivoting by rows, column
// order by increasing count). It appends the steps nstep.. to the factor and
// U row lists. ok is false if the submatrix is numerically singular, in
// which case rowActive/colActive mark the rows and columns left unpivoted.
func (f *Factor) denseKernel(nstep, m int, compact func(int), uRowCol []int, uRowVal []float64, uRowStart []int) (int, []int, []float64, []int, bool) {
	w := &f.mk
	rowActive, colActive := w.rowActive, w.colActive
	// Active rows and columns.
	rows := w.dRows[:0]
	cols := w.dCols[:0]
	for i := 0; i < m; i++ {
		if rowActive[i] {
			rows = append(rows, i)
		}
	}
	for c := 0; c < m; c++ {
		if colActive[c] {
			cols = append(cols, c)
		}
	}
	w.dRows, w.dCols = rows, cols
	nr, nc := len(rows), len(cols)
	// local row index
	lr := w.mark // -1 everywhere on entry
	for a, i := range rows {
		lr[i] = a
	}
	// Dense row-major storage A[a*nc+b].
	if cap(w.dA) < nr*nc {
		w.dA = make([]float64, nr*nc)
	}
	A := w.dA[:nr*nc]
	for k := range A {
		A[k] = 0
	}
	for b, c := range cols {
		compact(c)
		cv := w.cVal[c]
		for p, i := range w.cRows[c] {
			A[lr[i]*nc+b] = cv[p]
		}
	}
	for _, i := range rows {
		lr[i] = -1
	}
	// Row permutation (rp[a] = local row at position a), column permutation.
	rp := resizeInts(w.dRp, nr)
	cp := resizeInts(w.dCp, nc)
	w.dRp, w.dCp = rp, cp
	for a := range rp {
		rp[a] = a
	}
	for b := range cp {
		cp[b] = b
	}
	// Nonzero counts of rows and columns (local indices).
	rnz := resizeInts(w.dRnz, nr)
	cnz := resizeInts(w.dCnz, nc)
	w.dRnz, w.dCnz = rnz, cnz
	for a := range rnz {
		rnz[a] = 0
	}
	for b := range cnz {
		cnz[b] = 0
	}
	for a := 0; a < nr; a++ {
		row := A[a*nc : a*nc+nc]
		for b, v := range row {
			if v != 0 {
				rnz[a]++
				cnz[b]++
			}
		}
	}
	u := f.Threshold
	ok := true
	n := min(nr, nc)
	for s := 0; s < n; s++ {
		// Markowitz choice: columns by increasing count, rows by count
		// among entries passing the threshold.
		bestS, bestA := -1, -1
		bestCost := math.Inf(1)
		bestAbs := 0.0
		tried := 0
		for tried < nc-s && tried < 64 {
			// column with the smallest count not yet tried this step
			bb := -1
			for x := s; x < nc; x++ {
				b := cp[x]
				if cnz[b] < 0 {
					continue
				}
				if bb < 0 || cnz[b] < cnz[cp[bb]] {
					bb = x
				}
			}
			if bb < 0 {
				break
			}
			b := cp[bb]
			tried++
			mx := 0.0
			for a := s; a < nr; a++ {
				if v := math.Abs(A[rp[a]*nc+b]); v > mx {
					mx = v
				}
			}
			cc := cnz[b]
			cnz[b] = -1 - cc // mark tried
			if mx <= f.AbsTol {
				continue
			}
			for a := s; a < nr; a++ {
				v := math.Abs(A[rp[a]*nc+b])
				if v < u*mx || v <= f.AbsTol {
					continue
				}
				cost := float64(rnz[rp[a]]-1) * float64(cc-1)
				if cost < bestCost || (cost == bestCost && v > bestAbs) {
					bestS, bestA, bestCost, bestAbs = bb, a, cost, v
				}
			}
			if bestS >= 0 && (tried >= MarkowitzSearch || bestCost <= float64((cc-1)*(cc-1))) {
				break
			}
		}
		for x := s; x < nc; x++ {
			if b := cp[x]; cnz[b] < 0 {
				cnz[b] = -1 - cnz[b]
			}
		}
		if bestS < 0 {
			ok = false
			break
		}
		cp[s], cp[bestS] = cp[bestS], cp[s]
		rp[s], rp[bestA] = rp[bestA], rp[s]
		b := cp[s]
		pr := A[rp[s]*nc : rp[s]*nc+nc]
		piv := pr[b]
		r, c := rows[rp[s]], cols[b]
		t := nstep
		nstep++
		f.prow[t], f.pcol[t], f.piv[t] = r, c, piv
		rowActive[r], colActive[c] = false, false
		// U row.
		for bb := s + 1; bb < nc; bb++ {
			bc := cp[bb]
			if v := pr[bc]; v != 0 {
				cnz[bc]--
				uRowCol = append(uRowCol, cols[bc])
				uRowVal = append(uRowVal, v)
			}
		}
		uRowStart = append(uRowStart, len(uRowCol))
		// Pattern of the pivot row (columns after s).
		pat := w.dPat[:0]
		for bb := s + 1; bb < nc; bb++ {
			if bc := cp[bb]; pr[bc] != 0 {
				pat = append(pat, bc)
			}
		}
		w.dPat = pat
		// L column and elimination.
		for a := s + 1; a < nr; a++ {
			la := rp[a]
			ra := A[la*nc : la*nc+nc]
			if ra[b] == 0 {
				continue
			}
			l := ra[b] / piv
			ra[b] = 0
			rnz[la]--
			cnz[b]--
			f.lIdx = append(f.lIdx, rows[la])
			f.lVal = append(f.lVal, l)
			for _, bc := range pat {
				old := ra[bc]
				nv := old - l*pr[bc]
				if math.Abs(nv) <= f.DropTol {
					nv = 0
				}
				ra[bc] = nv
				if old == 0 && nv != 0 {
					rnz[la]++
					cnz[bc]++
				} else if old != 0 && nv == 0 {
					rnz[la]--
					cnz[bc]--
				}
			}
		}
		f.lStart = append(f.lStart, len(f.lIdx))
	}
	return nstep, uRowCol, uRowVal, uRowStart, ok
}
