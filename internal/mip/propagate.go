package mip

import "math"

// propagate performs activity-based bound tightening on the original rows.
// If changed is nil, all rows are processed initially. Returns false if
// infeasibility is detected.
func (s *solver) propagate(lo, up []float64, changed []int) bool {
	m := s.m
	if m == 0 || s.pricing() {
		// With column generation, activity bounds are not final.
		return true
	}
	if s.inQueue == nil || len(s.inQueue) != m {
		s.inQueue = make([]bool, m)
	}
	queue := s.queue[:0]
	if changed == nil {
		for i := 0; i < m; i++ {
			queue = append(queue, i)
			s.inQueue[i] = true
		}
	} else {
		for _, j := range changed {
			for k := s.p.ColStart[j]; k < s.p.ColStart[j+1]; k++ {
				i := s.p.RowIdx[k]
				if !s.inQueue[i] {
					s.inQueue[i] = true
					queue = append(queue, i)
				}
			}
		}
	}
	work := 0
	maxWork := 20*len(s.rowCol) + 1000
	feasible := true
	head := 0
	for objRound := 0; ; objRound++ {
		for ; head < len(queue); head++ {
			i := queue[head]
			s.inQueue[i] = false
			if !feasible || work > maxWork {
				continue
			}
			rl, ru := s.p.RowLo[i], s.p.RowUp[i]
			var minA, maxA float64
			minInf, maxInf := 0, 0
			st, en := s.rowStart[i], s.rowStart[i+1]
			work += en - st
			for k := st; k < en; k++ {
				j, a := s.rowCol[k], s.rowVal[k]
				l, u := lo[j], up[j]
				if a > 0 {
					if math.IsInf(l, -1) {
						minInf++
					} else {
						minA += a * l
					}
					if math.IsInf(u, 1) {
						maxInf++
					} else {
						maxA += a * u
					}
				} else {
					if math.IsInf(u, 1) {
						minInf++
					} else {
						minA += a * u
					}
					if math.IsInf(l, -1) {
						maxInf++
					} else {
						maxA += a * l
					}
				}
			}
			tolR := 1e-6 * math.Max(1, math.Max(math.Abs(minA), math.Abs(maxA)))
			if minInf == 0 && !math.IsInf(ru, 1) && minA > ru+tolR {
				feasible = false
				continue
			}
			if maxInf == 0 && !math.IsInf(rl, -1) && maxA < rl-tolR {
				feasible = false
				continue
			}
			canUp := !math.IsInf(ru, 1) && minInf <= 1
			canLo := !math.IsInf(rl, -1) && maxInf <= 1
			if !canUp && !canLo {
				continue
			}
			for k := st; k < en; k++ {
				j, a := s.rowCol[k], s.rowVal[k]
				l, u := lo[j], up[j]
				nl, nu := l, u
				if canUp {
					// a x_j <= ru - (minA - contribution_j)
					var res float64
					ok := true
					if a > 0 {
						if math.IsInf(l, -1) {
							ok = minInf == 1
							res = minA
						} else {
							ok = minInf == 0
							res = minA - a*l
						}
						if ok {
							nu = math.Min(nu, (ru-res)/a)
						}
					} else {
						if math.IsInf(u, 1) {
							ok = minInf == 1
							res = minA
						} else {
							ok = minInf == 0
							res = minA - a*u
						}
						if ok {
							nl = math.Max(nl, (ru-res)/a)
						}
					}
				}
				if canLo {
					var res float64
					ok := true
					if a > 0 {
						if math.IsInf(u, 1) {
							ok = maxInf == 1
							res = maxA
						} else {
							ok = maxInf == 0
							res = maxA - a*u
						}
						if ok {
							nl = math.Max(nl, (rl-res)/a)
						}
					} else {
						if math.IsInf(l, -1) {
							ok = maxInf == 1
							res = maxA
						} else {
							ok = maxInf == 0
							res = maxA - a*l
						}
						if ok {
							nu = math.Min(nu, (rl-res)/a)
						}
					}
				}
				if nl <= l && nu >= u {
					continue
				}
				if s.isInt[j] {
					nl = math.Ceil(nl - 1e-6)
					nu = math.Floor(nu + 1e-6)
				} else {
					// continuous: only accept significant tightenings
					width := u - l
					minStep := 1e-3 * math.Max(1, math.Abs(width))
					if math.IsInf(width, 1) {
						minStep = 1e-3 * math.Max(1, math.Max(math.Abs(nl), math.Abs(nu)))
					}
					if nl-l < minStep || math.Abs(nl) > 1e9 {
						nl = l
					}
					if u-nu < minStep || math.Abs(nu) > 1e9 {
						nu = u
					}
				}
				chg := false
				if nl > l+1e-9 {
					lo[j] = nl
					chg = true
				}
				if nu < u-1e-9 {
					up[j] = nu
					chg = true
				}
				if !chg {
					continue
				}
				if lo[j] > up[j]+1e-6 {
					feasible = false
					break
				}
				if lo[j] > up[j] {
					if s.isInt[j] {
						feasible = false
						break
					}
					lo[j] = up[j]
				}
				// Update activities incrementally would be faster; recompute by
				// re-queueing this row and the rows of j.
				for kk := s.p.ColStart[j]; kk < s.p.ColStart[j+1]; kk++ {
					r := s.p.RowIdx[kk]
					if !s.inQueue[r] {
						s.inQueue[r] = true
						queue = append(queue, r)
					}
				}
			}
		}
		if !feasible || objRound >= 3 {
			break
		}
		ch := s.propagateObjective(lo, up, &feasible)
		if !feasible || len(ch) == 0 {
			break
		}
		for _, j := range ch {
			for kk := s.p.ColStart[j]; kk < s.p.ColStart[j+1]; kk++ {
				r := s.p.RowIdx[kk]
				if !s.inQueue[r] {
					s.inQueue[r] = true
					queue = append(queue, r)
				}
			}
		}
	}
	for _, i := range queue {
		s.inQueue[i] = false
	}
	s.queue = queue[:0]
	return feasible
}

// propagateObjective tightens bounds with the objective cutoff
// c^T x <= cutoff (objective propagation). It returns the variables whose
// bounds changed and sets *feasible to false when the cutoff cannot be met.
func (s *solver) propagateObjective(lo, up []float64, feasible *bool) []int {
	cut := s.cutoffValue()
	if math.IsInf(cut, 1) || s.pricing() {
		return nil
	}
	if s.objIdx == nil {
		for j, c := range s.p.Obj {
			if c != 0 {
				s.objIdx = append(s.objIdx, j)
			}
		}
	}
	minAct := 0.0
	inf := 0
	infJ := -1
	for _, j := range s.objIdx {
		c := s.p.Obj[j]
		if c > 0 {
			if math.IsInf(lo[j], -1) {
				inf++
				infJ = j
				continue
			}
			minAct += c * lo[j]
		} else {
			if math.IsInf(up[j], 1) {
				inf++
				infJ = j
				continue
			}
			minAct += c * up[j]
		}
	}
	if inf > 1 {
		return nil
	}
	tol := 1e-9 * math.Max(1, math.Abs(cut))
	if inf == 0 && minAct > cut+tol {
		*feasible = false
		return nil
	}
	var changed []int
	for _, j := range s.objIdx {
		if inf == 1 && j != infJ {
			continue
		}
		c := s.p.Obj[j]
		var res float64 // slack available to x_j
		if c > 0 {
			if inf == 0 {
				res = cut - (minAct - c*lo[j])
			} else {
				res = cut - minAct
			}
			nu := res / c
			if s.isInt[j] {
				nu = math.Floor(nu + 1e-6)
			}
			if nu < up[j]-1e-7*math.Max(1, math.Abs(nu)) && (s.isInt[j] || up[j]-nu > 1e-3*math.Max(1, up[j]-lo[j])) {
				up[j] = nu
				changed = append(changed, j)
			}
		} else {
			if inf == 0 {
				res = cut - (minAct - c*up[j])
			} else {
				res = cut - minAct
			}
			nl := res / c
			if s.isInt[j] {
				nl = math.Ceil(nl - 1e-6)
			}
			if nl > lo[j]+1e-7*math.Max(1, math.Abs(nl)) && (s.isInt[j] || nl-lo[j] > 1e-3*math.Max(1, up[j]-lo[j])) {
				lo[j] = nl
				changed = append(changed, j)
			}
		}
		if lo[j] > up[j]+1e-6 {
			*feasible = false
			return nil
		}
		if lo[j] > up[j] {
			lo[j] = up[j]
		}
	}
	return changed
}
