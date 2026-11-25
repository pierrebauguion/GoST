package gost

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"gost/internal/mip"
)

// StartStatus tells what Solve did with the MIP start (see Model.SetStart).
type StartStatus int

const (
	StartNone     StartStatus = iota // no start was given
	StartAccepted                    // complete and feasible: first incumbent
	StartRepaired                    // partial or infeasible: completed by a sub-MIP
	StartRejected                    // could not be used (see Result.StartInfo)
)

func (s StartStatus) String() string { return mip.StartStatus(s).String() }

// SetStart gives a starting value to variable v for the next Solve (a "MIP
// start"). A start may be complete (every variable) or partial:
//
//   - a complete start that satisfies bounds, integrality and constraints
//     becomes the first incumbent: its objective is used for pruning from
//     the root on, and improvement heuristics (RINS) start from it;
//   - a partial or infeasible start is repaired: the integer variables
//     that were given are fixed to their rounded values and the rest of
//     the problem is solved by a node-limited sub-MIP; the solution found,
//     if any, becomes the first incumbent.
//
// The start never changes the result's optimality: it only helps the
// search. Lazy constraints, when present, are checked on it too. Result
// reports what happened in StartStatus and StartInfo.
func (m *Model) SetStart(v Var, val float64) {
	if len(m.start) < len(m.vars) {
		for len(m.start) < len(m.vars) {
			m.start = append(m.start, math.NaN())
		}
	}
	m.start[v.idx] = val
}

// SetStartValues sets a complete start, indexed like the variables
// (NaN entries mean "not given").
func (m *Model) SetStartValues(x []float64) {
	m.start = m.start[:0]
	for j := range m.vars {
		v := math.NaN()
		if j < len(x) {
			v = x[j]
		}
		m.start = append(m.start, v)
	}
}

// ClearStart removes the MIP start.
func (m *Model) ClearStart() { m.start = nil }

// startVector returns the start for the current variables (nil if none).
func (m *Model) startVector() []float64 {
	if m.start == nil {
		return nil
	}
	x := make([]float64, len(m.vars))
	for j := range x {
		x[j] = math.NaN()
		if j < len(m.start) {
			x[j] = m.start[j]
		}
	}
	return x
}

// ReadStart reads a MIP start from text: either one "name value" pair per
// line (names as given to AddVar or read from MPS), or one value per line
// in variable order. Empty lines and lines starting with '#' are ignored.
// It returns the number of values set.
func (m *Model) ReadStart(text string) (int, error) {
	byName := map[string]int{}
	for j, v := range m.vars {
		if v.name != "" {
			byName[v.name] = j
		}
	}
	n, pos := 0, 0
	for ln, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		f := strings.Fields(line)
		j := pos
		valStr := f[0]
		if len(f) >= 2 {
			k, ok := byName[f[0]]
			if !ok {
				// unnamed variables are written as x<index>
				if _, err := fmt.Sscanf(f[0], "x%d", &k); err != nil || k < 0 || k >= len(m.vars) {
					return n, fmt.Errorf("start line %d: unknown variable %q", ln+1, f[0])
				}
			}
			j, valStr = k, f[1]
		}
		val, err := strconv.ParseFloat(valStr, 64)
		if err != nil {
			return n, fmt.Errorf("start line %d: %v", ln+1, err)
		}
		if j >= len(m.vars) {
			return n, fmt.Errorf("start line %d: more values than variables", ln+1)
		}
		m.SetStart(Var{j}, val)
		pos++
		n++
	}
	return n, nil
}

// WriteSolution returns the solution of r as "name value" lines (one per
// variable; unnamed variables are written x<index>), a format accepted by
// ReadStart.
func (m *Model) WriteSolution(r *Result) string {
	var b strings.Builder
	for j, v := range m.vars {
		if x := r.Value(Var{j}); !math.IsNaN(x) {
			name := v.name
			if name == "" {
				name = fmt.Sprintf("x%d", j)
			}
			fmt.Fprintf(&b, "%s %.17g\n", name, x)
		}
	}
	return b.String()
}
