// Package mps reads and writes models in (free or fixed) MPS format.
package mps

import (
	"bufio"
	"compress/gzip"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"

	"gost"
)

// ReadFile reads an MPS file (optionally gzip-compressed, by .gz suffix or
// magic bytes).
func ReadFile(path string) (*gost.Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	br := bufio.NewReaderSize(f, 1<<16)
	magic, _ := br.Peek(2)
	var r io.Reader = br
	if len(magic) == 2 && magic[0] == 0x1f && magic[1] == 0x8b {
		gz, err := gzip.NewReader(br)
		if err != nil {
			return nil, err
		}
		defer gz.Close()
		r = gz
	}
	return Read(r)
}

type rowInfo struct {
	name  string
	typ   byte // N, L, G, E
	rhs   float64
	rng   float64
	hasR  bool
	index int // constraint index (-1 for objective / free rows)
	pos   int // position in ROWS section
}

type colInfo struct {
	name   string
	lo, up float64
	obj    float64
	integ  bool
	idx    []int
	val    []float64
	loSet  bool
	upSet  bool
}

// Read parses an MPS model from r.
func Read(r io.Reader) (*gost.Model, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	var (
		name     string
		section  string
		rows     []*rowInfo
		rowByNm  = map[string]*rowInfo{}
		cols     []*colInfo
		colByNm  = map[string]*colInfo{}
		objRow   string
		inInt    bool
		maximize bool
		objConst float64
		lineNo   int
	)
	num := func(s string) (float64, error) {
		v, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return 0, fmt.Errorf("mps line %d: bad number %q", lineNo, s)
		}
		return v, nil
	}
	for sc.Scan() {
		lineNo++
		line := sc.Text()
		if len(line) == 0 || line[0] == '*' {
			continue
		}
		if line[0] != ' ' && line[0] != '\t' {
			f := strings.Fields(line)
			section = strings.ToUpper(f[0])
			switch section {
			case "NAME":
				if len(f) > 1 {
					name = f[1]
				}
			case "OBJSENSE":
				if len(f) > 1 && strings.HasPrefix(strings.ToUpper(f[1]), "MAX") {
					maximize = true
				}
			case "ENDATA":
				goto done
			}
			continue
		}
		f := strings.Fields(line)
		if len(f) == 0 {
			continue
		}
		switch section {
		case "OBJSENSE":
			if strings.HasPrefix(strings.ToUpper(f[0]), "MAX") {
				maximize = true
			}
		case "OBJSENSE MAX", "OBJSENSE MAXIMIZE":
			maximize = true
		case "ROWS":
			if len(f) < 2 {
				return nil, fmt.Errorf("mps line %d: bad ROWS entry", lineNo)
			}
			t := strings.ToUpper(f[0])[0]
			ri := &rowInfo{name: f[1], typ: t, index: -1, pos: len(rows)}
			if t == 'N' {
				if objRow == "" {
					objRow = f[1]
				}
			} else if t != 'L' && t != 'G' && t != 'E' {
				return nil, fmt.Errorf("mps line %d: unknown row type %q", lineNo, f[0])
			}
			rows = append(rows, ri)
			rowByNm[f[1]] = ri
		case "COLUMNS":
			if len(f) >= 3 && strings.Contains(strings.ToUpper(line), "'MARKER'") {
				if strings.Contains(strings.ToUpper(line), "'INTORG'") {
					inInt = true
				} else if strings.Contains(strings.ToUpper(line), "'INTEND'") {
					inInt = false
				}
				continue
			}
			if len(f) < 3 || len(f)%2 == 0 {
				return nil, fmt.Errorf("mps line %d: bad COLUMNS entry", lineNo)
			}
			c := colByNm[f[0]]
			if c == nil {
				c = &colInfo{name: f[0], lo: 0, up: math.Inf(1), integ: inInt}
				cols = append(cols, c)
				colByNm[f[0]] = c
			}
			for k := 1; k+1 < len(f); k += 2 {
				v, err := num(f[k+1])
				if err != nil {
					return nil, err
				}
				rn := f[k]
				if rn == objRow {
					c.obj += v
					continue
				}
				ri := rowByNm[rn]
				if ri == nil {
					return nil, fmt.Errorf("mps line %d: unknown row %q", lineNo, rn)
				}
				if ri.typ == 'N' {
					continue
				}
				c.idx = append(c.idx, ri.pos)
				c.val = append(c.val, v)
			}
		case "RHS", "RANGES":
			k := 0
			if len(f)%2 == 1 {
				k = 1
			}
			for ; k+1 < len(f); k += 2 {
				v, err := num(f[k+1])
				if err != nil {
					return nil, err
				}
				if section == "RHS" && f[k] == objRow {
					objConst = -v
					continue
				}
				ri := rowByNm[f[k]]
				if ri == nil {
					return nil, fmt.Errorf("mps line %d: unknown row %q", lineNo, f[k])
				}
				if section == "RHS" {
					ri.rhs = v
				} else {
					ri.rng, ri.hasR = v, true
				}
			}
		case "BOUNDS":
			bt := strings.ToUpper(f[0])
			var cn string
			var v float64
			needVal := bt == "UP" || bt == "LO" || bt == "FX" || bt == "LI" || bt == "UI"
			switch {
			case needVal && len(f) >= 4:
				cn = f[2]
				x, err := num(f[3])
				if err != nil {
					return nil, err
				}
				v = x
			case needVal && len(f) == 3:
				cn = f[1]
				x, err := num(f[2])
				if err != nil {
					return nil, err
				}
				v = x
			case !needVal && len(f) >= 3:
				if _, ok := colByNm[f[2]]; ok {
					cn = f[2]
				} else {
					cn = f[1]
				}
			case !needVal && len(f) == 2:
				cn = f[1]
			default:
				return nil, fmt.Errorf("mps line %d: bad BOUNDS entry", lineNo)
			}
			c := colByNm[cn]
			if c == nil {
				return nil, fmt.Errorf("mps line %d: unknown column %q", lineNo, cn)
			}
			switch bt {
			case "UP", "UI":
				c.up, c.upSet = v, true
				if v < 0 && !c.loSet && c.lo == 0 {
					c.lo = math.Inf(-1)
				}
				if bt == "UI" {
					c.integ = true
				}
			case "LO", "LI":
				c.lo, c.loSet = v, true
				if bt == "LI" {
					c.integ = true
				}
			case "FX":
				c.lo, c.up = v, v
				c.loSet, c.upSet = true, true
			case "FR":
				c.lo, c.up = math.Inf(-1), math.Inf(1)
			case "MI":
				c.lo = math.Inf(-1)
			case "PL":
				c.up = math.Inf(1)
			case "BV":
				c.lo, c.up, c.integ = 0, 1, true
			default:
				return nil, fmt.Errorf("mps line %d: unsupported bound type %q", lineNo, bt)
			}
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
done:
	m := gost.NewModel(name)
	for _, c := range cols {
		t := gost.Continuous
		if c.integ {
			t = gost.Integer
			if c.lo == 0 && c.up == 1 {
				t = gost.Binary
			}
		}
		m.AddVar(c.lo, c.up, c.obj, t, c.name)
	}
	// Constraints in ROWS order (excluding N rows).
	cidx := 0
	for _, ri := range rows {
		if ri.typ == 'N' {
			continue
		}
		ri.index = cidx
		cidx++
	}
	type entry struct {
		j int
		v float64
	}
	rowEntries := make([][]entry, cidx)
	posToIndex := make([]int, len(rows))
	for p, ri := range rows {
		posToIndex[p] = ri.index
	}
	for j, c := range cols {
		for k, p := range c.idx {
			i := posToIndex[p]
			rowEntries[i] = append(rowEntries[i], entry{j, c.val[k]})
		}
	}
	for _, ri := range rows {
		if ri.typ == 'N' {
			continue
		}
		var lo, up float64
		switch ri.typ {
		case 'L':
			lo, up = math.Inf(-1), ri.rhs
			if ri.hasR {
				lo = ri.rhs - math.Abs(ri.rng)
			}
		case 'G':
			lo, up = ri.rhs, math.Inf(1)
			if ri.hasR {
				up = ri.rhs + math.Abs(ri.rng)
			}
		case 'E':
			lo, up = ri.rhs, ri.rhs
			if ri.hasR {
				if ri.rng > 0 {
					up = ri.rhs + ri.rng
				} else {
					lo = ri.rhs + ri.rng
				}
			}
		}
		e := gost.NewExpr()
		for _, en := range rowEntries[ri.index] {
			e.Add(en.v, m.VarByIndex(en.j))
		}
		m.AddRange(e, lo, up, ri.name)
	}
	m.SetObjOffset(objConst)
	if maximize {
		m.SetSense(gost.Maximize)
	}
	return m, nil
}

// Write writes the model in free MPS format.
func Write(w io.Writer, m *gost.Model) error {
	bw := bufio.NewWriter(w)
	fmt.Fprintf(bw, "NAME %s\n", nameOr(m.Name, "GOST"))
	if m.GetSense() == gost.Maximize {
		fmt.Fprintf(bw, "OBJSENSE\n    MAX\n")
	}
	fmt.Fprintf(bw, "ROWS\n N  OBJ\n")
	nr := m.NumConstrs()
	type ent struct {
		row int
		v   float64
	}
	colEnt := make([][]ent, m.NumVars())
	rnames := make([]string, nr)
	var rhsLines, rangeLines []string
	for i := 0; i < nr; i++ {
		name, lo, up, vars, coefs := m.ConstrInfo(m.ConstrByIndex(i))
		rn := nameOr(name, fmt.Sprintf("R%d", i))
		rnames[i] = rn
		for k, v := range vars {
			colEnt[v.Index()] = append(colEnt[v.Index()], ent{i, coefs[k]})
		}
		var t string
		switch {
		case lo == up:
			t = "E"
			if lo != 0 {
				rhsLines = append(rhsLines, fmt.Sprintf("    RHS %s %.17g", rn, lo))
			}
		case math.IsInf(lo, -1) && math.IsInf(up, 1):
			t = "N"
		case math.IsInf(lo, -1):
			t = "L"
			if up != 0 {
				rhsLines = append(rhsLines, fmt.Sprintf("    RHS %s %.17g", rn, up))
			}
		case math.IsInf(up, 1):
			t = "G"
			if lo != 0 {
				rhsLines = append(rhsLines, fmt.Sprintf("    RHS %s %.17g", rn, lo))
			}
		default:
			t = "G"
			if lo != 0 {
				rhsLines = append(rhsLines, fmt.Sprintf("    RHS %s %.17g", rn, lo))
			}
			rangeLines = append(rangeLines, fmt.Sprintf("    RNG %s %.17g", rn, up-lo))
		}
		fmt.Fprintf(bw, " %s  %s\n", t, rn)
	}
	fmt.Fprintf(bw, "COLUMNS\n")
	inInt := false
	var boundLines []string
	for j := 0; j < m.NumVars(); j++ {
		name, lo, up, obj, typ := m.VarInfo(m.VarByIndex(j))
		cn := nameOr(name, fmt.Sprintf("C%d", j))
		isInt := typ != gost.Continuous
		if isInt != inInt {
			if isInt {
				fmt.Fprintf(bw, "    MARKER 'MARKER' 'INTORG'\n")
			} else {
				fmt.Fprintf(bw, "    MARKER 'MARKER' 'INTEND'\n")
			}
			inInt = isInt
		}
		if obj != 0 {
			fmt.Fprintf(bw, "    %s OBJ %.17g\n", cn, obj)
		}
		for _, e := range colEnt[j] {
			fmt.Fprintf(bw, "    %s %s %.17g\n", cn, rnames[e.row], e.v)
		}
		if obj == 0 && len(colEnt[j]) == 0 {
			fmt.Fprintf(bw, "    %s OBJ 0\n", cn)
		}
		switch {
		case lo == up:
			boundLines = append(boundLines, fmt.Sprintf(" FX BND %s %.17g", cn, lo))
		case math.IsInf(lo, -1) && math.IsInf(up, 1):
			boundLines = append(boundLines, fmt.Sprintf(" FR BND %s", cn))
		default:
			if math.IsInf(lo, -1) {
				boundLines = append(boundLines, fmt.Sprintf(" MI BND %s", cn))
			} else if lo != 0 {
				boundLines = append(boundLines, fmt.Sprintf(" LO BND %s %.17g", cn, lo))
			}
			if !math.IsInf(up, 1) {
				boundLines = append(boundLines, fmt.Sprintf(" UP BND %s %.17g", cn, up))
			} else if isInt {
				boundLines = append(boundLines, fmt.Sprintf(" PL BND %s", cn))
			}
		}
	}
	if inInt {
		fmt.Fprintf(bw, "    MARKER 'MARKER' 'INTEND'\n")
	}
	fmt.Fprintf(bw, "RHS\n")
	if off := m.ObjOffset(); off != 0 {
		fmt.Fprintf(bw, "    RHS OBJ %.17g\n", -off)
	}
	for _, l := range rhsLines {
		fmt.Fprintln(bw, l)
	}
	if len(rangeLines) > 0 {
		fmt.Fprintf(bw, "RANGES\n")
		for _, l := range rangeLines {
			fmt.Fprintln(bw, l)
		}
	}
	if len(boundLines) > 0 {
		fmt.Fprintf(bw, "BOUNDS\n")
		for _, l := range boundLines {
			fmt.Fprintln(bw, l)
		}
	}
	fmt.Fprintf(bw, "ENDATA\n")
	return bw.Flush()
}

func nameOr(s, def string) string {
	if s == "" || strings.ContainsAny(s, " \t") {
		return def
	}
	return s
}
