// Package gost ("Golang Subject To") is a mixed-integer linear programming
// solver written in pure Go.
//
// Build a model, then call Solve:
//
//	m := gost.NewModel("knapsack")
//	x := m.AddVar(0, 1, 5, gost.Binary, "x")
//	y := m.AddVar(0, 1, 4, gost.Binary, "y")
//	m.AddConstr(gost.NewExpr().Add(3, x).Add(2, y), gost.LessEq, 4, "cap")
//	m.SetSense(gost.Maximize)
//	res, err := m.Solve(gost.DefaultParams())
package gost

import (
	"fmt"
	"math"
	"sort"
)

// Inf is positive infinity, used for unbounded variable and row bounds.
var Inf = math.Inf(1)

// VarType is the type of a decision variable.
type VarType int8

const (
	Continuous VarType = iota
	Integer
	Binary
)

// Sense is the optimization direction.
type Sense int8

const (
	Minimize Sense = iota
	Maximize
)

// ConstrSense is the sense of a constraint.
type ConstrSense int8

const (
	LessEq ConstrSense = iota
	GreaterEq
	Equal
)

// Var is a handle to a model variable.
type Var struct{ idx int }

// Index returns the column index of the variable in its model.
func (v Var) Index() int { return v.idx }

// Constr is a handle to a model constraint.
type Constr struct{ idx int }

// Index returns the row index of the constraint.
func (c Constr) Index() int { return c.idx }

// Expr is a linear expression sum(coef_k * var_k) + Constant.
type Expr struct {
	Vars     []Var
	Coefs    []float64
	Constant float64
}

// NewExpr returns an empty expression.
func NewExpr() *Expr { return &Expr{} }

// Add appends coef*v to the expression and returns it.
func (e *Expr) Add(coef float64, v Var) *Expr {
	e.Vars = append(e.Vars, v)
	e.Coefs = append(e.Coefs, coef)
	return e
}

// AddConst adds a constant term.
func (e *Expr) AddConst(c float64) *Expr {
	e.Constant += c
	return e
}

type variable struct {
	name   string
	lo, up float64
	obj    float64
	typ    VarType
}

type constraint struct {
	name   string
	lo, up float64
	idx    []int
	val    []float64
}

// Model is a mixed-integer linear program.
type Model struct {
	Name      string
	vars      []variable
	cons      []constraint
	sense     Sense
	objOffset float64
	colSig    map[string]int // column signature -> var index+1 (AddVarWithColumn)
	start     []float64      // MIP start (NaN = not given), see SetStart
}

// NewModel creates an empty model.
func NewModel(name string) *Model { return &Model{Name: name} }

// NumVars returns the number of variables.
func (m *Model) NumVars() int { return len(m.vars) }

// NumConstrs returns the number of constraints.
func (m *Model) NumConstrs() int { return len(m.cons) }

// AddVar adds a variable with bounds [lo, up], objective coefficient obj.
// Binary variables have their bounds intersected with [0, 1].
func (m *Model) AddVar(lo, up, obj float64, typ VarType, name string) Var {
	if typ == Binary {
		lo = math.Max(lo, 0)
		up = math.Min(up, 1)
	}
	m.vars = append(m.vars, variable{name: name, lo: lo, up: up, obj: obj, typ: typ})
	return Var{len(m.vars) - 1}
}

// AddConstr adds expr (sense) rhs. The expression constant is moved to the
// right-hand side and duplicate variables are merged.
func (m *Model) AddConstr(e *Expr, sense ConstrSense, rhs float64, name string) Constr {
	rhs -= e.Constant
	switch sense {
	case LessEq:
		return m.addRow(e, -Inf, rhs, name)
	case GreaterEq:
		return m.addRow(e, rhs, Inf, name)
	default:
		return m.addRow(e, rhs, rhs, name)
	}
}

// AddRange adds lo <= expr <= up.
func (m *Model) AddRange(e *Expr, lo, up float64, name string) Constr {
	return m.addRow(e, lo-e.Constant, up-e.Constant, name)
}

func (m *Model) addRow(e *Expr, lo, up float64, name string) Constr {
	acc := map[int]float64{}
	for k, v := range e.Vars {
		if v.idx < 0 || v.idx >= len(m.vars) {
			panic(fmt.Sprintf("gost: variable index %d out of range", v.idx))
		}
		acc[v.idx] += e.Coefs[k]
	}
	idx := make([]int, 0, len(acc))
	for j, c := range acc {
		if c != 0 {
			idx = append(idx, j)
		}
	}
	sort.Ints(idx)
	val := make([]float64, len(idx))
	for k, j := range idx {
		val[k] = acc[j]
	}
	m.cons = append(m.cons, constraint{name: name, lo: lo, up: up, idx: idx, val: val})
	return Constr{len(m.cons) - 1}
}

// SetObjective sets the objective to the expression (replacing all
// objective coefficients) and the direction.
func (m *Model) SetObjective(e *Expr, sense Sense) {
	for j := range m.vars {
		m.vars[j].obj = 0
	}
	for k, v := range e.Vars {
		m.vars[v.idx].obj += e.Coefs[k]
	}
	m.objOffset = e.Constant
	m.sense = sense
}

// SetSense sets the optimization direction.
func (m *Model) SetSense(s Sense) { m.sense = s }

// GetSense returns the optimization direction.
func (m *Model) GetSense() Sense { return m.sense }

// SetObjOffset sets the objective constant.
func (m *Model) SetObjOffset(c float64) { m.objOffset = c }

// ObjOffset returns the objective constant.
func (m *Model) ObjOffset() float64 { return m.objOffset }

// SetObjCoef sets the objective coefficient of v.
func (m *Model) SetObjCoef(v Var, c float64) { m.vars[v.idx].obj = c }

// SetBounds changes the bounds of v.
func (m *Model) SetBounds(v Var, lo, up float64) { m.vars[v.idx].lo, m.vars[v.idx].up = lo, up }

// SetVarType changes the type of v.
func (m *Model) SetVarType(v Var, t VarType) { m.vars[v.idx].typ = t }

// VarInfo returns the attributes of v.
func (m *Model) VarInfo(v Var) (name string, lo, up, obj float64, typ VarType) {
	x := m.vars[v.idx]
	return x.name, x.lo, x.up, x.obj, x.typ
}

// VarByIndex returns the variable handle for column j.
func (m *Model) VarByIndex(j int) Var { return Var{j} }

// ConstrInfo returns the attributes of constraint c.
func (m *Model) ConstrInfo(c Constr) (name string, lo, up float64, vars []Var, coefs []float64) {
	r := m.cons[c.idx]
	vars = make([]Var, len(r.idx))
	for k, j := range r.idx {
		vars[k] = Var{j}
	}
	return r.name, r.lo, r.up, vars, append([]float64(nil), r.val...)
}

// ConstrByIndex returns the constraint handle for row i.
func (m *Model) ConstrByIndex(i int) Constr { return Constr{i} }
