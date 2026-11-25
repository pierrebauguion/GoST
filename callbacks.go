package gost

import (
	"fmt"
	"math"

	"gost/internal/mip"
)

// Column describes the coefficients of a variable in existing constraints,
// for column-wise modelling (AddVarWithColumn) and column generation.
type Column struct {
	Constrs []Constr
	Coefs   []float64
}

// NewColumn returns an empty column.
func NewColumn() *Column { return &Column{} }

// Add appends coef in constraint c and returns the column.
func (col *Column) Add(coef float64, c Constr) *Column {
	col.Constrs = append(col.Constrs, c)
	col.Coefs = append(col.Coefs, coef)
	return col
}

// AddVarWithColumn adds a variable together with its coefficients in
// existing constraints (column-wise modelling, e.g. master problems).
func (m *Model) AddVarWithColumn(lo, up, obj float64, typ VarType, name string, col *Column) Var {
	v := m.AddVar(lo, up, obj, typ, name)
	if col == nil {
		return v
	}
	acc := map[int]float64{}
	order := []int{}
	for k, c := range col.Constrs {
		if c.idx < 0 || c.idx >= len(m.cons) {
			panic(fmt.Sprintf("gost: constraint index %d out of range", c.idx))
		}
		if _, ok := acc[c.idx]; !ok {
			order = append(order, c.idx)
		}
		acc[c.idx] += col.Coefs[k]
	}
	for _, i := range order {
		if a := acc[i]; a != 0 {
			m.cons[i].idx = append(m.cons[i].idx, v.idx)
			m.cons[i].val = append(m.cons[i].val, a)
		}
	}
	return v
}

// ---------------------------------------------------------------------------
// Constraint callbacks (lazy constraints and user cuts).

// ConstrContext is passed to the LazyConstraints and UserCuts callbacks. It
// exposes the candidate (lazy) or fractional LP (cuts) solution and lets the
// callback add constraints. Constraints added here are added to the model
// and stay in it after Solve returns.
type ConstrContext struct {
	m           *Model
	x           []float64
	depth, node int
	rows        []mip.RowSpec
}

// Value returns the value of v in the solution being examined.
func (c *ConstrContext) Value(v Var) float64 { return c.x[v.idx] }

// Values returns the solution being examined (indexed by variable).
func (c *ConstrContext) Values() []float64 { return append([]float64(nil), c.x...) }

// Depth returns the depth of the current node (0 at the root).
func (c *ConstrContext) Depth() int { return c.depth }

// Node returns the number of processed nodes.
func (c *ConstrContext) Node() int { return c.node }

// AddConstr adds the constraint e (sense) rhs.
func (c *ConstrContext) AddConstr(e *Expr, sense ConstrSense, rhs float64, name string) Constr {
	con := c.m.AddConstr(e, sense, rhs, name)
	c.record(con)
	return con
}

// AddRange adds lo <= e <= up.
func (c *ConstrContext) AddRange(e *Expr, lo, up float64, name string) Constr {
	con := c.m.AddRange(e, lo, up, name)
	c.record(con)
	return con
}

func (c *ConstrContext) record(con Constr) {
	r := c.m.cons[con.idx]
	c.rows = append(c.rows, mip.RowSpec{
		Idx: append([]int(nil), r.idx...), Val: append([]float64(nil), r.val...), Lo: r.lo, Up: r.up,
	})
}

// ---------------------------------------------------------------------------
// Pricing (column generation).

// PricingContext is passed to the Pricing callback after each LP solve. The
// callback reads the duals and adds variables with an attractive reduced
// cost (negative when minimizing, positive when maximizing). When it adds
// nothing, pricing has converged for the current node.
type PricingContext struct {
	m     *Model
	info  mip.PriceInfo
	sign  float64
	cols  []mip.ColSpec
	added int
}

// Dual returns the dual value of constraint c (model sense): the reduced
// cost of a column is obj - sum_i coef_i * Dual(c_i).
func (c *PricingContext) Dual(con Constr) float64 {
	if con.idx >= len(c.info.Duals) {
		return 0
	}
	return c.sign * c.info.Duals[con.idx]
}

// Duals returns the duals of all constraints (model sense).
func (c *PricingContext) Duals() []float64 {
	y := make([]float64, len(c.info.Duals))
	for i, v := range c.info.Duals {
		y[i] = c.sign * v
	}
	return y
}

// ReducedCost returns obj - sum coef*Dual for a candidate column.
func (c *PricingContext) ReducedCost(obj float64, col *Column) float64 {
	rc := obj
	for k, con := range col.Constrs {
		rc -= col.Coefs[k] * c.Dual(con)
	}
	return rc
}

// Value returns the LP value of an existing variable.
func (c *PricingContext) Value(v Var) float64 {
	if v.idx >= len(c.info.X) {
		return 0
	}
	return c.info.X[v.idx]
}

// LPObjective returns the current LP objective (model sense, with offset).
func (c *PricingContext) LPObjective() float64 {
	return c.sign*c.info.Obj + c.m.objOffset
}

// Depth returns the depth of the current node (0 at the root).
func (c *PricingContext) Depth() int { return c.info.Depth }

// Node returns the number of processed nodes.
func (c *PricingContext) Node() int { return c.info.Node }

// AddVar adds a new variable (column) to the model and the LP. Its bounds
// must contain 0 so that existing solutions remain valid.
func (c *PricingContext) AddVar(lo, up, obj float64, typ VarType, name string, col *Column) Var {
	if typ == Binary {
		lo, up = math.Max(lo, 0), math.Min(up, 1)
	}
	if lo > 0 || up < 0 {
		panic("gost: priced variables must have lo <= 0 <= up")
	}
	v := c.m.AddVarWithColumn(lo, up, obj, typ, name, col)
	spec := mip.ColSpec{Obj: c.sign * obj, Lo: lo, Up: up, Integer: typ != Continuous}
	if col != nil {
		acc := map[int]float64{}
		order := []int{}
		for k, con := range col.Constrs {
			if _, ok := acc[con.idx]; !ok {
				order = append(order, con.idx)
			}
			acc[con.idx] += col.Coefs[k]
		}
		for _, i := range order {
			if a := acc[i]; a != 0 {
				spec.Rows = append(spec.Rows, i)
				spec.Vals = append(spec.Vals, a)
			}
		}
	}
	c.cols = append(c.cols, spec)
	c.added++
	return v
}

// hooks builds the internal hooks from the parameters.
func (m *Model) hooks(p Params) *mip.Hooks {
	if p.LazyConstraints == nil && p.UserCuts == nil && p.Pricing == nil {
		return nil
	}
	h := &mip.Hooks{}
	sign := 1.0
	if m.sense == Maximize {
		sign = -1
	}
	if f := p.LazyConstraints; f != nil {
		h.Lazy = func(x []float64) []mip.RowSpec {
			ctx := &ConstrContext{m: m, x: x}
			f(ctx)
			return ctx.rows
		}
	}
	if f := p.UserCuts; f != nil {
		h.Cut = func(x []float64, depth, node int) []mip.RowSpec {
			ctx := &ConstrContext{m: m, x: x, depth: depth, node: node}
			f(ctx)
			return ctx.rows
		}
	}
	if f := p.Pricing; f != nil {
		h.Price = func(info mip.PriceInfo) []mip.ColSpec {
			ctx := &PricingContext{m: m, info: info, sign: sign}
			f(ctx)
			return ctx.cols
		}
	}
	return h
}
