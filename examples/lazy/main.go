// Lazy constraints: a constraint that is not in the model is checked on
// every candidate solution and added only when violated.
//
// Full model:
//
//	max  x + y
//	s.t. x + 2y ≤ 12          (lazy constraint, not in the initial model)
//	     0 ≤ x, y ≤ 10, integer
//
// The initial model only contains the bounds; the callback adds the
// constraint the first time a candidate solution violates it.
//
//	go run ./examples/lazy
package main

import (
	"fmt"

	"gost"
)

func main() {
	m := gost.NewModel("lazy")

	// --- Variables -------------------------------------------------------
	// x: integer variable between 0 and 10.
	x := m.AddVar(0, 10, 0, gost.Integer, "x")
	// y: integer variable between 0 and 10.
	y := m.AddVar(0, 10, 0, gost.Integer, "y")

	// --- Objective -------------------------------------------------------
	// Maximise x + y. Without the lazy constraint the optimum would be
	// x = y = 10.
	m.SetObjective(gost.NewExpr().Add(1, x).Add(1, y), gost.Maximize)

	// --- Lazy constraint -------------------------------------------------
	p := gost.DefaultParams()
	// LazyConstraints is called on every candidate integer solution
	// (before it becomes the incumbent).
	// ctx.Value(v) gives the value of v in that candidate;
	// ctx.AddConstr adds a constraint to the model and rejects the candidate.
	p.LazyConstraints = func(ctx *gost.ConstrContext) {
		// vx, vy: values of x and y in the candidate solution.
		vx, vy := ctx.Value(x), ctx.Value(y)
		fmt.Printf("candidate x=%.0f y=%.0f", vx, vy)
		// Constraint checked: x + 2y ≤ 12 (tolerance 1e-9).
		if vx+2*vy > 12+1e-9 {
			fmt.Println(" -> violates x + 2y <= 12, constraint added")
			// Add x + 2y ≤ 12 to the model for good, named "lazy".
			ctx.AddConstr(gost.NewExpr().Add(1, x).Add(2, y), gost.LessEq, 12, "lazy")
			return
		}
		// No constraint added: the candidate is accepted.
		fmt.Println(" -> accepted")
	}

	// --- Solve -----------------------------------------------------------
	res, _ := m.Solve(p)
	// m.NumConstrs() also counts the constraints added by the callback.
	fmt.Printf("%s: x=%.0f y=%.0f, objective %.0f, constraints in model: %d\n",
		res.Status, res.Value(x), res.Value(y), res.Objective, m.NumConstrs())
}
