// Production planning: a small integer program built with the modelling API.
//
// A workshop makes chairs and tables. Each product uses machine hours and
// labour hours, both available in limited quantity. We look for how many of
// each product to make in order to maximise profit.
//
// Model:
//
//	max  30·chairs + 50·tables                 (profit)
//	s.t. 2·chairs + 5·tables ≤ 40              (machine hours)
//	     3·chairs + 2·tables ≤ 36              (labour hours)
//	     chairs, tables ∈ ℕ
//
//	go run ./examples/production
package main

import (
	"fmt"

	"gost"
)

func main() {
	// m: the model, holding variables, constraints and the objective.
	m := gost.NewModel("production")

	// --- Variables -------------------------------------------------------
	// AddVar(lower bound, upper bound, objective cost, type, name).
	// The cost is left at 0 here because the objective is set below with
	// SetObjective (which replaces the costs).

	// chairs: number of chairs produced (integer ≥ 0, no upper bound).
	chairs := m.AddVar(0, gost.Inf, 0, gost.Integer, "chairs")
	// tables: number of tables produced (integer ≥ 0, no upper bound).
	tables := m.AddVar(0, gost.Inf, 0, gost.Integer, "tables")

	// --- Constraints -----------------------------------------------------
	// AddConstr(linear expression, sense, right-hand side, name).

	// machine: each chair needs 2 h of machine time, each table 5 h;
	// 40 machine hours are available.
	m.AddConstr(gost.NewExpr().Add(2, chairs).Add(5, tables), gost.LessEq, 40, "machine")
	// labour: each chair needs 3 h of labour, each table 2 h;
	// 36 labour hours are available.
	m.AddConstr(gost.NewExpr().Add(3, chairs).Add(2, tables), gost.LessEq, 36, "labour")

	// --- Objective -------------------------------------------------------
	// Unit profit: 30 per chair, 50 per table, to be maximised.
	m.SetObjective(gost.NewExpr().Add(30, chairs).Add(50, tables), gost.Maximize)

	// --- Solve -----------------------------------------------------------
	// DefaultParams: presolve, cuts, heuristics, no time limit.
	// res holds the status, the objective value and the variable values.
	res, err := m.Solve(gost.DefaultParams())
	if err != nil {
		panic(err)
	}
	fmt.Printf("status: %s\n", res.Status)
	fmt.Printf("profit: %.0f\n", res.Objective)
	// res.Value(v): value of variable v in the best solution found.
	fmt.Printf("chairs: %.0f, tables: %.0f\n", res.Value(chairs), res.Value(tables))
}
