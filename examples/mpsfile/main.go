// Solve a model read from an MPS file, with the solver log on stdout.
//
// MPS is the standard exchange format between solvers (CPLEX, Gurobi,
// HiGHS...). Variables and constraints are defined in the file: this
// program does not create any variable itself.
//
//	go run ./examples/mpsfile                       # testdata/p0201.mps.gz
//	go run ./examples/mpsfile path/to/model.mps.gz 60s
package main

import (
	"fmt"
	"os"
	"time"

	"gost"
	"gost/mps"
)

func main() {
	// path: MPS file to read (gzip-compressed or not). Defaults to a
	// MIPLIB instance shipped in testdata.
	path := "testdata/p0201.mps.gz"
	if len(os.Args) > 1 {
		path = os.Args[1]
	}
	// limit: time limit of the solve (2nd argument, in Go duration
	// format: "30s", "5m"...). 60 s by default.
	limit := 60 * time.Second
	if len(os.Args) > 2 {
		if d, err := time.ParseDuration(os.Args[2]); err == nil {
			limit = d
		}
	}

	// m: the model built from the file (variables, constraints, objective
	// and integer/continuous types as declared in the file).
	m, err := mps.ReadFile(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	// p: solve parameters.
	p := gost.DefaultParams()
	p.TimeLimit = limit // stop after this duration (best solution returned)
	p.Log = os.Stdout   // log: presolve, cuts, tree progress

	// res: status, best solution (Objective), bound (Bound) and number
	// of nodes explored.
	res, _ := m.Solve(p)
	fmt.Printf("\n%s: %s, objective %.10g, bound %.10g, %d nodes\n",
		m.Name, res.Status, res.Objective, res.Bound, res.Nodes)
}
