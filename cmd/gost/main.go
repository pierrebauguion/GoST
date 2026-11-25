// Command gost solves MPS models with the GoST solver.
package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"gost"
	"gost/mps"
)

func main() {
	tl := flag.Duration("time", 0, "time limit (e.g. 60s)")
	gap := flag.Float64("gap", 1e-4, "relative MIP gap")
	nopre := flag.Bool("nopresolve", false, "disable presolve")
	nocuts := flag.Bool("nocuts", false, "disable cuts")
	noheur := flag.Bool("noheur", false, "disable heuristics")
	quiet := flag.Bool("q", false, "quiet")
	nodes := flag.Int("nodes", 0, "node limit (0 = none)")
	rounds := flag.Int("cutrounds", -1, "root cut rounds (-1 = default)")
	sb := flag.Int("sb", -1, "strong branching candidates (-1 = default)")
	rel := flag.Int("rel", -1, "reliability threshold (-1 = default)")
	debugSol := flag.String("debugsol", "", "file with a known solution (one value per line, column order)")
	start := flag.String("start", "", "MIP start file (\"name value\" lines, or one value per line in column order; partial starts are repaired)")
	writeSol := flag.String("writesol", "", "write the best solution to this file (\"name value\" lines, usable with -start)")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: gost [flags] model.mps[.gz]")
		os.Exit(2)
	}
	t0 := time.Now()
	m, err := mps.ReadFile(flag.Arg(0))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if !*quiet {
		fmt.Printf("Read %s in %.2fs\n", flag.Arg(0), time.Since(t0).Seconds())
	}
	p := gost.DefaultParams()
	p.TimeLimit = *tl
	p.NodeLimit = *nodes
	if *sb >= 0 {
		p.StrongBranchCands = *sb
	}
	if *rel >= 0 {
		p.Reliability = *rel
	}
	if *rounds >= 0 {
		p.CutRounds = *rounds
	}
	p.RelGap = *gap
	p.Presolve = !*nopre
	p.Cuts = !*nocuts
	p.Heuristics = !*noheur
	if !*quiet {
		p.Log = os.Stdout
	}
	if *debugSol != "" {
		data, err := os.ReadFile(*debugSol)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		for _, f := range strings.Fields(string(data)) {
			v, _ := strconv.ParseFloat(f, 64)
			p.DebugSolution = append(p.DebugSolution, v)
		}
	}
	if *start != "" {
		data, err := os.ReadFile(*start)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		if _, err := m.ReadStart(string(data)); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
	r, _ := m.Solve(p)
	if *start != "" {
		fmt.Printf("MIP start: %s %s\n", r.StartStatus, r.StartInfo)
	}
	if *writeSol != "" && r.HasSolution() {
		if err := os.WriteFile(*writeSol, []byte(m.WriteSolution(r)), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, err)
		}
	}
	fmt.Printf("%s status=%s obj=%.10g bound=%.10g nodes=%d time=%.2fs\n", m.Name, r.Status, r.Objective, r.Bound, r.Nodes, r.Time.Seconds())
}
