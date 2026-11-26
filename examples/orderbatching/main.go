// Order batching in a warehouse: group customer orders into picking tours
// so as to minimise the total walking distance of the pickers
// (Gademann and van de Velde, 2005; de Koster et al., 2007).
//
// The warehouse has parallel aisles 0..A−1, all starting from a front
// cross-aisle where the depot is (in front of aisle 0). Each order consists
// of items stored at given aisles and depths. A picker cart holds at most
// Q items, so an order cannot be split but several orders can be picked in
// the same tour (a batch).
//
// Routing policy "return": the picker walks along the front aisle to the
// farthest aisle needed, enters each aisle containing items of the batch
// up to the deepest of them and comes back. The length of a tour is then
//
//	2·W·(farthest aisle index) + Σ_{aisles a visited} 2·(deepest item in a)
//
// with W the distance between two aisles. Combining orders that share
// aisles saves walking: that is the whole point of batching.
//
// Model (orders o, batches b = 0..n−1, aisles a):
//
//	min  Σ_b (2·W·F_b + Σ_a 2·D_ba)
//	s.t. Σ_b x_ob = 1                  for each order o     (assignment)
//	     Σ_o q_o·x_ob ≤ Q              for each batch b     (cart capacity)
//	     F_b ≥ far_o·x_ob              for each o, b        (farthest aisle)
//	     D_ba ≥ depth_oa·x_ob          for each o, b, a     (deepest item)
//	     x_ob = 0 for b > o                                 (symmetry breaking)
//	     x ∈ {0, 1}, F, D ≥ 0
//
// x_ob = 1 if order o is picked in batch b; F_b and D_ba are the maxima that
// define the tour length (minimisation pushes them down to the maxima).
// depth_oa is the deepest position of order o in aisle a, far_o its
// farthest aisle.
//
// Compared with the classic "first come, first served" batching and with a
// "seed" heuristic, then checked by recomputing each tour length directly.
//
//	go run ./examples/orderbatching
package main

import (
	"fmt"
	"math/rand"

	"gost"
)

const (
	nAisles = 10  // aisles 0..9
	depthMx = 20  // positions 1..20 along each aisle
	W       = 3.0 // distance between two adjacent aisles
	Q       = 12  // cart capacity (items)
	nOrders = 12  // number of orders
)

// order: its items as (aisle, depth) positions.
type order struct{ items [][2]int }

// tourLength computes the length of the return-policy tour that picks all
// the given orders together.
func tourLength(orders []order, batch []int) float64 {
	deepest := make([]int, nAisles) // deepest position needed in each aisle
	far := -1                       // farthest aisle needed
	for _, o := range batch {
		for _, it := range orders[o].items {
			deepest[it[0]] = max(deepest[it[0]], it[1])
			far = max(far, it[0])
		}
	}
	if far < 0 {
		return 0
	}
	l := 2 * W * float64(far)
	for _, d := range deepest {
		l += 2 * float64(d)
	}
	return l
}

func main() {
	// Fixed seed: the same instance on every run.
	rng := rand.New(rand.NewSource(4))
	// Orders of 2 to 6 items. Items are concentrated in a few "popular"
	// aisles, as in real warehouses, so that orders share aisles.
	orders := make([]order, nOrders)
	for o := range orders {
		k := 2 + rng.Intn(5)
		for i := 0; i < k; i++ {
			a := rng.Intn(nAisles)
			if rng.Intn(2) == 0 {
				a = rng.Intn(4) // popular aisles 0..3
			}
			orders[o].items = append(orders[o].items, [2]int{a, 1 + rng.Intn(depthMx)})
		}
	}
	q := func(o int) int { return len(orders[o].items) } // items of order o

	// --- Heuristic 1: first come, first served -------------------------------
	// Orders in arrival order, a new batch whenever the cart is full.
	fcfs, cur, used := 0.0, []int{}, 0
	for o := range orders {
		if used+q(o) > Q {
			fcfs += tourLength(orders, cur)
			cur, used = nil, 0
		}
		cur = append(cur, o)
		used += q(o)
	}
	fcfs += tourLength(orders, cur)

	// --- Heuristic 2: seed algorithm -----------------------------------------
	// Start a batch with the unassigned order whose own tour is longest (the
	// seed), then repeatedly add the order that increases the tour length
	// the least, while the cart has room.
	seed := 0.0
	done := make([]bool, nOrders)
	for left := nOrders; left > 0; {
		s, best := -1, -1.0
		for o := range orders {
			if !done[o] {
				if l := tourLength(orders, []int{o}); l > best {
					s, best = o, l
				}
			}
		}
		batch := []int{s}
		done[s], left, used = true, left-1, q(s)
		for {
			add, inc := -1, 0.0
			base := tourLength(orders, batch)
			for o := range orders {
				if !done[o] && used+q(o) <= Q {
					if d := tourLength(orders, append(batch, o)) - base; add < 0 || d < inc {
						add, inc = o, d
					}
				}
			}
			if add < 0 {
				break
			}
			batch = append(batch, add)
			done[add], left, used = true, left-1, used+q(add)
		}
		seed += tourLength(orders, batch)
	}
	fmt.Printf("%d orders, cart capacity %d items: first-come-first-served %.0f, seed heuristic %.0f\n",
		nOrders, Q, fcfs, seed)

	// --- Model -----------------------------------------------------------------
	m := gost.NewModel("order-batching")
	n := nOrders // at most one batch per order
	// x[o][b] for b ≤ o: order o in batch b (symmetry breaking: order o can
	// only go into the first o+1 batches).
	x := make([][]gost.Var, n)
	for o := range x {
		x[o] = make([]gost.Var, o+1)
		for b := range x[o] {
			x[o][b] = m.AddVar(0, 1, 0, gost.Binary, fmt.Sprintf("x%d_%d", o, b))
		}
	}
	// F[b]: farthest aisle of batch b (cost 2·W per aisle index).
	// D[b][a]: deepest position visited in aisle a by batch b (cost 2).
	F := make([]gost.Var, n)
	D := make([][]gost.Var, n)
	for b := 0; b < n; b++ {
		F[b] = m.AddVar(0, nAisles-1, 2*W, gost.Continuous, fmt.Sprintf("F%d", b))
		D[b] = make([]gost.Var, nAisles)
		for a := range D[b] {
			D[b][a] = m.AddVar(0, depthMx, 2, gost.Continuous, fmt.Sprintf("D%d_%d", b, a))
		}
	}
	for o := 0; o < n; o++ {
		// assign_o: each order in exactly one batch.
		e := gost.NewExpr()
		for b := range x[o] {
			e.Add(1, x[o][b])
		}
		m.AddConstr(e, gost.Equal, 1, fmt.Sprintf("assign%d", o))
	}
	for b := 0; b < n; b++ {
		// capacity_b: the items of batch b fit in the cart.
		e := gost.NewExpr()
		for o := b; o < n; o++ {
			e.Add(float64(q(o)), x[o][b])
		}
		m.AddConstr(e, gost.LessEq, Q, fmt.Sprintf("capacity%d", b))
		for o := b; o < n; o++ {
			// far_o_b and depth_o_b_a: F_b and D_ba are at least the values
			// required by each order of the batch.
			far, deep := 0, make([]int, nAisles)
			for _, it := range orders[o].items {
				far = max(far, it[0])
				deep[it[0]] = max(deep[it[0]], it[1])
			}
			m.AddConstr(gost.NewExpr().Add(1, F[b]).Add(-float64(far), x[o][b]), gost.GreaterEq, 0, fmt.Sprintf("far%d_%d", o, b))
			for a, dp := range deep {
				if dp > 0 {
					m.AddConstr(gost.NewExpr().Add(1, D[b][a]).Add(-float64(dp), x[o][b]), gost.GreaterEq, 0, fmt.Sprintf("depth%d_%d_%d", o, b, a))
				}
			}
		}
	}

	res, err := m.Solve(gost.DefaultParams())
	if err != nil {
		panic(err)
	}
	fmt.Printf("optimal batching: %s, total distance %.0f (%d nodes, %v)\n\n", res.Status, res.Objective, res.Nodes, res.Time.Round(1e6))

	// --- Independent check: recompute every tour from the batches ---------------
	total := 0.0
	for b := 0; b < n; b++ {
		var batch []int
		items := 0
		for o := b; o < n; o++ {
			if res.Value(x[o][b]) > 0.5 {
				batch = append(batch, o)
				items += q(o)
			}
		}
		if len(batch) == 0 {
			continue
		}
		l := tourLength(orders, batch)
		total += l
		fmt.Printf("tour %2d: orders %v, %2d/%d items, length %.0f\n", b, batch, items, Q, l)
	}
	fmt.Printf("independent check: total distance %.0f\n", total)
}
