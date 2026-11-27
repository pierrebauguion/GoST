package main

// Side-aware model of the 3E block of the Evreux warehouse (zone c4), built
// from the real floor plan (evreux_plan.go, Metabase atlas.warehouse_aisles).
//
// The graph of evreux.go joins aisles directly, whatever side the picker is
// on. In reality an aisle is a corridor between two racks: it is entered at
// one end and left at the other, and going from the end of one aisle to the
// start of another means walking along the cross-aisles, possibly around the
// racks. Moreover each aisle has a traversal direction (Metabase column
// traversal_direction: START_TO_END or END_TO_START): "operators can't turn
// back once they are in the aisle", so an aisle is a one-way arc.
//
// Layout of the 3E block (meters):
//
//	   x ≈ 60.5        62.0 ……… 70.8   71.8   72.8 ……… 81.7   82.6        91
//	 left corridor    B aisles (3Exx_B)  middle  A aisles (3Exx_A)  right    carts
//	 (along 4P01)                       corridor                 corridor   (depot)
//
// with a front corridor below the block (y = 20.5) and a back corridor above
// it (y = 96.5) joining the three vertical corridors. 4P01 is a long one-way
// aisle along the left wall (x = 58.86).
//
// Graph:
//   - each aisle a gives three nodes: entry, middle and exit (entry/exit at
//     its ends, in its traversal direction) and the one-way arcs
//     entry → middle → exit (half the aisle length each). Visiting the
//     middle node forces the traversal of the whole aisle in the right
//     direction, so the problem stays a Steiner TSP whose required nodes are
//     the middles of the aisles of the pick list (plus the depot);
//   - each vertical corridor is a chain of nodes (one per aisle row and per
//     junction) linked both ways by their distance; corridor → entry and
//     exit → corridor arcs join the aisles to it;
//   - the front and back corridors link the three vertical corridors;
//   - the depot (middle cart zone, x 89–93, y 30–50) is joined to the right
//     corridor at y = 40.
//
// Costs are walking distances in meters (Manhattan along the corridors).
// Assumptions to check on site: corridor positions (1 m from the rack ends),
// the front/back corridors, the depot position, and that 4P01 must also be
// walked entirely in its direction.

import (
	"math"
	"sort"
	"strings"
)

// geoNode: position (meters) and label (aisle code for aisle nodes).
type geoNode struct {
	x, y  float64
	label string
	kind  int // geoCorridor, geoEntry, geoMid, geoExit, geoDepot
}

const (
	geoCorridor = iota
	geoEntry
	geoMid
	geoExit
	geoDepot
)

// geoGraph: nodes, directed arcs, the middle node of each aisle (by code)
// and the depot node.
type geoGraph struct {
	nodes []geoNode
	arcs  []arc
	mid   map[string]int
	depot int
}

// evreuxTraversal: traversal_direction of the aisles of the block (Metabase).
var evreuxTraversal = map[string]string{
	"4P01": "END_TO_START",
}

func init() {
	// 3E aisles: the direction alternates every row (serpentine), the same
	// for the A and B halves of a row. Values from atlas.warehouse_aisles.
	rows := map[string]string{
		"01": "S", "02_03": "E", "04_05": "S", "06_07": "E", "08_09": "S", "10_11": "E",
		"12_13": "S", "14_15": "E", "16_17": "S", "18_19": "S", "20_21": "S", "22_23": "E",
		"24_25": "S", "26_27": "E", "28_29": "S", "30_31": "E", "32_33": "S", "34_35": "E",
		"36_37": "S", "38_39": "E", "40_41": "S", "42_43": "E", "44_45": "S", "46_47": "E",
		"48_49": "S", "50": "E",
	}
	for r, d := range rows {
		td := "START_TO_END"
		if d == "E" {
			td = "END_TO_START"
		}
		evreuxTraversal["3E"+r+"_A"] = td
		evreuxTraversal["3E"+r+"_B"] = td
	}
}

// buildEvreuxGeo builds the side-aware graph of the 3E block and 4P01.
func buildEvreuxGeo() *geoGraph {
	const (
		xL, xM, xR = 60.5, 71.8, 82.6 // vertical corridors
		yF, yB     = 20.5, 96.5       // front and back corridors
	)
	g := &geoGraph{mid: map[string]int{}}
	add := func(x, y float64, label string, kind int) int {
		g.nodes = append(g.nodes, geoNode{x, y, label, kind})
		return len(g.nodes) - 1
	}
	arcTo := func(u, v int) {
		c := math.Abs(g.nodes[u].x-g.nodes[v].x) + math.Abs(g.nodes[u].y-g.nodes[v].y)
		g.arcs = append(g.arcs, arc{u, v, c})
	}
	// Corridor nodes, created on demand at (column x, y).
	cols := map[float64]map[float64]int{xL: {}, xM: {}, xR: {}}
	corr := func(x, y float64) int {
		y = math.Round(y*100) / 100
		if v, ok := cols[x][y]; ok {
			return v
		}
		v := add(x, y, "", geoCorridor)
		cols[x][y] = v
		return v
	}
	nearestCol := func(x float64) float64 {
		best := xL
		for _, c := range []float64{xL, xM, xR} {
			if math.Abs(c-x) < math.Abs(best-x) {
				best = c
			}
		}
		return best
	}

	// Aisles: entry → middle → exit, joined to the nearest corridor.
	for _, a := range evreuxPlan {
		td, ok := evreuxTraversal[a.code]
		if !ok || !(strings.HasPrefix(a.code, "3E") || a.code == "4P01") {
			continue
		}
		x0, y0, x1, y1 := a.sx, a.sy, a.ex, a.ey
		if td == "END_TO_START" {
			x0, y0, x1, y1 = x1, y1, x0, y0
		}
		in := add(x0, y0, a.code, geoEntry)
		mid := add((x0+x1)/2, (y0+y1)/2, a.code, geoMid)
		out := add(x1, y1, a.code, geoExit)
		arcTo(in, mid)
		arcTo(mid, out)
		g.mid[a.code] = mid
		arcTo(corr(nearestCol(x0), y0), in)  // enter from the corridor
		arcTo(out, corr(nearestCol(x1), y1)) // leave to the corridor
	}
	// Front and back corridors, depot junction.
	for _, x := range []float64{xL, xM, xR} {
		corr(x, yF)
		corr(x, yB)
	}
	g.depot = add(91, 40, "depot", geoDepot)
	j := corr(xR, 40)
	arcTo(g.depot, j)
	arcTo(j, g.depot)
	// Vertical corridors: chains, both ways.
	for _, x := range []float64{xL, xM, xR} {
		m := cols[x]
		ys := make([]float64, 0, len(m))
		for y := range m {
			ys = append(ys, y)
		}
		sort.Float64s(ys)
		for i := 0; i+1 < len(ys); i++ {
			arcTo(m[ys[i]], m[ys[i+1]])
			arcTo(m[ys[i+1]], m[ys[i]])
		}
	}
	// Front and back corridors: between neighbouring columns, both ways.
	for _, y := range []float64{yF, yB} {
		for _, p := range [][2]float64{{xL, xM}, {xM, xR}} {
			u, v := corr(p[0], y), corr(p[1], y)
			arcTo(u, v)
			arcTo(v, u)
		}
	}
	return g
}
