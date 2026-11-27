package main

// Drawing the network and the tours as a self-contained HTML page (SVG).
//
// The data gives no coordinates, only travel costs between nodes, so a
// layout is computed from the costs: find 2-D points whose Euclidean
// distances are as close as possible to the (symmetrised) shortest-path
// distances. Nodes that are close in walking time end up close on the
// picture: it is a "travel-time map", not the exact floor plan.
//
// Two classic steps of graph drawing:
//  1. classical multidimensional scaling (MDS, Torgerson 1952) gives a
//     global starting picture;
//  2. stress majorization (Gansner, Koren and North, 2004) refines it by
//     minimising Σ w_ij (‖x_i − x_j‖ − d_ij)² with w_ij = 1/d_ij², which
//     favours the local structure (neighbouring aisles) over far pairs.

import (
	"fmt"
	"html"
	"math"
	"os"
	"sort"
	"strings"
)

// tourView is one pick list to draw: its required nodes, the optimal walk
// and the walk in the recorded order, with their lengths.
type tourView struct {
	name           string
	req            []int
	opt, actual    []int
	optLen, actLen float64
}

// graphLayout returns 2-D coordinates of the n nodes from the
// shortest-path distances d: classical MDS, then stress majorization.
func graphLayout(n int, d [][]float64) [][2]float64 {
	x := mdsLayout(n, d)
	// Target distances and weights. Pairs connected in neither direction
	// (the network is not strongly connected) get a weak weight towards a
	// large distance, so that they stay apart without distorting the rest.
	big := 0.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if !math.IsInf(d[i][j], 1) {
				big = math.Max(big, d[i][j])
			}
		}
	}
	T := make([][]float64, n)  // target distance
	Wt := make([][]float64, n) // weight
	for i := range T {
		T[i], Wt[i] = make([]float64, n), make([]float64, n)
		for j := range T[i] {
			if i == j {
				continue
			}
			v, ok := symDist(d, i, j)
			if !ok {
				T[i][j], Wt[i][j] = big, 0.1/(big*big)
				continue
			}
			v = math.Max(v, 1)
			T[i][j], Wt[i][j] = v, 1/(v*v)
		}
	}
	// Each sweep moves every node to the weighted average of the positions
	// that would put it at the right distance from each other node (the
	// localized majorization update).
	for sweep := 0; sweep < 300; sweep++ {
		for i := 0; i < n; i++ {
			var sx, sy, sw float64
			for j := 0; j < n; j++ {
				if j == i || Wt[i][j] == 0 {
					continue
				}
				dx, dy := x[i][0]-x[j][0], x[i][1]-x[j][1]
				l := math.Hypot(dx, dy)
				if l < 1e-9 {
					dx, dy, l = 1e-3*float64(i-j), 1e-3, math.Hypot(1e-3*float64(i-j), 1e-3)
				}
				w := Wt[i][j]
				sx += w * (x[j][0] + T[i][j]*dx/l)
				sy += w * (x[j][1] + T[i][j]*dy/l)
				sw += w
			}
			x[i] = [2]float64{sx / sw, sy / sw}
		}
	}
	return x
}

// depotPosition returns the mean position of the k nodes reached most
// cheaply from the depot.
func depotPosition(coords [][2]float64, arcs []arc, k int) [2]float64 {
	var out []arc
	for _, e := range arcs {
		if e.from == Depot {
			out = append(out, e)
		}
	}
	sort.Slice(out, func(a, b int) bool { return out[a].cost < out[b].cost })
	var p [2]float64
	k = min(k, len(out))
	for _, e := range out[:k] {
		p[0] += coords[e.to][0] / float64(k)
		p[1] += coords[e.to][1] / float64(k)
	}
	return p
}

// symDist is the distance used for drawing between i and j: the average of
// both directions, or the only finite one; ok is false if neither is.
func symDist(d [][]float64, i, j int) (float64, bool) {
	a, b := d[i][j], d[j][i]
	switch {
	case !math.IsInf(a, 1) && !math.IsInf(b, 1):
		return (a + b) / 2, true
	case !math.IsInf(a, 1):
		return a, true
	case !math.IsInf(b, 1):
		return b, true
	}
	return 0, false
}

// mdsLayout returns 2-D coordinates of the n nodes from the shortest-path
// distances d (classical MDS).
func mdsLayout(n int, d [][]float64) [][2]float64 {
	// Symmetrised distances (see symDist); pairs connected in neither
	// direction get the largest finite distance.
	big := 0.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if !math.IsInf(d[i][j], 1) {
				big = math.Max(big, d[i][j])
			}
		}
	}
	D2 := make([][]float64, n) // squared distances
	for i := range D2 {
		D2[i] = make([]float64, n)
		for j := range D2[i] {
			v, ok := symDist(d, i, j)
			if !ok {
				v = big
			}
			D2[i][j] = v * v
		}
	}
	// Double centring: B = −½·J·D²·J with J = I − 11ᵀ/n. B is the Gram
	// matrix of centred points whose distances are D.
	row := make([]float64, n)
	all := 0.0
	for i := range D2 {
		for j := range D2[i] {
			row[i] += D2[i][j]
		}
		all += row[i]
		row[i] /= float64(n)
	}
	all /= float64(n * n)
	B := make([][]float64, n)
	for i := range B {
		B[i] = make([]float64, n)
		for j := range B[i] {
			B[i][j] = -0.5 * (D2[i][j] - row[i] - row[j] + all)
		}
	}
	// The two leading eigenvectors of B (power iteration with deflation)
	// scaled by √eigenvalue are the coordinates.
	coords := make([][2]float64, n)
	var prev []float64
	for k := 0; k < 2; k++ {
		v := make([]float64, n)
		for i := range v {
			v[i] = math.Sin(float64(i*(k+1)) + 1) // deterministic start
		}
		lambda := 0.0
		for it := 0; it < 500; it++ {
			w := make([]float64, n)
			for i := range B {
				for j, b := range B[i] {
					w[i] += b * v[j]
				}
			}
			if prev != nil { // deflation: stay orthogonal to the first vector
				dot := 0.0
				for i := range w {
					dot += w[i] * prev[i]
				}
				for i := range w {
					w[i] -= dot * prev[i]
				}
			}
			norm := 0.0
			for _, x := range w {
				norm += x * x
			}
			norm = math.Sqrt(norm)
			lambda = norm
			for i := range w {
				v[i] = w[i] / norm
			}
		}
		for i := range v {
			coords[i][k] = v[i] * math.Sqrt(lambda)
		}
		prev = v
	}
	return coords
}

// writeGraphHTML writes the page: all arcs in light grey, the nodes (with
// their aisle code when known), and for each pick list the optimal walk
// (green) and the walk in the recorded order (orange), selectable.
func writeGraphHTML(path string, n int, arcs []arc, coords [][2]float64, labels map[int]string, tours []tourView) error {
	// Scale coordinates to the drawing area.
	const W, H, pad = 1100.0, 700.0, 30.0
	minX, maxX, minY, maxY := math.Inf(1), math.Inf(-1), math.Inf(1), math.Inf(-1)
	for _, c := range coords {
		minX, maxX = math.Min(minX, c[0]), math.Max(maxX, c[0])
		minY, maxY = math.Min(minY, c[1]), math.Max(maxY, c[1])
	}
	sc := math.Min((W-2*pad)/(maxX-minX), (H-2*pad)/(maxY-minY))
	px := func(v int) (float64, float64) {
		return pad + (coords[v][0]-minX)*sc, pad + (coords[v][1]-minY)*sc
	}

	var b strings.Builder
	b.WriteString(`<!DOCTYPE html><html><head><meta charset="utf-8"><title>Evreux network and pick lists</title>
<style>
body{font-family:system-ui,sans-serif;margin:16px;color:#222}
svg{border:1px solid #ddd;background:#fff}
.arc{stroke:#d6d6d6;stroke-width:.6}
.node{fill:#9aa;stroke:none}
.req{fill:#c62828}
.depot{fill:#111}
.opt{stroke:#2e7d32;stroke-width:2.6;fill:none;marker-end:url(#ag)}
.act{stroke:#ef6c00;stroke-width:2;stroke-dasharray:6 4;fill:none;marker-end:url(#ao)}
.lbl{font-size:10px;fill:#333}
.hide{display:none}
label{margin-right:14px}
#info{margin:8px 0;font-size:14px}
</style></head><body>
<h2>Evreux warehouse network</h2>
<p>Layout computed from walking costs (MDS + stress majorization): close nodes are close in travel cost, the picture is not the exact floor plan.
Hover a node for its id. Red: aisles of the pick list; black: depot (cart zone, node 223).
The depot merges several cart points and is linked to most aisles: it is drawn near the aisles it reaches most cheaply, and its arcs are hidden by default.
Arcs are drawn without direction.</p>
<div>`)
	for t, tv := range tours {
		checked := ""
		if t == 0 {
			checked = " checked"
		}
		fmt.Fprintf(&b, `<label><input type="radio" name="tour" value="%d"%s onchange="show(%d)"> %s</label>`, t, checked, t, html.EscapeString(tv.name))
	}
	b.WriteString(`<br><label><input type="checkbox" id="cOpt" checked onchange="upd()"> optimal tour (green)</label>
<label><input type="checkbox" id="cAct" checked onchange="upd()"> recorded order (orange, dashed)</label>
<label><input type="checkbox" id="cArc" checked onchange="upd()"> arcs between aisles</label>
<label><input type="checkbox" id="cDep" onchange="upd()"> depot arcs</label>
<button onclick="resetView()">reset zoom</button> <small>(mouse wheel: zoom, drag: move)</small></div><div id="info"></div>`)
	fmt.Fprintf(&b, `<svg id="g" width="%.0f" height="%.0f" viewBox="0 0 %.0f %.0f">
<defs>
<marker id="ag" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0L10,5L0,10z" fill="#2e7d32"/></marker>
<marker id="ao" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0L10,5L0,10z" fill="#ef6c00"/></marker>
</defs><g id="arcs">`, W, H, W, H)
	// Arcs, drawn once per pair of nodes; the depot's arcs (it is linked to
	// most aisles) go to a separate layer, hidden by default.
	seen := map[[2]int]bool{}
	var depotArcs strings.Builder
	for _, e := range arcs {
		k := [2]int{min(e.from, e.to), max(e.from, e.to)}
		if seen[k] {
			continue
		}
		seen[k] = true
		x1, y1 := px(e.from)
		x2, y2 := px(e.to)
		line := fmt.Sprintf(`<line class="arc" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>`, x1, y1, x2, y2)
		if e.from == Depot || e.to == Depot {
			depotArcs.WriteString(line)
		} else {
			b.WriteString(line)
		}
	}
	b.WriteString(`</g><g id="darcs" class="hide">` + depotArcs.String() + `</g>`)
	// Tours: one group per pick list (shown one at a time).
	for t, tv := range tours {
		fmt.Fprintf(&b, `<g id="t%d" class="tour%s">`, t, map[bool]string{true: "", false: " hide"}[t == 0])
		for _, w := range []struct {
			cls  string
			walk []int
		}{{"act", tv.actual}, {"opt", tv.opt}} {
			fmt.Fprintf(&b, `<g class="%sG">`, w.cls)
			for i := 0; i+1 < len(w.walk); i++ {
				x1, y1 := px(w.walk[i])
				x2, y2 := px(w.walk[i+1])
				// Shorten the segment so that the arrow head stays visible
				// next to the node.
				dx, dy := x2-x1, y2-y1
				l := math.Hypot(dx, dy)
				if l < 1e-9 {
					continue
				}
				f := math.Max(0, (l-6)/l)
				fmt.Fprintf(&b, `<line class="%s" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>`, w.cls, x1, y1, x1+dx*f, y1+dy*f)
			}
			b.WriteString(`</g>`)
		}
		b.WriteString(`</g>`)
	}
	// Nodes on top, with tooltips; required ones get their aisle code.
	reqOf := make([]map[int]bool, len(tours))
	for t, tv := range tours {
		reqOf[t] = map[int]bool{}
		for _, v := range tv.req {
			reqOf[t][v] = true
		}
	}
	for v := 0; v < n; v++ {
		x, y := px(v)
		title := fmt.Sprintf("node %d", v)
		if l, ok := labels[v]; ok {
			title += " — " + l
		}
		if v == Depot {
			fmt.Fprintf(&b, `<rect class="depot" x="%.1f" y="%.1f" width="12" height="12"><title>%s (depot)</title></rect>`, x-6, y-6, title)
			fmt.Fprintf(&b, `<text class="lbl" x="%.1f" y="%.1f">depot</text>`, x+8, y-8)
			continue
		}
		var reqIn []string
		for t := range tours {
			if reqOf[t][v] {
				reqIn = append(reqIn, fmt.Sprint(t))
			}
		}
		fmt.Fprintf(&b, `<circle class="node" data-req="%s" cx="%.1f" cy="%.1f" r="2.6"><title>%s</title></circle>`,
			strings.Join(reqIn, ","), x, y, html.EscapeString(title))
		if len(reqIn) > 0 {
			l := fmt.Sprint(v)
			if c, ok := labels[v]; ok {
				l += " " + c
			}
			fmt.Fprintf(&b, `<text class="lbl" data-req="%s" x="%.1f" y="%.1f">%s</text>`, strings.Join(reqIn, ","), x+5, y-5, html.EscapeString(l))
		}
	}
	b.WriteString(`</svg><script>
const tours = [`)
	for _, tv := range tours {
		fmt.Fprintf(&b, `{name:%q,opt:%.0f,act:%.0f},`, tv.name, tv.optLen, tv.actLen)
	}
	b.WriteString(`];
let cur = 0;
function show(t){ cur = t; upd(); }
function upd(){
  document.querySelectorAll('.tour').forEach((g,i)=>g.classList.toggle('hide', i!==cur));
  document.querySelectorAll('.optG').forEach(g=>g.classList.toggle('hide', !document.getElementById('cOpt').checked));
  document.querySelectorAll('.actG').forEach(g=>g.classList.toggle('hide', !document.getElementById('cAct').checked));
  document.getElementById('arcs').classList.toggle('hide', !document.getElementById('cArc').checked);
  document.getElementById('darcs').classList.toggle('hide', !document.getElementById('cDep').checked);
  document.querySelectorAll('[data-req]').forEach(e=>{
    const r = e.dataset.req ? e.dataset.req.split(',').map(Number) : [];
    const on = r.includes(cur);
    if (e.tagName==='circle'){ e.classList.toggle('req', on); e.setAttribute('r', on?5:2.6); }
    else e.classList.toggle('hide', !on);
  });
  const t = tours[cur];
  document.getElementById('info').textContent = t.name + ': optimal tour ' + t.opt + ', recorded order ' + t.act +
    ' (saving ' + (t.act-t.opt) + ', ' + (100*(t.act-t.opt)/t.act).toFixed(1) + '%)';
}
upd();
// Zoom with the mouse wheel and pan by dragging (changes the viewBox).
const svg = document.getElementById('g');
const vb0 = svg.getAttribute('viewBox').split(' ').map(Number);
let vb = vb0.slice();
function setVB(){ svg.setAttribute('viewBox', vb.join(' ')); }
function resetView(){ vb = vb0.slice(); setVB(); }
svg.addEventListener('wheel', e => {
  e.preventDefault();
  const r = svg.getBoundingClientRect();
  const mx = vb[0] + (e.clientX - r.left) / r.width * vb[2];
  const my = vb[1] + (e.clientY - r.top) / r.height * vb[3];
  const f = e.deltaY < 0 ? 0.8 : 1.25;
  vb = [mx - (mx - vb[0]) * f, my - (my - vb[1]) * f, vb[2] * f, vb[3] * f];
  setVB();
}, {passive: false});
let drag = null;
svg.addEventListener('mousedown', e => { drag = [e.clientX, e.clientY, vb[0], vb[1]]; });
window.addEventListener('mouseup', () => { drag = null; });
window.addEventListener('mousemove', e => {
  if (!drag) return;
  const r = svg.getBoundingClientRect();
  vb[0] = drag[2] - (e.clientX - drag[0]) / r.width * vb[2];
  vb[1] = drag[3] - (e.clientY - drag[1]) / r.height * vb[3];
  setVB();
});
</script></body></html>`)
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

// writePlanHTML draws the tours of the side-aware model (evreux_geo.go) on
// the real floor plan (evreux_plan.go): every aisle as a segment coloured
// by zone, the corridors of the model in light grey, the one-way aisles of
// the 3E block with an arrow giving their direction, the cart zones, and for
// each pick list the optimal tour and the recorded order, drawn node by node
// along the corridors.
func writePlanHTML(path string, g *geoGraph, tours []tourView) error {
	minX, maxX, minY, maxY := math.Inf(1), math.Inf(-1), math.Inf(1), math.Inf(-1)
	for _, a := range evreuxPlan {
		minX, maxX = math.Min(minX, math.Min(a.sx, a.ex)), math.Max(maxX, math.Max(a.sx, a.ex))
		minY, maxY = math.Min(minY, math.Min(a.sy, a.ey)), math.Max(maxY, math.Max(a.sy, a.ey))
	}
	// Screen: 1 m = 6 px, y axis pointing up.
	const s, pad = 6.0, 20.0
	W, H := (maxX-minX)*s+2*pad, (maxY-minY)*s+2*pad
	px := func(x, y float64) (float64, float64) { return pad + (x-minX)*s, pad + (maxY-y)*s }
	zoneColor := map[string]string{"c1_cava": "#8d6e63", "c1_aigle": "#a1887f", "c1_toz": "#bcaaa4",
		"c2": "#5c6bc0", "c3": "#9e9e9e", "c4": "#26a69a", "c5": "#ab47bc", "c5bis": "#ce93d8", "c6_meca": "#ce93d8"}

	var b strings.Builder
	b.WriteString(`<!DOCTYPE html><html><head><meta charset="utf-8"><title>Evreux floor plan and pick lists</title>
<style>
body{font-family:system-ui,sans-serif;margin:16px;color:#222}
svg{border:1px solid #ddd;background:#fff}
.opt{stroke:#2e7d32;stroke-width:3;fill:none;stroke-linejoin:round;marker-end:url(#ag)}
polyline.opt,polyline.act{marker-end:none}
.act{stroke:#ef6c00;stroke-width:2.2;stroke-dasharray:7 5;fill:none;stroke-linejoin:round;marker-end:url(#ao)}
.cor{stroke:#d0d0d0;stroke-width:1}
.lbl{font-size:10px;fill:#222}
.zl{font-size:16px;fill:#555;font-weight:600}
.hide{display:none}
label{margin-right:14px}
</style></head><body>
<h2>Evreux warehouse (EU-FRA-002): tours on the real floor plan</h2>
<p>Aisles from Metabase (atlas.warehouse_aisles), in meters, coloured by zone. Side-aware model of the 3E block (evreux_geo.go):
each aisle is one-way (small arrow, Metabase traversal direction) and is entered at one end and left at the other;
moves between aisles follow the corridors (light grey). Distances in meters. Black square: depot (middle cart zone).
Mouse wheel: zoom, drag: move.</p><div>`)
	for t, tv := range tours {
		checked := ""
		if t == 0 {
			checked = " checked"
		}
		fmt.Fprintf(&b, `<label><input type="radio" name="tour" value="%d"%s onchange="show(%d)"> %s</label>`, t, checked, t, html.EscapeString(tv.name))
	}
	b.WriteString(`<br><label><input type="checkbox" id="cOpt" checked onchange="upd()"> optimal tour (green)</label>
<label><input type="checkbox" id="cAct" checked onchange="upd()"> recorded order (orange, dashed)</label>
<button onclick="resetView()">reset zoom</button></div><div id="info"></div>`)
	fmt.Fprintf(&b, `<svg id="g" width="%.0f" height="%.0f" viewBox="0 0 %.0f %.0f">
<defs>
<marker id="ag" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0L10,5L0,10z" fill="#2e7d32"/></marker>
<marker id="ao" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0L10,5L0,10z" fill="#ef6c00"/></marker>
<marker id="ad" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10z" fill="#00695c"/></marker>
</defs>`, W, H, W, H)
	// Cart zones.
	for _, z := range evreuxCartZones {
		x0, y0 := px(z[0], z[3])
		x1, y1 := px(z[2], z[1])
		fmt.Fprintf(&b, `<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="#eee" stroke="#999"><title>cart zone</title></rect>`, x0, y0, x1-x0, y1-y0)
		fmt.Fprintf(&b, `<text class="lbl" x="%.1f" y="%.1f">carts</text>`, x0+2, y0+12)
	}
	// Aisles, and one label per zone.
	type acc struct{ x, y, n float64 }
	zc := map[string]*acc{}
	for _, a := range evreuxPlan {
		x1, y1 := px(a.sx, a.sy)
		x2, y2 := px(a.ex, a.ey)
		col := zoneColor[a.zone]
		if col == "" {
			col = "#bbb"
		}
		if x1 == x2 && y1 == y2 {
			fmt.Fprintf(&b, `<circle cx="%.1f" cy="%.1f" r="2" fill="%s"><title>%s</title></circle>`, x1, y1, col, a.code)
		} else {
			fmt.Fprintf(&b, `<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="2.5" stroke-linecap="round" opacity=".55"><title>%s (%s)</title></line>`, x1, y1, x2, y2, col, a.code, a.zone)
		}
		if zc[a.zone] == nil {
			zc[a.zone] = &acc{}
		}
		zc[a.zone].x += (x1 + x2) / 2
		zc[a.zone].y += (y1 + y2) / 2
		zc[a.zone].n++
	}
	zones := make([]string, 0, len(zc))
	for z := range zc {
		zones = append(zones, z)
	}
	sort.Strings(zones)
	for _, z := range zones {
		c := zc[z]
		fmt.Fprintf(&b, `<text class="zl" x="%.1f" y="%.1f" text-anchor="middle" opacity=".5">%s</text>`, c.x/c.n, c.y/c.n, z)
	}
	// Model: corridors (light grey) and the direction of each aisle (arrow
	// at its middle).
	for _, e := range g.arcs {
		u, v := g.nodes[e.from], g.nodes[e.to]
		x1, y1 := px(u.x, u.y)
		x2, y2 := px(v.x, v.y)
		switch {
		case u.kind == geoMid: // second half of an aisle: no extra drawing
		case v.kind == geoMid: // first half: direction arrow
			fmt.Fprintf(&b, `<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="none" marker-end="url(#ad)"/>`, x1, y1, x2, y2)
		default:
			fmt.Fprintf(&b, `<line class="cor" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>`, x1, y1, x2, y2)
		}
	}
	// Tours.
	for t, tv := range tours {
		fmt.Fprintf(&b, `<g class="tour%s">`, map[bool]string{true: "", false: " hide"}[t == 0])
		for k, w := range []struct {
			cls  string
			walk []int
		}{{"act", tv.actual}, {"opt", tv.opt}} {
			fmt.Fprintf(&b, `<g class="%sG">`, w.cls)
			// Small offset between the two tours so that both stay visible
			// on shared corridors.
			off := float64(k)*3 - 1.5
			// The walk as one polyline; arrows only where it enters an
			// aisle (towards the aisle middle), to show the direction
			// without cluttering the corridors.
			var pts []string
			for _, v := range w.walk {
				x, y := px(g.nodes[v].x, g.nodes[v].y)
				pts = append(pts, fmt.Sprintf("%.1f,%.1f", x+off, y+off))
			}
			fmt.Fprintf(&b, `<polyline class="%s" points="%s" marker-end="none"/>`, w.cls, strings.Join(pts, " "))
			for i := 0; i+1 < len(w.walk); i++ {
				u, v := g.nodes[w.walk[i]], g.nodes[w.walk[i+1]]
				if v.kind != geoMid {
					continue
				}
				x1, y1 := px(u.x, u.y)
				x2, y2 := px(v.x, v.y)
				fmt.Fprintf(&b, `<line class="%s" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>`, w.cls, x1+off, y1+off, x2+off, y2+off)
			}
			b.WriteString(`</g>`)
		}
		for _, v := range tv.req {
			nd := g.nodes[v]
			x, y := px(nd.x, nd.y)
			if nd.kind == geoDepot {
				fmt.Fprintf(&b, `<rect x="%.1f" y="%.1f" width="12" height="12" fill="#111"><title>depot</title></rect>`, x-6, y-6)
				continue
			}
			fmt.Fprintf(&b, `<circle cx="%.1f" cy="%.1f" r="5" fill="#c62828"><title>%s</title></circle>`, x, y, nd.label)
			fmt.Fprintf(&b, `<text class="lbl" x="%.1f" y="%.1f">%s</text>`, x+6, y-6, nd.label)
		}
		b.WriteString(`</g>`)
	}
	b.WriteString(`</svg><script>
const tours = [`)
	for _, tv := range tours {
		fmt.Fprintf(&b, `{name:%q,opt:%.0f,act:%.0f},`, tv.name, tv.optLen, tv.actLen)
	}
	b.WriteString(`];
let cur = 0;
function show(t){ cur = t; upd(); }
function upd(){
  document.querySelectorAll('.tour').forEach((g,i)=>g.classList.toggle('hide', i!==cur));
  document.querySelectorAll('.optG').forEach(g=>g.classList.toggle('hide', !document.getElementById('cOpt').checked));
  document.querySelectorAll('.actG').forEach(g=>g.classList.toggle('hide', !document.getElementById('cAct').checked));
  const t = tours[cur];
  document.getElementById('info').textContent = t.name + ': optimal tour ' + t.opt + ' m, recorded order ' + t.act +
    ' m (saving ' + (t.act-t.opt) + ' m, ' + (100*(t.act-t.opt)/t.act).toFixed(1) + '%)';
}
upd();
const svg = document.getElementById('g');
const vb0 = svg.getAttribute('viewBox').split(' ').map(Number);
let vb = vb0.slice();
function setVB(){ svg.setAttribute('viewBox', vb.join(' ')); }
function resetView(){ vb = vb0.slice(); setVB(); }
svg.addEventListener('wheel', e => {
  e.preventDefault();
  const r = svg.getBoundingClientRect();
  const mx = vb[0] + (e.clientX - r.left) / r.width * vb[2];
  const my = vb[1] + (e.clientY - r.top) / r.height * vb[3];
  const f = e.deltaY < 0 ? 0.8 : 1.25;
  vb = [mx - (mx - vb[0]) * f, my - (my - vb[1]) * f, vb[2] * f, vb[3] * f];
  setVB();
}, {passive: false});
let drag = null;
svg.addEventListener('mousedown', e => { drag = [e.clientX, e.clientY, vb[0], vb[1]]; });
window.addEventListener('mouseup', () => { drag = null; });
window.addEventListener('mousemove', e => {
  if (!drag) return;
  const r = svg.getBoundingClientRect();
  vb[0] = drag[2] - (e.clientX - drag[0]) / r.width * vb[2];
  vb[1] = drag[3] - (e.clientY - drag[1]) / r.height * vb[3];
  setVB();
});
// Start zoomed on the 3E block.
`)
	zx0, zy0 := px(50, 100)
	zx1, zy1 := px(100, 15)
	fmt.Fprintf(&b, "vb = [%.1f, %.1f, %.1f, %.1f]; setVB();\n</script></body></html>", zx0, zy0, zx1-zx0, zy1-zy0)
	return os.WriteFile(path, []byte(b.String()), 0o644)
}
