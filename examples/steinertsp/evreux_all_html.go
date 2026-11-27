package main

// Floor plan of the whole Evreux warehouse (grid model of evreux_all.go)
// with the pick lists whose walk improves the most: recorded walk and
// optimal walk drawn cell by cell, visit numbers on the aisles, and an
// animation that runs both pickers at the same speed.

import (
	"fmt"
	"html"
	"math"
	"os"
	"sort"
	"strings"
)

// pickShowcase chooses the lists to draw: the 3 largest savings (meters),
// then the best list of every other zone that has a saving.
func pickShowcase(results []listResult) []listResult {
	var ok []listResult
	for _, r := range results {
		if r.status == "ok" && r.recorded-r.optimal > 0.05 {
			ok = append(ok, r)
		}
	}
	sort.SliceStable(ok, func(i, j int) bool { return ok[i].recorded-ok[i].optimal > ok[j].recorded-ok[j].optimal })
	var out []listResult
	seen := map[string]bool{}
	for _, r := range ok {
		if len(out) < 3 {
			out = append(out, r)
			seen[r.zone] = true
		}
	}
	for _, r := range ok { // ok is sorted: the first list met in a zone is its best
		if !seen[r.zone] {
			out = append(out, r)
			seen[r.zone] = true
		}
	}
	return out
}

// walkPoints rebuilds the geometry of a walk: from the depot, for every
// aisle in order, the shortest path to its entry port then the aisle
// itself, and finally back to the depot. Returns the points (meters) and
// the length. cache keeps the shortest-path trees already computed.
func walkPoints(w *warehouse, depot int, codes []string, cache map[int][]int32) ([][2]float64, float64) {
	cur := w.depots[depot]
	x0, y0 := w.xy(cur)
	pts := [][2]float64{{x0, y0}}
	goTo := func(target int) {
		pred, ok := cache[cur]
		if !ok {
			_, pred = w.dijkstraPred(cur)
			cache[cur] = pred
		}
		var path []int
		for c := target; c != cur && c >= 0; c = int(pred[c]) {
			path = append(path, c)
		}
		for i := len(path) - 1; i >= 0; i-- {
			x, y := w.xy(path[i])
			pts = append(pts, [2]float64{x, y})
		}
		cur = target
	}
	for _, code := range codes {
		a := w.aisles[w.index[code]]
		goTo(a.entry)
		if a.exit != a.entry { // traverse the aisle along its axis
			x, y := w.xy(a.exit)
			pts = append(pts, [2]float64{a.px, a.py}, [2]float64{a.qx, a.qy}, [2]float64{x, y})
			cur = a.exit
		}
	}
	goTo(w.depots[depot])
	// Drop the points in the middle of straight runs.
	out := [][2]float64{pts[0]}
	for i := 1; i+1 < len(pts); i++ {
		p, q, r := out[len(out)-1], pts[i], pts[i+1]
		cross := (q[0]-p[0])*(r[1]-q[1]) - (q[1]-p[1])*(r[0]-q[0])
		dot := (q[0]-p[0])*(r[0]-q[0]) + (q[1]-p[1])*(r[1]-q[1])
		if math.Abs(cross) < 1e-9 && dot > 0 {
			continue
		}
		out = append(out, q)
	}
	out = append(out, pts[len(pts)-1])
	length := 0.0
	for i := 1; i < len(out); i++ {
		length += math.Hypot(out[i][0]-out[i-1][0], out[i][1]-out[i-1][1])
	}
	return out, length
}

// writeAllPlanHTML writes the floor plan with the lists of views.
func writeAllPlanHTML(path string, w *warehouse, views []listResult) error {
	const s, pad = 5.0, 15.0 // 1 m = 5 px
	minX, maxX, minY, maxY := -2.0, 300.0, -2.0, 106.0
	W, H := (maxX-minX)*s+2*pad, (maxY-minY)*s+2*pad
	px := func(x, y float64) (float64, float64) { return pad + (x-minX)*s, pad + (maxY-y)*s }
	zoneColor := map[string]string{"c1_cava": "#8d6e63", "c1_aigle": "#a1887f", "c1_toz": "#bcaaa4",
		"c2": "#5c6bc0", "c3": "#9e9e9e", "c4": "#26a69a", "c5": "#ab47bc", "c5bis": "#ce93d8", "c6_meca": "#ce93d8"}

	var b strings.Builder
	b.WriteString(`<!DOCTYPE html><html><head><meta charset="utf-8"><title>Evreux: most improved pick lists</title>
<style>
body{font-family:system-ui,sans-serif;margin:16px;color:#222}
svg{border:1px solid #ddd;background:#fff;cursor:grab}
.opt{stroke:#2e7d32;stroke-width:2.6;fill:none;stroke-linejoin:round;opacity:.9}
.act{stroke:#ef6c00;stroke-width:2;stroke-dasharray:6 4;fill:none;stroke-linejoin:round;opacity:.9}
.nopt,.nact{font-size:7px;fill:#fff;font-weight:700}
.zl{font-size:15px;fill:#555;font-weight:600}
.hide{display:none}
label{margin-right:14px}
select,button{font-size:14px;margin-right:8px}
#info{margin:8px 0;font-size:15px}
#anim{font-variant-numeric:tabular-nums}
</style></head><body>
<h2>Evreux (EU-FRA-002): the pick lists that gain the most, week of 2026-11-16</h2>
<p>Whole-warehouse grid model (evreux_all.go): racks in grey, aisles coloured by zone with their one-way direction (arrow),
an aisle is always walked end to end. Orange dashed: the walk actually done (aisles in the recorded order); green: the optimal walk
visiting the same aisles. Numbers: visit order (orange = recorded, green = optimal). Black square: cart zone (start and end).
Mouse wheel: zoom, drag: move. <b>Animate</b> runs both pickers at the same speed.</p>
<div><select id="sel" onchange="show(+this.value)">`)
	for i, r := range views {
		fmt.Fprintf(&b, `<option value="%d">%s (%s, %d aisles): %.0f → %.0f m (−%.0f%%)</option>`, i, html.EscapeString(r.rec.id), r.zone,
			r.distinct, r.recorded, r.optimal, 100*(r.recorded-r.optimal)/r.recorded)
	}
	b.WriteString(`</select>
<label><input type="checkbox" id="cAct" checked onchange="upd()"> recorded</label>
<label><input type="checkbox" id="cOpt" checked onchange="upd()"> optimal</label>
<button onclick="runAnim()">▶ Animate</button><button onclick="fit()">zoom on the list</button><button onclick="resetView()">whole warehouse</button>
<span id="anim"></span></div><div id="info"></div>`)
	fmt.Fprintf(&b, `<svg id="g" width="100%%" data-w="%.0f" data-h="%.0f" viewBox="0 0 %.0f %.0f">
<defs><marker id="ad" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto"><path d="M0,0L10,5L0,10z" fill="#555"/></marker></defs>`, W, H, W, H)

	// Racks: blocked cells, merged into horizontal runs.
	b.WriteString(`<g fill="#e3e3e3">`)
	for y := 0; y < w.ny; y++ {
		for x := 0; x < w.nx; {
			if !w.blocked[y*w.nx+x] {
				x++
				continue
			}
			x1 := x
			for x < w.nx && w.blocked[y*w.nx+x] {
				x++
			}
			cx0, cy0 := w.xy(y*w.nx + x1)
			sx, sy := px(cx0-gridRes/2, cy0+gridRes/2)
			fmt.Fprintf(&b, `<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f"/>`, sx, sy, float64(x-x1)*gridRes*s, gridRes*s)
		}
	}
	b.WriteString(`</g>`)
	// Cart zones.
	for _, z := range evreuxCartZones {
		x0, y0 := px(z[0], z[3])
		x1, y1 := px(z[2], z[1])
		fmt.Fprintf(&b, `<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="none" stroke="#999" stroke-dasharray="3 3"><title>cart zone</title></rect>`, x0, y0, x1-x0, y1-y0)
	}
	// Aisles (walking direction) and zone names.
	type acc struct{ x, y, n float64 }
	zc := map[string]*acc{}
	for _, a := range w.aisles {
		x1, y1 := px(a.px, a.py)
		x2, y2 := px(a.qx, a.qy)
		col := zoneColor[a.zone]
		if col == "" {
			col = "#bbb"
		}
		if a.entry == a.exit {
			fmt.Fprintf(&b, `<circle cx="%.1f" cy="%.1f" r="2" fill="%s"><title>%s</title></circle>`, x1, y1, col, a.code)
		} else {
			fmt.Fprintf(&b, `<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="2" opacity=".6"><title>%s (%s)</title></line>`, x1, y1, x2, y2, col, a.code, a.zone)
			fmt.Fprintf(&b, `<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="none" marker-end="url(#ad)"/>`, x1, y1, (x1+x2)/2, (y1+y2)/2)
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
		fmt.Fprintf(&b, `<text class="zl" x="%.1f" y="%.1f" text-anchor="middle" opacity=".45">%s</text>`, c.x/c.n, c.y/c.n, z)
	}

	// The lists.
	var js strings.Builder
	js.WriteString("const views = [")
	for i, r := range views {
		cache := map[int][]int32{}
		actPts, actLen := walkPoints(w, r.depot, r.rec.visits, cache)
		optPts, optLen := walkPoints(w, r.depot, r.optOrder, cache)
		fmt.Fprintf(&b, `<g class="view%s">`, map[bool]string{true: "", false: " hide"}[i == 0])
		bx0, by0, bx1, by1 := math.Inf(1), math.Inf(1), math.Inf(-1), math.Inf(-1)
		for k, wk := range []struct {
			cls string
			pts [][2]float64
		}{{"act", actPts}, {"opt", optPts}} {
			off := float64(k)*2.4 - 1.2 // keep both walks visible on shared corridors
			var ps []string
			for _, p := range wk.pts {
				x, y := px(p[0], p[1])
				bx0, by0, bx1, by1 = math.Min(bx0, x), math.Min(by0, y), math.Max(bx1, x), math.Max(by1, y)
				ps = append(ps, fmt.Sprintf("%.1f,%.1f", x+off, y+off))
			}
			fmt.Fprintf(&b, `<polyline id="%s%d" class="%s %sG" points="%s"/>`, wk.cls, i, wk.cls, wk.cls, strings.Join(ps, " "))
		}
		// Required aisles, then visit numbers: recorded on one side of the
		// aisle middle, optimal on the other.
		for _, code := range r.optOrder {
			a := w.aisles[w.index[code]]
			x1, y1 := px(a.px, a.py)
			x2, y2 := px(a.qx, a.qy)
			fmt.Fprintf(&b, `<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#c62828" stroke-width="4" stroke-linecap="round" opacity=".55"><title>%s</title></line>`, x1, y1, x2, y2, code)
		}
		// label writes the visit numbers of a walk on its aisles, at the
		// fraction f of the aisle length (recorded and optimal numbers at
		// different places, so that they do not overlap).
		label := func(cls, color string, codes []string, f float64) {
			nums := map[string][]string{}
			for k, code := range codes {
				nums[code] = append(nums[code], fmt.Sprint(k+1))
			}
			for code, ns := range nums {
				a := w.aisles[w.index[code]]
				x, y := px(a.px+f*(a.qx-a.px), a.py+f*(a.qy-a.py))
				t := strings.Join(ns, ",")
				wdt := 4.2*float64(len(t)) + 3
				fmt.Fprintf(&b, `<g class="%sG"><rect x="%.1f" y="%.1f" width="%.1f" height="8" rx="2" fill="%s"/><text class="n%s" x="%.1f" y="%.1f" text-anchor="middle">%s</text><title>%s: visit %s</title></g>`,
					cls, x-wdt/2, y-4, wdt, color, cls, x, y+2.5, t, code, t)
			}
		}
		label("act", "#ef6c00", r.rec.visits, 0.3)
		label("opt", "#2e7d32", r.optOrder, 0.7)
		dx, dy := px(w.depotXY[r.depot][0], w.depotXY[r.depot][1])
		fmt.Fprintf(&b, `<rect x="%.1f" y="%.1f" width="10" height="10" fill="#111"><title>cart zone (start / end)</title></rect>`, dx-5, dy-5)
		b.WriteString(`</g>`)
		const m = 40.0
		fmt.Fprintf(&js, "{id:%q,zone:%q,k:%d,rev:%d,act:%.1f,opt:%.1f,actG:%.1f,optG:%.1f,box:[%.0f,%.0f,%.0f,%.0f]},",
			r.rec.id, r.zone, r.distinct, r.revisits, r.recorded, r.optimal, actLen, optLen, bx0-m, by0-m, bx1-bx0+2*m, by1-by0+2*m)
	}
	js.WriteString("];\n")

	b.WriteString(`</svg><script>
`)
	b.WriteString(js.String())
	fmt.Fprintf(&b, "const S = %.1f;\n", s)
	b.WriteString(`let cur = 0, raf = null;
const svg = document.getElementById('g');
const vb0 = svg.getAttribute('viewBox').split(' ').map(Number);
let vb = vb0.slice();
function setVB(){ svg.setAttribute('viewBox', vb.join(' ')); }
function resetView(){ vb = vb0.slice(); setVB(); }
function fit(){ const b = views[cur].box.slice(); const r = svg.getBoundingClientRect();
  // keep the aspect ratio of the drawing
  let w = b[2], h = b[3]; const a = vb0[2]/vb0[3];
  if (w/h < a) { const nw = h*a; b[0] -= (nw-w)/2; w = nw; } else { const nh = w/a; b[1] -= (nh-h)/2; h = nh; }
  vb = [b[0], b[1], w, h]; setVB(); }
function show(i){ cur = i; stop(); upd(); fit(); }
function upd(){
  document.querySelectorAll('[class^=view]').forEach((g,i)=>g.classList.toggle('hide', i!==cur));
  document.querySelectorAll('.optG').forEach(g=>g.classList.toggle('hide', !document.getElementById('cOpt').checked));
  document.querySelectorAll('.actG').forEach(g=>g.classList.toggle('hide', !document.getElementById('cAct').checked));
  const v = views[cur];
  document.getElementById('info').innerHTML = '<b>' + v.id + '</b> (' + v.zone + ', ' + v.k + ' aisles' +
    (v.rev ? ', ' + v.rev + ' revisit(s)' : '') + '): recorded <span style="color:#ef6c00">' + v.act.toFixed(0) +
    ' m</span>, optimal <span style="color:#2e7d32">' + v.opt.toFixed(0) + ' m</span>, saving <b>' +
    (v.act-v.opt).toFixed(0) + ' m (' + (100*(v.act-v.opt)/v.act).toFixed(1) + ' %)</b>';
}
// Animation: both pickers walk at the same speed (1 m = 15 ms of animation).
function stop(){ if (raf) cancelAnimationFrame(raf); raf = null;
  document.querySelectorAll('.dot').forEach(d=>d.remove()); document.getElementById('anim').textContent=''; }
function runAnim(){
  stop();
  const pa = document.getElementById('act'+cur), po = document.getElementById('opt'+cur);
  const mk = (c)=>{ const d = document.createElementNS('http://www.w3.org/2000/svg','circle');
    d.setAttribute('r', 6); d.setAttribute('fill', c); d.setAttribute('stroke','#fff'); d.setAttribute('class','dot'); svg.appendChild(d); return d; };
  const da = mk('#ef6c00'), dop = mk('#2e7d32');
  const la = pa.getTotalLength(), lo = po.getTotalLength(), t0 = performance.now();
  function step(t){
    const m = (t - t0) / 15; // meters walked
    for (const [d,p,l] of [[da,pa,la],[dop,po,lo]]) { const q = p.getPointAtLength(Math.min(m*S, l)); d.setAttribute('cx', q.x); d.setAttribute('cy', q.y); }
    const v = views[cur];
    document.getElementById('anim').innerHTML = 'walked: <span style="color:#ef6c00">' + Math.min(m, la/S).toFixed(0) +
      ' m</span> / <span style="color:#2e7d32">' + Math.min(m, lo/S).toFixed(0) + ' m</span>' +
      (m*S >= lo ? ' — optimal picker back at the carts' : '') + (m*S >= la ? ' — both done' : '');
    if (m*S < la) raf = requestAnimationFrame(step); else raf = null;
  }
  raf = requestAnimationFrame(step);
}
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
show(0);
</script></body></html>`)
	return os.WriteFile(path, []byte(b.String()), 0o644)
}
