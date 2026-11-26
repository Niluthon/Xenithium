#include "html_assets.hpp"

#include <string>

namespace xenithium::bench {
namespace {

// The light theme's colours. Used three ways: when the reader picks it, when the reader's system prefers light and no
// theme was picked, and for print.
constexpr std::string_view kLightTheme =
    "--bg:#f7f8fb;--panel:#ffffff;--rule:#dfe2ea;--ink:#141824;--mute:#5d6578;--acc:#3457c9;--s0:#3f63d8;--s1:#1f9d57;--s2:#b98512;--ok:#1f9d57;--bad:#c93a32;"
    "--hl:rgba(52,87,201,.07);--tipbg:#ffffff;--shadow:0 10px 30px rgba(20,24,36,.16);color-scheme:light";

// The dark theme (the default) and the faces: numbers and labels in the system's monospace face, prose in its sans; no
// font is fetched. Colours are the launcher's (royal blue on ink, its green and yellow).
constexpr std::string_view kDarkTheme =
    ":root{--bg:#0b0d12;--panel:#10131b;--rule:#232838;--ink:#e7eaf3;--mute:#8a92a6;--acc:#7c9bff;--s0:#7c9bff;--s1:#7edc9a;--s2:#e8c46a;--ok:#7edc9a;--bad:#ff7b72;"
    "--hl:rgba(124,155,255,.09);--tipbg:#171b26;--shadow:0 10px 30px rgba(0,0,0,.45);"
    "--royal:#4169e1;--mono:ui-monospace,\"SF Mono\",\"Cascadia Mono\",\"JetBrains Mono\",Menlo,Consolas,monospace;"
    "--sans:system-ui,-apple-system,\"Segoe UI\",Roboto,sans-serif;color-scheme:dark}";

constexpr std::string_view kLayout = R"CSS(
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0 auto;max-width:1200px;padding:32px 28px 56px;background:var(--bg);color:var(--ink);font:15px/1.5 var(--sans)}
a{color:inherit}
button{font:inherit;color:inherit}
:focus-visible{outline:2px solid var(--acc);outline-offset:2px}

/* the masthead */
.top{display:flex;align-items:center;justify-content:space-between;gap:16px;min-height:28px}
.brand{font:600 13px var(--mono);letter-spacing:.14em;text-transform:uppercase}
.brand i{font-style:normal;color:var(--royal);margin-right:.5em}.brand b{color:var(--acc);font-weight:600}
.tools{display:flex;gap:8px}
.btn{border:1px solid var(--rule);background:var(--panel);border-radius:999px;padding:3px 12px;font:12px var(--mono);color:var(--mute);cursor:pointer}
.btn:hover{color:var(--ink);border-color:var(--mute)}
h1{font:650 34px/1.15 var(--sans);letter-spacing:-.02em;margin:22px 0 12px;overflow-wrap:anywhere}
.chips{display:flex;flex-wrap:wrap;gap:6px 8px;margin:0;padding:0;list-style:none;font:12px var(--mono);color:var(--mute)}
.chips li{border:1px solid var(--rule);border-radius:999px;padding:2px 10px;background:var(--panel);overflow-wrap:anywhere}
.chips b{color:var(--ink);font-weight:500}.chips .bad b{color:var(--bad)}
/* the run's facts: a label per row (engine, model, KV cache, ...), its facts as chips; the launch folded below */
.facts{display:grid;grid-template-columns:max-content 1fr;gap:8px 22px;margin:20px 0 0;align-items:start}
.facts dt{font:11px var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--mute);padding-top:4px}
.facts dd{margin:0}.facts .chips{color:var(--ink)}
details.launch{margin-top:18px;border:1px solid var(--rule);border-radius:12px;background:var(--panel);padding:10px 16px}
details.launch summary{font:600 13.5px var(--sans);color:var(--ink)}
details.launch pre{margin:10px 0 0;padding:8px 10px;border-radius:8px;background:var(--bg);font:12px/1.45 var(--mono);color:var(--mute);white-space:pre-wrap;overflow-wrap:anywhere}
details.launch p{color:var(--mute);margin:10px 0 2px}
details.rerun{border-color:var(--acc)}
details.rerun p.intro{margin:10px 0 4px;font:13px/1.5 var(--sans);color:var(--ink)}details.rerun p.intro b{font-weight:600}
details.rerun ol{margin:6px 0 4px;padding-left:22px}details.rerun ol>li{margin-top:16px}
details.rerun li p{margin:0;font:13px/1.45 var(--sans);color:var(--ink)}details.rerun li p b{color:var(--bad);font-weight:600}
details.rerun li p b.step{color:var(--ink);font-weight:650}
details.rerun li p.hint{margin-top:6px;color:var(--mute)}
details.rerun ul.files{margin:6px 0 0;padding-left:18px;font:13px/1.5 var(--sans);color:var(--ink)}
details.rerun a{color:var(--acc)}
details.rerun code{font:12px var(--mono);color:var(--acc)}
.cmdbox{position:relative}.cmdbox .copy{position:absolute;top:16px;right:8px}
details.rerun pre.cmd{color:var(--ink);padding-right:70px}
@media (max-width:640px){.facts{grid-template-columns:1fr;gap:4px}.facts dd{margin-bottom:8px}}

/* the section navigation: sticks to the top, marks the section in view */
nav{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:6px;margin:22px -8px 0;padding:7px 8px;background:var(--bg);
  background:color-mix(in srgb,var(--bg) 88%,transparent);backdrop-filter:blur(8px);border-bottom:1px solid var(--rule);font:12px var(--mono)}
nav .links{display:flex;gap:2px;overflow-x:auto;scrollbar-width:thin;flex:1;min-width:0}
nav a{color:var(--mute);text-decoration:none;padding:4px 9px;border-radius:6px;white-space:nowrap}
nav a:hover{color:var(--ink);background:var(--hl)}
nav a.on{color:var(--acc);background:var(--hl)}
nav a.err::after{content:"";display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--bad);margin-left:6px;vertical-align:1px}
nav .home{color:var(--royal);padding:4px 6px}

/* the headline figures */
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:26px 0 0}
.fig{display:block;text-decoration:none;padding:14px 16px 16px;border:1px solid var(--rule);border-radius:12px;background:var(--panel);transition:border-color .15s,transform .15s}
a.fig:hover{border-color:var(--acc);transform:translateY(-1px)}
.k{font:11px var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--mute)}
.n{font:600 32px/1.2 var(--mono);font-variant-numeric:tabular-nums;margin-top:6px}.n small{font:13px var(--mono);color:var(--mute);margin-left:6px}
.d{font-size:12.5px;color:var(--mute);margin-top:4px}

/* the checks */
.checks{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 0;padding:0;list-style:none;font:12px var(--mono)}
.checks li{display:inline-flex;align-items:center;gap:7px;border:1px solid var(--rule);border-radius:999px;padding:3px 11px 3px 9px;background:var(--panel)}
.checks li::before{content:"";flex:none;width:8px;height:8px;border-radius:50%;background:var(--ok)}
.checks li.bad{border-color:var(--bad)}.checks li.bad::before{background:var(--bad)}
.checks a{text-decoration:none}.checks a:hover{text-decoration:underline}
.checks span{color:var(--mute)}

/* sections */
section{margin-top:48px;scroll-margin-top:60px}
h2{display:flex;align-items:center;gap:8px;font:600 20px/1.3 var(--sans);letter-spacing:-.01em;margin:0 0 4px;padding-bottom:8px;border-bottom:1px solid var(--rule)}
h2 .ttl{flex:1}
h2 code{font:12px var(--mono);color:var(--mute);font-weight:400}
.fold{flex:none;border:0;background:none;color:var(--mute);cursor:pointer;width:24px;height:24px;border-radius:6px;padding:0;font-size:13px;transition:transform .15s}
.fold:hover{background:var(--hl);color:var(--ink)}
section.folded .fold{transform:rotate(-90deg)}
@media screen{section.folded>:not(h2){display:none}section.folded h2{border-bottom-style:dashed}}
.about{color:var(--mute);margin:8px 0 0;max-width:80ch;text-wrap:pretty}
.meta{font:12px var(--mono);color:var(--mute);margin:6px 0 0}.meta b{color:var(--bad);font-weight:600}
.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:20px}
figure{margin:0;min-width:0;padding:14px 16px 12px;border:1px solid var(--rule);border-radius:12px;background:var(--panel)}
figure.wide{margin-top:18px}
figcaption{font:12px var(--mono);color:var(--ink);margin-bottom:10px}figcaption span{color:var(--mute)}
.legend{font:12px var(--mono);color:var(--mute);margin:10px 0 0}
.legend i{display:inline-block;width:10px;height:10px;margin:0 6px 0 14px;background:var(--s0);vertical-align:-1px}.legend i:first-child{margin-left:0}
.legend i.area0{opacity:.85}.legend i.area1{opacity:.45}.legend i.area2{opacity:.25}.legend i.clk{height:2px;background:var(--s2);vertical-align:3px}
.hint{font:12px var(--mono);color:var(--mute);margin:8px 0 0}

/* tables */
.tbl{margin-top:18px;border:1px solid var(--rule);border-radius:12px;background:var(--panel);overflow:hidden}
.tbar{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:6px 10px 6px 14px;border-bottom:1px solid var(--rule);font:12px var(--mono);color:var(--ink);min-height:36px}
.tscroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font:12.5px var(--mono);font-variant-numeric:tabular-nums}
th,td{padding:7px 14px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--rule)}
th{font-weight:400;color:var(--mute);vertical-align:bottom;user-select:none}
th.sortable{cursor:pointer}th.sortable:hover{color:var(--ink)}
th[aria-sort=ascending]::after{content:" \25B4"}th[aria-sort=descending]::after{content:" \25BE"}
th.l,td.l{text-align:left}
tbody tr:last-child td{border-bottom:0}
tbody tr:hover td{background:var(--hl)}
.notes{margin:14px 0 0;padding:0;list-style:none;font-size:13px;color:var(--mute)}.notes li{margin-top:4px;overflow-wrap:anywhere}
.notes li.err{color:var(--bad)}

/* wrong answers: first in their section, in the failure colour */
.miss{margin-top:18px;border:1px solid var(--bad);border-left-width:4px;border-radius:12px;background:var(--panel);overflow:hidden}
.mbar{padding:9px 14px;border-bottom:1px solid var(--rule);font:13px var(--mono);color:var(--bad)}.mbar b{font-weight:600}
.miss ol{margin:0;padding:0;list-style:none}
.miss li{padding:11px 14px;border-bottom:1px solid var(--rule)}.miss li:last-child{border-bottom:0}
.mq{display:flex;align-items:baseline;gap:10px;font:12.5px var(--mono)}.mq code{color:var(--ink);font-weight:600}.mq span{color:var(--mute)}
.miss p{margin:5px 0 0;font-size:13px;color:var(--mute);overflow-wrap:anywhere}
.miss pre{margin:6px 0 0;padding:8px 10px;border-radius:8px;background:var(--bg);font:12px/1.45 var(--mono);color:var(--mute);white-space:pre-wrap;overflow-wrap:anywhere}
.miss dl{display:grid;grid-template-columns:max-content 1fr;gap:3px 14px;margin:8px 0 0;font:12.5px var(--mono)}
.miss dt{color:var(--mute)}.miss dd{margin:0;overflow-wrap:anywhere}
.miss dd.want{color:var(--ok)}.miss dd.got{color:var(--bad)}.miss dd.none{color:var(--bad);font-style:italic}
a.fig.bad{border-color:var(--bad)}.fig .w{font:12.5px var(--mono);color:var(--bad);margin-top:6px}

/* setup */
dl.setup{display:grid;grid-template-columns:max-content 1fr;gap:6px 22px;margin:16px 0 0;font:12.5px var(--mono)}
dl.setup dt{color:var(--mute)}dl.setup dd{margin:0;overflow-wrap:anywhere}
details{margin-top:16px;font:12.5px var(--mono)}summary{cursor:pointer;color:var(--mute)}details dl.setup{margin-top:10px}
footer{margin-top:64px;padding-top:16px;border-top:1px solid var(--rule);font:12px var(--mono);color:var(--mute);display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px 24px}

/* the tooltip the script draws */
#tip{position:fixed;z-index:20;left:0;top:0;pointer-events:none;max-width:360px;padding:9px 12px;border:1px solid var(--rule);border-radius:10px;background:var(--tipbg);
  box-shadow:var(--shadow);font:12px/1.45 var(--mono);color:var(--ink);opacity:0;transform:translateY(4px);transition:opacity .1s,transform .1s}
#tip.on{opacity:1;transform:none}
#tip b{display:block;font-weight:600;margin-bottom:4px;overflow-wrap:anywhere}
#tip dl{display:grid;grid-template-columns:auto auto;gap:2px 16px;margin:0}
#tip dt{color:var(--mute);display:flex;align-items:center;gap:6px}
#tip dd{margin:0;text-align:right;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.sw{display:inline-block;flex:none;width:9px;height:9px;border-radius:2px;background:var(--s0)}
.sw.s1{background:var(--s1)}.sw.s2{background:var(--s2)}.sw.a0{opacity:.85}.sw.a1{opacity:.45}.sw.a2{opacity:.25}

/* the charts' classes (svg_charts.hpp) */
svg{display:block;width:100%;height:auto;overflow:visible}
svg text{font:11px var(--mono);fill:var(--mute);pointer-events:none}svg .v{fill:var(--ink)}svg .spant{fill:#fff;font-weight:600}
svg rect{rx:1.5px}
.g{stroke:var(--rule)}.ax{stroke:var(--mute);opacity:.55}
.s0{fill:var(--s0)}.s1{fill:var(--s1)}.s2{fill:var(--s2)}
.l0,.l1,.l2{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}.l0{stroke:var(--s0)}.l1{stroke:var(--s1)}.l2{stroke:var(--s2)}
.hair{stroke:var(--mute);stroke-width:1.5}.band{fill:var(--s0);opacity:.6}.med{stroke:var(--ink);stroke-width:2}
.yes{fill:var(--ok)}.no{fill:none;stroke:var(--bad);stroke-width:1.6}
.span0{fill:var(--royal)}.span1{fill:var(--royal);opacity:.6}
.area0{fill:var(--s0);opacity:.85}.area1{fill:var(--s0);opacity:.45}.area2{fill:var(--s0);opacity:.25}
.cap{stroke:var(--mute);stroke-dasharray:4 4}.wait{fill:var(--rule)}
.hit{fill:transparent;rx:0}
.hov{outline:none;transition:opacity .12s}
.hov:focus-visible .hit{fill:var(--hl);stroke:var(--acc)}
.ch:has(.hov:hover) .hov:not(:hover){opacity:.4}
.bar:hover .hit,.rng:hover .hit,.dot:hover .hit{fill:var(--hl)}
.pt{stroke:var(--panel);stroke-width:1.5}
.guide{stroke:var(--mute);stroke-dasharray:3 3;opacity:0}
.col:hover .guide,.col:focus-visible .guide{opacity:.9}
.col:hover .pt,.col:focus-visible .pt{r:5px}
.span{cursor:pointer}.span:hover .span0,.span:hover .span1{filter:brightness(1.2)}
.lg{cursor:pointer;outline:none}.lg.off{opacity:.35}.lg:hover text{fill:var(--ink)}.lg:focus-visible .hit{stroke:var(--acc)}
.hid{display:none}
.trace{cursor:crosshair}.tguide{stroke:var(--ink);stroke-width:1;opacity:0}.trace.on .tguide{opacity:.6}

@media(max-width:820px){.charts{grid-template-columns:minmax(0,1fr)}h1{font-size:26px}body{padding:22px 14px 40px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}*{transition:none!important}}
@media print{nav,.tools,.fold,.copy,#tip{display:none}section,figure,.tbl{break-inside:avoid}body{max-width:none;padding:0}figure,.tbl,.fig,.chips li,.checks li{background:none}}
)CSS";

constexpr std::string_view kScript = R"JS(
(() => {
  'use strict';
  const root = document.documentElement;
  document.querySelectorAll('.hint[hidden]').forEach(hint => { hint.hidden = false; });  // they name what the script adds

  // ---- buttons in the masthead (made here: without the script they would do nothing) ----
  const tools = document.querySelector('.tools');
  const button = (label, title) => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'btn'; b.textContent = label; b.title = title;
    if (tools) tools.appendChild(b);
    return b;
  };

  // ---- light / dark: the reader's setting unless one is picked; the pick is kept for the next report ----
  const themeKey = 'xenithium-bench-theme', themes = ['auto', 'light', 'dark'];
  let theme = 'auto';
  try { theme = localStorage.getItem(themeKey) || 'auto'; } catch (e) { /* a file page may have no storage */ }
  const themeButton = button('', 'light, dark, or as the system is set');
  const applyTheme = () => {
    if (theme === 'auto') delete root.dataset.theme; else root.dataset.theme = theme;
    themeButton.textContent = 'theme: ' + theme;
  };
  applyTheme();
  themeButton.addEventListener('click', () => {
    theme = themes[(themes.indexOf(theme) + 1) % themes.length];
    try { localStorage.setItem(themeKey, theme); } catch (e) { /* not kept */ }
    applyTheme();
  });

  // ---- folding the scenarios' sections ----
  const setFolded = (section, folded) => {
    section.classList.toggle('folded', folded);
    const fold = section.querySelector('.fold');
    if (fold) { fold.setAttribute('aria-expanded', String(!folded)); fold.title = folded ? 'unfold' : 'fold'; }
  };
  const scenarios = [...document.querySelectorAll('section.sc')];
  for (const section of scenarios) {
    const fold = document.createElement('button');
    fold.type = 'button'; fold.className = 'fold'; fold.textContent = '\u25BE';
    fold.setAttribute('aria-expanded', 'true'); fold.title = 'fold';
    fold.addEventListener('click', () => setFolded(section, !section.classList.contains('folded')));
    section.querySelector('h2').prepend(fold);
  }
  if (scenarios.length) {
    const foldAll = button('fold all', 'fold or unfold every scenario');
    foldAll.addEventListener('click', () => {
      const fold = scenarios.some(s => !s.classList.contains('folded'));
      scenarios.forEach(s => setFolded(s, fold));
      foldAll.textContent = fold ? 'unfold all' : 'fold all';
    });
  }
  const reveal = id => {
    const section = id && document.getElementById(id);
    if (section && section.classList.contains('folded')) setFolded(section, false);
  };
  addEventListener('hashchange', () => reveal(decodeURIComponent(location.hash.slice(1))));
  reveal(decodeURIComponent(location.hash.slice(1)));

  // ---- the tooltip ----
  const tip = document.createElement('div');
  tip.id = 'tip';
  tip.setAttribute('role', 'tooltip');
  document.body.appendChild(tip);
  const escapeText = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
  // a series' colour (0, 1, 2 ...), a card's share of the power trace (-2, -3 ...), or none (-1)
  const swatch = s => (s >= 0 ? '<i class="sw s' + (s % 3) + '"></i>' : s <= -2 ? '<i class="sw a' + ((-2 - s) % 3) + '"></i>' : '');
  const render = d => {
    let html = '<b>' + escapeText(d.t) + '</b>';
    if (d.r && d.r.length) {
      html += '<dl>';
      for (const [label, value, series] of d.r) html += '<dt>' + swatch(series) + escapeText(label) + '</dt><dd>' + escapeText(value) + '</dd>';
      html += '</dl>';
    }
    return html;
  };
  const place = (x, y) => {
    const gap = 14, box = tip.getBoundingClientRect();
    let left = x + gap, top = y + gap;
    if (left + box.width > innerWidth - 8) left = x - gap - box.width;
    if (top + box.height > innerHeight - 8) top = y - gap - box.height;
    tip.style.left = Math.max(8, left) + 'px';
    tip.style.top = Math.max(8, top) + 'px';
  };
  const show = (data, x, y) => { tip.innerHTML = render(data); tip.classList.add('on'); place(x, y); };
  let pointed = null;  // the element whose tip is shown
  const hide = () => {
    tip.classList.remove('on');
    if (pointed) pointed.classList.remove('on');
    pointed = null;
  };
  const tipOf = el => { if (!el.tipData) el.tipData = JSON.parse(el.dataset.tip || el.dataset.trace); return el.tipData; };
  // a tipped mark's title element would open the browser's own tooltip beside this one: keep its text as the label only
  document.querySelectorAll('[data-tip] > title').forEach(t => { t.parentNode.setAttribute('aria-label', t.textContent); t.remove(); });

  // the power trace: the readings at the pointer's moment
  const clockText = s => { const m = Math.floor(s / 60), r = Math.round(s - m * 60); return m ? m + ' min ' + r + ' s' : r + ' s'; };
  const wattsText = w => (w >= 100 ? Math.round(w).toLocaleString('en-US') : w.toFixed(1)) + ' W';
  const traceAt = (group, event) => {
    const d = tipOf(group), matrix = group.ownerSVGElement.getScreenCTM();
    if (!matrix || !d.p.length) return null;
    const x = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse()).x;
    const t = ((x - d.x0) / d.w) * d.t;
    let lo = 0, hi = d.p.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (d.p[mid][0] < t) lo = mid + 1; else hi = mid; }
    if (lo > 0 && Math.abs(d.p[lo - 1][0] - t) < Math.abs(d.p[lo][0] - t)) lo--;
    const p = d.p[lo], at = d.x0 + (p[0] / d.t) * d.w, guide = group.querySelector('.tguide');
    guide.setAttribute('x1', at); guide.setAttribute('x2', at);
    const span = d.s.find(s => p[0] >= s[1] && p[0] <= s[2]);
    const rows = [];
    let total = 0;
    d.c.forEach((name, card) => { const w = p[2 + card] || 0; total += w; rows.push([name, wattsText(w), -2 - card]); });
    if (d.c.length > 1) rows.push(['all cards', wattsText(total), -1]);
    if (p[1] > 0) rows.push(['GPU clock', Math.round(p[1]).toLocaleString('en-US') + ' MHz', -1]);
    return {t: clockText(p[0]) + (span ? ' \u00b7 ' + span[0] : ''), r: rows};
  };

  const point = event => {
    const target = event.target instanceof Element ? event.target : null;
    const trace = target && target.closest('.trace');
    if (trace) {
      const data = traceAt(trace, event);
      if (!data) return;
      if (pointed !== trace) hide();
      pointed = trace;
      trace.classList.add('on');
      show(data, event.clientX, event.clientY);
      return;
    }
    const el = target && target.closest('[data-tip]');
    if (!el) { if (pointed) hide(); return; }
    if (el !== pointed) { hide(); pointed = el; show(tipOf(el), event.clientX, event.clientY); }
    else place(event.clientX, event.clientY);
  };
  document.addEventListener('pointermove', point);
  document.addEventListener('pointerdown', point);
  root.addEventListener('pointerleave', hide);
  addEventListener('scroll', () => { if (pointed && !pointed.contains(document.activeElement)) hide(); }, {passive: true});
  document.addEventListener('focusin', event => {
    const el = event.target instanceof Element ? event.target.closest('[data-tip]') : null;
    if (!el) return;
    const box = el.getBoundingClientRect();
    hide();
    pointed = el;
    show(tipOf(el), box.left + box.width / 2, box.bottom);
  });
  document.addEventListener('focusout', hide);

  // ---- a chart's legend shows or hides its series; a span of the timeline goes to its scenario ----
  const toggle = entry => {
    const off = entry.classList.toggle('off');
    entry.setAttribute('aria-pressed', String(!off));
    entry.ownerSVGElement.querySelectorAll('.k' + entry.dataset.k).forEach(mark => mark.classList.toggle('hid', off));
  };
  const activate = (el, run) => {
    el.addEventListener('click', run);
    el.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); run(); } });
  };
  document.querySelectorAll('.lg').forEach(entry => { entry.setAttribute('aria-pressed', 'true'); activate(entry, () => toggle(entry)); });
  document.querySelectorAll('[data-href]').forEach(el => activate(el, () => { reveal(el.dataset.href.slice(1)); location.hash = el.dataset.href; }));

  // ---- the navigation marks the section in view ----
  const strip = document.querySelector('nav .links');
  const links = new Map([...document.querySelectorAll('nav .links a')].map(a => [decodeURIComponent(a.hash.slice(1)), a]));
  if (strip && 'IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      for (const entry of entries) {
        const link = entry.isIntersecting && links.get(entry.target.id);
        if (!link) continue;
        links.forEach(a => a.classList.remove('on'));
        link.classList.add('on');
        const l = link.getBoundingClientRect(), s = strip.getBoundingClientRect();
        strip.scrollBy({left: l.left - s.left - s.width / 2 + l.width / 2, behavior: 'smooth'});
      }
    }, {rootMargin: '-30% 0px -65% 0px'});
    document.querySelectorAll('main section[id]').forEach(section => observer.observe(section));
  }

  // ---- tables: sort by a column (numbers by value: "41.5 / 900" by its median, "2/3" by its yes count), copy ----
  const numberIn = text => { const m = text.replace(/,/g, '').match(/-?\d+(?:\.\d+)?/); return m ? parseFloat(m[0]) : NaN; };
  document.querySelectorAll('.tbl').forEach(box => {
    const table = box.querySelector('table'), body = table && table.tBodies[0];
    if (!body || !table.tHead) return;
    [...body.rows].forEach((row, i) => { row.dataset.order = i; });
    const heads = [...table.tHead.rows[0].cells];
    heads.forEach((th, column) => {
      th.classList.add('sortable'); th.tabIndex = 0; th.title = 'sort by this column';
      activate(th, () => {
        const next = {none: 'ascending', ascending: 'descending', descending: 'none'}[th.getAttribute('aria-sort') || 'none'];
        heads.forEach(h => h.removeAttribute('aria-sort'));
        const rows = [...body.rows];
        const text = row => (row.cells[column] ? row.cells[column].textContent : '');
        if (next === 'none') rows.sort((a, b) => a.dataset.order - b.dataset.order);
        else {
          th.setAttribute('aria-sort', next);
          // a cell without a figure ("-": not measured) stays below the others in either direction
          const blank = row => ['', '-'].includes(text(row).trim());
          rows.sort((a, b) => {
            if (blank(a) !== blank(b)) return blank(a) ? 1 : -1;
            const x = numberIn(text(a)), y = numberIn(text(b));
            const order = Number.isNaN(x) || Number.isNaN(y) ? text(a).localeCompare(text(b)) : x - y;
            return next === 'ascending' ? order : -order;
          });
        }
        rows.forEach(row => body.appendChild(row));
      });
    });
    const bar = box.querySelector('.tbar');
    if (!bar) return;
    const copy = document.createElement('button');
    copy.type = 'button'; copy.className = 'btn copy'; copy.textContent = 'copy';
    copy.title = 'copy the table as tab-separated text (pastes into a spreadsheet as columns)';
    copy.addEventListener('click', () => {
      const tsv = [...table.rows].map(row => [...row.cells].map(c => c.textContent.trim()).join('\t')).join('\n');
      const done = ok => { copy.textContent = ok ? 'copied' : 'copy failed'; setTimeout(() => { copy.textContent = 'copy'; }, 1200); };
      const fallback = () => {
        const area = document.createElement('textarea');
        area.value = tsv; area.style.position = 'fixed'; area.style.opacity = '0';
        document.body.appendChild(area);
        area.select();
        let ok = false;
        try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
        area.remove();
        done(ok);
      };
      if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(tsv).then(() => done(true), fallback);
      else fallback();
    });
    bar.appendChild(copy);
  });
})();
(() => {
  // ---- the "run this yourself" commands: a copy button on each ----
  const put = (text, done) => {
    const fallback = () => {
      const area = document.createElement('textarea');
      area.value = text; area.style.position = 'fixed'; area.style.opacity = '0';
      document.body.appendChild(area);
      area.select();
      let ok = false;
      try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
      area.remove();
      done(ok);
    };
    if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(text).then(() => done(true), fallback);
    else fallback();
  };
  document.querySelectorAll('pre.cmd').forEach(pre => {
    const box = document.createElement('div');
    box.className = 'cmdbox';
    pre.replaceWith(box);
    box.appendChild(pre);
    const copy = document.createElement('button');
    copy.type = 'button'; copy.className = 'btn copy'; copy.textContent = 'copy';
    copy.title = 'copy the command';
    copy.addEventListener('click', () => put(pre.textContent, ok => {
      copy.textContent = ok ? 'copied' : 'copy failed';
      setTimeout(() => { copy.textContent = 'copy'; }, 1200);
    }));
    box.appendChild(copy);
  });
})();
)JS";

std::string style_sheet() {
  std::string css(kDarkTheme);
  css.append(":root[data-theme=light]{").append(kLightTheme).append("}");
  css.append("@media(prefers-color-scheme:light){:root:not([data-theme=dark]){").append(kLightTheme).append("}}");
  css.append("@media print{:root{").append(kLightTheme).append("}}");
  return css.append(kLayout);
}

}  // namespace

std::string_view page_style() {
  static const std::string kSheet = style_sheet();
  return kSheet;
}

std::string_view page_script() { return kScript; }

}  // namespace xenithium:bench
