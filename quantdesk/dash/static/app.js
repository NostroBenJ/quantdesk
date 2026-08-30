/* quantdesk research review — renders a run JSON. Computes nothing.
 *
 * Every figure here was produced by backtest/ and written to a run file. If a
 * number is missing the panel says so; it never falls back to a plausible
 * default, because a plausible default is indistinguishable from a result.
 */

const $ = (id) => document.getElementById(id);

/* JSON has no Infinity. results.py encodes it as the string "inf" so an
 * infinite profit factor survives the round trip and can be LABELLED rather
 * than silently becoming null or a large finite number. */
function num(v) {
  if (v === "inf") return Infinity;
  if (v === "-inf") return -Infinity;
  if (v === "nan" || v === null || v === undefined) return NaN;
  return v;
}
const has = (v) => Number.isFinite(num(v));

function fmt(v, dp = 2) {
  const n = num(v);
  if (n === Infinity) return "∞";
  if (n === -Infinity) return "−∞";
  if (!Number.isFinite(n)) return "—";
  return n.toLocaleString(undefined, { minimumFractionDigits: dp, maximumFractionDigits: dp });
}
function money(v) {
  const n = num(v);
  if (!Number.isFinite(n)) return "—";
  const sign = n < 0 ? "−" : n > 0 ? "+" : "";
  return sign + Math.abs(n).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
const pct = (v, dp = 1) => (Number.isFinite(num(v)) ? (num(v) * 100).toFixed(dp) + "%" : "—");
const dirClass = (v) => (num(v) > 0 ? "up" : num(v) < 0 ? "down" : "muted");

let chart = null, series = null;

/* ------------------------------------------------------------------ runs */

async function loadRuns() {
  const runs = await (await fetch("/api/runs")).json();
  const sel = $("runSelect");
  sel.innerHTML = "";
  if (!runs.length) {
    sel.innerHTML = '<option value="">no runs — python scripts/make_demo_run.py</option>';
    document.querySelectorAll(".grid .panel .body").forEach((b) => {
      if (!b.dataset.kept) b.innerHTML = '<div class="empty">no run selected</div>';
    });
    return;
  }
  for (const r of runs) {
    const opt = document.createElement("option");
    opt.value = r.id;
    const pf = fmt(r.profit_factor);  // fmt renders Infinity as ∞, not as —
    opt.textContent = `${r.grade.padEnd(11)} ${r.name}  ·  PF ${pf}  ·  n=${r.n ?? "—"}`;
    sel.appendChild(opt);
  }
  await loadRun(sel.value);
}

async function loadRun(id) {
  if (!id) return;
  const res = await fetch(`/api/run/${encodeURIComponent(id)}`);
  if (!res.ok) return;
  render(await res.json());
}

/* ---------------------------------------------------------------- render */

function render(d) {
  const net = d.net || {};
  const gross = d.gross || {};
  const cs = d.cost_sensitivity;
  const wf = d.walk_forward;

  $("runStamp").textContent =
    `${d.symbol} · ${d.timeframe.label} · ${d.module} · ${(d.created_at || "").replace("T", " ").slice(0, 19)}Z`;

  /* --- statbar --- */
  const grade = d.verdict.grade;
  const gradeEl = $("sbGrade");
  // "INSUFFICIENT DATA" wraps and pushes the statbar out of alignment. Show the
  // distinguishing word and carry the qualifier on the sub-line - the meaning is
  // "we cannot tell", and "blocked from paper" already says the consequence.
  const short = grade === "INSUFFICIENT DATA" ? "INSUFFICIENT" : grade;
  gradeEl.textContent = short;
  gradeEl.className = "v grade grade-" + grade.split(" ")[0];
  gradeEl.title = grade;
  $("sbGate").textContent = d.verdict.may_paper_trade ? "may go to paper" : "blocked from paper";

  $("sbNetPf").textContent = fmt(net.profit_factor);
  $("sbNetPf").className = "v " + (num(net.profit_factor) >= 1 ? "up" : "down");
  $("sbGrossPf").textContent = "gross " + fmt(gross.profit_factor);

  $("sbTrades").textContent = net.n ?? "—";
  const enough = net.n >= d.min_meaningful_trades;
  $("sbSample").textContent = enough ? `≥ ${d.min_meaningful_trades} ok` : `< ${d.min_meaningful_trades} too few`;
  $("sbSample").className = "sub " + (enough ? "" : "warn");

  $("sbEdgeCost").textContent = fmt(net.edge_to_cost_ratio);
  $("sbEdgeCost").className = "v " + (num(net.edge_to_cost_ratio) >= 2 ? "up" : "down");

  if (cs) {
    const be = num(cs.breakeven_multiple);
    $("sbBreakeven").textContent = be === Infinity ? "∞" : be === 0 ? "never" : fmt(be) + "×";
    $("sbBreakeven").className = "v " + (be >= 1.5 ? "up" : "down");
  } else {
    $("sbBreakeven").textContent = "—";
  }

  if (wf) {
    $("sbWindows").textContent = `${wf.consistent_windows}/${wf.total_windows}`;
    const frac = wf.total_windows ? wf.consistent_windows / wf.total_windows : 0;
    $("sbWindows").className = "v " + (frac >= 0.5 ? "up" : "down");
    $("sbDegradation").textContent = has(wf.degradation) ? `deg ${fmt(wf.degradation)}×` : "deg —";
  } else {
    $("sbWindows").textContent = "—";
    $("sbDegradation").textContent = "not run";
  }

  const es = d.equity_stats || {};
  // Past the point equity hits zero, a drawdown percentage is arithmetic on a
  // hypothetical - the account would have been closed out. Say RUINED instead.
  if (es.ruined) {
    $("sbDD").textContent = "RUINED";
    $("sbDD").className = "v down";
    $("sbSharpe").textContent = `equity hit ${money(es.min_equity)}`;
  } else {
    $("sbDD").textContent = pct(es.max_drawdown);
    $("sbDD").className = "v down";
    $("sbSharpe").textContent = `sharpe ${fmt(es.sharpe)}`;
  }

  /* --- verdict reasons --- */
  const bad = /below|disappears|only|curve fit|no conclusion|too few/i;
  // "retains 1.01x of in-sample" is NOT good news when in-sample was 0.19.
  // Only statements that assert a threshold was cleared get the tick.
  const good = /clears|survives costs|causal/i;
  $("reasons").innerHTML = d.verdict.reasons
    .map((r) => {
      const cls = bad.test(r) ? "bad" : good.test(r) ? "good" : "";
      return `<li class="${cls}">${escapeHtml(r)}</li>`;
    })
    .join("") || '<li class="muted">no reasons recorded</li>';
  $("verdictNote").textContent = d.name;

  /* --- gross vs net --- */
  const rows = [
    ["profit factor", fmt(gross.profit_factor), fmt(net.profit_factor), num(net.profit_factor) >= 1],
    ["expectancy", money(gross.expectancy), money(net.expectancy), num(net.expectancy) > 0],
    ["net P&L", money(gross.net_pnl), money(net.net_pnl), num(net.net_pnl) > 0],
    ["cost / trade", "—", money(net.cost_per_trade), false],
  ];
  $("compare").innerHTML = rows
    .map(
      ([label, g, n, ok]) =>
        `<div class="lab">${label}</div>
         <div class="g">${g}</div><div class="arrow">→</div>
         <div class="n ${ok ? "up" : "down"}">${n}</div>`
    )
    .join("");

  /* --- execution --- */
  const ex = d.execution || {};
  $("execTable").querySelector("tbody").innerHTML = [
    ["orders submitted", ex.orders_submitted],
    ["filled", ex.orders_filled],
    ["partially filled", ex.orders_partially_filled],
    ["unfilled", ex.orders_unfilled],
    ["expired (no next bar)", ex.orders_expired],
    ["fill rate", pct(ex.fill_rate)],
  ]
    .map(([k, v]) => `<tr><td class="muted">${k}</td><td class="num">${v ?? "—"}</td></tr>`)
    .join("") +
    (ex.forced_liquidation
      ? `<tr><td colspan="2"><span class="chip warn">forced liquidation at end of data</span></td></tr>`
      : "");

  /* --- cost sensitivity --- */
  const ctb = $("costTable").querySelector("tbody");
  if (!cs) {
    ctb.innerHTML = '<tr><td colspan="6" class="empty">cost sweep not run</td></tr>';
  } else {
    const maxPf = Math.max(2, ...cs.rows.map((r) => (has(r.profit_factor) ? num(r.profit_factor) : 0)));
    ctb.innerHTML = cs.rows
      .map((r) => {
        const pf = num(r.profit_factor);
        const ok = pf >= 1;
        const w = Math.min(100, (Math.max(pf, 0) / maxPf) * 100);
        const markLeft = (1 / maxPf) * 100;
        return `<tr class="${r.multiple === 0 ? "row-gross" : ""}">
          <td class="num">${fmt(r.multiple)}${r.multiple === 0 ? " <span class='dim'>gross</span>" : ""}</td>
          <td class="num ${ok ? "up" : "down"}">${fmt(r.profit_factor)}</td>
          <td><div class="pfbar">
                <i style="width:${w}%;background:var(--${ok ? "up" : "down"})"></i>
                <span class="mark" style="left:${markLeft}%"></span>
              </div></td>
          <td class="num ${dirClass(r.net_pnl)}">${money(r.net_pnl)}</td>
          <td class="num ${dirClass(r.expectancy)}">${money(r.expectancy)}</td>
          <td class="num muted">${money(r.total_costs)}</td>
        </tr>`;
      })
      .join("");
    const be = num(cs.breakeven_multiple);
    $("costNote").textContent =
      be === 0 ? "never profitable at any cost level"
      : be === Infinity ? "profitable across the whole sweep"
      : `breakeven at ${fmt(be)}× the estimate`;
  }

  /* --- module ledger --- */
  const mtb = $("moduleTable").querySelector("tbody");
  const mods = Object.entries(d.modules || {});
  mtb.innerHTML = mods.length
    ? mods
        .map(
          ([name, s]) => `<tr>
            <td>${escapeHtml(name)}${s.is_meaningful ? "" : ' <span class="chip warn">thin</span>'}</td>
            <td class="num">${s.n}</td>
            <td class="num ${num(s.profit_factor) >= 1 ? "up" : "down"}">${fmt(s.profit_factor)}</td>
            <td class="num">${pct(s.win_rate)}</td>
            <td class="num ${dirClass(s.expectancy)}">${money(s.expectancy)}</td>
            <td class="num ${num(s.edge_to_cost_ratio) >= 2 ? "up" : "down"}">${fmt(s.edge_to_cost_ratio)}</td>
          </tr>`
        )
        .join("")
    : '<tr><td colspan="6" class="empty">no modules recorded</td></tr>';

  /* --- walk-forward --- */
  const wtb = $("wfTable").querySelector("tbody");
  if (!wf) {
    wtb.innerHTML = '<tr><td colspan="10" class="empty">walk-forward not run — a single in-sample backtest is not a result</td></tr>';
    $("wfNote").textContent = "not run";
  } else {
    wtb.innerHTML = wf.windows
      .map((w) => {
        const good = num(w.oos_net) > 0;
        return `<tr>
          <td class="num">${w.index}</td>
          <td class="num muted">${w.train_size}</td>
          <td class="num muted">${w.embargo}</td>
          <td class="num muted">${w.test_size}</td>
          <td class="num">${fmt(w.is_profit_factor)}</td>
          <td class="num muted">${w.is_n}</td>
          <td class="num ${num(w.oos_profit_factor) >= 1 ? "up" : "down"}">${fmt(w.oos_profit_factor)}</td>
          <td class="num muted">${w.oos_n}</td>
          <td class="num ${dirClass(w.oos_net)}">${money(w.oos_net)}</td>
          <td>${good ? '<span class="chip ok">+ profitable</span>' : '<span class="chip bad">− loss</span>'}</td>
        </tr>`;
      })
      .join("");
    $("wfNote").textContent =
      `${wf.consistent_windows}/${wf.total_windows} profitable OOS · degradation ${fmt(wf.degradation)}× · ` +
      `embargo separates train from test`;
  }

  /* --- warnings --- */
  $("warnings").innerHTML = (d.warnings || [])
    .map((w) => `<p class="warning-box">${escapeHtml(w)}</p>`)
    .join("");

  drawEquity(d.equity_curve || []);
}

/* ---------------------------------------------------------------- chart */

function drawEquity(curve) {
  const holder = $("equityChart");
  const fallback = $("chartFallback");

  if (!curve.length) {
    holder.style.display = "none";
    fallback.style.display = "block";
    fallback.textContent = "no equity curve in this run";
    return;
  }
  // Rule 1 of the data-integrity list: when something fails to load, say so.
  // Do not leave an empty box that reads as "flat".
  if (typeof LightweightCharts === "undefined") {
    holder.style.display = "none";
    fallback.style.display = "block";
    fallback.textContent =
      "chart library did not load (offline?) — equity curve not rendered. " +
      "The numbers above are unaffected.";
    return;
  }
  holder.style.display = "block";
  fallback.style.display = "none";

  if (!chart) {
    chart = LightweightCharts.createChart(holder, {
      layout: { background: { color: "#0f141c" }, textColor: "#6b7888", fontSize: 11 },
      grid: { vertLines: { color: "#1e2733" }, horzLines: { color: "#1e2733" } },
      rightPriceScale: { borderColor: "#1e2733" },
      timeScale: { borderColor: "#1e2733", timeVisible: true },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
      height: 260,
    });
    series = chart.addAreaSeries({
      lineColor: "#5a8dee",
      topColor: "rgba(90, 141, 238, 0.22)",
      bottomColor: "rgba(90, 141, 238, 0.02)",
      lineWidth: 2,
      priceLineVisible: false,
    });
    new ResizeObserver(() => chart.applyOptions({ width: holder.clientWidth })).observe(holder);
  }

  // Lightweight Charts needs strictly increasing, de-duplicated timestamps.
  const seen = new Set();
  const data = [];
  for (const p of curve) {
    const t = Math.floor(new Date(p.ts).getTime() / 1000);
    if (seen.has(t)) continue;
    seen.add(t);
    data.push({ time: t, value: p.equity });
  }
  data.sort((a, b) => a.time - b.time);
  series.setData(data);
  chart.timeScale().fitContent();
  chart.applyOptions({ width: holder.clientWidth });
}

/* ---------------------------------------------------------------- health */

async function loadHealth() {
  const h = await (await fetch("/api/health")).json();
  const tb = $("healthTable").querySelector("tbody");
  const rows = [];
  rows.push(["runs on disk", h.runs]);
  if (h.store && h.store.series.length) {
    for (const s of h.store.series) {
      rows.push([
        `${s.symbol} ${s.timeframe}`,
        `${s.bars.toLocaleString()} bars`,
      ]);
      rows.push([
        `<span class="dim">coverage</span>`,
        `<span class="dim">${s.first.slice(0, 10)} → ${s.last.slice(0, 10)}</span>`,
      ]);
    }
  } else {
    rows.push(["bar store", '<span class="chip warn">empty</span>']);
  }
  tb.innerHTML = rows.map(([k, v]) => `<tr><td class="muted">${k}</td><td class="num">${v}</td></tr>`).join("");
  for (const e of h.errors || []) {
    tb.innerHTML += `<tr><td colspan="2"><span class="chip bad">${escapeHtml(e)}</span></td></tr>`;
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

$("runSelect").addEventListener("change", (e) => loadRun(e.target.value));
$("refreshBtn").addEventListener("click", () => { loadRuns(); loadHealth(); });

loadRuns();
loadHealth();
