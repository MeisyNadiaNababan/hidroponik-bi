/* HydroBI - logika front-end dashboard */
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
let META = null, ACTIVE = "monitor", MODEL = null, DEFAULT = null;
const charts = {};
const COLORS = ["#2e7d4f", "#2b6cb0", "#d69e2e", "#c05621", "#6b46c1", "#c53030", "#319795"];

// ---------- util
const pad = (n) => String(n).padStart(2, "0");
const toLocal = (d) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
const parse = (s) => new Date(s.replace(" ", "T"));
const fmt = (v, d = 2) => (v === null || v === undefined ? "-" : Number(v).toFixed(d));
const api = async (url, opt) => {
  const r = await fetch(url, opt);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  return r.json();
};
function mk(id, cfg) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(document.getElementById(id), cfg);
  return charts[id];
}
function qs(extra = {}) {
  const p = new URLSearchParams({
    device: $("#f-device").value, start: $("#f-start").value, end: $("#f-end").value,
    ph_min: $("#ph_min").value, ph_max: $("#ph_max").value,
    tds_min: $("#tds_min").value, tds_max: $("#tds_max").value, ...extra,
  });
  return p.toString();
}
const card = (t, v, s = "", cls = "") =>
  `<div class="card ${cls}"><div class="t">${t}</div><div class="v">${v}</div><div class="s">${s}</div></div>`;
const statusCls = (v, lo, hi) => (v === null ? "" : v >= lo && v <= hi ? "ok" : "bad");

// ---------- filter
function aturPreset() {
  const p = $("#f-preset").value;
  if (p === "custom") return;
  const end = parse(META.max);
  let start;
  if (p === "all") start = parse(META.min);
  else {
    const n = parseInt(p), unit = p.slice(-1);
    start = new Date(end.getTime() - n * (unit === "h" ? 3600e3 : 86400e3));
  }
  $("#f-start").value = toLocal(start);
  $("#f-end").value = toLocal(end);
}

// ---------- 1. Monitoring
async function loadMonitor() {
  const d = await api("/api/ringkasan?" + qs());
  if (d.kosong) { $("#m-cards").innerHTML = "<p>Tidak ada data pada rentang ini.</p>"; $("#m-kpi").innerHTML = ""; return; }
  const t = d.terkini, r = d.rentang, k = d.kpi;
  $("#m-waktu").textContent = "diperbarui " + t.waktu;
  const L = META.label;
  $("#m-cards").innerHTML =
    card(L.ph, fmt(t.ph), `optimal ${r.ph[0]} – ${r.ph[1]}`, statusCls(t.ph, ...r.ph)) +
    card(L.tds_ppm, fmt(t.tds_ppm, 0), `optimal ${r.tds_ppm[0]} – ${r.tds_ppm[1]}`, statusCls(t.tds_ppm, ...r.tds_ppm)) +
    card(L.suhu_air, fmt(t.suhu_air, 1)) + card(L.suhu_udara, fmt(t.suhu_udara, 1)) +
    card(L.kelembapan, fmt(t.kelembapan, 1)) + card(L.intensitas_cahaya, fmt(t.intensitas_cahaya, 0)) +
    card(L.ketinggian_air, fmt(t.ketinggian_air, 1), "batas isi ulang 15 cm", t.ketinggian_air > 15 ? "ok" : "bad");
  const p = k.pompa;
  $("#m-kpi").innerHTML =
    card("% Waktu Optimal pH", fmt(k.ph.persen_optimal, 1) + "%", `${k.ph.pelanggaran} pelanggaran`, k.ph.persen_optimal >= 95 ? "ok" : "warn") +
    card("% Waktu Optimal TDS", fmt(k.tds.persen_optimal, 1) + "%", `${k.tds.pelanggaran} pelanggaran`, k.tds.persen_optimal >= 95 ? "ok" : "warn") +
    card("Waktu Pemulihan Rata-rata TDS", fmt(k.tds.pemulihan_menit, 0) + " mnt", "keluar rentang → normal") +
    card("Waktu Pemulihan Rata-rata pH", fmt(k.ph.pemulihan_menit, 0) + " mnt") +
    card("Uptime Perangkat", fmt(k.uptime, 1) + "%", "data diterima ÷ data seharusnya", k.uptime >= 95 ? "ok" : "bad") +
    card("Pompa Nutrisi", p.nutrisi.kali + "×", p.nutrisi.detik + " detik total") +
    card("Pompa Asam", p.asam.kali + "×", p.asam.detik + " detik total") +
    card("Pompa Air", p.air.kali + "×", p.air.detik + " detik total");
  $("#m-note").textContent =
    `Pembersihan data (ETL): ${k.data_dibuang.kalibrasi} baris kalibrasi dikeluarkan dan ${k.data_dibuang.saturasi_adc} pembacaan pH dengan ADC 4095 (saturasi) dianggap tidak valid.`;
}

// ---------- 2. Tren
function initParams() {
  $("#t-params").innerHTML = Object.entries(META.label).map(([k, v]) =>
    `<label class="chk"><input type="checkbox" value="${k}" ${["ph", "tds_ppm", "suhu_air"].includes(k) ? "checked" : ""}> ${v}</label>`).join("");
  $$("#t-params input").forEach((c) => c.addEventListener("change", loadTren));
}
async function loadTren() {
  const params = $$("#t-params input:checked").map((c) => c.value);
  if (!params.length) { $("#t-charts").innerHTML = ""; return; }
  const d = await api("/api/tren?" + qs({ params: params.join(",") }));
  $("#t-agg").textContent = "Agregasi: " + d.agregasi;
  $("#t-charts").innerHTML = params.map((p) => `<div class="panel"><h3>${META.label[p]}</h3><canvas id="tc-${p}"></canvas></div>`).join("");
  const rng = { ph: [+$("#ph_min").value, +$("#ph_max").value], tds_ppm: [+$("#tds_min").value, +$("#tds_max").value] };
  params.forEach((p, i) => {
    const ds = [{ label: META.label[p], data: d.series[p], borderColor: COLORS[i % 7], borderWidth: 1.5, pointRadius: 0, spanGaps: false, tension: .2 }];
    if (rng[p]) rng[p].forEach((v, j) => ds.push({ label: j ? "batas atas" : "batas bawah", data: d.label.map(() => v), borderColor: "#c93c3c", borderDash: [5, 4], borderWidth: 1, pointRadius: 0 }));
    mk("tc-" + p, { type: "line", data: { labels: d.label.map((s) => s.slice(5, 16)), datasets: ds },
      options: { animation: false, plugins: { legend: { display: false } }, scales: { x: { ticks: { maxTicksLimit: 8 } } } } });
  });
  $("#t-stat").innerHTML = "<tr><th>Parameter</th><th>Rata-rata</th><th>Min</th><th>Maks</th><th>Simpangan baku</th></tr>" +
    params.map((p) => { const s = d.ringkas[p]; return `<tr><td>${META.label[p]}</td><td>${s.rata2}</td><td>${s.min}</td><td>${s.maks}</td><td>${s.std}</td></tr>`; }).join("");
}

// ---------- 3. Aktuator
async function loadAktuator() {
  const d = await api("/api/aktuator?" + qs());
  const nm = { nutrisi: "Nutrisi", asam: "Asam", air: "Air" }, col = { nutrisi: "#2e7d4f", asam: "#d69e2e", air: "#2b6cb0" };
  mk("a-chart", { type: "bar", data: { labels: d.hari.map((s) => s.slice(5)),
    datasets: Object.keys(nm).map((j) => ({ label: nm[j], data: d.harian[j], backgroundColor: col[j] })) },
    options: { scales: { x: { stacked: true }, y: { stacked: true } } } });
  $("#a-log").innerHTML = "<tr><th>Waktu</th><th>Pompa</th><th>Durasi (dtk)</th><th>Awal → Target</th></tr>" +
    d.log.map((r) => `<tr><td>${r.waktu.slice(5, 16)}</td><td>${nm[r.pompa]}</td><td>${r.durasi}</td><td>${r.awal === null ? "-" : r.awal + " → " + r.target}</td></tr>`).join("");
}

// ---------- 4. Anomali + alert
async function loadAnomali() {
  const [a, al] = await Promise.all([api("/api/anomali?" + qs()), api("/api/alerts?" + qs())]);
  $("#n-cards").innerHTML = card("Data Dianalisis", a.total_data) + card("Anomali Terdeteksi", a.total_anomali, a.persen + "% dari data", a.total_anomali ? "warn" : "ok") +
    card("Alert di Luar Rentang", Object.values(al.per_parameter).reduce((x, y) => x + y, 0), Object.entries(al.per_parameter).map(([k, v]) => `${k}: ${v}`).join(", "));
  mk("n-chart", { type: "bar", data: { labels: a.harian.hari.map((s) => s.slice(5)), datasets: [{ label: "Anomali", data: a.harian.jumlah, backgroundColor: "#c93c3c" }] }, options: { plugins: { legend: { display: false } } } });
  mk("n-reason", { type: "doughnut", data: { labels: Object.keys(a.per_alasan), datasets: [{ data: Object.values(a.per_alasan), backgroundColor: COLORS }] } });
  $("#n-table").innerHTML = "<tr><th>Waktu</th><th>pH</th><th>TDS</th><th>Suhu air</th><th>ADC pH</th><th>Skor</th><th>Dugaan penyebab</th></tr>" +
    a.daftar.map((r) => `<tr><td>${r.waktu.slice(5, 16)}</td><td>${r.ph}</td><td>${r.tds}</td><td>${r.suhu_air}</td><td>${r.adc_ph}</td><td>${r.skor}</td><td>${r.alasan}</td></tr>`).join("");
  $("#n-alert").innerHTML = al.daftar.length
    ? "<tr><th>Waktu</th><th>Parameter</th><th>Nilai</th><th>Ambang</th><th>Jenis</th><th>Status</th></tr>" +
      al.daftar.map((r) => `<tr><td>${r.waktu.slice(5, 16)}</td><td>${r.parameter}</td><td>${r.nilai}</td><td>${r.ambang}</td><td><span class="pill bad">${r.jenis}</span></td><td>${r.status}</td></tr>`).join("")
    : "<tr><td>Tidak ada alert pada rentang ini.</td></tr>";
}

// ---------- 5. ML
function isiForm() {
  if (!DEFAULT) return;
  const j = $("#p-jenis").value, k = DEFAULT.konteks, r = DEFAULT.rekomendasi[j];
  $("#p-awal").value = r.nilai_sekarang; $("#p-target").value = r.target;
  $("#p-vol").value = k.volume_air_l; $("#p-sa").value = k.suhu_air; $("#p-su").value = k.suhu_udara;
  $("#p-lux").value = k.intensitas_cahaya; $("#p-tinggi").value = k.ketinggian_air; $("#p-jam").value = k.jam;
}
async function loadML() {
  [DEFAULT, MODEL] = await Promise.all([api("/api/pompa/default?" + qs()), api("/api/model")]);
  const nm = { nutrisi: "Pompa Nutrisi", asam: "Pompa Asam" }, un = { nutrisi: "ppm", asam: "pH" };
  $("#ml-rek").innerHTML = Object.entries(DEFAULT.rekomendasi).map(([j, r]) =>
    card(nm[j], r.perlu ? `${r.durasi} detik` : "Tidak perlu",
      r.perlu ? `${j === "nutrisi" ? "TDS" : "pH"} ${r.nilai_sekarang} → target ${r.target}` : `${j === "nutrisi" ? "TDS" : "pH"} ${r.nilai_sekarang} masih dalam rentang`, r.perlu ? "warn" : "ok")).join("");
  isiForm();
  renderEval();
}
function renderEval() {
  if (!MODEL || !MODEL.pompa) return;
  const j = $("#e-jenis").value, m = MODEL.pompa[j];
  $("#e-table").innerHTML = "<tr><th>Model</th><th>MAE val</th><th>MAE uji</th><th>RMSE</th><th>MAPE %</th><th>R²</th></tr>" +
    Object.entries(m.hasil).map(([n, h]) => { const b = n === m.model_terbaik ? "best" : "";
      return `<tr><td class="${b}">${n}</td><td class="${b}">${h.val.MAE}</td><td class="${b}">${h.test.MAE}</td><td class="${b}">${h.test.RMSE}</td><td class="${b}">${h.test.MAPE}</td><td class="${b}">${h.test.R2}</td></tr>`; }).join("") +
    `<tr><td colspan="6" class="note">Data: ${m.n_data} kejadian (latih ${m.n_train} / validasi ${m.n_val} / uji ${m.n_test}), dibagi kronologis. Satuan MAE/RMSE: detik.</td></tr>`;
  mk("e-chart", { type: "line", data: { labels: m.sampel_uji.map((_, i) => i + 1),
    datasets: [{ label: "Aktual", data: m.sampel_uji.map((s) => s.aktual), borderColor: "#2b6cb0" },
               { label: "Prediksi", data: m.sampel_uji.map((s) => s.prediksi), borderColor: "#c05621", borderDash: [4, 3] }] },
    options: { scales: { x: { title: { display: true, text: "Kejadian pada data uji" } }, y: { title: { display: true, text: "Durasi (detik)" } } } } });
  const imp = m.pentingnya_fitur;
  mk("e-imp", { type: "bar", data: { labels: Object.keys(imp), datasets: [{ label: "Permutation importance (kenaikan MAE, detik)", data: Object.values(imp), backgroundColor: "#2e7d4f" }] },
    options: { indexAxis: "y" } });
  const an = MODEL.anomali;
  $("#e-anom").innerHTML = "<tr><th>Model</th><th>Precision</th><th>Recall</th><th>F1</th><th>TP</th><th>FP</th><th>FN</th></tr>" +
    Object.entries(an.hasil).map(([n, h]) => { const b = n === an.model_terbaik ? "best" : "";
      return `<tr><td class="${b}">${n}</td><td class="${b}">${h.precision}</td><td class="${b}">${h.recall}</td><td class="${b}">${h.f1}</td><td>${h.TP}</td><td>${h.FP}</td><td>${h.FN}</td></tr>`; }).join("");
  $("#e-note").textContent = `Data uji: ${an.n_test} pembacaan, ${an.anomali_di_uji} di antaranya anomali hasil simulasi. ${MODEL.catatan}`;
}
async function prediksi() {
  const body = { jenis: $("#p-jenis").value, nilai_awal: $("#p-awal").value, nilai_target: $("#p-target").value,
    volume_air_l: $("#p-vol").value, suhu_air: $("#p-sa").value, suhu_udara: $("#p-su").value,
    intensitas_cahaya: $("#p-lux").value, ketinggian_air: $("#p-tinggi").value, jam: $("#p-jam").value };
  try {
    const r = await api("/api/pompa/prediksi", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    $("#p-hasil").innerHTML = `Pompa perlu menyala sekitar<div class="big">${r.durasi_detik} detik</div><span class="muted">Model: ${r.model} • galat tipikal pada data uji ± ${r.galat_tipikal} detik</span>`;
  } catch (e) { $("#p-hasil").textContent = "Gagal: " + e.message; }
}

// ---------- kontrol
const LOADERS = { monitor: loadMonitor, tren: loadTren, aktuator: loadAktuator, anomali: loadAnomali, ml: loadML };
async function muat() {
  $("#f-info").textContent = "memuat…";
  try { await LOADERS[ACTIVE](); $("#f-info").textContent = ""; }
  catch (e) { $("#f-info").textContent = "Error: " + e.message; }
}
$$("#tabs button").forEach((b) => b.addEventListener("click", () => {
  ACTIVE = b.dataset.tab;
  $$("#tabs button").forEach((x) => x.classList.toggle("on", x === b));
  $$(".tab").forEach((x) => x.classList.toggle("on", x.id === "tab-" + ACTIVE));
  muat();
}));
$("#f-preset").addEventListener("change", () => { aturPreset(); muat(); });
["#f-start", "#f-end"].forEach((s) => $(s).addEventListener("change", () => { $("#f-preset").value = "custom"; }));
$("#f-apply").addEventListener("click", muat);
$("#f-device").addEventListener("change", muat);
$("#p-jenis").addEventListener("change", isiForm);
$("#e-jenis").addEventListener("change", renderEval);
$("#p-btn").addEventListener("click", prediksi);

(async () => {
  META = await api("/api/meta");
  $("#f-device").innerHTML = META.devices.map((d) => `<option>${d}</option>`).join("");
  initParams(); aturPreset(); muat();
})();
