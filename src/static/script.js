// script.js — EmoteVision 3D Interactive Frontend
// Socket.IO real-time · Chart.js trend · Particle neural net · Mouse-tilt cards

'use strict';

// ── Config ──────────────────────────────────────────────────────────
let LABELS = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise'];
let EMOTION_COLORS = {
    Angry:'#ff1744', Disgust:'#00e676', Fear:'#d500f9',
    Happy:'#ffea00', Neutral:'#b0bec5', Sad:'#00b0ff', Surprise:'#ff6d00'
};
const EMOJI_MAP = {
    Angry:'\ud83d\ude20', Disgust:'\ud83e\udd22', Fear:'\ud83d\ude28',
    Happy:'\ud83d\ude0a', Neutral:'\ud83d\ude10', Sad:'\ud83d\ude22', Surprise:'\ud83d\ude2e'
};

// ── State ───────────────────────────────────────────────────────────
let trendChart = null;
let trendData  = {};
let trendLabels = [];
const TREND_MAX = 60;
let lastEmotion = '';

// ═══════════════════════════════════════════════════
// SOCKET.IO
// ═══════════════════════════════════════════════════
const socket = io({
    transports: ['websocket', 'polling'],
    reconnectionAttempts: 10,
    reconnectionDelay: 1000,
});

socket.on('connect', () => {
    setStatus('\ud83d\udfe2 LIVE', '#00ff88');
    const ws = document.getElementById('ws-badge');
    if (ws) ws.textContent = '\u26a1 Live';
    console.log('\u2705 WebSocket connected');
});

socket.on('disconnect', () => {
    setStatus('\ud83d\udd34 DISCONNECTED', '#ff1744');
    const ws = document.getElementById('ws-badge');
    if (ws) ws.textContent = '\u26a0 Offline';
});

socket.on('connect_error', () => {
    setStatus('\ud83d\udfe1 RECONNECTING\u2026', '#ffd700');
});

socket.on('emotion_update', (data) => {
    if (data.emotion) {
        updateEmotionDisplay(data);
        updateProbabilities(data.smoothed_probs || data.probabilities || {});
        updateEngagement(data.engagement ?? 50);
        pushTrendPoint(data.smoothed_probs || {});
    } else {
        setStatus('\ud83d\udfe1 SCANNING\u2026', '#ffd700');
    }
    updateStats(data);
});

// ═══════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const cfg = await fetchWithTimeout('/api/config');
        if (cfg.labels)         LABELS         = cfg.labels;
        if (cfg.emotion_colors) EMOTION_COLORS = cfg.emotion_colors;
    } catch (_) {}

    try {
        const stats = await fetchWithTimeout('/api/stats');
        if (stats.model_accuracy != null) {
            const el = document.getElementById('footer-accuracy');
            if (el) el.textContent = `Model Accuracy: ${(stats.model_accuracy * 100).toFixed(2)}% | Trained on 35,000+ images`;
        }
    } catch (_) {}

    initProbabilityBars();
    initTrendChart();
    initTheme();
    wireButtons();
    initParticles();
    init3DTilt();

    console.log('\ud83d\ude80 EmoteVision initialised');
});

// ═══════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════
async function fetchWithTimeout(url, opts = {}, ms = 4000) {
    const ctrl = new AbortController();
    const tid = setTimeout(() => ctrl.abort(), ms);
    try {
        const r = await fetch(url, { ...opts, signal: ctrl.signal });
        clearTimeout(tid);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
    } catch (e) { clearTimeout(tid); throw e; }
}

function setStatus(text, color) {
    const el = document.getElementById('status');
    if (!el) return;
    el.textContent = text;
    // Map color to a data-state attribute for CSS-driven styling
    if (color === '#00ff88') el.dataset.state = 'active';
    else if (color === '#ffd700') el.dataset.state = 'warning';
    else if (color === '#ff1744') el.dataset.state = 'error';
    else {
        el.dataset.state = '';
        el.style.color = color;
        el.style.borderColor = color;
        el.style.background = color.replace(')', ',0.15)').replace('rgb(', 'rgba(');
    }
}

function animateCount(el, from, to, ms = 350) {
    const step = (to - from) / (ms / 16);
    let cur = from;
    const timer = setInterval(() => {
        cur += step;
        const done = step > 0 ? cur >= to : cur <= to;
        el.textContent = Math.round(done ? to : cur);
        if (done) clearInterval(timer);
    }, 16);
}

// ═══════════════════════════════════════════════════
// EMOTION DISPLAY
// ═══════════════════════════════════════════════════
function updateEmotionDisplay(data) {
    const { emotion, confidence } = data;
    const color  = EMOTION_COLORS[emotion] || '#00d4ff';
    const color2 = color + 'aa';

    setStatus('\ud83d\udfe2 FACE DETECTED', '#00ff88');

    // Emoji with bounce animation
    const emojiEl = document.getElementById('emoji');
    if (emojiEl) {
        const newEmoji = EMOJI_MAP[emotion] || '\ud83d\ude10';
        if (newEmoji !== emojiEl.textContent) {
            emojiEl.classList.remove('bounce');
            void emojiEl.offsetWidth;          // force reflow
            emojiEl.textContent = newEmoji;
            emojiEl.classList.add('bounce');
        }
    }

    // Emoji orb ring color theming
    document.querySelectorAll('.emoji-orb-ring, .emoji-orb-ring-2').forEach(ring => {
        ring.style.borderColor = color + '55';
    });
    const emojiBg = document.querySelector('.emoji-bg');
    if (emojiBg) emojiBg.style.background = `radial-gradient(circle, ${color}22 0%, transparent 70%)`;

    // Name gradient
    const nameEl = document.getElementById('emotion-name');
    if (nameEl) {
        nameEl.textContent = emotion.toUpperCase();
        nameEl.style.background = `linear-gradient(135deg, ${color}, ${color2})`;
        nameEl.style.webkitBackgroundClip = 'text';
        nameEl.style.webkitTextFillColor  = 'transparent';
        nameEl.style.backgroundClip       = 'text';
    }

    // Confidence bar (.conf-track > .conf-fill#confidence-fill)
    const pct = confidence.toFixed(1);
    const fill = document.getElementById('confidence-fill');
    if (fill) {
        fill.style.width      = `${pct}%`;
        fill.style.background = `linear-gradient(90deg, ${color}, ${color2})`;
    }
    const confText = document.getElementById('confidence-text');
    if (confText) {
        confText.textContent = `${pct}%`;
        confText.style.color = color;
    }

    lastEmotion = emotion;
}

// ═══════════════════════════════════════════════════
// PROBABILITY BARS  (uses .probability-bar-track from new CSS)
// ═══════════════════════════════════════════════════
function initProbabilityBars() {
    const container = document.getElementById('probabilities');
    if (!container) return;
    container.innerHTML = '';
    LABELS.forEach(emotion => {
        const color = EMOTION_COLORS[emotion] || '#00d4ff';
        const row = document.createElement('div');
        row.className = 'probability-row';
        row.innerHTML = `
            <div class="probability-label" style="color:${color}">${emotion.toUpperCase()}</div>
            <div class="probability-bar-track">
                <div class="probability-bar" id="bar-${emotion.toLowerCase()}"
                     style="width:0%;background:linear-gradient(90deg,${color},${color}aa)"></div>
            </div>
            <div class="probability-value" id="val-${emotion.toLowerCase()}">0.0%</div>`;
        container.appendChild(row);
    });
}

function updateProbabilities(probs) {
    LABELS.forEach(emotion => {
        const val  = (probs[emotion] || 0).toFixed(1);
        const bar  = document.getElementById(`bar-${emotion.toLowerCase()}`);
        const valEl= document.getElementById(`val-${emotion.toLowerCase()}`);
        if (bar) bar.style.width = `${val}%`;
        if (valEl) {
            valEl.textContent = `${val}%`;
            valEl.style.color = parseFloat(val) > 50
                ? (EMOTION_COLORS[emotion] || '#00d4ff') : '';
            valEl.style.fontWeight = parseFloat(val) > 50 ? '700' : '';
        }
    });
}

// ═══════════════════════════════════════════════════
// ENGAGEMENT GAUGE  (SVG id="gauge-fill")
// ═══════════════════════════════════════════════════
function updateEngagement(score) {
    const clipped = Math.min(100, Math.max(0, score));
    const engEl = document.getElementById('engagement');
    if (engEl) engEl.textContent = Math.round(clipped);

    const arc = document.getElementById('gauge-fill');
    if (arc) {
        const offset = 157 * (1 - clipped / 100);
        arc.style.strokeDashoffset = offset;
        const hue = Math.round(clipped * 1.2);
        arc.style.stroke = `hsl(${hue},90%,55%)`;
    }
}

// ═══════════════════════════════════════════════════
// STATS
// ═══════════════════════════════════════════════════
function updateStats(data) {
    if (data.latency != null) {
        const el = document.getElementById('latency');
        if (el) el.textContent = `${Math.round(data.latency)}ms`;
    }
    if (data.faces_detected != null) {
        const el = document.getElementById('faces');
        if (el) el.textContent = data.faces_detected;
    }
    if (data.total_predictions != null) {
        const el = document.getElementById('total-predictions');
        if (el) {
            const cur = parseInt(el.textContent) || 0;
            if (data.total_predictions > cur) animateCount(el, cur, data.total_predictions);
        }
    }
    if (data.uptime != null) {
        const u = data.uptime;
        const h = Math.floor(u / 3600);
        const m = Math.floor((u % 3600) / 60);
        const s = u % 60;
        const el = document.getElementById('uptime');
        if (el) el.textContent = h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${s}s` : `${s}s`;
    }
}

// ═══════════════════════════════════════════════════
// CHART.JS TREND
// ═══════════════════════════════════════════════════
function initTrendChart() {
    LABELS.forEach(l => { trendData[l] = new Array(TREND_MAX).fill(null); });
    trendLabels = new Array(TREND_MAX).fill('');

    const el = document.getElementById('trend-chart');
    if (!el) return;
    const ctx = el.getContext('2d');
    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';

    trendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: trendLabels,
            datasets: LABELS.map(label => ({
                label,
                data: trendData[label],
                borderColor: EMOTION_COLORS[label] || '#888',
                backgroundColor: (EMOTION_COLORS[label] || '#888') + '18',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.4,
                fill: false,
            })),
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            plugins: {
                legend: { labels: { color: isDark ? '#8888aa' : '#444', boxWidth: 12, font: { size: 11 } } },
                tooltip: { mode: 'index', intersect: false },
            },
            scales: {
                x: { ticks: { display: false }, grid: { color: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.06)' } },
                y: {
                    min: 0, max: 100,
                    ticks: { color: isDark ? '#666' : '#888', font: { size: 11 }, callback: v => `${v}%` },
                    grid: { color: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.06)' },
                },
            },
        },
    });
}

function pushTrendPoint(probs) {
    if (!trendChart) return;
    trendLabels.shift();
    trendLabels.push('');
    trendChart.data.datasets.forEach(ds => {
        trendData[ds.label].shift();
        trendData[ds.label].push(parseFloat((probs[ds.label] || 0).toFixed(1)));
        ds.data = trendData[ds.label];
    });
    trendChart.update();
}

// ═══════════════════════════════════════════════════
// THEME TOGGLE
// ═══════════════════════════════════════════════════
function initTheme() {
    const saved = localStorage.getItem('emotevision-theme') || 'dark';
    applyTheme(saved);
    document.getElementById('theme-toggle')?.addEventListener('click', () => {
        const cur = document.documentElement.getAttribute('data-theme');
        const next = cur === 'dark' ? 'light' : 'dark';
        applyTheme(next);
        localStorage.setItem('emotevision-theme', next);
    });
}

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    const btn = document.getElementById('theme-toggle');
    if (btn) btn.textContent = theme === 'dark' ? '\ud83c\udf19' : '\u2600\ufe0f';

    if (trendChart) {
        const dark = theme === 'dark';
        const gc = dark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.06)';
        trendChart.options.scales.x.grid.color = gc;
        trendChart.options.scales.y.grid.color = gc;
        trendChart.options.scales.y.ticks.color = dark ? '#666' : '#888';
        trendChart.options.plugins.legend.labels.color = dark ? '#8888aa' : '#444';
        trendChart.update();
    }
}

// ═══════════════════════════════════════════════════
// BUTTONS
// ═══════════════════════════════════════════════════
function wireButtons() {
    document.getElementById('export-btn')?.addEventListener('click', async () => {
        try {
            const resp = await fetch('/api/session/export');
            if (resp.status === 404) { alert('No session data yet.'); return; }
            const blob = await resp.blob();
            const url = URL.createObjectURL(blob);
            const a = Object.assign(document.createElement('a'), { href: url, download: 'emotevision_session.csv' });
            document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
        } catch (e) { console.error('Export failed', e); }
    });

    document.getElementById('reset-btn')?.addEventListener('click', async () => {
        if (!confirm('Reset session data?')) return;
        await fetch('/api/session/reset', { method: 'POST' });
        LABELS.forEach(l => { trendData[l] = new Array(TREND_MAX).fill(null); });
        trendChart?.update();
        const el = document.getElementById('total-predictions');
        if (el) el.textContent = '0';
    });
}

// ═══════════════════════════════════════════════════
// PARTICLE NEURAL NETWORK CANVAS
// ═══════════════════════════════════════════════════
function initParticles() {
    const cvs = document.getElementById('particle-canvas');
    if (!cvs) return;
    const ctx = cvs.getContext('2d');

    let w, h, particles, mouse = { x: -9999, y: -9999 };
    const COUNT = 80;
    const LINK_DIST = 150;
    const MOUSE_DIST = 200;

    function resize() {
        w = cvs.width  = window.innerWidth;
        h = cvs.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    document.addEventListener('mousemove', e => { mouse.x = e.clientX; mouse.y = e.clientY; });

    class P {
        constructor() { this.reset(); }
        reset() {
            this.x = Math.random() * w;
            this.y = Math.random() * h;
            this.vx = (Math.random() - 0.5) * 0.5;
            this.vy = (Math.random() - 0.5) * 0.5;
            this.r  = Math.random() * 2 + 1;
        }
        update() {
            this.x += this.vx;
            this.y += this.vy;
            if (this.x < 0 || this.x > w) this.vx *= -1;
            if (this.y < 0 || this.y > h) this.vy *= -1;

            // drift towards mouse gently
            const dx = mouse.x - this.x, dy = mouse.y - this.y;
            const d = Math.sqrt(dx*dx + dy*dy);
            if (d < MOUSE_DIST && d > 0) {
                this.vx += (dx / d) * 0.015;
                this.vy += (dy / d) * 0.015;
            }
            // dampen
            this.vx *= 0.998; this.vy *= 0.998;
        }
    }

    particles = Array.from({ length: COUNT }, () => new P());

    function frame() {
        ctx.clearRect(0, 0, w, h);

        // links
        for (let i = 0; i < COUNT; i++) {
            for (let j = i + 1; j < COUNT; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const d = Math.sqrt(dx*dx + dy*dy);
                if (d < LINK_DIST) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    const alpha = 1 - d / LINK_DIST;
                    ctx.strokeStyle = `rgba(0,212,255,${(alpha * 0.22).toFixed(3)})`;
                    ctx.lineWidth = alpha * 1.2;
                    ctx.stroke();
                }
            }
        }

        // dots
        for (const p of particles) {
            p.update();
            const dMouse = Math.sqrt((p.x - mouse.x)**2 + (p.y - mouse.y)**2);
            const glow = dMouse < MOUSE_DIST ? 0.7 : 0.35;
            const radius = dMouse < MOUSE_DIST ? p.r * 1.6 : p.r;
            ctx.beginPath();
            ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(0,212,255,${glow})`;
            ctx.fill();
            if (dMouse < MOUSE_DIST) {
                ctx.beginPath();
                ctx.arc(p.x, p.y, radius + 3, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(0,212,255,${(glow * 0.15).toFixed(3)})`;
                ctx.fill();
            }
        }

        requestAnimationFrame(frame);
    }
    frame();
}

// ═══════════════════════════════════════════════════
// 3D MOUSE-TILT CARDS + SPOTLIGHT + RIPPLE
// ═══════════════════════════════════════════════════
function init3DTilt() {
    const TILT_MAX = 6; // degrees
    const els = document.querySelectorAll('.card, .stat-card');

    els.forEach(el => {
        // Add spotlight div for cursor-following glow
        const spot = document.createElement('div');
        spot.className = 'card-spotlight';
        el.style.position = 'relative';
        el.appendChild(spot);

        el.addEventListener('mousemove', e => {
            const rect = el.getBoundingClientRect();
            const cx = rect.left + rect.width  / 2;
            const cy = rect.top  + rect.height / 2;
            const rx = ((e.clientY - cy) / (rect.height / 2)) * -TILT_MAX;
            const ry = ((e.clientX - cx) / (rect.width  / 2)) *  TILT_MAX;
            el.style.transform = `perspective(${rect.width > 400 ? 1200 : 900}px) rotateX(${rx.toFixed(2)}deg) rotateY(${ry.toFixed(2)}deg) scale3d(1.02,1.02,1.02)`;

            // Move spotlight to cursor position
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            spot.style.left = x + 'px';
            spot.style.top  = y + 'px';
        });

        el.addEventListener('mouseleave', () => {
            el.style.transform = '';
        });

        // Click ripple effect
        el.addEventListener('click', e => {
            const rect = el.getBoundingClientRect();
            const ripple = document.createElement('div');
            ripple.className = 'card-ripple';
            const size = Math.max(rect.width, rect.height);
            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = (e.clientX - rect.left - size / 2) + 'px';
            ripple.style.top  = (e.clientY - rect.top  - size / 2) + 'px';
            el.appendChild(ripple);
            ripple.addEventListener('animationend', () => ripple.remove());
        });
    });
}

// ═══════════════════════════════════════════════════
// VISIBILITY & KEYBOARD
// ═══════════════════════════════════════════════════
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && !socket.connected) socket.connect();
});

document.addEventListener('keydown', e => {
    if (e.ctrlKey || e.metaKey) return;
    if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        fetchWithTimeout('/api/predict').then(data => {
            if (data.emotions?.[0]) updateEmotionDisplay(data.emotions[0]);
        }).catch(console.error);
    }
    if (e.key === 'h' || e.key === 'H') { e.preventDefault(); window.location.href = '/'; }
});

// Parallax blobs on scroll
window.addEventListener('scroll', () => {
    const y = window.scrollY;
    document.querySelectorAll('.blob').forEach((b, i) => {
        const speed = 0.03 + i * 0.015;
        b.style.transform = `translateY(${y * speed}px)`;
    });
}, { passive: true });

window.addEventListener('error', e => console.error('Unhandled:', e.error));
