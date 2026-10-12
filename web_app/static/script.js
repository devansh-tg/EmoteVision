"use strict";

let LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"];
let EMOTION_COLORS = {
    Angry: "#E5484D",
    Disgust: "#30A46C",
    Fear: "#8E4EC6",
    Happy: "#F5A524",
    Neutral: "#8B949E",
    Sad: "#3E8EDE",
    Surprise: "#F76808",
};

const TREND_MAX = 60;
const TREND_FRAME_MS = 120;
let trendChart = null;
let trendData = {};
let trendLabels = new Array(TREND_MAX).fill("");
let restPollTimer = null;
let trendTimer = null;
let latestTrendProbs = null;

function colorFor(label) {
    return EMOTION_COLORS[label] || "#4fb3d8";
}

async function fetchJson(url, opts = {}, ms = 4000) {
    const ctrl = new AbortController();
    const tid = setTimeout(() => ctrl.abort(), ms);
    try {
        const response = await fetch(url, { ...opts, signal: ctrl.signal });
        clearTimeout(tid);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
    } catch (error) {
        clearTimeout(tid);
        throw error;
    }
}

function setStatus(text, state) {
    const el = document.getElementById("status");
    if (!el) return;
    el.dataset.state = state;
    el.innerHTML = `<span class="status-dot"></span>${text}`;
}

function setSocketBadge(text) {
    const el = document.getElementById("ws-badge");
    if (el) el.textContent = text;
}

function setFeedState(text, state = "active") {
    const el = document.getElementById("feed-state");
    if (!el) return;
    el.textContent = text;
    el.dataset.state = state;
}

function formatUptime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    if (h) return `${h}h ${m}m`;
    if (m) return `${m}m ${s}s`;
    return `${s}s`;
}

function updateStats(data) {
    if (data.latency != null) {
        const el = document.getElementById("latency");
        if (el) el.textContent = `${Math.round(data.latency)}ms`;
    }
    if (data.faces_detected != null) {
        const el = document.getElementById("faces");
        if (el) el.textContent = data.faces_detected;
    }
    if (data.total_predictions != null) {
        const el = document.getElementById("total-predictions");
        if (el) el.textContent = data.total_predictions;
    }
    if (data.uptime != null) {
        const el = document.getElementById("uptime");
        if (el) el.textContent = formatUptime(data.uptime);
    }
}

function updateEngagement(score) {
    const clipped = Math.min(100, Math.max(0, Number(score) || 0));
    const value = document.getElementById("engagement");
    if (value) value.textContent = Math.round(clipped);

    const arc = document.getElementById("gauge-fill");
    if (arc) {
        arc.style.strokeDashoffset = 157 * (1 - clipped / 100);
        arc.style.stroke = clipped >= 70 ? "#7ccf9b" : clipped >= 40 ? "#f5a524" : "#e5484d";
    }
}

function updateEmotionDisplay(data) {
    const emotion = data.emotion || "Waiting";
    const confidence = Number(data.confidence) || 0;
    const color = colorFor(emotion);

    const name = document.getElementById("emotion-name");
    if (name) name.textContent = emotion;

    const metricName = document.getElementById("emotion-name-metric");
    if (metricName) metricName.textContent = emotion;

    const token = document.getElementById("emotion-token");
    if (token) {
        token.textContent = emotion === "Waiting" ? "NE" : emotion.slice(0, 2).toUpperCase();
        token.style.color = color;
        token.style.background = `${color}22`;
    }

    const text = document.getElementById("confidence-text");
    if (text) text.textContent = `${confidence.toFixed(1)}%`;

    const metricConfidence = document.getElementById("confidence-metric");
    if (metricConfidence) metricConfidence.textContent = `${confidence.toFixed(1)}%`;

    const fill = document.getElementById("confidence-fill");
    if (fill) {
        fill.style.width = `${Math.min(100, Math.max(0, confidence))}%`;
        fill.style.background = color;
    }
}

function initProbabilityBars() {
    const container = document.getElementById("probabilities");
    if (!container) return;
    container.innerHTML = "";

    LABELS.forEach((emotion) => {
        const row = document.createElement("div");
        row.className = "probability-row";
        row.innerHTML = `
            <div class="probability-label">${emotion}</div>
            <div class="probability-bar-track">
                <div class="probability-bar" id="bar-${emotion.toLowerCase()}"></div>
            </div>
            <div class="probability-value" id="val-${emotion.toLowerCase()}">0.0%</div>
        `;
        container.appendChild(row);
    });
}

function updateProbabilities(probs) {
    LABELS.forEach((emotion) => {
        const value = Number(probs[emotion] || 0);
        const bar = document.getElementById(`bar-${emotion.toLowerCase()}`);
        const text = document.getElementById(`val-${emotion.toLowerCase()}`);
        if (bar) {
            bar.style.width = `${Math.min(100, Math.max(0, value))}%`;
            bar.style.background = colorFor(emotion);
        }
        if (text) text.textContent = `${value.toFixed(1)}%`;
    });
}

function initTrendChart() {
    if (typeof Chart === "undefined") return;

    LABELS.forEach((label) => {
        trendData[label] = new Array(TREND_MAX).fill(null);
    });

    const canvas = document.getElementById("trend-chart");
    if (!canvas) return;

    const dark = document.documentElement.getAttribute("data-theme") === "dark";
    trendChart = new Chart(canvas.getContext("2d"), {
        type: "line",
        data: {
            labels: trendLabels,
            datasets: LABELS.map((label) => ({
                label,
                data: trendData[label],
                borderColor: colorFor(label),
                borderWidth: 2.2,
                pointRadius: 0,
                tension: 0.45,
                cubicInterpolationMode: "monotone",
            })),
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 180,
                easing: "easeOutQuad",
            },
            plugins: {
                legend: {
                    labels: {
                        color: dark ? "#9aa4af" : "#59636f",
                        boxWidth: 10,
                        font: { size: 11 },
                    },
                },
            },
            interaction: {
                intersect: false,
                mode: "index",
            },
            scales: {
                x: {
                    ticks: { display: false },
                    grid: { color: dark ? "#252c34" : "#dbe1e8" },
                },
                y: {
                    min: 0,
                    max: 100,
                    ticks: {
                        color: dark ? "#9aa4af" : "#59636f",
                        callback: (value) => `${value}%`,
                    },
                    grid: { color: dark ? "#252c34" : "#dbe1e8" },
                },
            },
        },
    });
}

function pushTrendPoint(probs, mode = "active") {
    if (!trendChart) return;
    trendLabels.shift();
    trendLabels.push("");
    trendChart.data.datasets.forEach((dataset) => {
        trendData[dataset.label].shift();
        trendData[dataset.label].push(Number(probs[dataset.label] || 0));
        dataset.data = trendData[dataset.label];
    });
    trendChart.update(mode);
}

function ensureTrendPump() {
    if (trendTimer != null) return;
    trendTimer = window.setInterval(() => {
        if (!latestTrendProbs) return;
        pushTrendPoint(latestTrendProbs, "active");
        latestTrendProbs = null;
    }, TREND_FRAME_MS);
}

function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    const button = document.getElementById("theme-toggle");
    if (button) button.textContent = theme === "dark" ? "D" : "L";

    if (trendChart) {
        const dark = theme === "dark";
        const grid = dark ? "#252c34" : "#dbe1e8";
        const tick = dark ? "#9aa4af" : "#59636f";
        trendChart.options.scales.x.grid.color = grid;
        trendChart.options.scales.y.grid.color = grid;
        trendChart.options.scales.y.ticks.color = tick;
        trendChart.options.plugins.legend.labels.color = tick;
        trendChart.update("none");
    }
}

function initTheme() {
    const saved = localStorage.getItem("emotevision-theme") || "dark";
    applyTheme(saved);
    document.getElementById("theme-toggle")?.addEventListener("click", () => {
        const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
        localStorage.setItem("emotevision-theme", next);
        applyTheme(next);
    });
}

function wireButtons() {
    document.getElementById("export-btn")?.addEventListener("click", async () => {
        try {
            const response = await fetch("/api/session/export");
            if (response.status === 404) {
                setSocketBadge("No session data");
                return;
            }
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const link = Object.assign(document.createElement("a"), {
                href: url,
                download: "emotevision_session.csv",
            });
            document.body.appendChild(link);
            link.click();
            link.remove();
            URL.revokeObjectURL(url);
        } catch (error) {
            console.error("Export failed", error);
            setSocketBadge("Export failed");
        }
    });

    document.getElementById("reset-btn")?.addEventListener("click", async () => {
        try {
            await fetch("/api/session/reset", { method: "POST" });
            LABELS.forEach((label) => {
                trendData[label] = new Array(TREND_MAX).fill(null);
            });
            latestTrendProbs = null;
            trendChart?.update("none");
            document.getElementById("total-predictions").textContent = "0";
            setSocketBadge("Session reset");
        } catch (error) {
            console.error("Reset failed", error);
            setSocketBadge("Reset failed");
        }
    });
}

function normalizeRestPrediction(payload) {
    const primary = payload.emotions?.[0];
    if (!primary) {
        return {
            emotion: null,
            faces_detected: payload.faces_detected ?? 0,
            engagement: payload.engagement ?? 50,
            total_predictions: payload.stats?.total_predictions,
            uptime: payload.stats?.uptime,
        };
    }
    return {
        emotion: primary.emotion,
        confidence: primary.confidence,
        probabilities: primary.probabilities,
        smoothed_probs: payload.smoothed_probs,
        engagement: payload.engagement,
        latency: primary.latency,
        faces_detected: payload.faces_detected,
        total_predictions: payload.stats?.total_predictions,
        uptime: payload.stats?.uptime,
    };
}

function startRestPolling() {
    if (restPollTimer) return;

    const poll = async () => {
        let delay = 1200;
        try {
            const payload = await fetchJson("/api/predict", {}, 2500);
            const data = normalizeRestPrediction(payload);
            if (data) handleEmotionUpdate(data);
            setSocketBadge("Inference online");
            setFeedState("Live camera", "active");
        } catch (error) {
            setStatus("Camera online", "warning");
            setSocketBadge("Inference unavailable");
            setFeedState("Camera live · inference unavailable", "warning");
            delay = 2500;
        }
        restPollTimer = window.setTimeout(poll, delay);
    };

    poll();
}

function handleEmotionUpdate(data) {
    if (data.error) {
        setStatus(data.error, "error");
        setSocketBadge("Inference unavailable");
        setFeedState("Camera live · inference unavailable", "warning");
        updateStats(data);
        return;
    }
    if (data.emotion) {
        setStatus("Face detected", "active");
        setFeedState("Live camera · face detected", "active");
        updateEmotionDisplay(data);
        const probs = data.smoothed_probs || data.probabilities || {};
        updateProbabilities(probs);
        updateEngagement(data.engagement ?? 50);
        latestTrendProbs = probs;
        ensureTrendPump();
    } else {
        setStatus("Scanning", "warning");
        setFeedState("Live camera · scanning", "active");
    }
    updateStats(data);
}

function initSocket() {
    setStatus("Camera online", "active");
    setSocketBadge("Inference initializing");
    startRestPolling();
}

async function initConfig() {
    try {
        const cfg = await fetchJson("/api/config");
        if (cfg.labels) LABELS = cfg.labels;
        if (cfg.emotion_colors) EMOTION_COLORS = cfg.emotion_colors;
    } catch (error) {
        console.warn("Using bundled UI config", error);
    }
}

async function initStats() {
    try {
        const stats = await fetchJson("/api/stats");
        if (stats.model_accuracy != null) {
            const el = document.getElementById("footer-accuracy");
            if (el) el.textContent = `Validation accuracy ${(stats.model_accuracy * 100).toFixed(2)}%`;
        }
    } catch (error) {
        console.warn("Stats unavailable", error);
    }
}

document.addEventListener("DOMContentLoaded", async () => {
    initTheme();
    await initConfig();
    initProbabilityBars();
    initTrendChart();
    wireButtons();
    initSocket();
    initStats();
    const feed = document.getElementById("video-feed");
    feed?.addEventListener("load", () => setFeedState("Live camera", "active"));
    feed?.addEventListener("error", () => {
        setStatus("Camera unavailable", "error");
        setFeedState("Camera unavailable", "error");
    });
});

document.addEventListener("keydown", (event) => {
    if (event.ctrlKey || event.metaKey) return;
    if (event.key === "r" || event.key === "R") {
        event.preventDefault();
        fetchJson("/api/predict")
            .then(normalizeRestPrediction)
            .then((data) => {
                if (data) handleEmotionUpdate(data);
            })
            .catch((error) => console.error("Manual prediction failed", error));
    }
});
