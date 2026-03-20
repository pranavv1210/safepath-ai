const PAST_STEPS = 4;
const FUTURE_STEPS = 6;

const PATH_STYLE = ["safe", "caution", "risk"];
const PATH_COLORS = {
  safe: "#22b573",
  caution: "#f4b63d",
  risk: "#ef5d6c",
  past: "#2f80ff",
};

const DEFAULT_BOUNDS = {
  minX: -8,
  maxX: 8,
  minY: -6,
  maxY: 6,
};

const canvas = document.getElementById("trajectory-canvas");
const ctx = canvas.getContext("2d");
const playButton = document.getElementById("play-button");
const stepButton = document.getElementById("step-button");
const resetButton = document.getElementById("reset-button");
const pathCards = document.getElementById("path-cards");
const statusMessage = document.getElementById("status-message");
const loadingIndicator = document.getElementById("loading-indicator");
const liveIndicator = document.getElementById("live-indicator");
const frameIndicator = document.getElementById("frame-indicator");
const footerFrameIndicator = document.getElementById("footer-frame-indicator");
const behaviorLabel = document.getElementById("behavior-label");
const directionLabel = document.getElementById("direction-label");
const targetName = document.getElementById("target-name");
const targetNameSummary = document.getElementById("target-name-summary");
const primaryRiskChip = document.getElementById("primary-risk-chip");
const targetVelocity = document.getElementById("target-velocity");
const targetDistance = document.getElementById("target-distance");
const maxCollisionProbability = document.getElementById("max-collision-probability");
const collisionProgress = document.getElementById("collision-progress");
const timeToCollision = document.getElementById("time-to-collision");
const latencyMetric = document.getElementById("latency-metric");
const timelineProgress = document.getElementById("timeline-progress");
const timelineThumb = document.getElementById("timeline-thumb");
const adeMetric = document.getElementById("ade-metric");
const fdeMetric = document.getElementById("fde-metric");
const observationCount = document.getElementById("observation-count");
const observedCoordinates = document.getElementById("observed-coordinates");
const predictedCoordinates = document.getElementById("predicted-coordinates");
const copyObservedButton = document.getElementById("copy-observed-button");
const copyPredictedButton = document.getElementById("copy-predicted-button");
const headerObservedWindow = document.getElementById("header-observed-window");
const headerBehaviorChip = document.getElementById("header-behavior-chip");
const headerRiskChip = document.getElementById("header-risk-chip");

let observedHistory = [];
let latestPrediction = null;
let revealStep = 0;
let currentBounds = { ...DEFAULT_BOUNDS };

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
}

function getModelWindow() {
  return observedHistory.slice(-PAST_STEPS);
}

function withVelocity(points) {
  const result = points.map((point) => [...point]);
  for (let i = 0; i < result.length; i += 1) {
    if (i === 0) {
      result[i][2] = 0;
      result[i][3] = 0;
    } else {
      result[i][2] = Number((result[i][0] - result[i - 1][0]).toFixed(4));
      result[i][3] = Number((result[i][1] - result[i - 1][1]).toFixed(4));
    }
  }
  return result;
}

function normalizePoints(points) {
  const flat = points.flat().filter(Boolean);
  if (!flat.length) {
    return { ...DEFAULT_BOUNDS };
  }

  const xs = flat.map((point) => point[0]);
  const ys = flat.map((point) => point[1]);
  return {
    minX: Math.min(...xs, DEFAULT_BOUNDS.minX),
    maxX: Math.max(...xs, DEFAULT_BOUNDS.maxX),
    minY: Math.min(...ys, DEFAULT_BOUNDS.minY),
    maxY: Math.max(...ys, DEFAULT_BOUNDS.maxY),
  };
}

function projectPoint(point, bounds) {
  const padding = 58;
  const width = canvas.clientWidth - padding * 2;
  const height = canvas.clientHeight - padding * 2;
  const x = padding + ((point[0] - bounds.minX) / (bounds.maxX - bounds.minX + 1e-6)) * width;
  const y =
    canvas.clientHeight -
    (padding + ((point[1] - bounds.minY) / (bounds.maxY - bounds.minY + 1e-6)) * height);
  return [x, y];
}

function drawAxisTicks(bounds) {
  const padding = 58;
  const innerWidth = canvas.clientWidth - padding * 2;
  const innerHeight = canvas.clientHeight - padding * 2;
  const xStart = Math.ceil(bounds.minX);
  const xEnd = Math.floor(bounds.maxX);
  const yStart = Math.ceil(bounds.minY);
  const yEnd = Math.floor(bounds.maxY);

  ctx.save();
  ctx.font = '11px "IBM Plex Mono"';
  ctx.fillStyle = "rgba(23, 36, 55, 0.82)";
  ctx.strokeStyle = "rgba(23, 36, 55, 0.18)";
  ctx.lineWidth = 1;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  for (let xValue = xStart; xValue <= xEnd; xValue += 1) {
    const ratio = (xValue - bounds.minX) / Math.max(bounds.maxX - bounds.minX, 1e-6);
    const x = padding + ratio * innerWidth;
    ctx.beginPath();
    ctx.moveTo(x, padding + 2);
    ctx.lineTo(x, padding + 8);
    ctx.moveTo(x, canvas.clientHeight - padding - 8);
    ctx.lineTo(x, canvas.clientHeight - padding - 2);
    ctx.stroke();
    ctx.fillText(String(xValue), x, canvas.clientHeight - padding - 18);
  }

  ctx.textAlign = "right";
  for (let yValue = yStart; yValue <= yEnd; yValue += 1) {
    const ratio = (yValue - bounds.minY) / Math.max(bounds.maxY - bounds.minY, 1e-6);
    const y = canvas.clientHeight - (padding + ratio * innerHeight);
    ctx.beginPath();
    ctx.moveTo(padding + 2, y);
    ctx.lineTo(padding + 8, y);
    ctx.moveTo(canvas.clientWidth - padding - 8, y);
    ctx.lineTo(canvas.clientWidth - padding - 2, y);
    ctx.stroke();
    ctx.fillText(String(yValue), padding + 28, y);
  }

  ctx.restore();
}

function unprojectPoint(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const padding = 58;
  const xPx = clientX - rect.left;
  const yPx = clientY - rect.top;
  const usableWidth = rect.width - padding * 2;
  const usableHeight = rect.height - padding * 2;
  const x =
    currentBounds.minX + ((xPx - padding) / Math.max(usableWidth, 1)) * (currentBounds.maxX - currentBounds.minX);
  const y =
    currentBounds.minY +
    ((rect.height - yPx - padding) / Math.max(usableHeight, 1)) * (currentBounds.maxY - currentBounds.minY);

  return [
    Number(Math.max(currentBounds.minX, Math.min(currentBounds.maxX, x)).toFixed(3)),
    Number(Math.max(currentBounds.minY, Math.min(currentBounds.maxY, y)).toFixed(3)),
  ];
}

function drawPath(points, color, options = {}) {
  const { dashed = false, visiblePoints = points.length, dots = false, width = 4 } = options;
  if (!points.length) return;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  if (dashed) ctx.setLineDash([10, 10]);

  const visible = points.slice(0, Math.max(visiblePoints, 1));
  ctx.beginPath();
  visible.forEach((point, index) => {
    const [x, y] = projectPoint(point, currentBounds);
    if (index === 0) {
      ctx.moveTo(x, y);
    } else if (index === visible.length - 1) {
      ctx.lineTo(x, y);
    } else {
      const [nextX, nextY] = projectPoint(visible[index + 1], currentBounds);
      const cx = (x + nextX) / 2;
      const cy = (y + nextY) / 2;
      ctx.quadraticCurveTo(x, y, cx, cy);
    }
  });
  ctx.stroke();

  if (dots) {
    points.slice(0, Math.max(visiblePoints, 1)).forEach((point) => {
      const [x, y] = projectPoint(point, currentBounds);
      ctx.beginPath();
      ctx.fillStyle = color;
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.55)";
      ctx.lineWidth = 2;
      ctx.stroke();
    });
  }

  ctx.restore();
}

function buildLabelRect(x, y, width, height) {
  return {
    left: x,
    top: y,
    right: x + width,
    bottom: y + height,
  };
}

function rectsOverlap(a, b) {
  return !(a.right < b.left || a.left > b.right || a.bottom < b.top || a.top > b.bottom);
}

function pickLabelPlacement(x, y, labelWidth, labelHeight, occupiedRects) {
  const candidates = [
    { x: x + 14, y: y - labelHeight - 10 },
    { x: x + 14, y: y + 8 },
    { x: x - labelWidth - 14, y: y - labelHeight - 10 },
    { x: x - labelWidth - 14, y: y + 8 },
    { x: x - labelWidth / 2, y: y - labelHeight - 16 },
    { x: x - labelWidth / 2, y: y + 12 },
  ];
  const maxX = canvas.clientWidth - labelWidth - 8;
  const maxY = canvas.clientHeight - labelHeight - 8;

  for (const candidate of candidates) {
    const clampedX = Math.min(Math.max(8, candidate.x), maxX);
    const clampedY = Math.min(Math.max(8, candidate.y), maxY);
    const rect = buildLabelRect(clampedX, clampedY, labelWidth, labelHeight);
    if (!occupiedRects.some((existing) => rectsOverlap(existing, rect))) {
      occupiedRects.push(rect);
      return rect;
    }
  }

  const fallback = buildLabelRect(Math.min(Math.max(8, x + 14), maxX), Math.min(Math.max(8, y + 10), maxY), labelWidth, labelHeight);
  occupiedRects.push(fallback);
  return fallback;
}

function drawLabeledPoints(points, color, options = {}) {
  const {
    visiblePoints = points.length,
    labelFormatter = (index) => `${index + 1}`,
    pointRadius = 5,
    occupiedRects = [],
    pointIndices = null,
  } = options;
  const visible = points.slice(0, Math.max(visiblePoints, 1));
  if (!visible.length) return;

  ctx.save();
  ctx.font = '12px "IBM Plex Mono"';
  ctx.textBaseline = "middle";

  const indicesToDraw = pointIndices ?? visible.map((_, index) => index);

  indicesToDraw.forEach((index) => {
    const point = visible[index];
    if (!point) return;
    const [x, y] = projectPoint(point, currentBounds);

    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.arc(x, y, pointRadius, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "rgba(255, 255, 255, 0.95)";
    ctx.beginPath();
    ctx.arc(x, y, pointRadius <= 5 ? 1.8 : 2.2, 0, Math.PI * 2);
    ctx.fill();

    const label = labelFormatter(index, point);
    const labelWidth = ctx.measureText(label).width + 16;
    const labelHeight = 22;
    const rect = pickLabelPlacement(x, y, labelWidth, labelHeight, occupiedRects);

    ctx.fillStyle = "rgba(255, 255, 255, 0.92)";
    ctx.strokeStyle = "rgba(24, 34, 48, 0.08)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(rect.left, rect.top, labelWidth, labelHeight, 10);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#182230";
    ctx.fillText(label, rect.left + 8, rect.top + labelHeight / 2);
  });

  ctx.restore();
}

function drawRoadContext() {
  const stageWidth = canvas.clientWidth;
  const stageHeight = canvas.clientHeight;
  const centerX = stageWidth * 0.5;
  const centerY = stageHeight * 0.54;

  ctx.save();
  ctx.fillStyle = "rgba(61, 77, 101, 0.12)";
  ctx.fillRect(centerX - 105, 0, 210, stageHeight);
  ctx.fillRect(0, centerY - 82, stageWidth, 164);

  ctx.fillStyle = "rgba(255,255,255,0.34)";
  for (let i = 0; i < 8; i += 1) {
    ctx.fillRect(centerX - 12, 34 + i * 72, 24, 34);
  }
  for (let i = 0; i < Math.max(8, Math.floor(stageWidth / 76)); i += 1) {
    ctx.fillRect(52 + i * 76, centerY - 10, 34, 20);
  }

  ctx.strokeStyle = "rgba(255,255,255,0.24)";
  ctx.lineWidth = 2;
  ctx.setLineDash([14, 14]);
  ctx.beginPath();
  ctx.moveTo(centerX, 0);
  ctx.lineTo(centerX, stageHeight);
  ctx.moveTo(0, centerY);
  ctx.lineTo(stageWidth, centerY);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "rgba(255,255,255,0.2)";
  for (let i = 0; i < 7; i += 1) {
    ctx.fillRect(centerX - 150 + i * 18, centerY + 92, 10, 44);
    ctx.fillRect(centerX - 150 + i * 18, centerY - 136, 10, 44);
  }
  ctx.restore();
}

function classifyDirection(vx, vy) {
  if (Math.abs(vy) < 0.12) return "Forward";
  return vy < 0 ? "Right" : "Left";
}

function classifyBehavior(points) {
  if (points.length < 2) return "Waiting";
  const [sx, sy] = points[0];
  const [ex, ey] = points[points.length - 1];
  const dx = ex - sx;
  const dy = ey - sy;
  const speed = Math.sqrt(dx * dx + dy * dy) / Math.max(points.length - 1, 1);

  if (speed < 0.18) return "Stopping";
  if (Math.abs(dy) > Math.abs(dx) * 0.55) return "Turning";
  return "Walking Straight";
}

function updateObservationCount() {
  const count = Math.min(observedHistory.length, PAST_STEPS);
  headerObservedWindow.textContent = `${count} / ${PAST_STEPS} points`;
  observationCount.textContent =
    observedHistory.length < PAST_STEPS
      ? `Click map to add observed points (${count} / ${PAST_STEPS})`
      : `Live window ready (${PAST_STEPS} / ${PAST_STEPS}) - using latest 4 points`;
}

function updateInsightSummary() {
  const modelWindow = withVelocity(getModelWindow());
  if (!modelWindow.length) {
    behaviorLabel.textContent = "Waiting";
    headerBehaviorChip.textContent = "Waiting";
    directionLabel.textContent = "Forward";
    targetVelocity.textContent = "0.0 m/s";
    targetDistance.textContent = "0.0 m";
    targetName.textContent = "Live Agent";
    targetNameSummary.textContent = "Live Agent";
    updateObservationCount();
    return;
  }

  const last = modelWindow[modelWindow.length - 1];
  const speed = Math.sqrt(last[2] ** 2 + last[3] ** 2);
  const direction = classifyDirection(last[2], last[3]);
  const behavior = classifyBehavior(modelWindow.map(([x, y]) => [x, y]));
  const distance = Math.sqrt(last[0] ** 2 + last[1] ** 2);

  behaviorLabel.textContent = behavior;
  headerBehaviorChip.textContent = behavior;
  directionLabel.textContent = direction;
  targetVelocity.textContent = `${speed.toFixed(2)} m/s`;
  targetDistance.textContent = `${distance.toFixed(2)} m`;
  targetName.textContent = "Live Agent";
  targetNameSummary.textContent = "Live Agent";
  updateObservationCount();
}

function renderCanvas() {
  resizeCanvas();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const observed = getModelWindow().map(([x, y]) => [x, y]);
  const allPoints = [observed];
  if (latestPrediction) {
    latestPrediction.paths.forEach((path) => allPoints.push(path));
  }
  currentBounds = normalizePoints(allPoints);
  const occupiedRects = [];

  drawRoadContext();
  drawAxisTicks(currentBounds);
  drawPath(observed, PATH_COLORS.past, { dots: true, width: 5 });
  drawLabeledPoints(observed, PATH_COLORS.past, {
    visiblePoints: observed.length,
    labelFormatter: (index) => (index === 0 ? "P1 start" : `P${index + 1} end`),
    pointRadius: 6,
    occupiedRects,
    pointIndices:
      observed.length > 1 ? [0, observed.length - 1] : observed.length === 1 ? [0] : [],
  });

  if (latestPrediction) {
    latestPrediction.paths.forEach((path, index) => {
      const style = PATH_STYLE[index] || "safe";
      drawPath(path, PATH_COLORS[style], {
        dashed: true,
        visiblePoints: Math.min(revealStep, path.length),
        width: index === 0 ? 5 : 4,
      });

      const visibleCount = Math.min(revealStep, path.length);
      drawLabeledPoints(path, PATH_COLORS[style], {
        visiblePoints: visibleCount,
        labelFormatter: () => `${String.fromCharCode(65 + index)} end`,
        pointRadius: 5.5,
        occupiedRects,
        pointIndices: visibleCount > 0 ? [visibleCount - 1] : [],
      });
    });
  }
}

function formatCoordinate(point) {
  return `(${point[0].toFixed(2)}, ${point[1].toFixed(2)})`;
}

function renderObservedCoordinates() {
  const observed = getModelWindow().map(([x, y]) => [x, y]);
  if (!observed.length) {
    observedCoordinates.innerHTML = `<div class="coord-row empty">No observed points yet</div>`;
    return;
  }

  observedCoordinates.innerHTML = observed
    .map(
      (point, index) => `
        <div class="coord-row">
          <span class="coord-label">P${index + 1}</span>
          <span class="coord-value">${formatCoordinate(point)}</span>
        </div>
      `,
    )
    .join("");
}

function renderPredictedCoordinates() {
  if (!latestPrediction) {
    predictedCoordinates.innerHTML = `<div class="coord-row empty">Run prediction to view future coordinates</div>`;
    return;
  }

  predictedCoordinates.innerHTML = latestPrediction.paths
    .map((path, index) => {
      const style = PATH_STYLE[index] || "safe";
      const title =
        index === 0 ? "Path A (Most Likely)" : index === 1 ? "Path B (Alternative)" : "Path C (Risky)";
      const probability = `${(latestPrediction.probabilities[index] * 100).toFixed(1)}%`;
      const rows = path
        .map(
          (point, pointIndex) => `
            <div class="coord-row">
              <span class="coord-label">F${pointIndex + 1}</span>
              <span class="coord-value">${formatCoordinate(point)}</span>
            </div>
          `,
        )
        .join("");

      return `
        <div class="pred-group ${style}">
          <div class="pred-head">
            <span class="path-title">${title}</span>
            <span class="path-probability">${probability}</span>
          </div>
          ${rows}
        </div>
      `;
    })
    .join("");
}

function updateCoordinatePanels() {
  renderObservedCoordinates();
  renderPredictedCoordinates();
}

function setLoadingState(isLoading) {
  loadingIndicator.classList.toggle("hidden", !isLoading);
  liveIndicator.textContent = isLoading ? "Predicting" : latestPrediction ? "Ready" : "Idle";
  liveIndicator.classList.toggle("running", isLoading || Boolean(latestPrediction));
}

function updateTimeline() {
  footerFrameIndicator.textContent = `${revealStep} / ${FUTURE_STEPS}`;
  const percent = (revealStep / FUTURE_STEPS) * 100;
  timelineProgress.style.width = `${percent}%`;
  timelineThumb.style.left = `${percent}%`;
  frameIndicator.textContent = `Past Window ${Math.min(getModelWindow().length, PAST_STEPS)} / ${PAST_STEPS}`;
}

function resetCards() {
  latestPrediction = null;
  revealStep = 0;
  pathCards.innerHTML = "";
  primaryRiskChip.className = "risk-chip risk-low";
  primaryRiskChip.textContent = "LOW";
  headerRiskChip.textContent = "Low";
  maxCollisionProbability.textContent = "0.0%";
  collisionProgress.style.width = "0%";
  timeToCollision.textContent = "No collision window";
  latencyMetric.textContent = "0 ms";
  adeMetric.textContent = "0.0000";
  fdeMetric.textContent = "0.0000";
  statusMessage.textContent = "Click 4 live points on the map to start forecasting.";
  updateTimeline();
  updateCoordinatePanels();
  renderCanvas();
}

function renderPathCards() {
  if (!latestPrediction) {
    pathCards.innerHTML = "";
    return;
  }

  pathCards.innerHTML = latestPrediction.paths
    .map((path, index) => {
      const probability = latestPrediction.probabilities[index];
      const risk = latestPrediction.risk[index];
      const style = PATH_STYLE[index] || "safe";
      const label = index === 0 ? "Path A" : index === 1 ? "Path B" : "Path C";
      return `
        <article class="path-card ${style}">
          <h4>${label}: ${(probability * 100).toFixed(0)}%</h4>
          <p>Risk level: ${risk.risk_level}</p>
          <p>Collision probability: ${(risk.collision_probability * 100).toFixed(1)}%</p>
          <p>TTC: ${risk.time_to_collision === null ? "No collision window" : `${risk.time_to_collision}s`}</p>
          <div class="prob-bar">
            <div class="prob-fill" style="width:${(probability * 100).toFixed(1)}%; background:${PATH_COLORS[style]}"></div>
          </div>
        </article>
      `;
    })
    .join("");
}

function updateRiskSummary() {
  if (!latestPrediction) return;

  const topRisk = latestPrediction.risk.reduce((best, current) => {
    return current.collision_probability > best.collision_probability ? current : best;
  });

  const riskClass = topRisk.risk_level.toLowerCase();
  primaryRiskChip.className = `risk-chip risk-${riskClass}`;
  primaryRiskChip.textContent = topRisk.risk_level;
  headerRiskChip.textContent = topRisk.risk_level;
  maxCollisionProbability.textContent = `${(topRisk.collision_probability * 100).toFixed(1)}%`;
  collisionProgress.style.width = `${(topRisk.collision_probability * 100).toFixed(1)}%`;
  timeToCollision.textContent =
    topRisk.time_to_collision === null ? "No collision window" : `${topRisk.time_to_collision}s`;
  latencyMetric.textContent = `${latestPrediction.meta.latency_ms} ms`;
  adeMetric.textContent = latestPrediction.meta.ade.toFixed(4);
  fdeMetric.textContent = latestPrediction.meta.fde.toFixed(4);
  statusMessage.textContent = "Live forecast updated from the latest 4 observed positions.";
}

async function runPrediction() {
  const modelWindow = getModelWindow();
  if (modelWindow.length < PAST_STEPS) {
    statusMessage.textContent = `Need ${PAST_STEPS} observed points. Click ${PAST_STEPS - modelWindow.length} more on the map.`;
    return;
  }

  const payload = { trajectory: withVelocity(modelWindow) };
  setLoadingState(true);
  statusMessage.textContent = "Running live multi-path forecast...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    latestPrediction = await response.json();
    revealStep = 1;
    updateRiskSummary();
    renderPathCards();
    updateTimeline();
    updateCoordinatePanels();
    renderCanvas();
  } catch (error) {
    latestPrediction = null;
    statusMessage.textContent = `Inference unavailable: ${error.message}`;
    renderPathCards();
    updateCoordinatePanels();
    renderCanvas();
  } finally {
    setLoadingState(false);
  }
}

function nextFrame() {
  if (!latestPrediction) return;
  revealStep = Math.min(FUTURE_STEPS, revealStep + 1);
  updateTimeline();
  renderCanvas();
}

function resetLiveSession() {
  observedHistory = [];
  updateInsightSummary();
  resetCards();
}

canvas.addEventListener("click", async (event) => {
  const point = unprojectPoint(event.clientX, event.clientY);
  observedHistory.push([point[0], point[1], 0, 0]);
  updateInsightSummary();
  latestPrediction = null;
  revealStep = 0;
  renderPathCards();
  updateTimeline();
  updateCoordinatePanels();
  renderCanvas();

  if (getModelWindow().length >= PAST_STEPS) {
    await runPrediction();
  } else {
    statusMessage.textContent = `Observed point captured. ${PAST_STEPS - getModelWindow().length} more needed for prediction.`;
  }
});

playButton.addEventListener("click", runPrediction);
stepButton.addEventListener("click", nextFrame);
resetButton.addEventListener("click", resetLiveSession);
copyObservedButton.addEventListener("click", async () => {
  const observed = getModelWindow().map(([x, y]) => formatCoordinate([x, y])).join("\n");
  await navigator.clipboard.writeText(observed || "No observed points yet");
});
copyPredictedButton.addEventListener("click", async () => {
  if (!latestPrediction) {
    await navigator.clipboard.writeText("No predicted trajectories yet");
    return;
  }
  const text = latestPrediction.paths
    .map((path, index) => {
      const title = index === 0 ? "Path A" : index === 1 ? "Path B" : "Path C";
      const probability = `${(latestPrediction.probabilities[index] * 100).toFixed(1)}%`;
      const coords = path.map((point) => formatCoordinate(point)).join("\n");
      return `${title} - ${probability}\n${coords}`;
    })
    .join("\n\n");
  await navigator.clipboard.writeText(text);
});

window.addEventListener("resize", () => {
  renderCanvas();
});

updateInsightSummary();
resetCards();
