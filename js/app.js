import {
  FilesetResolver,
  HandLandmarker,
  ImageSegmenter,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const SEGMENTER_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite";

const LILY_FRAME_COUNT = 150;
const CHRYS_FRAME_COUNT = 134;
const LILY_DIR = "assets/flowers/lily";
const CHRYS_DIR = "assets/flowers/chrysanthemum";
const FLOWER_W = 960;
const FLOWER_H = 540;

// Bouquet composition, carried over from the flower-comparison-v3 tool.
const LILY_LAYOUT = { offsetXFrac: -0.2, scale: 0.9 };
const CHRYS_LAYOUT = { offsetXFrac: 0.22, scale: 0.62 };

// The source frames' stems are cropped hard right at the bottom edge of
// their own image (no natural taper), so anchoring them at the canvas
// bottom shows an abrupt cutoff line. Fading the last stretch of the
// flower canvas to transparent dissolves that hard edge away instead.
const FLOWER_BOTTOM_FADE_START_FRAC = 0.8;

// Pinch-openness calibration: ratio of (thumb-tip <-> index-tip 3D distance)
// to (wrist <-> middle-MCP 3D distance, i.e. palm size). Tune these two
// numbers against the live debug readout if bloom feels too eager/lazy.
const PINCH_RATIO_CLOSED = 0.15;
const PINCH_RATIO_OPEN = 1.05;

// Exponential smoothing factor applied to the bloom fraction each frame.
// Higher = snappier/more jittery, lower = smoother/more laggy.
const BLOOM_SMOOTHING = 0.35;

// Per-pixel temporal smoothing applied to the segmentation confidence mask,
// to kill frame-to-frame edge flicker. Lower = smoother but laggier edges.
const MASK_TEMPORAL_SMOOTHING = 0.45;

// Contrast curve applied to the (temporally smoothed) mask value before it
// becomes alpha: values below LOW go fully transparent, above HIGH go fully
// opaque, with a smooth ramp between. Narrowing the gap fixes background
// bleeding through semi-transparent edges (hair, arms) at the cost of a
// harder-edged cutout; widening it softens the edge but lets more of the
// background show through as translucency.
const MASK_CONTRAST_LOW = 0.45;
const MASK_CONTRAST_HIGH = 0.7;

// "Is she holding up a flower?" detector: MobileNet (v1, alpha 1.0 — must
// match the version used to train the head below) produces a 1024-d
// embedding of the current frame; a small custom-trained head classifies
// that embedding into 5 flower species or "not_flower". See
// scratchpad/flower-train/ for the training script (tf_flowers + a sample of
// imagenette as negatives).
const FLOWER_HEAD_MODEL_URL = "assets/models/flower-head/model.json";
const FLOWER_LABELS_URL = "assets/models/flower-head/labels.json";
const FLOWER_CHECK_INTERVAL_MS = 600;
const FLOWER_CONFIDENCE_THRESHOLD = 0.93;
const FLOWER_ON_STREAK = 3; // consecutive positive checks before triggering
const FLOWER_OFF_STREAK = 2; // consecutive negative checks before clearing
// The classifier was trained on tight, centered flower photos. A flower held
// at arm's length only fills a small part of the full webcam frame, so we
// digitally zoom into a centered square crop before classifying — this is
// pure framing, not resolution: it does not fix a flower held very small/far
// away, but it means she no longer has to press it against the lens.
const FLOWER_CROP_ZOOM = 1.6;
const FLOWER_CROP_SIZE = 224;

// Typewriter reveal speed for moment overlay text, in ms per character.
const TYPEWRITER_MS_PER_CHAR = 45;
const FLOWER_MESSAGE = "when I am with you it feels like flowers are blooming";
const HEART_MESSAGE = "Happy birthday Mari";

// Two-hand heart gesture: bring both thumb tips together and both index
// fingertips together (the classic finger-heart shape). Ratios are each
// pair's 3D distance normalized by the average of both hands' palm sizes.
const HEART_THUMB_RATIO = 0.35;
const HEART_INDEX_RATIO = 0.35;
const HEART_ON_STREAK = 4;
const HEART_OFF_STREAK = 3;

// Auto brightness compensation: samples average scene luminance and nudges
// exposure back toward TARGET_LUMINANCE (0-255 scale) so a too-bright or
// too-dark room doesn't wash out or crush the composite.
const TARGET_LUMINANCE = 128;
const BRIGHTNESS_MIN = 0.6;
const BRIGHTNESS_MAX = 1.6;
const BRIGHTNESS_SMOOTHING = 0.06;

// Subtle parallax: the flower background shifts a small fraction opposite
// her on-screen movement, so the scene doesn't look perfectly flat when the
// camera (or she) shifts.
const PARALLAX_STRENGTH = 0.12;
const PARALLAX_SMOOTHING = 0.08;

// Late-90s/early-2000s digicam looks, toggled by pressing 1-5 (0 clears back
// to normal). Each one emulates a cheap CCD point-and-shoot rather than a
// film-camera vintage look: blown flash highlights, off color casts, visible
// compression-era noise, low native resolution, and the classic amber
// digital timestamp burned into the corner.
const PIXELATE_FACTOR = 3.2; // >1 = how chunky the "low-res sensor" look is
const FILTER_PRESETS = {
  "1": { name: "CyberShot 2003", cssFilter: "contrast(1.12) saturate(1.25) brightness(1.03) hue-rotate(-3deg)", vignette: 0.15, grain: 0.12, pixelate: false, timestamp: true },
  "2": { name: "Flash Snap", cssFilter: "contrast(1.3) saturate(1.1) brightness(1.22)", vignette: 0.35, grain: 0.08, pixelate: false, timestamp: true },
  "3": { name: "Low-Res Cam", cssFilter: "contrast(1.15) saturate(1.2) brightness(1.02)", vignette: 0.1, grain: 0.06, pixelate: true, timestamp: true },
  "4": { name: "Disposable Digicam", cssFilter: "contrast(0.92) saturate(1.15) brightness(1.1) hue-rotate(5deg)", vignette: 0.3, grain: 0.18, pixelate: false, timestamp: true },
  "5": { name: "Y2K Webcam", cssFilter: "saturate(0.85) contrast(1.05) hue-rotate(-8deg) brightness(1.0)", vignette: 0.12, grain: 0.22, pixelate: true, timestamp: true },
};

// ---------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------

const video = document.getElementById("video");
const outputCanvas = document.getElementById("output");
const outputCtx = outputCanvas.getContext("2d");
const startOverlay = document.getElementById("startOverlay");
const startBtn = document.getElementById("startBtn");
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingText = document.getElementById("loadingText");
const modeButtons = Array.from(document.querySelectorAll(".mode-btn"));
const momentOverlayEl = document.getElementById("momentOverlay");
const momentTextEl = document.getElementById("momentText");
const afCorners = Array.from(document.querySelectorAll(".af-corner"));
const recDotEl = document.getElementById("recDot");
const powerLedEl = document.getElementById("powerLed");
const hudFrameCountEl = document.getElementById("hudFrameCount");
const shutterBtn = document.getElementById("shutterBtn");
const flashOverlayEl = document.getElementById("flashOverlay");

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let handLandmarker = null;
let imageSegmenter = null;
let lilyFrames = [];
let chrysFrames = [];
let mode = "bouquet"; // "lily" | "chrys" | "bouquet"
let bloomLily = 0; // smoothed, 0 = bud, 1 = full bloom
let bloomChrys = 0;

const flowerCanvas = document.createElement("canvas");
flowerCanvas.width = FLOWER_W;
flowerCanvas.height = FLOWER_H;
const flowerCtx = flowerCanvas.getContext("2d");

let personCanvas = null;
let personCtx = null;
let maskCanvas = null;
let maskCtx = null;
let smoothedMask = null; // Float32Array, persists across frames for temporal smoothing
let smoothedMaskW = 0;
let smoothedMaskH = 0;

let mobilenetModel = null;
let flowerHeadModel = null;
let flowerLabels = [];
let flowerOnStreak = 0;
let flowerOffStreak = 0;

let heartOnStreak = 0;
let heartOffStreak = 0;

// Shared full-screen "moment" overlay (dim + typewriter text), used by both
// the flower detector and the heart gesture. Tracks which one currently owns
// it so one can't clear a moment the other one triggered.
let activeMomentSource = null; // null | "flower" | "heart"
let typewriterHandle = null;

let captureCount = 0;

const flowerCropCanvas = document.createElement("canvas");
flowerCropCanvas.width = FLOWER_CROP_SIZE;
flowerCropCanvas.height = FLOWER_CROP_SIZE;
const flowerCropCtx = flowerCropCanvas.getContext("2d");

// Full unfiltered composite (background + flower + person), before the
// active vintage filter and its overlays are blitted onto the visible canvas.
let sceneCanvas = null;
let sceneCtx = null;

const BRIGHTNESS_SAMPLE_SIZE = 16;
const brightnessSampleCanvas = document.createElement("canvas");
brightnessSampleCanvas.width = BRIGHTNESS_SAMPLE_SIZE;
brightnessSampleCanvas.height = BRIGHTNESS_SAMPLE_SIZE;
const brightnessSampleCtx = brightnessSampleCanvas.getContext("2d", { willReadFrequently: true });
let brightnessFactor = 1;

let personCentroidValid = false;
let personCentroidRawX = 0.5;
let personCentroidRawY = 0.5;
let parallaxX = 0;
let parallaxY = 0;

let activeFilterKey = null; // null = no filter

let pixelateCanvas = null;
let pixelateCtx = null;

const grainTile = buildGrainTile(128);
const grainPattern = outputCtx.createPattern(grainTile, "repeat");

// ---------------------------------------------------------------------------
// Digicam filter helpers, built once at load
// ---------------------------------------------------------------------------

function buildGrainTile(size) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  const imageData = ctx.createImageData(size, size);
  for (let i = 0; i < imageData.data.length; i += 4) {
    const v = Math.floor(Math.random() * 255);
    imageData.data[i] = v;
    imageData.data[i + 1] = v;
    imageData.data[i + 2] = v;
    imageData.data[i + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

function drawDigicamTimestamp(ctx, W, H) {
  const now = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  const text = `${pad(now.getMonth() + 1)} ${pad(now.getDate())} ${now.getFullYear()}  ${pad(now.getHours())}:${pad(now.getMinutes())}`;
  const fontSize = Math.max(14, Math.round(H * 0.032));
  ctx.save();
  ctx.font = `${fontSize}px "Courier New", monospace`;
  ctx.textAlign = "right";
  ctx.textBaseline = "bottom";
  const x = W - fontSize * 0.8;
  const y = H - fontSize * 0.8;
  ctx.fillStyle = "rgba(0, 0, 0, 0.45)";
  ctx.fillText(text, x + 1, y + 1);
  ctx.fillStyle = "rgba(255, 149, 0, 0.9)";
  ctx.fillText(text, x, y);
  ctx.restore();
}

function updateBrightnessCorrection() {
  brightnessSampleCtx.drawImage(video, 0, 0, BRIGHTNESS_SAMPLE_SIZE, BRIGHTNESS_SAMPLE_SIZE);
  const data = brightnessSampleCtx.getImageData(0, 0, BRIGHTNESS_SAMPLE_SIZE, BRIGHTNESS_SAMPLE_SIZE).data;
  let sum = 0;
  const pixelCount = data.length / 4;
  for (let i = 0; i < data.length; i += 4) {
    sum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  const avgLuminance = sum / pixelCount;
  const rawFactor = TARGET_LUMINANCE / Math.max(avgLuminance, 1);
  const targetFactor = Math.min(BRIGHTNESS_MAX, Math.max(BRIGHTNESS_MIN, rawFactor));
  brightnessFactor += (targetFactor - brightnessFactor) * BRIGHTNESS_SMOOTHING;
}

// ---------------------------------------------------------------------------
// Asset loading
// ---------------------------------------------------------------------------

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = (err) => reject(new Error(`Failed to load ${src}: ${err}`));
    img.src = src;
  });
}

async function loadFrameSequence(dir, count, onProgress) {
  const frames = new Array(count);
  let loaded = 0;
  await Promise.all(
    Array.from({ length: count }, async (_, i) => {
      const name = `frame_${String(i).padStart(3, "0")}.webp`;
      frames[i] = await loadImage(`${dir}/${name}`);
      loaded++;
      if (onProgress) onProgress(loaded, count);
    })
  );
  return frames;
}

// ---------------------------------------------------------------------------
// MediaPipe setup, with GPU -> CPU delegate fallback
// ---------------------------------------------------------------------------

async function createHandLandmarker(vision) {
  const options = {
    baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numHands: 2,
  };
  try {
    return await HandLandmarker.createFromOptions(vision, options);
  } catch (err) {
    console.error("HandLandmarker GPU init failed, falling back to CPU:", err);
    options.baseOptions.delegate = "CPU";
    return await HandLandmarker.createFromOptions(vision, options);
  }
}

async function createImageSegmenter(vision) {
  const options = {
    baseOptions: { modelAssetPath: SEGMENTER_MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    outputCategoryMask: false,
    outputConfidenceMasks: true,
  };
  try {
    return await ImageSegmenter.createFromOptions(vision, options);
  } catch (err) {
    console.error("ImageSegmenter GPU init failed, falling back to CPU:", err);
    options.baseOptions.delegate = "CPU";
    return await ImageSegmenter.createFromOptions(vision, options);
  }
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise((resolve) => {
    video.onloadedmetadata = () => resolve();
  });
  await video.play();

  outputCanvas.width = video.videoWidth;
  outputCanvas.height = video.videoHeight;
  personCanvas = document.createElement("canvas");
  personCanvas.width = video.videoWidth;
  personCanvas.height = video.videoHeight;
  personCtx = personCanvas.getContext("2d");
  maskCanvas = document.createElement("canvas");
  maskCanvas.width = video.videoWidth;
  maskCanvas.height = video.videoHeight;
  maskCtx = maskCanvas.getContext("2d");

  sceneCanvas = document.createElement("canvas");
  sceneCanvas.width = video.videoWidth;
  sceneCanvas.height = video.videoHeight;
  sceneCtx = sceneCanvas.getContext("2d");

  pixelateCanvas = document.createElement("canvas");
  pixelateCanvas.width = Math.max(1, Math.round(video.videoWidth / PIXELATE_FACTOR));
  pixelateCanvas.height = Math.max(1, Math.round(video.videoHeight / PIXELATE_FACTOR));
  pixelateCtx = pixelateCanvas.getContext("2d");
}

// ---------------------------------------------------------------------------
// Pinch openness
// ---------------------------------------------------------------------------

function dist3D(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = (a.z ?? 0) - (b.z ?? 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function computePinchOpenness(landmarks) {
  const wrist = landmarks[0];
  const middleMcp = landmarks[9];
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];

  const palmSize = dist3D(wrist, middleMcp);
  if (palmSize < 1e-6) return null;

  const pinchDist = dist3D(thumbTip, indexTip);
  const ratio = pinchDist / palmSize;

  const t = (ratio - PINCH_RATIO_CLOSED) / (PINCH_RATIO_OPEN - PINCH_RATIO_CLOSED);
  return Math.min(1, Math.max(0, t));
}

function detectHeartGesture(landmarksA, landmarksB) {
  const palmA = dist3D(landmarksA[0], landmarksA[9]);
  const palmB = dist3D(landmarksB[0], landmarksB[9]);
  const avgPalm = (palmA + palmB) / 2;
  if (avgPalm < 1e-6) return false;

  const thumbDist = dist3D(landmarksA[4], landmarksB[4]) / avgPalm;
  const indexDist = dist3D(landmarksA[8], landmarksB[8]) / avgPalm;
  return thumbDist < HEART_THUMB_RATIO && indexDist < HEART_INDEX_RATIO;
}

// ---------------------------------------------------------------------------
// Shared "moment" overlay (dim + typewriter text)
// ---------------------------------------------------------------------------

function startTypewriter(text) {
  if (typewriterHandle) clearInterval(typewriterHandle);
  let shown = 0;
  momentTextEl.textContent = "";
  typewriterHandle = setInterval(() => {
    shown++;
    momentTextEl.textContent = text.slice(0, shown);
    if (shown >= text.length) {
      clearInterval(typewriterHandle);
      typewriterHandle = null;
    }
  }, TYPEWRITER_MS_PER_CHAR);
}

function showMoment(source, text) {
  if (activeMomentSource === source) return;
  activeMomentSource = source;
  momentOverlayEl.classList.add("active");
  startTypewriter(text);
}

function hideMoment(source) {
  if (activeMomentSource !== source) return;
  activeMomentSource = null;
  momentOverlayEl.classList.remove("active");
  if (typewriterHandle) {
    clearInterval(typewriterHandle);
    typewriterHandle = null;
  }
}

// ---------------------------------------------------------------------------
// "Is she showing a flower to the camera?" detector
// ---------------------------------------------------------------------------

async function loadFlowerDetector() {
  try {
    mobilenetModel = await mobilenet.load({ version: 1, alpha: 1.0 });
    flowerHeadModel = await tf.loadLayersModel(FLOWER_HEAD_MODEL_URL);
    flowerLabels = await (await fetch(FLOWER_LABELS_URL)).json();
  } catch (err) {
    console.error("Flower detector failed to load — feature disabled:", err);
    mobilenetModel = null;
    flowerHeadModel = null;
  }
}

function drawFlowerCropFrame() {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const side = Math.min(vw, vh) / FLOWER_CROP_ZOOM;
  const sx = (vw - side) / 2;
  const sy = (vh - side) / 2;
  flowerCropCtx.drawImage(video, sx, sy, side, side, 0, 0, FLOWER_CROP_SIZE, FLOWER_CROP_SIZE);
}

function classifyFlowerFrame() {
  drawFlowerCropFrame();
  return tf.tidy(() => {
    const embedding = mobilenetModel.infer(flowerCropCanvas, true);
    const logits = flowerHeadModel.predict(embedding);
    const data = logits.dataSync();
    let bestIdx = 0;
    for (let i = 1; i < data.length; i++) {
      if (data[i] > data[bestIdx]) bestIdx = i;
    }
    return { label: flowerLabels[bestIdx], confidence: data[bestIdx] };
  });
}

function startFlowerDetectionLoop() {
  setInterval(() => {
    try {
      const { label, confidence } = classifyFlowerFrame();
      const isFlower = label !== "not_flower" && confidence >= FLOWER_CONFIDENCE_THRESHOLD;

      if (isFlower) {
        flowerOnStreak++;
        flowerOffStreak = 0;
      } else {
        flowerOffStreak++;
        flowerOnStreak = 0;
      }

      if (activeMomentSource !== "flower" && flowerOnStreak >= FLOWER_ON_STREAK) {
        showMoment("flower", FLOWER_MESSAGE);
      }
      if (activeMomentSource === "flower" && flowerOffStreak >= FLOWER_OFF_STREAK) {
        hideMoment("flower");
      }
    } catch (err) {
      console.error("Error in flower detection loop:", err);
    }
  }, FLOWER_CHECK_INTERVAL_MS);
}

// ---------------------------------------------------------------------------
// Flower compositing
// ---------------------------------------------------------------------------

function drawFlowerFrame(ctx, img, offsetXFrac, scale) {
  const originX = FLOWER_W / 2 + offsetXFrac * FLOWER_W;
  const originY = FLOWER_H;
  ctx.save();
  ctx.translate(originX, originY);
  ctx.scale(scale, scale);
  ctx.drawImage(img, -FLOWER_W / 2, -FLOWER_H, FLOWER_W, FLOWER_H);
  ctx.restore();
}

function renderFlowerBackground(lilyFraction, chrysFraction) {
  flowerCtx.clearRect(0, 0, FLOWER_W, FLOWER_H);

  if (mode === "lily") {
    const idx = Math.floor(lilyFraction * (lilyFrames.length - 1));
    drawFlowerFrame(flowerCtx, lilyFrames[idx], 0, 1);
  } else if (mode === "chrys") {
    const idx = Math.floor(chrysFraction * (chrysFrames.length - 1));
    drawFlowerFrame(flowerCtx, chrysFrames[idx], 0, 1);
  } else {
    const lilyIdx = Math.floor(lilyFraction * (lilyFrames.length - 1));
    const chrysIdx = Math.floor(chrysFraction * (chrysFrames.length - 1));
    drawFlowerFrame(flowerCtx, lilyFrames[lilyIdx], LILY_LAYOUT.offsetXFrac, LILY_LAYOUT.scale);
    drawFlowerFrame(flowerCtx, chrysFrames[chrysIdx], CHRYS_LAYOUT.offsetXFrac, CHRYS_LAYOUT.scale);
  }

  const fadeStart = FLOWER_H * FLOWER_BOTTOM_FADE_START_FRAC;
  const fadeGrad = flowerCtx.createLinearGradient(0, fadeStart, 0, FLOWER_H);
  fadeGrad.addColorStop(0, "rgba(0,0,0,0)");
  fadeGrad.addColorStop(1, "rgba(0,0,0,1)");
  flowerCtx.globalCompositeOperation = "destination-out";
  flowerCtx.fillStyle = fadeGrad;
  flowerCtx.fillRect(0, fadeStart, FLOWER_W, FLOWER_H - fadeStart);
  flowerCtx.globalCompositeOperation = "source-over";
}

function capturePhoto() {
  flashOverlayEl.classList.remove("flash");
  // Force reflow so re-adding the class restarts the animation on rapid clicks.
  void flashOverlayEl.offsetWidth;
  flashOverlayEl.classList.add("flash");

  outputCanvas.toBlob((blob) => {
    if (!blob) {
      console.error("Failed to capture photo: canvas toBlob returned null");
      return;
    }
    captureCount++;
    hudFrameCountEl.textContent = String(captureCount).padStart(3, "0");
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    a.href = url;
    a.download = `bloom-${stamp}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }, "image/png");
}

// ---------------------------------------------------------------------------
// Person cutout compositing
// ---------------------------------------------------------------------------

function sharpenMaskValue(v) {
  // Smoothstep contrast boost: pushes mid-confidence pixels toward fully
  // transparent or fully opaque, instead of leaving them semi-transparent.
  const t = Math.min(1, Math.max(0, (v - MASK_CONTRAST_LOW) / (MASK_CONTRAST_HIGH - MASK_CONTRAST_LOW)));
  return t * t * (3 - 2 * t);
}

function compositePersonCutout(confidenceMask) {
  const w = personCanvas.width;
  const h = personCanvas.height;

  // Draw the raw (unmirrored) video frame.
  personCtx.globalCompositeOperation = "source-over";
  personCtx.drawImage(video, 0, 0, w, h);

  // The mask's own resolution can differ from the video's — build the alpha
  // mask at its native size, then let drawImage scale it onto personCanvas.
  const maskW = confidenceMask.width;
  const maskH = confidenceMask.height;
  const maskData = confidenceMask.getAsFloat32Array();

  // Temporally smooth each pixel's confidence across frames to kill flicker.
  // Reset the buffer if the mask's resolution ever changes.
  if (!smoothedMask || smoothedMaskW !== maskW || smoothedMaskH !== maskH) {
    smoothedMask = Float32Array.from(maskData);
    smoothedMaskW = maskW;
    smoothedMaskH = maskH;
  } else {
    for (let i = 0; i < maskData.length; i++) {
      smoothedMask[i] += (maskData[i] - smoothedMask[i]) * MASK_TEMPORAL_SMOOTHING;
    }
  }

  if (maskCanvas.width !== maskW || maskCanvas.height !== maskH) {
    maskCanvas.width = maskW;
    maskCanvas.height = maskH;
  }
  const maskImageData = maskCtx.createImageData(maskW, maskH);
  let sumX = 0;
  let sumY = 0;
  let sumWeight = 0;
  for (let i = 0; i < smoothedMask.length; i++) {
    const a = sharpenMaskValue(smoothedMask[i]);
    maskImageData.data[i * 4 + 3] = a * 255;
    sumX += (i % maskW) * a;
    sumY += Math.floor(i / maskW) * a;
    sumWeight += a;
  }
  maskCtx.putImageData(maskImageData, 0, 0);

  // Track her silhouette's centroid (in raw/unmirrored space) for parallax.
  if (sumWeight > 1) {
    personCentroidRawX = sumX / sumWeight / maskW;
    personCentroidRawY = sumY / sumWeight / maskH;
    personCentroidValid = true;
  } else {
    personCentroidValid = false;
  }

  // Keep only the person pixels, weighted by the sharpened mask confidence.
  personCtx.globalCompositeOperation = "destination-in";
  personCtx.drawImage(maskCanvas, 0, 0, maskW, maskH, 0, 0, w, h);
  personCtx.globalCompositeOperation = "source-over";
}

// ---------------------------------------------------------------------------
// Main render loop
// ---------------------------------------------------------------------------

function renderFrame(now) {
  try {
    const handResult = handLandmarker.detectForVideo(video, now);

    // Collect each visible hand's pinch openness plus its on-screen (mirrored)
    // x position, so in bouquet mode two hands can each drive one flower.
    const hands = [];
    if (handResult.landmarks) {
      for (const landmarks of handResult.landmarks) {
        const openness = computePinchOpenness(landmarks);
        if (openness !== null) {
          hands.push({ openness, displayX: 1 - landmarks[0].x, landmarks });
        }
      }
    }
    hands.sort((a, b) => a.displayX - b.displayX);

    afCorners.forEach((el) => el.classList.toggle("active", hands.length > 0));

    const heartDetected = hands.length === 2 && detectHeartGesture(hands[0].landmarks, hands[1].landmarks);
    if (heartDetected) {
      heartOnStreak++;
      heartOffStreak = 0;
    } else {
      heartOffStreak++;
      heartOnStreak = 0;
    }
    if (activeMomentSource !== "heart" && heartOnStreak >= HEART_ON_STREAK) {
      showMoment("heart", HEART_MESSAGE);
    }
    if (activeMomentSource === "heart" && heartOffStreak >= HEART_OFF_STREAK) {
      hideMoment("heart");
    }

    let targetLily = bloomLily;
    let targetChrys = bloomChrys;

    if (mode === "bouquet") {
      if (hands.length >= 2) {
        targetLily = hands[0].openness; // leftmost hand -> lily (left flower)
        targetChrys = hands[hands.length - 1].openness; // rightmost -> chrys
      } else if (hands.length === 1) {
        targetLily = hands[0].openness;
        targetChrys = hands[0].openness;
      }
    } else if (mode === "lily" && hands.length > 0) {
      targetLily = hands[0].openness;
    } else if (mode === "chrys" && hands.length > 0) {
      targetChrys = hands[0].openness;
    }

    bloomLily += (targetLily - bloomLily) * BLOOM_SMOOTHING;
    bloomChrys += (targetChrys - bloomChrys) * BLOOM_SMOOTHING;
    bloomLily = Math.min(1, Math.max(0, bloomLily));
    bloomChrys = Math.min(1, Math.max(0, bloomChrys));

    renderFlowerBackground(bloomLily, bloomChrys);

    imageSegmenter.segmentForVideo(video, now, (segResult) => {
      try {
        const masks = segResult.confidenceMasks;
        if (masks && masks.length > 0) {
          const personMask = masks.length > 1 ? masks[1] : masks[0];
          compositePersonCutout(personMask);
          masks.forEach((m) => m.close());
        }

        // Track a smoothed parallax offset from her silhouette's centroid, in
        // mirrored/display space (positive devX = she's toward screen-right).
        if (personCentroidValid) {
          const devX = (1 - personCentroidRawX) - 0.5;
          const devY = personCentroidRawY - 0.5;
          parallaxX += (devX - parallaxX) * PARALLAX_SMOOTHING;
          parallaxY += (devY - parallaxY) * PARALLAX_SMOOTHING;
        }

        updateBrightnessCorrection();

        const W = outputCanvas.width;
        const H = outputCanvas.height;

        // Compose the unfiltered scene: her real (mirrored, brightness-
        // corrected) background -> flower (with subtle parallax) -> her
        // cutout on top. The flower artwork itself skips brightness
        // correction — only the camera-derived layers need it.
        sceneCtx.clearRect(0, 0, W, H);

        sceneCtx.save();
        sceneCtx.filter = `brightness(${brightnessFactor})`;
        sceneCtx.translate(W, 0);
        sceneCtx.scale(-1, 1);
        sceneCtx.drawImage(video, 0, 0, W, H);
        sceneCtx.restore();

        const parallaxOffsetX = -parallaxX * PARALLAX_STRENGTH * W;
        const parallaxOffsetY = -parallaxY * PARALLAX_STRENGTH * H;
        sceneCtx.drawImage(flowerCanvas, 0, 0, FLOWER_W, FLOWER_H, parallaxOffsetX, parallaxOffsetY, W, H);

        if (personCanvas) {
          sceneCtx.save();
          sceneCtx.filter = `brightness(${brightnessFactor})`;
          sceneCtx.translate(W, 0);
          sceneCtx.scale(-1, 1);
          sceneCtx.drawImage(personCanvas, 0, 0, W, H);
          sceneCtx.restore();
        }

        // Blit through the active digicam filter (if any), then layer its
        // vignette/grain/timestamp overlays on top, unaffected by the filter.
        const preset = activeFilterKey ? FILTER_PRESETS[activeFilterKey] : null;
        outputCtx.clearRect(0, 0, W, H);

        if (preset && preset.pixelate) {
          // Downsample (with the color grade baked in) then blow back up
          // with no smoothing, for a chunky low-res "old sensor" look.
          const pw = pixelateCanvas.width;
          const ph = pixelateCanvas.height;
          pixelateCtx.filter = preset.cssFilter;
          pixelateCtx.clearRect(0, 0, pw, ph);
          pixelateCtx.drawImage(sceneCanvas, 0, 0, pw, ph);
          pixelateCtx.filter = "none";
          outputCtx.imageSmoothingEnabled = false;
          outputCtx.drawImage(pixelateCanvas, 0, 0, pw, ph, 0, 0, W, H);
          outputCtx.imageSmoothingEnabled = true;
        } else {
          outputCtx.filter = preset ? preset.cssFilter : "none";
          outputCtx.drawImage(sceneCanvas, 0, 0);
          outputCtx.filter = "none";
        }

        if (preset) {
          if (preset.vignette > 0) {
            const grad = outputCtx.createRadialGradient(W / 2, H / 2, H * 0.35, W / 2, H / 2, H * 0.9);
            grad.addColorStop(0, "rgba(0,0,0,0)");
            grad.addColorStop(1, `rgba(0,0,0,${preset.vignette})`);
            outputCtx.fillStyle = grad;
            outputCtx.fillRect(0, 0, W, H);
          }
          if (preset.grain > 0 && grainPattern) {
            const ox = Math.floor(Math.random() * 128);
            const oy = Math.floor(Math.random() * 128);
            outputCtx.save();
            outputCtx.globalAlpha = preset.grain;
            outputCtx.globalCompositeOperation = "overlay";
            outputCtx.translate(-ox, -oy);
            outputCtx.fillStyle = grainPattern;
            outputCtx.fillRect(ox, oy, W, H);
            outputCtx.restore();
          }
          if (preset.timestamp) {
            drawDigicamTimestamp(outputCtx, W, H);
          }
        }
      } catch (err) {
        console.error("Error compositing segmentation result:", err);
      }
    });
  } catch (err) {
    console.error("Error in render loop:", err);
  }

  requestAnimationFrame(renderFrame);
}

// ---------------------------------------------------------------------------
// Boot sequence
// ---------------------------------------------------------------------------

function setLoading(msg) {
  loadingOverlay.hidden = false;
  loadingText.textContent = msg;
}

async function boot() {
  startOverlay.hidden = true;
  setLoading("loading flowers…");

  try {
    const [lily, chrys] = await Promise.all([
      loadFrameSequence(LILY_DIR, LILY_FRAME_COUNT, (n, total) =>
        setLoading(`loading lily frames… ${n}/${total}`)
      ),
      loadFrameSequence(CHRYS_DIR, CHRYS_FRAME_COUNT, (n, total) =>
        setLoading(`loading chrysanthemum frames… ${n}/${total}`)
      ),
    ]);
    lilyFrames = lily;
    chrysFrames = chrys;

    setLoading("loading on-device vision models…");
    const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
    handLandmarker = await createHandLandmarker(vision);
    imageSegmenter = await createImageSegmenter(vision);

    setLoading("loading flower recognizer…");
    await loadFlowerDetector();

    setLoading("starting camera…");
    await setupCamera();

    loadingOverlay.hidden = true;
    modeButtons.forEach((btn) => {
      btn.disabled = false;
    });
    recDotEl.classList.add("live");
    powerLedEl.classList.add("on");
    shutterBtn.disabled = false;

    requestAnimationFrame(renderFrame);
    if (mobilenetModel && flowerHeadModel) startFlowerDetectionLoop();
  } catch (err) {
    console.error("Boot failed:", err);
    setLoading("something went wrong — check the console and reload.");
  }
}

modeButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    mode = btn.dataset.mode;
    modeButtons.forEach((b) => b.classList.toggle("active", b === btn));
  });
});

startBtn.addEventListener("click", boot);
shutterBtn.addEventListener("click", capturePhoto);

window.addEventListener("keydown", (e) => {
  if (e.key === "0") {
    activeFilterKey = null;
  } else if (FILTER_PRESETS[e.key]) {
    activeFilterKey = e.key;
  }
});
