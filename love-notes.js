// --- Scroll-reveal ---

const revealObserver = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
      }
    }
  },
  { threshold: 0.25 }
);

// --- Photo prompts: click to pick a photo, preview it, remember it locally ---

const PHOTO_STORAGE_PREFIX = "love-notes-photo-";

function applyPhoto(frame, dataUrl) {
  const preview = frame.querySelector(".photo-preview");
  preview.src = dataUrl;
  preview.hidden = false;
  frame.classList.add("has-photo");
}

function wirePhotoFrame(frame) {
  const trackId = frame.dataset.trackPhoto;
  const input = frame.querySelector(".photo-input");
  const storageKey = PHOTO_STORAGE_PREFIX + trackId;

  const saved = localStorage.getItem(storageKey);
  if (saved) {
    applyPhoto(frame, saved);
  }

  input.addEventListener("change", () => {
    const file = input.files && input.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result;
      applyPhoto(frame, dataUrl);
      try {
        localStorage.setItem(storageKey, dataUrl);
      } catch (err) {
        console.error("Couldn't save photo locally (storage full?):", err);
      }
    };
    reader.onerror = () => {
      console.error("Failed to read the selected photo:", reader.error);
    };
    reader.readAsDataURL(file);
  });
}

// --- Lyrics: fetched from lrcmux.dev (free, no key, real CORS, full lyrics
// with per-line timestamps — see love-notes project memory for why this
// provider specifically). ---

async function fetchLyricLines(artist, title) {
  const url = `https://api.lrcmux.dev/get?artist=${encodeURIComponent(artist)}&title=${encodeURIComponent(title)}&level=line&format=json`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`server said HTTP ${res.status}`);
  const data = await res.json();
  if (!Array.isArray(data.lines) || data.lines.length === 0) throw new Error(data.detail || "no lyrics found");
  return data.lines.map((l) => l.text);
}

function normalizeWord(w) {
  return w.toLowerCase().replace(/[^\p{L}\p{N}]/gu, "");
}

// Different lyric sources transcribe the same song slightly differently
// (punctuation, "ooh" vs "ooo", "ya" vs "you"...), so an exact substring
// search is too brittle. Instead score every line by how many of the
// highlight phrase's words it contains, and take the best match.
function findBestMatchingLineIndex(lines, highlightText) {
  const targetWords = highlightText.split(/\s+/).map(normalizeWord).filter(Boolean);
  if (targetWords.length === 0) return -1;

  let bestIdx = -1;
  let bestScore = 0;
  lines.forEach((line, i) => {
    const lineWords = new Set(line.split(/\s+/).map(normalizeWord).filter(Boolean));
    let score = 0;
    for (const w of targetWords) {
      if (lineWords.has(w)) score++;
    }
    if (score > bestScore) {
      bestScore = score;
      bestIdx = i;
    }
  });

  const minRequired = Math.max(2, Math.ceil(targetWords.length * 0.5));
  return bestScore >= minRequired ? bestIdx : -1;
}

function renderLyricLines(box, lines, highlightIdx, translation) {
  box.textContent = "";
  lines.forEach((text, i) => {
    const p = document.createElement("p");
    p.className = "lyric-line";
    p.textContent = text;
    if (i === highlightIdx) {
      p.classList.add("lyric-highlight");
      if (translation) {
        p.appendChild(document.createElement("br"));
        const t = document.createElement("span");
        t.className = "lyric-translation";
        t.textContent = translation;
        p.appendChild(t);
      }
    }
    box.appendChild(p);
  });
}

function loadLyricsFor(card, lyricsArtist, lyricsTitle, highlight) {
  const box = card.querySelector(".lyrics-text");

  box.textContent = "loading lyrics…";
  box.scrollTop = 0;
  fetchLyricLines(lyricsArtist, lyricsTitle)
    .then((lines) => {
      const idx = highlight ? findBestMatchingLineIndex(lines, highlight.text) : -1;
      if (highlight && idx === -1) {
        console.error(`Highlight line not found in fetched lyrics for "${lyricsTitle}":`, highlight.text);
      }
      // Once we know where the highlighted line is, start the lyrics right
      // there instead of making her scroll past the rest of the song to
      // find it — everything before it is dropped, not just scrolled past.
      const visibleLines = idx >= 0 ? lines.slice(idx) : lines;
      const visibleHighlightIdx = idx >= 0 ? 0 : -1;
      renderLyricLines(box, visibleLines, visibleHighlightIdx, highlight && highlight.translation);
      box.scrollTop = 0;
    })
    .catch((err) => {
      console.error("Lyrics fetch failed:", err);
      box.textContent = `lyrics not available for this one (${err.message}). `;
      box.scrollTop = 0;
      const retryBtn = document.createElement("button");
      retryBtn.type = "button";
      retryBtn.className = "lyrics-retry";
      retryBtn.textContent = "retry";
      retryBtn.addEventListener("click", () => loadLyricsFor(card, lyricsArtist, lyricsTitle, highlight));
      box.appendChild(retryBtn);
    });
}

// --- Build each track card from TRACKS (see tracks-data.js) ---

function buildTrackCard(track, index) {
  const displayTitle = track.displayTitle || track.title;
  const isReverse = index % 2 === 1;

  const section = document.createElement("section");
  section.className = `track-card reveal${isReverse ? " reverse" : ""}`;
  section.dataset.track = String(track.order);

  section.innerHTML = `
    <div class="record-wrap">
      <div class="record">
        <div class="record-label">
          <label class="photo-frame placeholder" data-track-photo="${track.order}">
            <input type="file" accept="image/*" class="photo-input" hidden />
            <img class="photo-preview" alt="" hidden />
            <span class="placeholder-prompt">
              <span class="placeholder-icon">+</span>
              <span class="placeholder-text">add a photo</span>
            </span>
          </label>
        </div>
      </div>
    </div>
    <div class="note-wrap">
      <p class="track-index">${String(track.order).padStart(2, "0")}</p>
      <h2 class="track-title"></h2>
      <p class="track-artist"></p>
      <div class="spotify-embed">
        <iframe
          style="border-radius: 12px"
          src="https://open.spotify.com/embed/track/${track.spotifyTrackId}?utm_source=generator&theme=0"
          width="100%"
          height="152"
          frameborder="0"
          allowfullscreen
          allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
          loading="lazy"
        ></iframe>
      </div>
      <div class="lyrics-text">loading lyrics…</div>
    </div>
  `;

  section.querySelector(".track-title").textContent = displayTitle;
  section.querySelector(".track-artist").textContent = track.artist;

  return section;
}

const reel = document.getElementById("reel");
const finaleCard = document.querySelector(".finale-card");

document.querySelectorAll(".reveal").forEach((el) => revealObserver.observe(el));

TRACKS.slice()
  .sort((a, b) => a.order - b.order)
  .forEach((track, index) => {
    const section = buildTrackCard(track, index);
    reel.insertBefore(section, finaleCard);
    revealObserver.observe(section);
    section.querySelectorAll(".photo-frame").forEach(wirePhotoFrame);

    const lyricsArtist = track.lyricsArtist || track.artist;
    const lyricsTitle = track.lyricsTitle || track.title;
    // Stagger requests slightly so firing them all at once doesn't trip a
    // rate limit on the free API.
    setTimeout(() => loadLyricsFor(section, lyricsArtist, lyricsTitle, track.highlightLyric), index * 300);
  });
