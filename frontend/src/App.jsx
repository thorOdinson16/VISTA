import { useEffect, useRef, useState } from "react";

// Restrained palette: every object reads as gold, with enough variation in
// tone to tell neighbouring boxes apart.
const TONES = ["#f3d97a", "#d4af37", "#e8c766", "#fff1c1", "#b8962e", "#f0e2b0"];

function colorFor(category = "") {
  let h = 0;
  for (let i = 0; i < category.length; i++) h = (h * 31 + category.charCodeAt(i)) >>> 0;
  return TONES[h % TONES.length];
}

function confClass(c) {
  if (c >= 0.7) return "conf-ok";
  if (c >= 0.4) return "conf-mid";
  return "conf-low";
}

function fmtTime(s) {
  if (s === null || s === undefined) return "-";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, "0")}`;
}

// Nearest observation bounding box within a short window, else null.
function boxAt(object, t, window = 1.5) {
  const obs = object.observations || [];
  if (!obs.length) return null;
  let best = null;
  let bestDelta = Infinity;
  for (const o of obs) {
    const d = Math.abs(o.timestamp - t);
    if (d < bestDelta) {
      bestDelta = d;
      best = o;
    }
  }
  if (!best || bestDelta > window) return null;
  return best.bounding_box;
}

async function api(url, options) {
  const r = await fetch(url, options);
  let body = null;
  try {
    body = await r.json();
  } catch {
    /* non-JSON error body */
  }
  if (!r.ok) {
    throw new Error((body && body.detail) || `Request failed (${r.status})`);
  }
  return body;
}

const SUGGESTIONS = [
  "What is shown in the video?",
  "Where is the coin?",
  "Where is my phone?",
  "What happened in the video?",
];

export default function App() {
  const [status, setStatus] = useState({
    objects: 0,
    events: 0,
    video: null,
    fps: null,
    duration: null,
  });
  const [ingesting, setIngesting] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [question, setQuestion] = useState("");
  const [querying, setQuerying] = useState(false);
  const [lastQuestion, setLastQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [timeline, setTimeline] = useState([]);
  const [objects, setObjects] = useState([]);
  const [scene, setScene] = useState(null);
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [selectedObject, setSelectedObject] = useState(null);
  const [similar, setSimilar] = useState([]);
  const [videoUrl, setVideoUrl] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [videoSize, setVideoSize] = useState({ w: 0, h: 0 });

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileRef = useRef(null);

  async function refreshStatus() {
    const s = await api("/api/status");
    setStatus(s);
    if (s.video) setVideoUrl(`/video?t=${Date.now()}`);
    return s;
  }

  async function refreshMemory() {
    const [tl, ob, sc] = await Promise.all([
      api("/api/timeline"),
      api("/api/objects"),
      api("/api/scene"),
    ]);
    setTimeline(tl.events || []);
    setObjects(ob.objects || []);
    setScene(sc.state);
  }

  useEffect(() => {
    // Restore an already-ingested session after a page refresh.
    refreshStatus()
      .then((s) => (s.video ? refreshMemory() : null))
      .catch(() => {});
  }, []);

  // Draw bounding boxes whenever playback time, objects or video size change.
  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const ctx = canvas.getContext("2d");
    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!video.videoWidth || !videoSize.w) return;
    const sx = canvas.width / videoSize.w;
    const sy = canvas.height / videoSize.h;

    for (const object of objects) {
      const box = boxAt(object, currentTime);
      if (!box) continue;
      const color = colorFor(object.category);
      const x = box.x_min * sx;
      const y = box.y_min * sy;
      const w = (box.x_max - box.x_min) * sx;
      const h = (box.y_max - box.y_min) * sy;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
      const label = `${object.category} · ${object.object_id}`;
      ctx.font = "600 11px Inter, sans-serif";
      const tw = ctx.measureText(label).width + 12;
      const ly = Math.max(0, y - 20);
      ctx.fillStyle = color;
      ctx.fillRect(x, ly, tw, 20);
      ctx.fillStyle = "#0a0a0a";
      ctx.fillText(label, x + 6, ly + 14);
    }
  }, [currentTime, objects, videoSize]);

  async function ingest(file) {
    if (!file) return;
    setIngesting(true);
    setError(null);
    setAnswer(null);
    setSelectedObject(null);
    setSelectedEvent(null);
    setSimilar([]);
    try {
      const form = new FormData();
      form.append("file", file);
      await api("/api/ingest", { method: "POST", body: form });
      await refreshStatus();
      await refreshMemory();
    } catch (e) {
      setError(e.message);
    } finally {
      setIngesting(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function ask(text) {
    const q = (text ?? question).trim();
    if (!q || querying) return;
    setQuestion(q);
    setLastQuestion(q);
    setQuerying(true);
    setError(null);
    try {
      const d = await api("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      setAnswer(d);
      if (d.searched_on_demand) {
        // Memory was rebuilt with the new vocabulary; sync the panels.
        await refreshStatus();
        await refreshMemory();
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setQuerying(false);
    }
  }

  async function showSimilar(objectId) {
    try {
      const d = await api(
        `/api/similar?object_id=${encodeURIComponent(objectId)}`
      );
      setSimilar(d.similar || []);
    } catch (e) {
      setError(e.message);
    }
  }

  function seek(t) {
    if (videoRef.current) {
      videoRef.current.currentTime = t;
      videoRef.current.play();
    }
  }

  function selectEvent(ev) {
    setSelectedEvent(ev);
    seek(ev.start_time);
  }

  function selectObject(o) {
    setSelectedObject((prev) => (prev === o.object_id ? null : o.object_id));
    setSimilar([]);
  }

  const selected = objects.find((o) => o.object_id === selectedObject) || null;
  const selectedEvents = selected
    ? timeline
        .filter((ev) => (selected.event_refs || []).includes(ev.event_id))
        .sort((a, b) => a.start_time - b.start_time)
    : [];

  function evidenceImage(ev) {
    if (ev.event_refs && ev.event_refs.length) {
      return `/evidence/${ev.event_refs[0]}.jpg`;
    }
    if (ev.object_refs && ev.object_refs.length) {
      return `/evidence/${ev.object_refs[0]}.jpg`;
    }
    return null;
  }

  const hasVideo = Boolean(status.video && videoUrl);
  const span = status.duration && status.duration > 0 ? status.duration : 180;

  function onDrop(e) {
    e.preventDefault();
    setDragOver(false);
    ingest(e.dataTransfer.files && e.dataTransfer.files[0]);
  }

  const fileInput = (
    <input
      ref={fileRef}
      className="file-input"
      type="file"
      accept="video/*"
      onChange={(e) => ingest(e.target.files[0])}
    />
  );

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <h1>VISTA</h1>
          <span className="tagline">Visual Intelligence · Spatial Temporal Archive</span>
        </div>
        <div className="spacer" />
        {hasVideo && !ingesting && (
          <div className="chips">
            <span className="chip">
              <span className="pulse" />
              <b>{status.detector_name || status.detector_mode}</b>
            </span>
            <span className="chip">
              <b>{status.objects}</b> objects
            </span>
            <span className="chip">
              <b>{status.events}</b> events
            </span>
            {status.duration ? (
              <span className="chip">
                <b>{fmtTime(status.duration)}</b>
              </span>
            ) : null}
          </div>
        )}
        {hasVideo && !ingesting && (
          <button className="btn-ghost" onClick={() => fileRef.current?.click()}>
            New video
          </button>
        )}
        {fileInput}
      </header>

      {ingesting && (
        <div className="hero">
          <div className="processing">
            <div className="spinner" />
            <div>Building visual memory — detecting, tracking and indexing.</div>
          </div>
        </div>
      )}

      {!ingesting && !hasVideo && (
        <div className="hero">
          <h2>
            Video that <em>remembers</em>.
          </h2>
          <p>
            Drop in footage and ask where things are, what happened and when.
            Every answer comes with the frame that proves it.
          </p>
          <div
            className={`dropzone ${dragOver ? "drag" : ""}`}
            onClick={() => fileRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
          >
            <button className="btn-gold" type="button">
              Choose a video
            </button>
            <div className="hint">or drag and drop it here</div>
          </div>
          {error && <div className="error">{error}</div>}
        </div>
      )}

      {!ingesting && hasVideo && (
        <>
          <div className="grid">
            <div className="card">
              <h2 className="card-title">Footage</h2>
              <div className="stage">
                <video
                  ref={videoRef}
                  src={videoUrl}
                  controls
                  onTimeUpdate={(e) => setCurrentTime(e.target.currentTime)}
                  onLoadedMetadata={(e) =>
                    setVideoSize({
                      w: e.target.videoWidth,
                      h: e.target.videoHeight,
                    })
                  }
                />
                <canvas ref={canvasRef} className="overlay" />
              </div>

              {timeline.length > 0 && (
                <>
                  <div className="timeline">
                    <div
                      className="playhead"
                      style={{ left: `${Math.min(100, (currentTime / span) * 100)}%` }}
                    />
                    {timeline.map((ev, i) => (
                      <div
                        key={ev.event_id || i}
                        className={`marker ${
                          selectedEvent && selectedEvent.event_id === ev.event_id
                            ? "marker-active"
                            : ""
                        }`}
                        style={{ left: `${Math.min(99, (ev.start_time / span) * 100)}%` }}
                        title={`${fmtTime(ev.start_time)} · ${ev.type}`}
                        onClick={() => selectEvent(ev)}
                      />
                    ))}
                  </div>
                  <div className="timeline-caption">
                    {selectedEvent
                      ? `${fmtTime(selectedEvent.start_time)} — ${selectedEvent.type} · ${(
                          selectedEvent.confidence * 100
                        ).toFixed(0)}%`
                      : `${timeline.length} events — select a marker to jump to it`}
                  </div>
                </>
              )}
            </div>

            <div className="stack">
              <div className="card">
                <h2 className="card-title">Ask</h2>
                <div className="ask">
                  <input
                    type="text"
                    placeholder="Where is the coin?"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && ask()}
                  />
                  <button
                    className="btn-gold"
                    onClick={() => ask()}
                    disabled={querying || !question.trim()}
                  >
                    {querying ? "…" : "Ask"}
                  </button>
                </div>
                <div className="suggestions">
                  {SUGGESTIONS.map((s) => (
                    <button
                      key={s}
                      className="suggestion"
                      disabled={querying}
                      onClick={() => ask(s)}
                    >
                      {s}
                    </button>
                  ))}
                </div>
                {error && <div className="error">{error}</div>}
              </div>

              <div className="card">
                <h2 className="card-title">Answer</h2>
                {querying ? (
                  <div className="processing" style={{ alignItems: "flex-start" }}>
                    <div className="spinner" />
                    <div>Searching the footage for “{lastQuestion}”…</div>
                  </div>
                ) : answer ? (
                  <div>
                    <p className="answer-text">{answer.answer.text}</p>
                    <div className="meta-row">
                      <span className={`badge ${confClass(answer.answer.confidence)}`}>
                        {(answer.answer.confidence * 100).toFixed(0)}% confidence
                      </span>
                      <span className="badge">{answer.query_type}</span>
                      {answer.target_category && (
                        <span className="badge">{answer.target_category}</span>
                      )}
                    </div>

                    {answer.searched_on_demand && (
                      <div className="notice">
                        “{answer.target_category}” wasn’t in the detector’s
                        vocabulary, so VISTA re-scanned the video looking for it.
                      </div>
                    )}

                    {answer.answer.observed.length > 0 && (
                      <>
                        <div className="section-label">Observed</div>
                        <ul className="list">
                          {answer.answer.observed.map((o, i) => (
                            <li key={i}>{o}</li>
                          ))}
                        </ul>
                      </>
                    )}

                    {answer.answer.inferred.length > 0 && (
                      <>
                        <div className="section-label">Inferred</div>
                        <ul className="list">
                          {answer.answer.inferred.map((o, i) => (
                            <li key={i}>{o}</li>
                          ))}
                        </ul>
                      </>
                    )}

                    {answer.answer.poster && (
                      <>
                        <div className="section-label">Frame</div>
                        <div className="evidence-grid">
                          <div
                            className="evidence-card"
                            onClick={() => seek(0)}
                            title="Representative frame"
                          >
                            <img
                              src={answer.answer.poster}
                              alt="scene frame"
                              onError={(e) => {
                                e.target.style.display = "none";
                              }}
                            />
                          </div>
                        </div>
                      </>
                    )}

                    {answer.evidence.length > 0 && (
                      <>
                        <div className="section-label">Evidence</div>
                        <div className="evidence-grid">
                          {answer.evidence.map((ev) => {
                            const src = evidenceImage(ev);
                            return (
                              <div
                                key={ev.evidence_id}
                                className="evidence-card"
                                onClick={() => seek(ev.timestamp_start)}
                              >
                                {src ? (
                                  <img
                                    src={src}
                                    alt="evidence"
                                    onError={(e) => {
                                      e.target.style.display = "none";
                                    }}
                                  />
                                ) : (
                                  <div className="empty" style={{ padding: 12 }}>
                                    frame {ev.frame_start}
                                  </div>
                                )}
                                <span className="meta">{fmtTime(ev.timestamp_start)}</span>
                              </div>
                            );
                          })}
                        </div>
                      </>
                    )}
                  </div>
                ) : (
                  <div className="empty">
                    Ask about anything in the footage — objects don’t need to be
                    on a fixed list.
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="grid-2">
            <div className="card">
              <h2 className="card-title">
                Objects <span className="count">{objects.length}</span>
              </h2>
              {objects.length > 0 ? (
                <ul className="list">
                  {objects.map((o) => (
                    <li
                      key={o.object_id}
                      className={`object-item ${
                        selectedObject === o.object_id ? "object-active" : ""
                      }`}
                      onClick={() => selectObject(o)}
                    >
                      <span className="dot" style={{ background: colorFor(o.category) }} />
                      <strong>{o.category}</strong>
                      <span className="tag">{o.object_id}</span>
                      <span className="tag">{(o.confidence * 100).toFixed(0)}%</span>
                      {o.attributes && o.attributes.color && (
                        <span className="tag">{o.attributes.color}</span>
                      )}
                      <button
                        className="mini"
                        onClick={(e) => {
                          e.stopPropagation();
                          showSimilar(o.object_id);
                        }}
                      >
                        similar
                      </button>
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="empty">
                  No objects detected. Ask for something specific and VISTA
                  will go looking for it.
                </div>
              )}

              {similar.length > 0 && (
                <>
                  <div className="section-label">Visually similar</div>
                  <ul className="list">
                    {similar.map((s) => (
                      <li key={s.object_id}>
                        {s.object_id} <span className="tag">{s.score.toFixed(2)}</span>
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </div>

            <div className="card">
              {selected ? (
                <>
                  <h2 className="card-title">
                    {selected.category} · {selected.object_id}
                  </h2>
                  <div className="history">
                    <img
                      className="object-thumb"
                      src={`/evidence/${selected.object_id}.jpg`}
                      alt="object"
                      onError={(e) => {
                        e.target.style.display = "none";
                      }}
                    />
                    <div>
                      <div className="section-label" style={{ marginTop: 0 }}>
                        Events ({selectedEvents.length})
                      </div>
                      {selectedEvents.length > 0 ? (
                        <ul className="list">
                          {selectedEvents.map((ev) => (
                            <li
                              key={ev.event_id}
                              className="clickable"
                              onClick={() => seek(ev.start_time)}
                            >
                              {fmtTime(ev.start_time)} — {ev.type}{" "}
                              <span className="tag">{ev.kind}</span>{" "}
                              <span className="tag">
                                {(ev.confidence * 100).toFixed(0)}%
                              </span>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <div className="empty">No linked events.</div>
                      )}
                    </div>
                  </div>
                </>
              ) : (
                <>
                  <h2 className="card-title">
                    Scene
                    {scene ? <span className="count">@ {fmtTime(scene.timestamp)}</span> : null}
                  </h2>
                  {scene && scene.spatial_relationships.length > 0 ? (
                    <ul className="list">
                      {scene.spatial_relationships.map((r, i) => (
                        <li key={i}>{r}</li>
                      ))}
                    </ul>
                  ) : (
                    <div className="empty">
                      Select an object to see its history, or ask a question.
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
