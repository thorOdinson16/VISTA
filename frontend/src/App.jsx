import { useEffect, useRef, useState } from "react";

const CATEGORY_COLORS = {
  person: "#ff6bcb",
  notebook: "#ff5252",
  book: "#ff5252",
  mug: "#4c8dff",
  cup: "#4c8dff",
  backpack: "#3fb96f",
  laptop: "#e0a53e",
  "cell phone": "#9b6bff",
  phone: "#9b6bff",
  chair: "#7ad0c0",
};

function colorFor(category) {
  return CATEGORY_COLORS[category] || "#4c8dff";
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

export default function App() {
  const [status, setStatus] = useState({
    objects: 0,
    events: 0,
    video: null,
    fps: null,
    duration: null,
  });
  const [ingesting, setIngesting] = useState(false);
  const [question, setQuestion] = useState("");
  const [querying, setQuerying] = useState(false);
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

  async function refreshStatus() {
    const r = await fetch("/api/status");
    const s = await r.json();
    setStatus(s);
    if (s.video) setVideoUrl(`/video?t=${Date.now()}`);
  }

  async function refreshTimeline() {
    const r = await fetch("/api/timeline");
    const d = await r.json();
    setTimeline(d.events || []);
  }

  async function refreshObjects() {
    const r = await fetch("/api/objects");
    const d = await r.json();
    setObjects(d.objects || []);
  }

  async function refreshScene() {
    const r = await fetch("/api/scene");
    const d = await r.json();
    setScene(d.state);
  }

  useEffect(() => {
    refreshStatus();
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
      ctx.strokeStyle = colorFor(object.category);
      ctx.lineWidth = 2;
      ctx.strokeRect(
        box.x_min * sx,
        box.y_min * sy,
        (box.x_max - box.x_min) * sx,
        (box.y_max - box.y_min) * sy
      );
      const label = `${object.category} ${object.object_id}`;
      ctx.font = "12px sans-serif";
      const w = ctx.measureText(label).width + 8;
      ctx.fillStyle = colorFor(object.category);
      ctx.fillRect(box.x_min * sx, box.y_min * sy - 16, w, 16);
      ctx.fillStyle = "#0f1115";
      ctx.fillText(label, box.x_min * sx + 4, box.y_min * sy - 4);
    }
  }, [currentTime, objects, videoSize]);

  async function onFile(e) {
    const file = e.target.files[0];
    if (!file) return;
    setIngesting(true);
    setAnswer(null);
    setSelectedObject(null);
    setSelectedEvent(null);
    const form = new FormData();
    form.append("file", file);
    await fetch("/api/ingest", { method: "POST", body: form });
    await refreshStatus();
    await refreshTimeline();
    await refreshObjects();
    await refreshScene();
    setIngesting(false);
  }

  async function ask() {
    if (!question.trim()) return;
    setQuerying(true);
    const r = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const d = await r.json();
    setAnswer(d);
    setQuerying(false);
  }

  async function showSimilar(objectId) {
    setSelectedObject(objectId);
    const r = await fetch(`/api/similar?object_id=${encodeURIComponent(objectId)}`);
    const d = await r.json();
    setSimilar(d.similar || []);
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

  const selected =
    objects.find((o) => o.object_id === selectedObject) || null;
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

  return (
    <div className="app">
      <header>
        <h1>VISTA</h1>
        <div className="subtitle">
          Visual Intelligence &amp; Spatial Temporal Archive
        </div>
      </header>

      <div className="panel">
        <h2>Ingest</h2>
        <input type="file" accept="video/*" onChange={onFile} disabled={ingesting} />
        <div style={{ marginTop: 8 }} className="empty">
          {ingesting
            ? "Processing video..."
            : status.video
            ? `Indexed: ${status.objects} objects, ${status.events} events`
            : "No video ingested yet."}
        </div>
        {status.detector_mode && (
          <div className="empty" style={{ marginTop: 4, fontSize: 12 }}>
            Detector: {status.detector_name || status.detector_mode}
            {status.detector_name ? ` (${status.detector_mode})` : ""}
            {status.saliency_fallback ? " + saliency fallback" : ""}
          </div>
        )}
      </div>

      <div className="row">
        <div className="panel">
          <h2>Video</h2>
          {videoUrl ? (
            <div className="video-wrap">
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
          ) : (
            <div className="empty">Upload a video to begin.</div>
          )}
        </div>

        <div className="panel">
          <h2>Ask</h2>
          <input
            type="text"
            placeholder='e.g. "Who moved the notebook?"'
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && ask()}
          />
          <button onClick={ask} disabled={querying} style={{ marginTop: 10 }}>
            {querying ? "Thinking..." : "Ask"}
          </button>
        </div>
      </div>

      {timeline.length > 0 && (
        <div className="panel">
          <h2>Timeline</h2>
          <div className="timeline">
            {timeline.map((ev, i) => {
              const span =
                status.duration && status.duration > 0 ? status.duration : 180;
              const left = Math.min(99, (ev.start_time / span) * 100);
              return (
                <div
                  key={i}
                  className={`marker ${
                    selectedEvent && selectedEvent.event_id === ev.event_id
                      ? "marker-active"
                      : ""
                  }`}
                  style={{ left: `${left}%` }}
                  title={`${fmtTime(ev.start_time)} - ${ev.type}`}
                  onClick={() => selectEvent(ev)}
                />
              );
            })}
          </div>
          <div className="empty" style={{ marginTop: 8 }}>
            {selectedEvent
              ? `${fmtTime(selectedEvent.start_time)} — ${selectedEvent.type} (${selectedEvent.confidence.toFixed(2)})`
              : "Click a marker to seek."}
          </div>
        </div>
      )}

      <div className="row">
        <div className="panel">
          <h2>Answer</h2>
          {answer ? (
            <div>
              <div className="answer-text">{answer.answer.text}</div>
              <span className={`status-badge ${confClass(answer.answer.confidence)}`}>
                confidence {(answer.answer.confidence * 100).toFixed(0)}%
              </span>
              <span className="tag">{answer.query_type}</span>
              {answer.target_category && (
                <span className="tag">{answer.target_category}</span>
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
                            <div className="empty">frame {ev.frame_start}</div>
                          )}
                          <div className="meta">{fmtTime(ev.timestamp_start)}</div>
                        </div>
                      );
                    })}
                  </div>
                </>
              )}
            </div>
          ) : (
            <div className="empty">Ask a question about the video.</div>
          )}
        </div>

        <div className="panel">
          <h2>Objects</h2>
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
                  <span
                    className="dot"
                    style={{ background: colorFor(o.category) }}
                  />
                  <strong>{o.category}</strong>{" "}
                  <span className="tag">{o.object_id}</span>
                  <span className="tag">
                    {(o.confidence * 100).toFixed(0)}%
                  </span>
                  <span className="tag">{o.trajectory.length} pts</span>
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
            <div className="empty">No objects yet.</div>
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

          {scene && scene.spatial_relationships.length > 0 && (
            <>
              <div className="section-label">
                Scene @ {fmtTime(scene.timestamp)}
              </div>
              <ul className="list">
                {scene.spatial_relationships.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </>
          )}
        </div>
      </div>

      {selected && (
        <div className="panel">
          <h2>Object history — {selected.category} {selected.object_id}</h2>
          <div className="row">
            <div>
              <img
                className="object-thumb"
                src={`/evidence/${selected.object_id}.jpg`}
                alt="object"
                onError={(e) => {
                  e.target.style.display = "none";
                }}
              />
            </div>
            <div>
              <div className="section-label">
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
                      <span className="tag">{ev.kind}</span>
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
        </div>
      )}
    </div>
  );
}
