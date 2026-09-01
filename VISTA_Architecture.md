# VISTA --- System Architecture Document

## 1. Overview

**VISTA (Visual Intelligence & Spatial Temporal Archive)** is a
persistent visual-memory system that converts continuous video into a
structured, queryable representation of objects, people, locations,
interactions, and events.

### Core idea

> **VISTA does not merely store video. It remembers what happened.**

A user provides a video or camera stream. VISTA processes it through an
OpenCV-centric perception pipeline, maintains persistent object
identities and scene state over time, extracts events and relationships,
and exposes that memory through natural-language queries with
evidence-backed answers.

------------------------------------------------------------------------

## 2. Goals

### Primary goals

1.  Build persistent visual memory from video.
2.  Maintain object identity across time, motion, viewpoint changes, and
    temporary occlusion.
3.  Represent events and spatial relationships explicitly.
4.  Support temporal and relational queries.
5.  Return answers grounded in specific visual evidence.
6.  Make OpenCV 5 a substantive part of the perception pipeline.
7.  Support both offline video and future live-camera operation.
8.  Keep uncertainty explicit rather than forcing unsupported
    conclusions.

### Non-goals

-   General-purpose video generation.
-   Replacing a foundation model with VISTA.
-   Treating an LLM as the primary perception system.
-   Storing every frame as the only form of memory.
-   Claiming identity or causality when visual evidence is insufficient.

------------------------------------------------------------------------

## 3. High-Level Architecture

``` text
                         ┌───────────────────────┐
                         │      VIDEO INPUT      │
                         │ Camera / Video File   │
                         └───────────┬───────────┘
                                     │
                                     ▼
              ┌─────────────────────────────────────────┐
              │            PERCEPTION LAYER             │
              │                 OpenCV 5                │
              │                                         │
              │ Frame sampling                          │
              │ Motion / optical flow                   │
              │ Feature extraction & matching           │
              │ Object detection                        │
              │ Multi-object tracking                   │
              │ Scene / region analysis                 │
              └───────────────────┬─────────────────────┘
                                  │
                                  ▼
              ┌─────────────────────────────────────────┐
              │        TEMPORAL ASSOCIATION            │
              │                                         │
              │ Track stitching                         │
              │ Object re-identification                │
              │ Occlusion handling                       │
              │ Spatial consistency                     │
              │ Temporal consistency                     │
              └───────────────────┬─────────────────────┘
                                  │
                                  ▼
              ┌─────────────────────────────────────────┐
              │            VISUAL MEMORY                │
              │                                         │
              │ Object Memory                            │
              │ Event Memory                             │
              │ Scene State                              │
              │ Relationship Graph                       │
              └───────────────────┬─────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
          ┌──────────────────┐        ┌──────────────────┐
          │ Semantic /       │        │ Structured       │
          │ Vector Retrieval │        │ Graph / DB       │
          └────────┬─────────┘        └────────┬─────────┘
                   │                           │
                   └────────────┬──────────────┘
                                ▼
              ┌─────────────────────────────────────────┐
              │              QUERY ENGINE               │
              │                                         │
              │ Query parsing                            │
              │ Entity / event retrieval                 │
              │ Temporal filtering                       │
              │ Relationship traversal                   │
              │ Evidence retrieval                       │
              └───────────────────┬─────────────────────┘
                                  │
                                  ▼
              ┌─────────────────────────────────────────┐
              │          REASONING / ANSWERING          │
              │                                         │
              │ Evidence synthesis                       │
              │ Temporal reasoning                       │
              │ Uncertainty handling                     │
              │ Natural-language generation              │
              └───────────────────┬─────────────────────┘
                                  │
                                  ▼
              ┌─────────────────────────────────────────┐
              │                 VISTA UI                 │
              │                                         │
              │ Answer                                   │
              │ Timeline                                 │
              │ Evidence frames                          │
              │ Bounding boxes / tracks                  │
              │ Confidence / uncertainty                 │
              └─────────────────────────────────────────┘
```

------------------------------------------------------------------------

## 4. Architectural Principles

### 4.1 Perception before reasoning

The system should first establish visual evidence through computer
vision. Language-model reasoning operates on structured observations
rather than raw, unverified assumptions.

### 4.2 Persistent identity

Objects should have identities that survive across frames and, where
possible, temporary occlusion.

### 4.3 Event-centric memory

VISTA should store meaningful changes and interactions rather than
relying exclusively on frame-level embeddings.

### 4.4 Evidence-backed answers

Every factual answer should be traceable to one or more observations and
source frames.

### 4.5 Explicit uncertainty

If evidence is insufficient, VISTA should say so.

### 4.6 Efficient memory

Continuous video should not require storing every frame indefinitely.
VISTA should retain representative frames, tracks, events, embeddings,
and references to source video.

------------------------------------------------------------------------

## 5. Core Components

## 5.1 Video Ingestion

Responsibilities:

-   Read video files and camera streams.
-   Normalize frame rate and resolution.
-   Maintain timestamps and frame indices.
-   Create seekable references to source footage.
-   Produce frames for downstream processing.

Input:

``` text
Video stream
```

Output:

``` text
Frame {
    frame_id
    timestamp
    image
    source_id
}
```

------------------------------------------------------------------------

## 5.2 OpenCV 5 Perception Layer

OpenCV should provide substantive functionality rather than being used
only for I/O.

Potential responsibilities:

-   Frame decoding and preprocessing.
-   Motion estimation.
-   Optical flow.
-   Feature detection and description.
-   Feature matching.
-   Geometric verification.
-   Tracking support.
-   Region and scene analysis.
-   Image transformations.
-   Temporal frame selection.

The exact algorithms can evolve during implementation.

------------------------------------------------------------------------

## 5.3 Object Detection

The detector identifies entities relevant to the selected environment.

Example classes:

-   person
-   backpack
-   laptop
-   cup
-   notebook
-   phone
-   chair
-   tool

The detector can be an external model, but OpenCV should remain
responsible for significant downstream visual processing.

Output:

``` text
Detection {
    class
    bounding_box
    confidence
    frame_id
}
```

------------------------------------------------------------------------

## 5.4 Multi-Object Tracking

Tracking converts frame-level detections into persistent trajectories.

Responsibilities:

-   Associate detections across frames.
-   Maintain track IDs.
-   Handle short-term missed detections.
-   Estimate object motion.
-   Record trajectory history.
-   Trigger re-identification when tracks are interrupted.

Output:

``` text
Track {
    track_id
    object_class
    observations[]
    trajectory[]
    first_seen
    last_seen
    status
}
```

------------------------------------------------------------------------

## 5.5 Temporal Association and Re-identification

This is one of VISTA's core technical components.

When an object disappears temporarily, VISTA should determine whether it
has:

1.  left the scene,
2.  become occluded,
3.  been missed by the detector,
4.  changed appearance/viewpoint,
5.  reappeared as an existing object.

Association can combine:

-   appearance similarity,
-   local feature matching,
-   spatial continuity,
-   motion continuity,
-   temporal constraints,
-   scene geometry,
-   contextual relationships.

The result should include an association confidence.

------------------------------------------------------------------------

## 5.6 Visual Memory

Visual Memory is the central data layer.

It consists of four complementary representations.

### Object Memory

Stores persistent entities and their histories.

``` text
Object
├── identity
├── category
├── appearance descriptors
├── observations
├── locations
├── trajectory
├── interactions
└── confidence
```

### Event Memory

Stores meaningful changes.

Examples:

-   entered
-   exited
-   picked up
-   placed
-   moved
-   approached
-   interacted
-   appeared
-   disappeared
-   scene changed

``` text
Event
├── type
├── timestamp
├── subject
├── actor
├── source_state
├── destination_state
├── confidence
└── evidence_refs
```

### Scene Memory

Stores snapshots of the world state.

``` text
SceneState
├── timestamp
├── visible_objects
├── object_locations
├── active_people
└── spatial_relationships
```

### Relationship Graph

Represents relationships between entities.

``` text
Person_A ──picked_up──> Notebook
Notebook ──located_on──> Desk
Person_A ──near──> Desk
```

------------------------------------------------------------------------

## 5.7 Evidence Store

VISTA should preserve evidence references for important observations.

Evidence may include:

-   source frame IDs,
-   short frame ranges,
-   representative images,
-   bounding boxes,
-   track segments,
-   source video timestamps.

A final answer should be able to point back to evidence.

------------------------------------------------------------------------

## 5.8 Query Engine

The query engine converts natural-language questions into retrieval
operations.

Example:

> "What happened to the red notebook?"

Possible retrieval plan:

``` text
1. Resolve "red notebook" → Object
2. Retrieve its observation history
3. Retrieve related events
4. Sort events temporally
5. Retrieve evidence frames
6. Construct answer
```

Example query types:

### Object queries

-   Where is the laptop?
-   When was the mug last seen?

### Temporal queries

-   What happened before the bag disappeared?
-   What happened after Person A entered?

### Relationship queries

-   Who moved the notebook?
-   What did Person B interact with?

### State-change queries

-   What changed while I was away?
-   Which objects moved?

### Investigation queries

-   Why is the screwdriver no longer on the table?

------------------------------------------------------------------------

## 5.9 Reasoning / Answering Layer

The reasoning layer synthesizes retrieved evidence.

It should:

-   reason over timelines,
-   combine multiple observations,
-   distinguish observation from inference,
-   identify contradictions,
-   express uncertainty,
-   avoid unsupported conclusions.

A response should conceptually separate:

``` text
OBSERVED
"In frame 18420, the screwdriver is on the workbench."

INFERRED
"Person A appears to move it."

CONFIDENCE
0.88
```

------------------------------------------------------------------------

## 5.10 VISTA User Interface

The UI should prioritize visual verification.

Core elements:

1.  Video player.
2.  Natural-language query box.
3.  Answer panel.
4.  Event timeline.
5.  Evidence cards.
6.  Object history.
7.  Bounding-box overlays.
8.  Confidence / uncertainty indicator.

Selecting an event should seek directly to the relevant video timestamp.

------------------------------------------------------------------------

# 6. Data Flow

``` text
Video
  ↓
Frame extraction
  ↓
Detection + visual features
  ↓
Tracking
  ↓
Temporal association
  ↓
Object observations
  ↓
Event extraction
  ↓
Memory update
  ↓
Query
  ↓
Structured retrieval
  ↓
Evidence retrieval
  ↓
Reasoning
  ↓
Answer + evidence
```

------------------------------------------------------------------------

# 7. Storage Architecture

A practical implementation can use three storage layers.

### Object / Event Database

Stores structured entities, events, timestamps, and relationships.

### Vector Store

Stores visual and semantic embeddings for approximate retrieval.

### Object / Evidence Storage

Stores representative frames and source-video references.

Conceptually:

``` text
                    VISTA MEMORY
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     Structured DB   Vector Store   Video / Images
          │              │              │
     objects/events   embeddings     evidence
     relationships    semantic       source frames
```

------------------------------------------------------------------------

# 8. AWS Deployment

AWS should provide meaningful infrastructure without obscuring the CV
contribution.

A possible deployment:

``` text
                    AWS
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
       Compute                Storage
          │                     │
       EC2 /                   S3
     Graviton                  │
          │                     ├── source video
          │                     ├── evidence frames
          │                     └── artifacts
          ▼
    VISTA Processing
          │
          ├── OpenCV 5
          ├── tracking
          ├── event extraction
          └── memory service
          │
          ▼
       Query API
          │
          ▼
        VISTA UI
```

The final AWS architecture should be chosen based on measurable
requirements such as throughput, latency, cost, and reproducibility.

------------------------------------------------------------------------

# 9. Reliability and Failure Handling

The system should explicitly handle:

-   detector misses,
-   temporary occlusion,
-   duplicate tracks,
-   object reappearance,
-   low-confidence observations,
-   ambiguous queries,
-   missing evidence,
-   conflicting observations.

Example:

``` text
Observation A: Notebook on desk
Observation B: Notebook not visible
Observation C: Notebook on shelf

Conclusion:
Notebook was likely moved from desk to shelf.

Confidence:
0.82

Evidence:
A + B + C
```

The system should not claim:

> "Person A definitely moved the notebook"

unless the evidence supports that conclusion.

------------------------------------------------------------------------

# 10. Extensibility

VISTA should be modular enough to support future capabilities:

-   live camera streams,
-   multiple cameras,
-   active perception,
-   cross-camera identity association,
-   anomaly detection,
-   scene comparison,
-   agentic investigation,
-   voice queries,
-   mobile clients,
-   COOL / Graviton optimization.

------------------------------------------------------------------------

# 11. Competition Differentiation

The project should emphasize:

### OpenCV depth

OpenCV is involved in the actual visual understanding pipeline.

### Persistent visual memory

The system maintains identity and state over time.

### Temporal reasoning

Questions can involve events before, after, and between observations.

### Evidence grounding

Answers link back to visual evidence.

### Agentic potential

The query system can evolve from passive retrieval into active
investigation.

### Real-world usefulness

The same architecture can support personal memory, workplace monitoring,
asset tracking, incident analysis, and other domains without changing
its fundamental memory engine.
