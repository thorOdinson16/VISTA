# VISTA --- Software Requirements Document

## 1. Document Purpose

This document defines the functional, technical, performance, and
quality requirements for **VISTA (Visual Intelligence & Spatial Temporal
Archive)**.

VISTA is a visual-memory system that transforms video into a structured
representation of objects, people, locations, events, and relationships,
allowing users to ask questions about what happened in the video and
receive evidence-backed answers.

------------------------------------------------------------------------

# 2. Product Vision

> **VISTA remembers what happened in visual environments.**

The system should bridge the gap between raw video and useful episodic
memory.

Instead of requiring users to manually scrub through video, VISTA should
allow questions such as:

-   "Where was the red notebook last seen?"
-   "Who moved the mug?"
-   "What happened before the laptop disappeared?"
-   "What changed while I was away?"
-   "Show me every time the backpack was moved."

Answers must be grounded in visual evidence.

------------------------------------------------------------------------

# 3. Scope

## 3.1 MVP Scope

The first implementation must support:

1.  Video-file ingestion.
2.  Frame extraction and timestamp preservation.
3.  Object detection.
4.  Multi-object tracking.
5.  Persistent object histories.
6.  Basic temporal association.
7.  Event extraction.
8.  Structured visual memory.
9.  Natural-language querying.
10. Evidence retrieval.
11. Timeline-based visualization.
12. Video seeking to evidence timestamps.
13. Confidence / uncertainty representation.

## 3.2 Future Scope

The following are desirable but not required for MVP:

-   Live camera input.
-   Multi-camera memory.
-   Active camera control.
-   Voice interface.
-   Cross-video memory.
-   Fully autonomous investigation.
-   Edge deployment.
-   COOL / Graviton optimization.
-   Mobile application.

------------------------------------------------------------------------

# 4. Users

## 4.1 Primary User

A user who wants to understand or retrieve information from previously
observed video without manually reviewing the entire recording.

## 4.2 Secondary Users

Potential future users include:

-   researchers,
-   operators,
-   investigators,
-   facility managers,
-   accessibility users,
-   robotics systems,
-   security / monitoring systems.

------------------------------------------------------------------------

# 5. Functional Requirements

## FR-001 --- Video Ingestion

The system shall accept supported video files.

The system shall preserve:

-   source identifier,
-   frame index,
-   timestamp,
-   frame rate,
-   resolution.

The system shall reject unsupported or corrupted inputs gracefully.

------------------------------------------------------------------------

## FR-002 --- Frame Processing

The system shall process video frames using a configurable sampling
strategy.

The system shall avoid unnecessary processing of visually redundant
frames where possible.

The system shall retain references allowing important moments to be
recovered from the original video.

------------------------------------------------------------------------

## FR-003 --- OpenCV Processing

The system shall use OpenCV 5 for substantive computer-vision
processing.

At minimum, the implementation should use OpenCV for multiple meaningful
components such as:

-   video processing,
-   feature extraction,
-   feature matching,
-   motion analysis,
-   optical flow,
-   geometric verification,
-   tracking support,
-   temporal analysis.

OpenCV shall not be limited to video decoding or display.

------------------------------------------------------------------------

## FR-004 --- Object Detection

The system shall identify configured object categories in processed
frames.

Each detection shall contain:

-   object category,
-   bounding box,
-   confidence,
-   frame identifier,
-   timestamp.

------------------------------------------------------------------------

## FR-005 --- Object Tracking

The system shall associate detections across consecutive frames.

Each active object shall receive a persistent internal track identifier.

The system shall record:

-   trajectory,
-   first observed timestamp,
-   last observed timestamp,
-   observations,
-   confidence.

------------------------------------------------------------------------

## FR-006 --- Occlusion Handling

The system shall tolerate short periods in which an object is not
directly visible.

When an object reappears, the system shall attempt to determine whether
it corresponds to an existing object track.

The association decision shall include a confidence score.

------------------------------------------------------------------------

## FR-007 --- Object Re-identification

The system shall support re-association of an object after track
interruption using available evidence.

Possible evidence includes:

-   visual appearance,
-   local features,
-   spatial continuity,
-   motion continuity,
-   temporal constraints,
-   contextual relationships.

------------------------------------------------------------------------

## FR-008 --- Persistent Object Memory

The system shall maintain an object record independent of individual
frames.

An object record should contain:

``` text
Object ID
Category
Appearance information
First seen
Last seen
Observation history
Location history
Trajectory
Related events
Confidence
```

------------------------------------------------------------------------

## FR-009 --- Scene State

The system shall maintain snapshots of the observed environment.

A scene state should include:

-   visible objects,
-   approximate locations,
-   active people,
-   relevant spatial relationships.

------------------------------------------------------------------------

## FR-010 --- Event Detection

The system shall identify meaningful temporal events.

Initial event types should include:

-   object appeared,
-   object disappeared,
-   person entered,
-   person exited,
-   object picked up,
-   object placed,
-   object moved,
-   object carried,
-   object approached another object.

Event types shall be extensible.

------------------------------------------------------------------------

## FR-011 --- Event Representation

Each event shall contain, where available:

``` text
Event ID
Event type
Timestamp / time range
Subject
Actor
Source state
Destination state
Confidence
Evidence references
```

------------------------------------------------------------------------

## FR-012 --- Relationship Graph

The system shall represent important relationships between entities.

Example:

``` text
Person A
   │
   ├── picked_up ──> Notebook
   │
   └── moved_to ──> Shelf

Notebook
   └── located_on ──> Shelf
```

Relationships shall be queryable.

------------------------------------------------------------------------

## FR-013 --- Evidence Storage

Important observations and events shall retain references to source
evidence.

Evidence references shall include enough information to locate the
relevant source frames or timestamp range.

------------------------------------------------------------------------

## FR-014 --- Natural-Language Queries

The system shall accept natural-language questions.

Examples:

-   "Where is the red notebook?"
-   "Who moved the mug?"
-   "When was the laptop last seen?"
-   "What happened before the bag disappeared?"
-   "What changed in the room?"

------------------------------------------------------------------------

## FR-015 --- Query Resolution

The system shall convert a user query into one or more retrieval
operations.

The query system shall be able to resolve:

-   objects,
-   people,
-   events,
-   locations,
-   time ranges,
-   relationships,
-   state changes.

------------------------------------------------------------------------

## FR-016 --- Temporal Retrieval

The system shall support temporal operators such as:

-   before,
-   after,
-   during,
-   between,
-   first,
-   last,
-   nearest in time.

Example:

> "What happened immediately before the notebook disappeared?"

------------------------------------------------------------------------

## FR-017 --- Relational Retrieval

The system shall support questions involving relationships.

Examples:

-   "Who interacted with the backpack?"
-   "What did Person A move?"
-   "Which objects were on the table?"

------------------------------------------------------------------------

## FR-018 --- Evidence-Grounded Answering

Every factual answer should be associated with supporting evidence.

The answer interface shall provide access to the relevant evidence.

------------------------------------------------------------------------

## FR-019 --- Evidence Navigation

The user shall be able to select an evidence item and jump directly to
its corresponding timestamp in the video.

------------------------------------------------------------------------

## FR-020 --- Timeline

The system shall display significant events on a chronological timeline.

Selecting an event shall expose its associated object(s), timestamp, and
evidence.

------------------------------------------------------------------------

## FR-021 --- Uncertainty

The system shall preserve confidence values for uncertain observations
and inferences.

If evidence is insufficient, the system shall communicate uncertainty
rather than produce a definitive unsupported claim.

------------------------------------------------------------------------

## FR-022 --- Observation vs Inference

The system should distinguish between directly observed facts and
higher-level inferences.

Example:

``` text
Observed:
The notebook is visible on the desk at 10:02.

Observed:
Person A is holding the notebook at 10:04.

Inference:
Person A likely moved the notebook.
```

------------------------------------------------------------------------

## FR-023 --- Contradiction Handling

When observations conflict, the system shall avoid silently selecting
one interpretation.

It should surface uncertainty or identify the conflicting observations.

------------------------------------------------------------------------

# 6. User Interface Requirements

## UI-001 --- Video Viewer

The interface shall provide a video player with timestamp navigation.

## UI-002 --- Query Interface

The interface shall provide a natural-language query input.

## UI-003 --- Answer Panel

The interface shall display the generated answer prominently.

## UI-004 --- Evidence

The interface shall display supporting evidence frames or clips.

## UI-005 --- Timeline

The interface shall provide a chronological event timeline.

## UI-006 --- Object History

The interface should allow users to inspect the history of a selected
object.

## UI-007 --- Visual Overlay

The interface should display bounding boxes and track identifiers when
viewing relevant evidence.

## UI-008 --- Confidence

The interface should communicate confidence or uncertainty for
non-trivial conclusions.

------------------------------------------------------------------------

# 7. Data Requirements

## 7.1 Object Record

``` text
object_id
category
appearance_embedding
first_seen
last_seen
observations
trajectory
location_history
event_refs
confidence
```

## 7.2 Observation Record

``` text
observation_id
object_id
frame_id
timestamp
bounding_box
appearance_features
confidence
```

## 7.3 Event Record

``` text
event_id
type
start_time
end_time
subject_refs
actor_refs
source_state
destination_state
confidence
evidence_refs
```

## 7.4 Evidence Record

``` text
evidence_id
source_video
frame_start
frame_end
timestamp_start
timestamp_end
object_refs
event_refs
```

------------------------------------------------------------------------

# 8. Performance Requirements

These targets are initial engineering goals and may be revised after
benchmarking.

## PR-001 --- Processing

The MVP should support processing ordinary 720p video without requiring
manual intervention.

## PR-002 --- Memory Efficiency

The system should avoid retaining every decoded frame in active memory.

## PR-003 --- Query Latency

For an already-indexed video, common object and event queries should
return results within a few seconds.

## PR-004 --- Evidence Navigation

Selecting an evidence item should navigate to the corresponding video
location with minimal delay.

## PR-005 --- Scalability

The memory representation should allow longer videos without linear
growth in retained raw-frame memory.

------------------------------------------------------------------------

# 9. Accuracy Requirements

## AR-001 --- Tracking

The system should maintain stable object tracks under ordinary movement.

## AR-002 --- Re-identification

The system should recover object identity after short occlusions where
sufficient visual evidence exists.

## AR-003 --- Event Accuracy

Events should only be created when supporting evidence exceeds a
configurable confidence threshold.

## AR-004 --- Evidence Integrity

Every generated event should retain a source reference.

## AR-005 --- Answer Grounding

The answer generator shall not introduce factual claims that cannot be
supported by retrieved memory or evidence.

------------------------------------------------------------------------

# 10. Reliability Requirements

The system shall gracefully handle:

-   corrupted video,
-   missing frames,
-   detector failures,
-   temporary occlusion,
-   duplicate tracks,
-   ambiguous objects,
-   low-confidence detections,
-   empty search results,
-   ambiguous natural-language queries.

For an unsupported or unanswered query, the system should respond with
an explicit limitation rather than fabricate an answer.

------------------------------------------------------------------------

# 11. Security and Privacy Requirements

The system should minimize unnecessary retention of raw video.

Where possible:

-   store references instead of duplicate video,
-   isolate user datasets,
-   protect stored evidence,
-   avoid exposing unrelated video segments in query results.

Any deployment involving people should include appropriate consent,
access control, and data-retention policies.

------------------------------------------------------------------------

# 12. AWS Requirements

The system shall include a meaningful AWS component for competition
deployment.

The AWS implementation should support:

-   compute for processing,
-   object storage for source/evidence data,
-   API access to VISTA services.

The final deployment shall document:

-   AWS services used,
-   reason for each service,
-   deployment steps,
-   resource requirements,
-   reproducibility considerations.

If performance data supports it, the project may additionally evaluate
AWS Graviton / COOL optimization.

------------------------------------------------------------------------

# 13. Testing Requirements

Testing shall occur at multiple levels.

## Unit tests

Test:

-   timestamp handling,
-   track association,
-   event generation,
-   query parsing,
-   memory updates.

## Integration tests

Test:

``` text
Video
→ perception
→ tracking
→ memory
→ query
→ evidence
```

## Scenario tests

Prepare controlled videos for:

1.  object enters scene,
2.  object exits scene,
3.  object moves,
4.  object is picked up,
5.  object is placed,
6.  object becomes briefly occluded,
7.  object reappears,
8.  multiple similar objects coexist,
9.  person-object interaction,
10. ambiguous / insufficient evidence.

## Evaluation metrics

Track at least:

-   detection accuracy,
-   tracking accuracy,
-   re-identification accuracy,
-   event precision / recall,
-   query answer accuracy,
-   evidence retrieval accuracy,
-   query latency,
-   processing throughput.

------------------------------------------------------------------------

# 14. MVP Acceptance Criteria

The MVP is considered successful when the system can process a
controlled video and correctly answer questions such as:

### Object location

> "Where was the red notebook last seen?"

### Object history

> "Show me the history of the notebook."

### Actor identification

> "Who moved the mug?"

### Temporal reasoning

> "What happened before the laptop disappeared?"

### Change detection

> "What changed while I was away?"

### Evidence

For each answer, the user can inspect the relevant video evidence.

------------------------------------------------------------------------

# 15. Demonstration Scenario

The primary competition demonstration should use a controlled
environment containing several objects and people.

Example sequence:

``` text
00:00
Notebook on desk.

00:35
Person A enters.

00:48
Person A picks up notebook.

01:10
Person A walks behind an occluding object.

01:17
Person A reappears.

01:24
Notebook placed on shelf.

02:00
Person A leaves.

03:00
User asks:

"What happened to the notebook?"
```

VISTA should produce:

``` text
The red notebook was moved from the desk
to the shelf by Person A.

Timeline:
00:48 — picked up
01:10 — carried behind occlusion
01:24 — placed on shelf

Evidence:
[00:48] [01:10] [01:24]
```

This demonstrates the central value proposition:

> **VISTA remembers an object's history rather than merely recognizing
> it in individual frames.**

------------------------------------------------------------------------

# 16. Future Research Directions

Potential future capabilities include:

1.  Active perception.
2.  Multi-camera visual memory.
3.  Cross-video identity persistence.
4.  Automatic scene graph construction.
5.  Long-term anomaly detection.
6.  Visual memory compression.
7.  Counterfactual scene reasoning.
8.  Autonomous visual investigation.
9.  Real-time streaming memory.
10. Edge / Graviton optimization.

------------------------------------------------------------------------

# 17. Success Definition

VISTA succeeds if a user can provide a video, walk away from it, and
later ask:

> **"What happened?"**

and receive a useful answer that is:

-   temporally coherent,
-   visually grounded,
-   explainable,
-   evidence-backed,
-   uncertainty-aware,
-   and substantially easier to obtain than manually reviewing the
    video.
