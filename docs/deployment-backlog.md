# Atlas deployment backlog

Local filing list for later Linear issues. Nothing here has been posted.
Do not refile an id in [Already on Linear](#already-on-linear) under the same
title. Where a gap is still open, the issue below cites that id and states the
remaining work.

Top-level issues are the `BL-` headings. Nested `Sub-issues` are delegable
slices of that parent, not extra top-level tickets.

A trained model used as a tool (MAAT) accepts a compatible data-acquisition
input and returns an analysis a later step can use. The agent sandbox in
ATLAS-27 stays closed: a session cannot execute or register caller-supplied
code, and it cannot gain a tool by naming one.

## Already on Linear

Do not restate these as new issues.

| Id | Why it is cited, not copied |
| --- | --- |
| ATLAS-14 | STAC pagination and empty `Scene.source` are still open. BL-17 is only the cursor on a pull manifest. |
| ATLAS-15 | Antimeridian bbox coverage is done. BL-56 is the non-STAC clients that ticket did not cover. |
| ATLAS-16 | Incomplete GIBS, GOES, Himawari, and Maxar snapshots are still open. |
| ATLAS-17 | Data-acquisition pentest epic. Children include ATLAS-14, 16, and 20. |
| ATLAS-18 | Aggregation epic for FLDAS and RED/NIR. |
| ATLAS-19 | FLDAS training table. Merged on main; do not refile the extract. |
| ATLAS-20 | Catalog and registry drift is in review. BL-16 is the registration test for the next source. |
| ATLAS-21 | Training-chip pipeline was canceled. BL-07 is AOI clipping, not that pipeline. |
| ATLAS-22 | Cloud-minimum filter is done. |
| ATLAS-23 | `item_to_scene` hardening is done. |
| ATLAS-24 | RED/NIR exposure is in progress. BL-58 is the index math after those bands exist. |
| ATLAS-25 | Agent runtime epic. |
| ATLAS-26 | Finish the agent loop. |
| ATLAS-27 | Sandbox. BL-26 and BL-59 depend on it and must not weaken it. |
| ATLAS-28 | Core tools epic. Bash (ATLAS-42) and write (ATLAS-44) are done. |
| ATLAS-29 | Backend session metrics. The UI surface is ATLAS-49. |
| ATLAS-30 | Context compaction. This backlog does not redesign it. BL-41 only warns. |
| ATLAS-31 | Tool discovery. |
| ATLAS-32 | Skill loading. BL-31 is how a loaded skill is followed. |
| ATLAS-33 | Skill authoring. BL-32 is the capture rules. |
| ATLAS-34 | Load `AGENTS.md`. BL-34 is the no-new-tools constraint. |
| ATLAS-35 | Unified `atlas data` CLI is done. BL-01 is the agent calling that same CLI. |
| ATLAS-36 | Debloat `atlas.agent` after behavior changes. |
| ATLAS-37 | Image format enum. BL-30 is the MAAT check that uses it. |
| ATLAS-38 | System prompt file. BL-63 is the sandbox and citation text. |
| ATLAS-39 | Small-model prompt classifier. BL-62 is the closed tool list. |
| ATLAS-40 | Attach local image bytes to the model. BL-29 is the shared chip with MAAT. |
| ATLAS-41 | Retire `LocalArtifactStore`. BL-64 is the fail-closed replacement rule. |
| ATLAS-43 | List tool, still open. |
| ATLAS-45 | Edit tool, still open. |
| ATLAS-46 | Scene limit modes, still open. BL-18 records the mode that was used. |
| ATLAS-47 – ATLAS-53 | UI / TUI Design basics (`/model`, context gauge, tokens and spend, session or user key, `.atlas/` layout, scoped remove, HTTP status and latency). Do not refile. |
| ATLAS-54 | MAAT epic. Children ATLAS-55, 56, 57, 58, and 60 are the first local models. |
| ATLAS-59 | YOLO detector was canceled. Do not reopen it here. |
| ATLAS-5 – ATLAS-13 | Older copies of the data epic. File against ATLAS-17 and ATLAS-18, not these. |

## Issues

### BL-01. Let the agent use the separate data-acquisition CLI

**Problem:** `atlas data` is a separate CLI (ATLAS-35, done) and `consolidate()` is already agent-agnostic, but a session still cannot search or pull through that same entry. A person and the agent can disagree about the AOI, the dates, and the output directory.

**End goal:** One registered tool runs the existing data-acquisition CLI contract for a caller-chosen AOI and date window, writes under the project `.atlas/data`, and returns scene ids plus paths. It does not grow a second downloader.

**Already on Linear:** ATLAS-35 is the CLI. This issue is the agent entry to that CLI.

**Sub-issues:**

- **BL-01a. Search tool.** Problem: the agent cannot run a metadata search. End goal: a tool returns the same scene list `atlas data` would print, with no download.
- **BL-01b. Pull tool.** Problem: downloads have no tool. End goal: a tool writes products under `.atlas/data` and returns relative paths only.
- **BL-01c. Root lock.** Problem: a tool argument could point the output outside the project. End goal: the pull rejects absolute paths and `..`, same as the artifact store.

### BL-02. Join every available data type with optical satellite imagery

**Problem:** Search fans out across the registry, but the product a person gets is still a per-source scene list plus an optional Planetary Computer mosaic. Tabular sources (FLDAS, FIRMS, SMAP) and optical scenes for the same place and time are not one joined result.

**End goal:** Joining every available data type with optical satellite imagery for one AOI and one time window attaches every returned row to the optical scene or scenes for that same place and time. A source that fails is named and the optical result is kept.

**Already on Linear:** ATLAS-18 and ATLAS-19 cover FLDAS extraction, not this join.

**Sub-issues:**

- **BL-02a. Join key.** Problem: scenes from different collections have no shared identity. End goal: a key of AOI, time window, and footprint that both optical items and tables use.
- **BL-02b. Tables stay tables.** Problem: FLDAS and FIRMS are not images. End goal: they attach as tables on the join, not as fake scenes.
- **BL-02c. Partial failure.** Problem: one bad source can look like an empty pull. End goal: the join lists the error and still returns the optical scenes that succeeded.

### BL-03. Index satellite information for model reasoning

**Problem:** A search returns STAC-shaped metadata that is too large and too raw to put in a model context. The agent cannot look up "clearest Sentinel-2 over this AOI in this week" without another full search.

**End goal:** Indexable satellite information the model can query: scene id, source, datetime, cloud, bands, footprint, and workspace-relative path. Answers cite index ids. This is the index that supports model reasoning.

**Sub-issues:**

- **BL-03a. Scene cards.** Problem: there is no small record per scene. End goal: one card per kept scene, capped in size.
- **BL-03b. Query.** Problem: the only read path is the original search. End goal: query cards by place, date, source, and cloud without calling the provider again.
- **BL-03c. Citations.** Problem: later tools do not point at a card. End goal: MAAT and report steps include the card id they used.

### BL-04. Put monthly tables and daily imagery on one analysis window

**Problem:** FLDAS is monthly and optical scenes are daily. A naive join either drops the table or pretends a month is one timestamp.

**End goal:** The join states the table's native period and which optical dates fall inside it, and it refuses a silent mismatch.

### BL-05. Harmonize CRS and resolution before a model reads a stack

**Problem:** Sentinel-2, Landsat, MODIS, and FLDAS do not share a grid. A MAAT input that assumes one shape will read misaligned pixels.

**End goal:** A stack records source CRS, resolution, and the resample used. The MAAT adapter rejects a stack that was not harmonized to the plugin's declared shape.

### BL-06. Mask cloud, shadow, snow, and saturation per pixel

**Problem:** `select` keeps or drops a scene from metadata cloud cover. Scenes with no cloud field (SAR) stay. SCL and QA bands are not applied, so a "clear" scene can still be mostly shadow or snow.

**End goal:** An optional mask product from the scene's own QA or SCL band, written next to the chip, with the unmasked fraction stored on the scene card.

### BL-07. Clip pulled assets to the requested AOI

**Problem:** Several clients still hand back a full granule. Training and viewing then mix pixels outside the region the person named. ATLAS-21 tracked a chip pipeline and was canceled; the clip itself is still missing.

**End goal:** Every raster and table written for a pull is clipped to the request AOI. The manifest stores both the granule id and the clip geometry.

**Already on Linear:** Do not reopen ATLAS-21. This is the clip only.

### BL-08. Write a pull manifest the session can reopen

**Problem:** After a pull, the workspace has files and no record of the request, the scene ids, or the tool versions that produced them.

**End goal:** `.atlas/data/<pull-id>/manifest.json` stores AOI, start and end dates, sources, scene ids, and relative output paths, with no API keys and no host absolute paths.

### BL-09. Deduplicate one granule returned by several catalogs

**Problem:** Planetary Computer, Earth Search, USGS, and CDSE can all return the same Sentinel or Landsat granule. The model then reasons about duplicates as if they were independent observations.

**End goal:** The index keeps one card per physical granule and lists the catalogs that offered it.

### BL-10. Read a COG window instead of the whole granule

**Problem:** A small AOI still downloads a full scene. Long sessions run out of disk and time before any analysis starts.

**End goal:** When the asset is a COG, the pull reads the AOI window and records bytes read versus granule size on the manifest.

### BL-11. Estimate size and quota before a pull starts

**Problem:** A wide date range across the whole registry can request far more data than the machine or the provider quota allows. Failure shows up mid-download.

**End goal:** The CLI prints an estimate (scene count, bytes, sources skipped) and starts the download only after the person confirms, or when an explicit `--yes` is passed.

### BL-12. Retry a rate-limited source without aborting the others

**Problem:** A 429 from one provider currently fails that source. Aggressive retries would stall every other source in the fan-out.

**End goal:** Each source retries with a bounded backoff. The join waits only for that source's budget, then records the final HTTP status and moves on.

### BL-13. Name the missing credential when a source cannot search

**Problem:** Earthdata, FIRMS, and CDSE fail with provider errors that do not say which key is absent. The rest of the pull is hard to trust.

**End goal:** A missing credential becomes a source error that names the variable (`EARTHDATA_TOKEN`, `FIRMS_MAP_KEY`, or the CDSE pair) and does not print the value.

### BL-14. Accept a polygon AOI

**Problem:** `PullRequest` is a bbox. A watershed or a coastline bbox includes a lot of ground the person did not ask about, and the dateline case is only special-cased for coverage.

**End goal:** A pull accepts a polygon, stores it on the manifest, and documents the bbox used for catalogs that only support a bbox.

### BL-15. Publish units, scale, offset, and nodata for every asset

**Problem:** A table value or a scaled reflectance pixel can be handed to a model with no units. The model then treats raw digital numbers as physical values.

**End goal:** Each asset on a scene card lists units, scale, offset, and nodata. The MAAT adapter refuses a band whose units are missing when the plugin declares a unit.

### BL-16. Make a new source pass one registration test

**Problem:** Sources are registered by hand in `SOURCES`. Drift between the catalog, the client, and the exports is a recurring review (ATLAS-20). The next source has no single test that proves it can be added the same way.

**End goal:** A source is registered only if a test sees `search(PullRequest)`, a catalog row, and an export from `atlas.data`, using a fake HTTP client.

**Already on Linear:** ATLAS-20 is the current drift. This issue is the gate for the next source.

### BL-17. Store the STAC page cursor on the pull manifest

**Problem:** STAC search still does not page, and `Scene.source` can stay empty (ATLAS-14). Even after paging exists, a resumed pull has nowhere to store the cursor.

**End goal:** The manifest records next-page cursors per source. A resume continues from that cursor instead of repeating page one.

**Already on Linear:** ATLAS-14 owns the pager. Do not refile it.

### BL-18. Record which scene limit mode a pull used

**Problem:** Limit modes (first, last, evenly spaced, all) are specified in ATLAS-46 and are not saved. A resumed session cannot tell which scenes were dropped on purpose.

**End goal:** The manifest stores the mode and the scene ids it kept and dropped.

**Already on Linear:** ATLAS-46 owns the modes.

### BL-19. Fall back to Sentinel-1 when optical coverage is too cloudy

**Problem:** A person asks what changed, and every optical scene in the window is cloudy. The session stops or answers from a bad chip. Sentinel-1 is already registered and is not offered as the fallback.

**End goal:** When optical coverage is below a stated threshold, the join marks SAR as the analysis input and says that optical was skipped for cloud.

### BL-20. Attach a license before an artifact is shown or exported

**Problem:** Maxar Open Data and some other collections restrict redistribution. The workspace treats every downloaded file as freely showable.

**End goal:** The scene card stores the license id. Export and the human-readable report refuse, or watermark, an artifact whose license does not allow the action.

### BL-21. Close the basic Atlas tooling set

**Problem:** The session can read, write, and run a bounded shell. List (ATLAS-43) and span edit (ATLAS-45) are still open, and a tool result has no byte cap, so one listing can fill the context window.

**End goal:** A default session can read, list, write, edit one unique span, and run a bounded shell, and each tool result is truncated to a fixed byte budget with the relative path only.

**Already on Linear:** ATLAS-28, ATLAS-42, ATLAS-43, ATLAS-44, ATLAS-45.

**Sub-issues:**

- **BL-21a. List.** Problem: there is no list tool. End goal: land ATLAS-43, workspace-relative names only.
- **BL-21b. Edit.** Problem: there is no span edit. End goal: land ATLAS-45, unchanged file when the span is missing or repeated.
- **BL-21c. Result cap.** Problem: tool text is unbounded. End goal: truncate with a visible marker and never include a host path.

### BL-22. Adapt a data-acquisition product into a MAAT input

**Problem:** Model-as-a-tool (MAAT) means a trained deep-learning model takes a compatible data-acquisition input and returns an analysis. Plugins declare a tensor shape, but nothing builds that tensor from a scene chip or a table.

**End goal:** An adapter turns one indexed chip or table into the plugin's declared input, or returns a tool error that names the mismatch (bands, dtype, shape, units).

**Already on Linear:** ATLAS-54 is the epic. This issue is the adapter, not another network.

**Sub-issues:**

- **BL-22a. Chip adapter.** Problem: a scene path is not a tensor. End goal: build the declared image input from a workspace-relative chip.
- **BL-22b. Table adapter.** Problem: FLDAS and FIRMS rows have no tensor path. End goal: build the declared table input from manifest rows.
- **BL-22c. Mismatch error.** Problem: a bad chip can crash the runtime. End goal: a validation error, and the tool is not swapped for a different one.

### BL-23. Return a MAAT analysis the next step can use

**Problem:** A local model that only draws a preview PNG does not give the agent a value it can cite. The next step needs masks, labels, or rows plus the input ids.

**End goal:** Every MAAT tool returns a `ToolResult` whose text is the analysis (class, score, row values, or mask summary) and whose artifacts are workspace-relative outputs. The input scene-card ids are included.

### BL-24. Use trained models for harder analysis

**Problem:** Metadata search cannot answer change, water, cloud, or built-up questions. The first trained models are already filed and are not all wired to a pull.

**End goal:** A session can run the trained land-cover, building, water, cloud, and change models on a compatible pull and get an analysis back. Canceled detectors stay canceled.

**Already on Linear:** ATLAS-54, ATLAS-55, ATLAS-56, ATLAS-57, ATLAS-58, ATLAS-60. ATLAS-59 stays canceled.

### BL-25. Add a model plugin, or replace its weights, without editing the loop

**Problem:** Adding a model still means a new package plus a registry walk that the session does not explain. Replacing weights has a sha256 check and no rollback. The agent must not gain a runtime by naming one.

**End goal:** A person adds `plugin.toml`, weights, and a tool module. The next session advertises it only after the adapter accepts the spec. A bad weight file keeps the previous file active.

**Sub-issues:**

- **BL-25a. Spec check.** Problem: a bad `plugin.toml` fails late. End goal: refuse the plugin before it is advertised.
- **BL-25b. Weight rollback.** Problem: a failed export can leave a half-written file. End goal: swap weights only after the hash matches, and keep the previous file.
- **BL-25c. Closed registry.** Problem: a model id in a prompt is not a tool. End goal: unknown runtimes raise, as they do now (ATLAS-27).

### BL-26. Let Atlas draft a new tool, including a MAAT, without running it

**Problem:** The product goal is that Atlas can build new tools as needed, including MAAT, and can build new models or update older tools. The sandbox forbids executing caller-supplied code. Those two rules conflict if a draft runs inside the session.

**End goal:** The agent may write a tool or model proposal under `.atlas/proposals`. Nothing in that directory is imported, executed, or registered until a person accepts it outside the session.

**Already on Linear:** ATLAS-27 is the sandbox. Do not weaken it.

**Sub-issues:**

- **BL-26a. Proposal files.** Problem: there is nowhere to put a draft. End goal: a proposal directory that the tool loader ignores.
- **BL-26b. Human accept.** Problem: acceptance is undefined. End goal: a developer command copies an accepted proposal into the source tree.
- **BL-26c. No exec.** Problem: a proposal could be imported by path. End goal: discovery skips the proposal directory, and the session has no import or exec tool.

### BL-27. Score a candidate MAAT before it is advertised

**Problem:** A newly trained model can be registered with no held-out score. The session then treats it as ready for an arbitrary region.

**End goal:** Registration records a score on a held-out set that does not overlap the chips used to train it. Below a stated bar, the tool stays `opt_in` and the description says so.

### BL-28. Make a MAAT abstain when the input is not usable

**Problem:** A cloudy chip, a nodata wedge, or a low `coverage_fraction` still produces a confident class. The person cannot tell abstention from an answer.

**End goal:** The tool returns an abstention reason (cloud, nodata, coverage, or shape) and no class label when the declared check fails.

### BL-29. Hand the same chip to the language model and to MAAT

**Problem:** MAAT can read a workspace image while the language model only sees a path string (ATLAS-40). The two steps then disagree about what was analyzed.

**End goal:** One chip path is both the MAAT input and the image attached to the model turn, and the tool result cites that path.

**Already on Linear:** ATLAS-40 owns ingestion.

### BL-30. Refuse an image format the store and the plugin do not both allow

**Problem:** The artifact store's image suffixes and a plugin's declared input can diverge (ATLAS-37). A file that lists as an image may still be unreadable by the model.

**End goal:** The format enum is the only list. A MAAT call rejects a suffix outside that list before it opens the file.

**Already on Linear:** ATLAS-37 owns the enum.

### BL-31. Use a skill during a session

**Problem:** Skill utilization is filed (ATLAS-32) and does not say what the running session is allowed to do with the text. A skill that adds a tool would break the sandbox.

**End goal:** A loaded skill is instructions in the prompt: which pull, which MAAT, and how to cite. It does not register a tool and it does not raise the tool-call budget.

**Already on Linear:** ATLAS-32.

### BL-32. Create a skill from a finished analysis

**Problem:** Skill creation is filed (ATLAS-33). A captured skill can easily store an API key, a host path, or a one-off scene id that will not exist next month.

**End goal:** The person can save a skill that records AOI, dates, sources, and model ids. The file contains no secrets and no host paths, and it stays disabled until a person enables it.

**Already on Linear:** ATLAS-33.

**Sub-issues:**

- **BL-32a. Capture.** Problem: the successful steps are only in the transcript. End goal: write a skill body from the manifest and the MAAT names.
- **BL-32b. Redaction.** Problem: transcripts can echo a key. End goal: the writer drops key-shaped strings and absolute paths.
- **BL-32c. Enable.** Problem: a new file might load on the next launch. End goal: skills load only from an enabled list.

### BL-33. Create and modify session memory

**Problem:** There is no memory of the region, the dates, or the findings. Context compaction (ATLAS-30) is a different problem and is not a memory store. A memory file that keeps secrets is worse than none.

**End goal:** The person, or a bounded tool, can create a memory entry, modify one entry, and delete one entry. Memory stores the AOI, the dates, the finding, and the manifest id. It never stores a key.

**Sub-issues:**

- **BL-33a. Create.** Problem: nothing records a finding. End goal: append one entry under the project `.atlas` memory file.
- **BL-33b. Modify.** Problem: an entry cannot be corrected. End goal: replace one entry by id and leave the others.
- **BL-33c. Delete.** Problem: a stale finding stays forever. End goal: delete by id.
- **BL-33d. Secret fence.** Problem: a finding might include a token. End goal: reject a write that contains a configured secret or an absolute path.

### BL-34. Load project instructions without granting tools

**Problem:** `AGENTS.md` is local guidance and is not loaded into the loop (ATLAS-34). If that file can name a new tool, the sandbox is gone.

**End goal:** The session prepends project instructions as text. Unknown tool names in that file are not registered.

**Already on Linear:** ATLAS-34.

### BL-35. Keep a short or a long session inside a budget

**Problem:** A turn stops at `max_tool_calls` (default 256) and does not stop on tokens or dollars. A long analysis can run past the spend the person allowed. Compaction is ATLAS-30 and is out of scope here.

**End goal:** A short-to-long analysis session stays inside a budget the person sets, in tokens and in dollars. Hitting either budget stops the turn with a fixed sentence that names which budget fired, and it does not invent a final analysis.

### BL-36. Analyze change in an arbitrary region from the index

**Problem:** Change questions re-search the world and then guess. The index from BL-03 and the change model from ATLAS-60 are not connected, so an arbitrary region has no before/after pair.

**End goal:** Given an AOI and two dates, the session loads indexed scenes for that region and runs the change MAAT. If either date is missing from the index, it says so and does not substitute a different date.

**Sub-issues:**

- **BL-36a. Date pair.** Problem: "what changed" has no two dates. End goal: the request stores both dates on the manifest.
- **BL-36b. Change tool.** Problem: the pair is not a model input. End goal: call the trained change model (ATLAS-60) through the adapter.
- **BL-36c. Missing date.** Problem: a gap looks like "no change". End goal: an explicit missing-scene result.

### BL-37. Resume a session with the same region, dates, and earlier results

**Problem:** Quitting loses the AOI, the dates, and the tool results. The next launch is a new conversation even when the person is still on the same region.

**End goal:** Reopening a session id restores the region, the dates, the manifest, and the earlier tool results. The tool-call budget for the new turn starts fresh. The model stays the one last selected.

**Sub-issues:**

- **BL-37a. Persist identity.** Problem: a session has no id tied to a manifest. End goal: store session id, AOI, and dates, without keys.
- **BL-37b. Reload results.** Problem: earlier tool output is gone. End goal: reload the saved results and do not re-download unless asked.
- **BL-37c. Fresh budget.** Problem: a resume could inherit a spent tool budget. End goal: the new turn gets a full tool budget, matching today's resume rule.

### BL-38. Return an analysis a person can understand

**Problem:** A turn ends as model prose plus tool lines. A person cannot see what changed, where it changed, or which data supported it.

**End goal:** The human-readable result has three labeled parts: what changed, where (AOI and dates, not a host path), and which data supported it (source, scene id, datetime, artifact). Failed sources are listed. A low coverage fraction is stated as uncertainty.

**Sub-issues:**

- **BL-38a. What changed.** Problem: the claim is unstructured. End goal: one short paragraph tied to a MAAT result or an explicit "not enough data".
- **BL-38b. Where.** Problem: paths leak into the answer. End goal: place and time from the manifest only.
- **BL-38c. Which data.** Problem: citations are optional. End goal: every claim lists source, scene id, and artifact.
- **BL-38d. Gaps.** Problem: failures and thin coverage disappear. End goal: list failed sources and the coverage fraction.

### BL-39. Show the cited chip or table next to the claim

**Problem:** Even a cited path is hard to check. The person has to open a file the answer does not point at in the TUI.

**End goal:** The report lists a workspace-relative chip or table for each claim, and the TUI can open that relative path from the project artifacts directory.

### BL-40. Keep new session files in the project store and secrets in the user store

**Problem:** Sessions, artifacts, and pulls have a project home, and keys, weights, and defaults have a user home (ATLAS-51, ATLAS-52). Memory, manifests, and proposals do not yet follow that split, so a later feature will put a key next to a scene.

**End goal:** New session, artifact, data, memory, and proposal files go under the project `.atlas/`. Keys, weights, and user defaults go under `~/.atlas/`. Existing `.atlas/data` and `~/.atlas/weights` keep working.

**Already on Linear:** ATLAS-51 and ATLAS-52 define the layout and the remove commands.

### BL-41. Warn when the session nears the selected model's context limit

**Problem:** The context gauge (ATLAS-48) can show used versus the advertised limit and still not tell the person that the next turn is likely to fail. Compaction is a separate algorithm (ATLAS-30).

**End goal:** The status line warns at a stated fraction of the selected model's advertised context length. It does not compact, and it does not substitute a hardcoded window when the catalog has no length.

**Already on Linear:** ATLAS-48 and ATLAS-30.

### BL-42. Keep HTTP status, latency, and spend on a resumed session

**Problem:** Token totals, spend, tokens per second, and HTTP status are session-UI fields (ATLAS-49, ATLAS-53). A resumed session starts that history over, so a long analysis hides the earlier failures.

**End goal:** The session file stores the recent call list (status, elapsed seconds, token counts, spend when the provider sent it) and the TUI shows it again on resume. Secrets stay out of the file.

**Already on Linear:** ATLAS-29, ATLAS-49, ATLAS-53.

### BL-43. Store Earthdata and FIRMS keys like the OpenRouter key

**Problem:** OpenRouter session-or-user key handling is ATLAS-50. Earthdata and FIRMS keys still have nowhere to live except the project `.env`, which is easy to commit.

**End goal:** Each of those keys can be set for the session only, or for the user under `~/.atlas/keys`. The session value wins. The project tree never receives the file, and the value is never printed.

**Already on Linear:** ATLAS-50 is OpenRouter. This issue is the other pull credentials.

**Sub-issues:**

- **BL-43a. Session scope.** Problem: a one-off token has no memory-only slot. End goal: keep it in the process and drop it on exit.
- **BL-43b. User scope.** Problem: there is no user file. End goal: write `~/.atlas/keys/` with user-only permissions.
- **BL-43c. Silence.** Problem: errors echo the token. End goal: messages name the variable and redact the value.

### BL-44. Resolve a reprocessed scene by place and time

**Problem:** A dynamic reprocess changes scene ids. A manifest that only stores ids cannot find the replacement, so a resumed region looks empty.

**End goal:** The manifest stores collection, datetime, and footprint beside the id. A refresh may replace the id and must say that it did.

### BL-45. Do not reuse an older file when the requested window is empty

**Problem:** A source that has not published "today", or a sensor that was offline, can leave the previous CSV or chip in place. The report then describes the wrong day.

**End goal:** An empty window writes a failed manifest entry and does not present the older file as the result of this request.

### BL-46. Separate seasonal change from the change the person asked about

**Problem:** Snow, green-up, and harvest move pixels every year. A change model will report that as the event.

**End goal:** The report labels a seasonal baseline when the only difference matches a stored phenology or snow flag, and it does not call that baseline the requested change.

### BL-47. Record resampling when the sensors do not share a grid

**Problem:** A 10 m optical chip, a 30 m Landsat chip, a MODIS pixel, and a 0.1° FLDAS cell cannot be stacked silently.

**End goal:** The stack metadata names the target grid and the resample method. The report repeats that method next to any cross-sensor claim.

### BL-48. Pick one optical scene when several collections overlap

**Problem:** Landsat 8, Landsat 9, HLS, and USGS can all return the same day. Landsat 8 and 9 already share a collection and must keep the platform filter. The index still has no rule for which card is the optical anchor.

**End goal:** The join picks one optical anchor, records the platforms it rejected, and does not drop the Landsat platform filter.

### BL-49. Run from cached pulls and local models when OpenRouter is unreachable

**Problem:** A deployed session with no outbound network cannot open the agent, even if the pull and the MAAT weights are already on disk.

**End goal:** An explicit offline mode answers only from the index, cached artifacts, and local MAAT plugins. It does not call OpenRouter, and it says that the language-model turn was skipped.

### BL-50. Keep concurrent sessions from mixing artifacts

**Problem:** Two sessions in one project share one artifact directory. One remove or one pull can delete the other session's chips.

**End goal:** Each session id has its own artifact directory. A remove of one id does not delete the other, and it does not touch `~/.atlas/`.

### BL-51. Export a cited bundle

**Problem:** A person cannot hand someone else the analysis without also handing them the whole workspace, including whatever else is on disk.

**End goal:** An export contains the report, the manifest, the model id, the weight sha256, and the cited artifacts, plus the license ids. It omits keys and uncited files.

### BL-52. Delete a pull without deleting keys or weights

**Problem:** Disk fills up with granules. A careless recursive delete of `.atlas` or `~/.atlas` also removes keys and weights.

**End goal:** A delete command removes one pull id under the project data directory. Keys and weights are removed only when that command names them.

### BL-53. Show a cost preview before a long pull or a long session

**Problem:** Spend is visible after the fact. A long date range or a long tool loop is the expensive choice, and the person sees it too late.

**End goal:** Before a pull over a stated scene count, or a session over a stated token budget, the CLI or TUI prints the estimate and waits for confirmation unless `--yes` is set.

### BL-54. Write a redacted audit log

**Problem:** After a bad answer there is no durable list of tool names, model ids, and HTTP statuses. Transcripts are easy to lose and sometimes contain secrets.

**End goal:** The project session directory appends tool name, model id, HTTP status, and elapsed time. The log redacts key material and host paths.

### BL-55. Record the model id so two days can be compared

**Problem:** A later session on the same region can use another model and another sample setting. A difference in the prose is then mistaken for a difference on the ground.

**End goal:** The session stores model id, the MAAT weight hashes, and the skill id. The comparison view shows those before it shows the findings.

### BL-56. Handle dateline AOIs in the non-STAC clients

**Problem:** Antimeridian bboxes now affect coverage for the STAC path (ATLAS-15, done). GIBS, GOES, Himawari, and FIRMS still need their own request split or they return an empty or wrapped scene.

**End goal:** Each of those clients documents and tests a dateline AOI, and a failure is a source error rather than a silent empty image.

**Already on Linear:** ATLAS-15 is done for coverage. ATLAS-16 is the incomplete-snapshot epic. This issue is the dateline request only.

### BL-57. Tell the agent why a mosaic was skipped

**Problem:** Mosaics are Planetary Computer only. A skipped mosaic can look like a failed pull even when the scene list is fine. Main reports skipped mosaics in the CLI; the agent does not.

**End goal:** The tool result includes the skip reason in one sentence. The scene list is still returned.

### BL-58. Compute NDVI, NDWI, and NBR once RED and NIR are available

**Problem:** ATLAS-24 exposes RED and NIR and does not compute indices. Both the CLI and the agent need the same numbers, or they will each invent a formula.

**End goal:** One function returns NDVI, NDWI, and NBR from the exposed bands, with nodata propagated. The CLI and the agent both call it.

**Already on Linear:** ATLAS-24 owns band exposure. Do not refile that issue.

### BL-59. Keep the shell inside the workspace and away from user keys

**Problem:** Bash is a default tool and can reach `~/.atlas/keys` or another absolute path unless the sandbox stops it (ATLAS-27). That would expose the user key the TUI is careful not to print.

**End goal:** The shell's working directory is the workspace, absolute paths and `..` that leave it fail, and `~/.atlas/keys` is denied even if a path resolves there. A drafted tool still does not run.

**Already on Linear:** ATLAS-27.

**Sub-issues:**

- **BL-59a. Key directory.** Problem: the shell can read the user key file. End goal: that directory is denied and the error does not include the key.
- **BL-59b. Escape.** Problem: `..` and absolute paths leave the root. End goal: those commands fail closed.
- **BL-59c. Drafts.** Problem: a proposal could be executed by the shell. End goal: the proposal directory is not a runnable tool path for the session.

### BL-60. Keep the launch split obvious in deployment

**Problem:** Bare `atlas` opens the Rust TUI and `atlas data` stays the data CLI. A missing Rust toolchain or a non-interactive terminal can look like a hung analysis if the error is unclear.

**End goal:** `atlas` and `atlas data` keep those roles. A missing terminal prints the existing interactive-terminal error. A missing cargo prints the existing toolchain error. Neither path starts an HTTP client.

### BL-61. Use the same user and project split on Windows and Linux

**Problem:** The layout is described as `~/.atlas` and a project `.atlas`. Windows deployments will invent a second path unless the helper is the only place that resolves the user directory.

**End goal:** One helper resolves the user directory on macOS, Linux, and Windows. Keys never go in the project tree on any of them.

### BL-62. Classify a prompt without adding a tool

**Problem:** A small model in front of the loop (ATLAS-39) can become a second place that grants capabilities. That conflicts with the explicit registry.

**End goal:** The classifier may choose a skill or a refusal. It cannot register a tool or raise `max_tool_calls`.

**Already on Linear:** ATLAS-39.

### BL-63. State the sandbox and the citation rules in the system prompt file

**Problem:** The system prompt lives in the runtime module (ATLAS-38). It does not yet say that tools are relative to the workspace, that MAAT output must be cited, or that a draft tool is not executable.

**End goal:** The prompt file states those three rules. The runtime loads the file and does not keep a second copy.

**Already on Linear:** ATLAS-38.

### BL-64. Retire the artifact store without opening the path rule

**Problem:** `LocalArtifactStore` is the path glue ATLAS-41 wants retired. A replacement that returns host paths, or that resolves `..`, would undo the sandbox.

**End goal:** Path checks live in one helper used by tools, pulls, and deletes. Errors name the relative path only. Absolute paths and escapes still fail.

**Already on Linear:** ATLAS-41.

### BL-65. Compare two saved sessions for one region

**Problem:** A person can resume one session and still cannot see how today's finding differs from last week's on the same AOI.

**End goal:** A compare view takes two session ids, checks that the AOI matches, and lists findings that were added, removed, or reworded, with the model ids from BL-55.

### BL-66. Join nighttime lights, thermal, and SAR with the optical scene

**Problem:** Night lights (GIBS), land-surface temperature (SLSTR), and Sentinel-1 are registered and are not part of the optical join. A change at night or under cloud is invisible.

**End goal:** When those sources return for the same place and time, their cards attach to the optical anchor. An empty source is a listed gap, not a fake zero.

### BL-67. Keep training chips out of the region being scored

**Problem:** Workspace models ship with training chips. Scoring the same AOI those chips came from inflates the analysis.

**End goal:** The MAAT call records whether the request AOI overlaps a training footprint. On overlap it abstains or marks the score as leaked, and it says which footprint hit.

### BL-68. Alert when a remembered region crosses a threshold

**Problem:** Memory can store a region and still never tell the person that a later pull changed. The person has to remember to ask.

**End goal:** A memory entry may store a numeric threshold on a named MAAT output. A later pull for that AOI appends an alert when the value crosses it, and the alert cites the two manifests.

## Theme checklist

These themes are issues above, not section titles.

| Theme | Issue |
| --- | --- |
| Separate data-acquisition CLI | BL-01 |
| Join every available data type with optical satellite imagery at the same place and time | BL-02 |
| Indexable satellite information for model reasoning | BL-03 |
| Basic Atlas tooling | BL-21 |
| MAAT: a trained deep-learning model takes a compatible data-acquisition input and returns an analysis | BL-22, BL-23 |
| Trained models for harder analysis | BL-24 |
| Atlas builds new tools, including MAAT, and builds new models or updates older tools | BL-25, BL-26 |
| Skill utilization and skill creation | BL-31, BL-32 |
| Memory creation and modification | BL-33 |
| Short-to-long sessions inside a budget | BL-35 |
| Change in an arbitrary region from indexed data | BL-36 |
| Resume a session with the same region, dates, and earlier results | BL-37 |
| What changed, where, and which data supported it | BL-38 |
| Deployment and dynamic satellite-analysis gaps | BL-44 through BL-68 |
