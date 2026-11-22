//! Core TUI state, live streaming transcript, and v2 server message reducer.

use std::path::PathBuf;
use std::time::Instant;

use ratatui::layout::Rect;
use ratatui::text::Line;
use tui_textarea::{CursorMove, TextArea};

use crate::approval::PermissionPrompt;
use crate::commands::{completions_for, fuzzy_filter, CompletionList};
use crate::history::PromptHistory;
use crate::markdown::render_markdown;
use crate::protocol::{
    clip_chars, format_duration, CatalogEntry, DonePayload, ErrorPayload, ReplayMessage,
    ServerMessage, SessionInfo, StatusSnapshot, StepPayload,
};
use crate::selection::TextSelection;
use crate::theme::Theme;

pub const STILL_STOPPING: &str =
    "Still stopping. Ctrl+C quits; the conversation is saved up to the last finished step.";

/// Cooperative tools poll the cancel flag between steps of their own work.
const COOPERATIVE_TOOLS: [&str; 10] = [
    "get_satellite_data",
    "prepare_stack",
    "time_series",
    "composite",
    "band_math",
    "classify",
    "change_detect",
    "area_table",
    "render_map",
    "export_geotiff",
];

/// Words the first Esc by what is running; never promises more than the tool class delivers.
pub fn cancel_message(activity: &Option<Activity>) -> String {
    match activity {
        Some(Activity::Thinking | Activity::Working | Activity::ToolCalling(_)) => {
            "cancelling...".into()
        }
        Some(Activity::ToolExecuting(name)) => match name.as_str() {
            "bash" | "bash_tool" | "run_tests" => "cancelling the running command...".into(),
            tool if COOPERATIVE_TOOLS.contains(&tool) => {
                "cancel requested; the tool stops at its next check".into()
            }
            other => format!(
                "cancel requested; {other} is still running and the turn stops right after it returns"
            ),
        },
        None => "cancel requested; the turn stops at the next step".into(),
    }
}

pub fn session_row(info: &SessionInfo) -> String {
    let prompt = clip_chars(&info.prompt, 60);
    let mut row = format!(
        "{}  {}  {}  {} msgs  {}",
        info.id, info.saved, info.model, info.messages, prompt
    );
    if info.interrupted {
        row.push_str(" [interrupted]");
    }
    if info.stopped_for_limit {
        row.push_str(" [limit]");
    }
    row
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Status {
    #[default]
    Ready,
    Waiting,
    Stopped,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Activity {
    Thinking,
    Working,
    ToolCalling(String),
    ToolExecuting(String),
}

pub struct ModelPickerState {
    pub loading: bool,
    pub filter: TextArea<'static>,
    pub index: usize,
}

pub struct KeyInputState {
    pub provider: String,
    pub scope: String,
    pub editor: TextArea<'static>,
}

#[derive(Default)]
pub enum Mode {
    #[default]
    Compose,
    ModelPicker(ModelPickerState),
    Help(usize),
    KeyInput(KeyInputState),
}

pub struct ToolHit {
    pub y: u16,
    pub height: u16,
    pub index: usize,
}

#[derive(Debug, Clone, Default)]
pub struct ToolEntry {
    pub data: StepPayload,
    pub running: bool,
    pub live_output: String,
    pub expanded: bool,
}

impl ToolEntry {
    pub fn collapsed_label(&self) -> String {
        let d = &self.data;
        let mut s = d.name.clone();
        let raw = d.summary.trim();
        let cleaned = raw
            .strip_prefix(&format!("{}:", d.name))
            .or_else(|| raw.strip_prefix(&format!("{} ", d.name)))
            .map(str::trim)
            .unwrap_or_else(|| if raw == d.name { "" } else { raw });
        let summary = clip_chars(cleaned, 60);
        if !summary.is_empty() {
            s.push(' ');
            s.push_str(&summary);
        }
        if self.running {
            s.push('…');
        } else {
            if !d.ok {
                s.push_str(" failed");
            }
            if d.duration_ms > 0 {
                s.push_str(&format!(" ({})", format_duration(d.duration_ms)));
            }
        }
        s
    }

    pub fn has_details(&self) -> bool {
        let d = &self.data;
        let ne = |o: &Option<String>| o.as_deref().is_some_and(|s| !s.is_empty());
        !d.text.is_empty()
            || !self.live_output.is_empty()
            || ne(&d.error)
            || ne(&d.diff)
            || ne(&d.omitted)
            || !d.artifacts.is_empty()
            || d.notes.iter().any(|n| !n.trim().is_empty())
            || d.arguments
                .as_ref()
                .is_some_and(|v| !v.is_null() && v.as_object().is_none_or(|m| !m.is_empty()))
    }
}

pub enum TranscriptEntry {
    User(String),
    Thought {
        text: String,
        expanded: bool,
        streaming: bool,
    },
    Assistant {
        text: String,
        streaming: bool,
        cached_width: usize,
        cached_lines: Vec<Line<'static>>,
    },
    Tool(Box<ToolEntry>),
    Notice {
        text: String,
        is_error: bool,
    },
}

impl TranscriptEntry {
    pub fn assistant(text: String, streaming: bool) -> Self {
        Self::Assistant {
            text,
            streaming,
            cached_width: 0,
            cached_lines: Vec::new(),
        }
    }

    pub fn assistant_lines(&mut self, width: usize, theme: &Theme) -> &[Line<'static>] {
        let Self::Assistant {
            text,
            cached_width: cw,
            cached_lines: cl,
            ..
        } = self
        else {
            return &[];
        };
        if *cw != width || cl.is_empty() {
            (*cl, *cw) = (render_markdown(text, width, theme), width);
        }
        cl
    }
}

#[derive(Default)]
pub struct App {
    pub workspace: PathBuf,
    pub session_id: String,
    pub tools: Vec<String>,
    pub key_set: bool,
    pub status_snapshot: StatusSnapshot,
    pub transcript: Vec<TranscriptEntry>,
    pub composer: TextArea<'static>,
    pub prompt_history: PromptHistory,
    pub mode: Mode,
    pub status: Status,
    pub waiting_from_stopped: bool,
    pub session_open: bool,
    pub last_stderr: Option<String>,
    pub activity: Option<Activity>,
    pub activity_since: Option<Instant>,
    pub cancel_stage: u8,
    pub pending_permission: Option<PermissionPrompt>,
    pub cached_models: Vec<CatalogEntry>,
    pub known_skills: Vec<String>,
    pub session_names: Vec<String>,
    pub preview_sha: Option<(String, String)>,
    pub menu_dismissed: bool,
    pub menu_index: usize,
    pub menu_interacted: bool,
    pub scroll: usize,
    pub scroll_max: usize,
    pub viewport_height: usize,
    pub follow: bool,
    pub unseen_output: bool,
    pub selected: Option<usize>,
    pub reveal_selected: bool,
    pub tool_hits: Vec<ToolHit>,
    pub screen_area: Rect,
    pub text_selection: Option<TextSelection>,
    pub copy_selection: bool,
    pub status_notice: Option<String>,
    pub ticks: u32,
    pub dirty: bool,
    pub lookup_context: bool,
    pub compacting: bool,
}

impl App {
    pub fn new(workspace: PathBuf, prompt_history: PromptHistory) -> Self {
        let mut composer = TextArea::default();
        composer.set_tab_length(2);
        Self {
            workspace,
            key_set: true,
            composer,
            prompt_history,
            session_open: true,
            viewport_height: 20,
            follow: true,
            dirty: true,
            ..Default::default()
        }
    }

    pub fn in_flight(&self) -> bool {
        matches!(self.status, Status::Waiting)
    }

    /// Sets the activity; the elapsed clock restarts only when it changes.
    pub fn set_activity(&mut self, next: Option<Activity>) {
        if next != self.activity {
            self.activity_since = next.as_ref().map(|_| Instant::now());
        }
        self.activity = next;
    }

    pub fn composer_text(&self) -> String {
        self.composer.lines().join("\n")
    }

    pub fn set_composer_text(&mut self, text: &str) {
        self.composer = TextArea::default();
        self.composer.set_tab_length(2);
        if !text.is_empty() {
            self.composer.insert_str(text);
        }
        (self.menu_index, self.menu_interacted, self.dirty) = (0, false, true);
    }

    pub fn clear_composer(&mut self) {
        self.set_composer_text("");
        self.menu_dismissed = false;
    }

    pub fn set_composer_cursor_byte(&mut self, byte_offset: usize) {
        let text = self.composer_text();
        let pre = &text[..byte_offset.min(text.len())];
        let row = pre.bytes().filter(|&b| b == b'\n').count();
        let col = pre[pre.rfind('\n').map_or(0, |i| i + 1)..].chars().count();
        self.composer
            .move_cursor(CursorMove::Jump(row as u16, col as u16));
    }

    pub fn active_completions(&self) -> Option<CompletionList> {
        if !matches!(self.mode, Mode::Compose)
            || self.menu_dismissed
            || self.pending_permission.is_some()
        {
            return None;
        }
        let (row, col) = self.composer.cursor();
        let lines = self.composer.lines();
        let prev: usize = lines.iter().take(row).map(|l| l.len() + 1).sum();
        let cur: usize = lines
            .get(row)
            .map_or(0, |l| l.chars().take(col).map(char::len_utf8).sum());
        completions_for(
            &self.composer_text(),
            prev + cur,
            &self.workspace,
            &self.known_skills,
            &self.session_names,
        )
    }

    pub fn filtered_models(&self, query: &str) -> Vec<&CatalogEntry> {
        fuzzy_filter(&self.cached_models, query, |m| m.id.as_str())
    }

    pub fn push_notice(&mut self, text: impl Into<String>, is_error: bool) {
        self.transcript.push(TranscriptEntry::Notice {
            text: text.into(),
            is_error,
        });
        self.unseen_output |= !self.follow;
        self.dirty = true;
    }

    fn note_or(&mut self, note: Option<String>, fallback: String, is_error: bool) {
        self.push_notice(note.filter(|s| !s.is_empty()).unwrap_or(fallback), is_error);
    }

    fn opt_note(&mut self, note: Option<String>) {
        if let Some(n) = note.filter(|s| !s.is_empty()) {
            self.push_notice(n, false);
        }
    }

    pub fn last_reply(&self) -> Option<&str> {
        self.transcript.iter().rev().find_map(|e| match e {
            TranscriptEntry::Assistant { text, .. } if !text.is_empty() => Some(text.as_str()),
            _ => None,
        })
    }

    pub fn expandable_indices(&self) -> Vec<usize> {
        self.transcript
            .iter()
            .enumerate()
            .filter_map(|(i, e)| match e {
                TranscriptEntry::Tool(t) if t.has_details() => Some(i),
                TranscriptEntry::Thought { text, .. } if !text.is_empty() => Some(i),
                _ => None,
            })
            .collect()
    }

    pub fn set_expanded(&mut self, index: usize, open: Option<bool>) {
        let slot = match self.transcript.get_mut(index) {
            Some(TranscriptEntry::Tool(t)) if t.has_details() => &mut t.expanded,
            Some(TranscriptEntry::Thought { expanded, text, .. }) if !text.is_empty() => expanded,
            _ => return,
        };
        let target = open.unwrap_or(!*slot);
        if *slot != target {
            *slot = target;
            (self.selected, self.reveal_selected, self.dirty) = (Some(index), true, true);
        }
    }

    pub fn toggle_selected(&mut self, open: Option<bool>) {
        if let Some(i) = self.selected {
            self.set_expanded(i, open);
        }
    }

    pub fn select_expandable(&mut self, delta: isize) {
        let items = self.expandable_indices();
        if items.is_empty() {
            return;
        }
        let pos = self
            .selected
            .and_then(|c| items.iter().position(|&i| i == c));
        let next = match pos {
            Some(p) if delta < 0 => items[p.saturating_sub(1)],
            Some(p) => items[(p + 1).min(items.len() - 1)],
            None if delta < 0 => *items.last().unwrap(),
            None => items[0],
        };
        (self.selected, self.reveal_selected, self.dirty) = (Some(next), true, true);
    }

    pub fn scroll_by(&mut self, delta: isize) {
        if delta < 0 {
            (self.follow, self.scroll) = (false, self.scroll.saturating_sub(delta.unsigned_abs()));
        } else {
            let next = self.scroll.saturating_add(delta as usize);
            (self.follow, self.scroll) = (next >= self.scroll_max, next.min(self.scroll_max));
            self.unseen_output &= !self.follow;
        }
        self.dirty = true;
    }

    pub fn jump_scroll(&mut self, bottom: bool) {
        (self.follow, self.scroll) = (bottom, [0, self.scroll_max][usize::from(bottom)]);
        (self.unseen_output, self.dirty) = (self.unseen_output && !bottom, true);
    }

    pub fn reset_conversation(&mut self) {
        self.transcript.clear();
        (self.status, self.waiting_from_stopped, self.cancel_stage) = (Status::Ready, false, 0);
        self.set_activity(None);
        (self.pending_permission, self.selected) = (None, None);
        (self.reveal_selected, self.scroll, self.scroll_max) = (false, 0, 0);
        (self.follow, self.unseen_output, self.dirty) = (true, false, true);
    }

    pub fn note_session_closed(&mut self) {
        if !std::mem::replace(&mut self.session_open, false) {
            return;
        }
        self.set_activity(None);
        (
            self.pending_permission,
            self.status,
            self.waiting_from_stopped,
            self.cancel_stage,
        ) = (None, Status::Ready, false, 0);
        let d = self.last_stderr.as_deref().unwrap_or("session ended");
        self.push_notice(format!("error: {d}"), true);
    }

    fn finalize_streams(&mut self) {
        for e in &mut self.transcript {
            if let TranscriptEntry::Assistant { streaming, .. }
            | TranscriptEntry::Thought { streaming, .. } = e
            {
                *streaming = false;
            }
        }
    }

    fn find_tool(&mut self, id: &str, running_only: bool) -> Option<&mut ToolEntry> {
        (!id.is_empty()).then(|| {
            self.transcript.iter_mut().rev().find_map(|e| match e {
                TranscriptEntry::Tool(t)
                    if t.data.call_id == id && (!running_only || t.running) =>
                {
                    Some(t.as_mut())
                }
                _ => None,
            })
        })?
    }

    fn adopt_meta(
        &mut self,
        mut st: StatusSnapshot,
        model: String,
        goal: String,
        skills: Vec<String>,
    ) {
        if st.model.is_empty() {
            st.model = model;
        }
        if st.goal.is_none() && !goal.is_empty() {
            st.goal = Some(goal);
        }
        if st.active_skills.is_empty() {
            st.active_skills = skills;
        }
        self.status_snapshot = st;
    }

    pub fn apply(&mut self, msg: ServerMessage) {
        use ServerMessage as S;
        use TranscriptEntry as E;
        self.dirty = true;
        match msg {
            S::Ready(r) => {
                let mut st = r.status;
                self.session_id = r.session_id;
                if !r.workspace.is_empty() {
                    self.workspace = PathBuf::from(&r.workspace);
                    st.workspace.get_or_insert(r.workspace);
                }
                st.yolo_enabled |= r.yolo_enabled;
                (self.tools, self.key_set, self.lookup_context) =
                    (r.tools, r.key_set, r.key_set && st.context_limit.is_none());
                self.adopt_meta(st, r.model, r.goal, r.active_skills);
            }
            S::Phase { phase, detail } => self.apply_phase(&phase, detail.as_deref().unwrap_or("")),
            S::ThoughtDelta { text } if !text.is_empty() => {
                match self.transcript.last_mut() {
                    Some(E::Thought {
                        text: b,
                        streaming: true,
                        ..
                    }) => b.push_str(&text),
                    _ => {
                        self.transcript.push(E::Thought {
                            text,
                            expanded: false,
                            streaming: true,
                        });
                        self.selected = Some(self.transcript.len() - 1);
                    }
                }
                self.unseen_output |= !self.follow;
            }
            S::ContentDelta { text } if !text.is_empty() => {
                if let Some(E::Thought { streaming, .. }) = self.transcript.last_mut() {
                    *streaming = false;
                }
                match self.transcript.last_mut() {
                    Some(E::Assistant {
                        text: b,
                        streaming: true,
                        cached_width,
                        ..
                    }) => {
                        b.push_str(&text);
                        *cached_width = 0;
                    }
                    _ => self.transcript.push(E::assistant(text, true)),
                }
                self.unseen_output |= !self.follow;
            }
            S::ToolStart {
                call_id,
                name,
                summary,
            } => {
                self.finalize_streams();
                let data = StepPayload {
                    call_id,
                    name,
                    summary,
                    ..Default::default()
                };
                self.transcript.push(E::Tool(Box::new(ToolEntry {
                    data,
                    running: true,
                    ..Default::default()
                })));
                self.selected = Some(self.transcript.len() - 1);
                self.unseen_output |= !self.follow;
            }
            S::ToolOutputDelta { call_id, chunk } if !chunk.is_empty() => {
                if let Some(t) = self.find_tool(&call_id, false) {
                    t.live_output.push_str(&chunk);
                    self.unseen_output |= !self.follow;
                }
            }
            S::Step(s) => self.apply_step(*s),
            S::PermissionRequest(p) => {
                self.pending_permission = Some(PermissionPrompt::new(*p));
                self.status = Status::Waiting;
                self.set_activity(None);
            }
            S::Status { status } => {
                if std::mem::take(&mut self.compacting) {
                    self.push_notice("context compacted", false);
                }
                self.status_snapshot = status;
            }
            S::Done(d) => self.apply_done(*d),
            S::Compacted { compaction, status } => {
                (self.compacting, self.status_notice) = (false, None);
                if !status.model.is_empty()
                    || status.context_used > 0
                    || status.context_limit.is_some()
                {
                    let prev_model = self.status_snapshot.model.clone();
                    self.status_snapshot = status;
                    if self.status_snapshot.model.is_empty() {
                        self.status_snapshot.model = prev_model;
                    }
                }
                let note = compaction
                    .notice()
                    .unwrap_or_else(|| "nothing to compact".into());
                self.push_notice(note, false);
            }
            S::SessionLoaded(s) => {
                self.reset_conversation();
                self.session_id = s.session_id.clone();
                let saved_model = s.model.clone();
                let count = s.messages.len();
                self.adopt_meta(s.status, s.model, s.goal, s.active_skills);
                if s.omitted > 0 {
                    self.push_notice(format!("{} earlier items not shown", s.omitted), false);
                }
                self.transcript
                    .extend(s.messages.into_iter().map(replay_to_entry));
                let model_tag = if saved_model.is_empty() {
                    String::new()
                } else {
                    format!(" ({saved_model})")
                };
                self.push_notice(
                    format!("resumed {}: {count} items{model_tag}", s.session_id),
                    false,
                );
                self.status = if s.stopped_for_limit || s.interrupted {
                    Status::Stopped
                } else {
                    Status::Ready
                };
            }
            S::Undo { ok, path, message } => {
                let fb = match (ok, path) {
                    (true, Some(p)) => format!("restored {p}"),
                    (false, Some(p)) => format!("undo failed for {p}"),
                    (true, None) => "checkpoint restored".into(),
                    (false, None) => "nothing to undo".into(),
                };
                self.note_or((!message.is_empty()).then_some(message), fb, !ok);
                (self.status, self.waiting_from_stopped) = (Status::Ready, false);
                self.set_activity(None);
            }
            S::Models {
                mut models,
                selected,
            } => {
                if !selected.is_empty() && self.status_snapshot.model.is_empty() {
                    self.status_snapshot.model = selected;
                }
                let cur = &self.status_snapshot.model;
                if self.status_snapshot.context_limit.is_none() {
                    self.status_snapshot.context_limit = models
                        .iter()
                        .find(|m| &m.id == cur)
                        .and_then(|m| m.context_length);
                }
                if !cur.is_empty() && !models.iter().any(|m| &m.id == cur) {
                    models.push(CatalogEntry {
                        id: cur.clone(),
                        supports_tools: true,
                        ..Default::default()
                    });
                }
                self.cached_models = models;
                if let Mode::ModelPicker(st) = &mut self.mode {
                    (st.loading, st.index) = (false, 0);
                }
            }
            S::ModelSelected {
                model,
                context_length,
                mut status,
            } => {
                if status.model.is_empty() {
                    status.model = model.clone();
                }
                status.context_limit = status.context_limit.or(context_length);
                (self.status_snapshot, self.mode) = (status, Mode::Compose);
                self.push_notice(format!("model {model}"), false);
            }
            S::List {
                kind,
                items,
                note,
                sessions,
            } => {
                if kind == "sessions" && !sessions.is_empty() {
                    self.session_names = sessions.iter().map(|s| s.id.clone()).collect();
                    for info in &sessions {
                        self.push_notice(session_row(info), false);
                    }
                    self.opt_note(note);
                    return;
                }
                if kind == "sessions" {
                    self.session_names = items.clone();
                } else if kind == "skills" {
                    self.known_skills = items.clone();
                }
                let joined = items.join(", ");
                let body = [joined.as_str(), "(none)"][usize::from(joined.is_empty())];
                self.push_notice(format!("{kind}: {body}"), false);
                self.opt_note(note);
            }
            S::Goal {
                text,
                cleared,
                status,
            } => {
                self.status_snapshot = status;
                let msg = match (cleared, text.is_empty()) {
                    (true, _) => "goal cleared".into(),
                    (_, true) => "no goal set".into(),
                    _ => format!("goal: {text}"),
                };
                self.push_notice(msg, false);
            }
            S::Skill {
                name,
                active,
                note,
                status,
            } => {
                self.status_snapshot = status;
                if !name.is_empty() && !self.known_skills.contains(&name) {
                    self.known_skills.push(name.clone());
                }
                let fb = [
                    format!("skill {name} inactive"),
                    format!("using skill {name}"),
                ][usize::from(active)]
                .clone();
                self.note_or(note, fb, false);
            }
            S::SaveSkill { name, path, note } => {
                self.push_notice(format!("saved skill {name} ({path})"), false);
                self.opt_note(note);
            }
            S::EnablePreview {
                name,
                sha256,
                files,
                note,
            } => {
                self.preview_sha = Some((name.clone(), sha256.clone()));
                let (s, n) = (&sha256[..sha256.len().min(8)], files.len());
                self.push_notice(
                    format!(
                        "draft skill {name} ({s}, {n} files) — `/enable-skill {name} confirm {s}`"
                    ),
                    false,
                );
                self.opt_note(note);
            }
            S::EnabledSkill {
                name,
                enabled,
                note,
            } => {
                let fb = [
                    format!("skill {name} not enabled"),
                    format!("enabled skill {name}"),
                ][usize::from(enabled)]
                .clone();
                self.note_or(note, fb, !enabled);
            }
            S::KeySet {
                provider,
                key_set,
                note,
            } => {
                if provider.eq_ignore_ascii_case("openrouter") || provider.is_empty() {
                    self.key_set = key_set;
                    self.lookup_context = key_set && self.status_snapshot.context_limit.is_none();
                }
                self.mode = Mode::Compose;
                let p = [provider.as_str(), "API"][usize::from(provider.is_empty())];
                let verb = ["removed", "configured"][usize::from(key_set)];
                self.note_or(note, format!("{p} key {verb}"), false);
            }
            S::Yolo {
                enabled,
                mut status,
            } => {
                status.yolo_enabled = enabled;
                self.status_snapshot = status;
                if enabled {
                    self.pending_permission = None;
                }
                let msg = [
                    "YOLO off: tool calls require approval",
                    "YOLO on: tool calls run without approval",
                ][usize::from(enabled)];
                self.push_notice(msg, false);
            }
            S::Error(e) => self.apply_error(*e),
            _ => {}
        }
    }

    fn apply_phase(&mut self, phase: &str, d: &str) {
        let next = match phase {
            "thinking" => Some(Activity::Thinking),
            "working" => Some(Activity::Working),
            "tool_calling" => Some(Activity::ToolCalling(d.into())),
            "tool_executing" => Some(Activity::ToolExecuting(d.into())),
            "idle" => {
                if std::mem::take(&mut self.compacting) && !d.is_empty() {
                    self.push_notice(d, false);
                }
                None
            }
            _ => None,
        };
        self.set_activity(next);
    }

    fn apply_step(&mut self, mut s: StepPayload) {
        if self
            .pending_permission
            .as_ref()
            .is_some_and(|p| p.payload.id == s.call_id)
        {
            self.pending_permission = None;
        }
        self.finalize_streams();
        if let Some(t) = self.find_tool(&s.call_id, true) {
            if s.summary.is_empty() {
                s.summary = std::mem::take(&mut t.data.summary);
            }
            (t.data, t.running) = (s, false);
        } else {
            self.transcript
                .push(TranscriptEntry::Tool(Box::new(ToolEntry {
                    data: s,
                    ..Default::default()
                })));
            self.selected = Some(self.transcript.len() - 1);
        }
        self.unseen_output |= !self.follow;
    }

    fn apply_done(&mut self, d: DonePayload) {
        use TranscriptEntry as E;
        self.set_activity(None);
        (
            self.status_notice,
            self.pending_permission,
            self.cancel_stage,
        ) = (None, None, 0);
        (self.waiting_from_stopped, self.status_snapshot) = (false, d.status);
        let has_thought = self
            .transcript
            .iter()
            .rev()
            .take_while(|e| !matches!(e, E::User(_)))
            .any(|e| matches!(e, E::Thought { .. }));
        let trailing_assistant = matches!(
            self.transcript.last(),
            Some(E::Assistant {
                streaming: true,
                ..
            })
        );
        if let Some(t) = d.thoughts.filter(|s| !has_thought && !s.trim().is_empty()) {
            let entry = E::Thought {
                text: t,
                expanded: false,
                streaming: false,
            };
            if trailing_assistant {
                self.transcript.insert(self.transcript.len() - 1, entry);
            } else {
                self.transcript.push(entry);
            }
        }
        if !d.response.is_empty() {
            match self.transcript.last_mut() {
                Some(E::Assistant {
                    text,
                    streaming: true,
                    cached_width,
                    ..
                }) => {
                    if text.is_empty() || d.response.len() > text.len() {
                        (*text, *cached_width) = (d.response, 0);
                    }
                }
                _ => self.transcript.push(E::assistant(d.response, false)),
            }
        }
        self.finalize_streams();
        for (hit, msg, err) in [
            (d.cancelled, "turn cancelled; conversation kept", false),
            (
                d.stopped_for_budget,
                "stopped at token or spend budget in .atlas/agent.toml",
                false,
            ),
            (
                d.stopped_for_context,
                "stopped: the conversation no longer fits the model context; run /compact or start /new",
                true,
            ),
        ] {
            if hit {
                self.push_notice(msg, err);
            }
        }
        if let Some(notice) = d.compaction.notice() {
            self.push_notice(notice, false);
        }
        self.status = [Status::Ready, Status::Stopped][usize::from(d.stopped_for_limit)];
    }

    fn apply_error(&mut self, e: ErrorPayload) {
        self.set_activity(None);
        (
            self.status_notice,
            self.pending_permission,
            self.cancel_stage,
            self.compacting,
        ) = (None, None, 0, false);
        if matches!(self.mode, Mode::ModelPicker(ref s) if s.loading) {
            self.mode = Mode::Compose;
        }
        self.finalize_streams();
        if let Some(s) = e.status {
            self.status_snapshot = s;
        }
        let mut detail = format!("error: {}", e.message);
        if let Some(c) = e.http_status {
            detail.push_str(&format!(" (HTTP {c})"));
        }
        if let Some(s) = e.elapsed_seconds {
            detail.push_str(&format!(" [{s:.1}s]"));
        }
        if let Some(b) = e.error_body.filter(|b| !b.trim().is_empty()) {
            detail.push('\n');
            detail.extend(b.chars().take(240));
        }
        self.push_notice(detail, true);
        self.status = [Status::Ready, Status::Stopped]
            [usize::from(std::mem::take(&mut self.waiting_from_stopped))];
        self.follow = true;
    }
}

fn replay_to_entry(msg: ReplayMessage) -> TranscriptEntry {
    match msg.role.as_str() {
        "user" => TranscriptEntry::User(msg.text),
        "thought" => TranscriptEntry::Thought {
            text: msg.text,
            expanded: false,
            streaming: false,
        },
        "tool" => {
            let data = StepPayload {
                name: msg.name.unwrap_or_else(|| "tool".into()),
                summary: msg.summary.unwrap_or_default(),
                ok: msg.ok.unwrap_or(true),
                text: msg.text,
                diff: msg.diff,
                ..Default::default()
            };
            TranscriptEntry::Tool(Box::new(ToolEntry {
                data,
                ..Default::default()
            }))
        }
        _ => TranscriptEntry::assistant(msg.text, false),
    }
}
