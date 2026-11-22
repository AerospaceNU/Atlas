//! Frozen v2 JSON-lines protocol between `atlas-tui` and Python `atlas.agent`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(default)]
pub struct StatusSnapshot {
    pub model: String,
    pub context_used: i64,
    pub context_limit: Option<i64>,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
    pub cached_tokens: i64,
    pub reasoning_tokens: i64,
    pub spend: Option<f64>,
    pub tokens_per_second: Option<f64>,
    pub yolo_enabled: bool,
    pub estimated: bool,
    pub branch: Option<String>,
    pub workspace: Option<String>,
    pub goal: Option<String>,
    pub active_skills: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct CatalogEntry {
    pub id: String,
    pub name: Option<String>,
    pub context_length: Option<i64>,
    pub prompt_price: Option<String>,
    pub completion_price: Option<String>,
    #[serde(default = "default_true")]
    pub supports_tools: bool,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(default)]
pub struct Compaction {
    pub occurred: bool,
    pub tokens_reclaimed: i64,
    pub tool_results_shortened: i64,
    pub messages_dropped: i64,
    pub digest_inserted: bool,
    pub digest_chars: i64,
}

impl Compaction {
    pub fn notice(&self) -> Option<String> {
        if !self.occurred {
            return None;
        }
        let mut p = vec![format!(
            "context compacted: {} tokens reclaimed",
            self.tokens_reclaimed
        )];
        if self.messages_dropped > 0 {
            p.push(format!("{} messages dropped", self.messages_dropped));
        }
        if self.tool_results_shortened > 0 {
            p.push(format!(
                "{} tool results shortened",
                self.tool_results_shortened
            ));
        }
        if self.digest_inserted {
            p.push(format!("digest kept ({} chars)", self.digest_chars));
        }
        Some(p.join(", "))
    }
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(default)]
pub struct SessionInfo {
    pub id: String,
    pub saved: String,
    pub model: String,
    pub messages: i64,
    pub prompt: String,
    pub interrupted: bool,
    pub stopped_for_limit: bool,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(default)]
pub struct ReplayMessage {
    pub role: String,
    pub text: String,
    pub name: Option<String>,
    pub summary: Option<String>,
    pub ok: Option<bool>,
    pub diff: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(default)]
pub struct ReadyPayload {
    pub session_id: String,
    pub model: String,
    pub workspace: String,
    pub tools: Vec<String>,
    pub key_set: bool,
    pub status: StatusSnapshot,
    pub goal: String,
    pub active_skills: Vec<String>,
    pub yolo_enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(default)]
pub struct StepPayload {
    pub call_id: String,
    pub name: String,
    pub summary: String,
    pub ok: bool,
    pub duration_ms: u64,
    pub text: String,
    pub error: Option<String>,
    pub error_kind: Option<String>,
    pub diff: Option<String>,
    pub lines: i64,
    pub omitted: Option<String>,
    pub arguments: Option<Value>,
    pub artifacts: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(default)]
pub struct PermissionPayload {
    pub id: String,
    pub tool: String,
    pub summary: String,
    pub preview: String,
    pub preview_kind: String,
    pub arguments: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(default)]
pub struct DonePayload {
    pub response: String,
    pub thoughts: Option<String>,
    pub cancelled: bool,
    pub stopped_for_limit: bool,
    pub stopped_for_budget: bool,
    pub stopped_for_context: bool,
    pub compaction: Compaction,
    pub empty_replies: u64,
    pub intent_nudges: u64,
    pub status: StatusSnapshot,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(default)]
pub struct SessionLoadedPayload {
    pub session_id: String,
    pub model: String,
    pub goal: String,
    pub active_skills: Vec<String>,
    pub status: StatusSnapshot,
    pub messages: Vec<ReplayMessage>,
    pub stopped_for_limit: bool,
    pub interrupted: bool,
    pub omitted: i64,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(default)]
pub struct ErrorPayload {
    pub message: String,
    pub http_status: Option<u16>,
    pub elapsed_seconds: Option<f64>,
    pub error_body: Option<String>,
    pub status: Option<StatusSnapshot>,
}

/// Server -> Client v2 JSON-lines message.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    Ready(Box<ReadyPayload>),
    Phase {
        phase: String,
        detail: Option<String>,
    },
    ContentDelta {
        text: String,
    },
    ThoughtDelta {
        text: String,
    },
    ToolStart {
        call_id: String,
        name: String,
        #[serde(default)]
        summary: String,
    },
    ToolOutputDelta {
        call_id: String,
        chunk: String,
    },
    Step(Box<StepPayload>),
    PermissionRequest(Box<PermissionPayload>),
    Status {
        status: StatusSnapshot,
    },
    Done(Box<DonePayload>),
    Compacted {
        #[serde(default)]
        compaction: Compaction,
        #[serde(default)]
        status: StatusSnapshot,
    },
    SessionLoaded(Box<SessionLoadedPayload>),
    Undo {
        ok: bool,
        path: Option<String>,
        #[serde(default)]
        message: String,
    },
    Models {
        #[serde(default)]
        models: Vec<CatalogEntry>,
        #[serde(default)]
        selected: String,
    },
    ModelSelected {
        model: String,
        context_length: Option<i64>,
        #[serde(default)]
        status: StatusSnapshot,
    },
    List {
        kind: String,
        #[serde(default)]
        items: Vec<String>,
        note: Option<String>,
        #[serde(default)]
        sessions: Vec<SessionInfo>,
    },
    Goal {
        #[serde(default)]
        text: String,
        #[serde(default)]
        cleared: bool,
        #[serde(default)]
        status: StatusSnapshot,
    },
    Skill {
        name: String,
        #[serde(default)]
        active: bool,
        note: Option<String>,
        #[serde(default)]
        status: StatusSnapshot,
    },
    SaveSkill {
        name: String,
        #[serde(default)]
        path: String,
        note: Option<String>,
    },
    EnablePreview {
        name: String,
        #[serde(default)]
        sha256: String,
        #[serde(default)]
        files: Vec<String>,
        note: Option<String>,
    },
    EnabledSkill {
        name: String,
        #[serde(default)]
        enabled: bool,
        note: Option<String>,
    },
    KeySet {
        #[serde(default)]
        provider: String,
        #[serde(default)]
        key_set: bool,
        note: Option<String>,
    },
    Yolo {
        enabled: bool,
        #[serde(default)]
        status: StatusSnapshot,
    },
    Error(Box<ErrorPayload>),
    #[serde(skip)]
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionDecision {
    Allow,
    AllowSession,
    Deny,
}

/// Client -> Server v2 JSON-lines message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    User {
        text: String,
    },
    Resume,
    Cancel,
    Permission {
        id: String,
        decision: PermissionDecision,
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    Yolo {
        #[serde(skip_serializing_if = "Option::is_none")]
        enabled: Option<bool>,
    },
    LoadSession {
        id: String,
    },
    Undo,
    New,
    Compact,
    Models {
        #[serde(skip_serializing_if = "Option::is_none")]
        refresh: Option<bool>,
    },
    SelectModel {
        model: String,
    },
    Goal {
        #[serde(skip_serializing_if = "Option::is_none")]
        text: Option<String>,
    },
    Skill {
        name: String,
    },
    SaveSkill {
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    EnableSkill {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        sha256: Option<String>,
    },
    List {
        kind: String,
    },
    SetKey {
        key: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        scope: Option<String>,
    },
    DeleteKey {
        #[serde(skip_serializing_if = "Option::is_none")]
        provider: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        scope: Option<String>,
    },
    Quit,
}

pub fn parse_line(line: &str) -> Result<Option<ServerMessage>, serde_json::Error> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    match serde_json::from_str::<ServerMessage>(trimmed) {
        Ok(msg) => Ok(Some(msg)),
        Err(err) => match serde_json::from_str::<Value>(trimmed) {
            Ok(Value::Object(_)) => Ok(Some(ServerMessage::Unknown)),
            _ => Err(err),
        },
    }
}

pub fn format_tokens(tokens: i64) -> String {
    if tokens <= 0 {
        return "0".into();
    }
    let v = tokens as f64;
    for (unit, suffix) in [(1e9, "B"), (1e6, "M"), (1e3, "K")] {
        if v >= unit {
            let s = v / unit;
            let p = if s >= 100.0 {
                0
            } else {
                usize::from(s < 10.0) + 1
            };
            return format!("{s:.p$}{suffix}");
        }
    }
    tokens.to_string()
}

pub fn price_per_million(token_price: &str) -> Option<String> {
    let val: f64 = token_price.trim().parse().ok()?;
    (val.is_finite() && val >= 0.0).then(|| {
        format!("{:.6}", val * 1e6)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .into()
    })
}

pub fn format_duration(ms: u64) -> String {
    match ms {
        0..=999 => format!("{ms}ms"),
        1_000..=59_999 => format!("{:.1}s", ms as f64 / 1e3),
        _ => format!("{}m {}s", ms / 60_000, (ms / 1_000) % 60),
    }
}

/// `7s`, `1m12s` or `1h02m`.
pub fn format_elapsed(secs: u64) -> String {
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m{:02}s", secs / 60, secs % 60)
    } else {
        format!("{}h{:02}m", secs / 3600, (secs % 3600) / 60)
    }
}

/// First `max` characters with newlines flattened, `...` when cut.
pub fn clip_chars(text: &str, max: usize) -> String {
    let flat: String = text
        .trim()
        .chars()
        .map(|ch| if matches!(ch, '\n' | '\r') { ' ' } else { ch })
        .collect();
    if flat.chars().count() <= max {
        return flat;
    }
    let cut: String = flat.chars().take(max).collect();
    format!("{cut}...")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_v2_messages_and_serializes_client_messages() {
        assert_eq!(
            parse_line(r#"{"type":"future_event"}"#).unwrap(),
            Some(ServerMessage::Unknown)
        );
        assert!(parse_line("   ").unwrap().is_none() && parse_line("{bad").is_err());
        let perm = serde_json::to_value(ClientMessage::Permission {
            id: "c1".into(),
            decision: PermissionDecision::AllowSession,
            reason: Some("ok".into()),
        })
        .unwrap();
        assert_eq!(
            perm,
            serde_json::json!({"type":"permission","id":"c1","decision":"allow_session","reason":"ok"})
        );
        assert_eq!(
            (
                format_tokens(1_048_576),
                price_per_million("0.0000000198"),
                format_duration(12),
                format_elapsed(7),
                format_elapsed(72),
                format_elapsed(3720),
                clip_chars("hello\nworld", 20),
                clip_chars("abcdef", 4),
            ),
            (
                "1.05M".into(),
                Some("0.0198".into()),
                "12ms".into(),
                "7s".into(),
                "1m12s".into(),
                "1h02m".into(),
                "hello world".into(),
                "abcd...".into(),
            )
        );
    }

    #[test]
    fn parses_017_compacted_done_step_sessions_and_session_loaded() {
        let Some(ServerMessage::Compacted { compaction, status }) = parse_line(
            r#"{"type":"compacted","compaction":{"occurred":true,"tokens_reclaimed":50,"tool_results_shortened":1,"messages_dropped":2,"digest_inserted":true,"digest_chars":10},"status":{"context_used":7,"context_limit":1000}}"#,
        )
        .unwrap() else {
            panic!("expected compacted");
        };
        let note = compaction.notice().unwrap();
        assert!(
            note.contains("50 tokens reclaimed")
                && note.contains("2 messages dropped")
                && note.contains("1 tool results shortened")
                && note.contains("digest kept (10 chars)")
                && (status.context_used, status.context_limit) == (7, Some(1000))
        );
        assert!(matches!(
            parse_line(r#"{"type":"compacted"}"#).unwrap(),
            Some(ServerMessage::Compacted { .. })
        ));
        assert!(matches!(
            parse_line(r#"{"type":"done","response":"","stopped_for_context":true,"empty_replies":2,"intent_nudges":1,"compaction":{"occurred":true,"tokens_reclaimed":123}}"#).unwrap(),
            Some(ServerMessage::Done(d)) if d.stopped_for_context && d.compaction.occurred && d.empty_replies == 2 && d.intent_nudges == 1
        ));
        assert!(matches!(
            parse_line(r#"{"type":"step","name":"bash","ok":true,"notes":["repeated result"],"artifacts":["out.png"]}"#).unwrap(),
            Some(ServerMessage::Step(s)) if s.notes == ["repeated result"] && s.artifacts == ["out.png"]
        ));
        assert!(matches!(
            parse_line(r#"{"type":"list","kind":"sessions","items":["s1"],"sessions":[{"id":"s1","saved":"2026-11-22","model":"m","messages":4,"prompt":"hello","interrupted":true,"stopped_for_limit":true}]}"#).unwrap(),
            Some(ServerMessage::List { sessions, .. }) if sessions.len() == 1 && sessions[0].interrupted && sessions[0].stopped_for_limit
        ));
        assert!(matches!(
            parse_line(r#"{"type":"session_loaded","session_id":"s1","model":"m","interrupted":true,"omitted":5,"messages":[]}"#).unwrap(),
            Some(ServerMessage::SessionLoaded(s)) if s.interrupted && !s.stopped_for_limit && s.omitted == 5
        ));
    }
}
