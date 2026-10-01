//! Parsing for one line of the Atlas agent session protocol.

use serde::Deserialize;
use serde_json::Value;

/// A single server -> client message.
#[derive(Debug, Clone, PartialEq)]
pub enum ServerMessage {
    Ready {
        tools: Vec<ToolInfo>,
        key_set: bool,
        status: StatusSnapshot,
    },
    Step {
        name: String,
        call_id: String,
        ok: bool,
        text: Option<String>,
        error: Option<String>,
        error_kind: Option<String>,
        artifacts: Vec<String>,
    },
    Done {
        response: String,
        stopped_for_limit: bool,
        status: StatusSnapshot,
    },
    Error {
        message: String,
        http_status: Option<u16>,
        elapsed_seconds: Option<f64>,
        error_body: Option<String>,
        status_line: String,
    },
    Models {
        ok: bool,
        message: String,
        models: Vec<CatalogEntry>,
        status: StatusSnapshot,
    },
    ModelSelected {
        id: String,
        context_length: Option<i64>,
        status: StatusSnapshot,
    },
    KeySet {
        scope: String,
        saved: bool,
    },
    Listing {
        scope: String,
        kind: String,
        names: Vec<String>,
    },
    Removed {
        scope: String,
        kind: String,
        name: String,
    },
    Phase {
        phase: String,
        name: Option<String>,
    },
    /// A line we do not understand yet.
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct StatusSnapshot {
    #[serde(default)]
    pub status_line: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub context_used: i64,
    #[serde(default)]
    pub context_limit: Option<i64>,
    #[serde(default)]
    pub prompt_tokens: i64,
    #[serde(default)]
    pub completion_tokens: i64,
    #[serde(default)]
    pub total_tokens: i64,
    #[serde(default)]
    pub spend: Option<f64>,
    #[serde(default)]
    pub tokens_per_second: Option<f64>,
    #[serde(default)]
    pub http_status: Option<u16>,
    #[serde(default)]
    pub elapsed_seconds: Option<f64>,
    #[serde(default)]
    pub time_to_first_token_seconds: Option<f64>,
    #[serde(default)]
    pub recent: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CatalogEntry {
    pub id: String,
    #[serde(default)]
    pub context_length: Option<i64>,
    #[serde(default)]
    pub prompt_price: Option<String>,
    #[serde(default)]
    pub completion_price: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ToolInfo {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub parameters: Value,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum RawMessage {
    #[serde(rename = "ready")]
    Ready {
        tools: Vec<ToolInfo>,
        #[serde(default)]
        key_set: bool,
        #[serde(flatten)]
        status: StatusSnapshot,
    },
    #[serde(rename = "step")]
    Step {
        name: String,
        #[serde(default)]
        call_id: String,
        #[serde(default)]
        ok: bool,
        #[serde(default)]
        text: Option<String>,
        #[serde(default)]
        error: Option<String>,
        #[serde(default)]
        error_kind: Option<String>,
        #[serde(default)]
        artifacts: Vec<String>,
    },
    #[serde(rename = "done")]
    Done {
        #[serde(default)]
        response: String,
        #[serde(default)]
        stopped_for_limit: bool,
        #[serde(flatten)]
        status: StatusSnapshot,
    },
    #[serde(rename = "error")]
    Error {
        message: String,
        #[serde(default)]
        http_status: Option<u16>,
        #[serde(default)]
        elapsed_seconds: Option<f64>,
        #[serde(default)]
        error_body: Option<String>,
        #[serde(default)]
        status_line: String,
    },
    #[serde(rename = "models")]
    Models {
        #[serde(default)]
        ok: bool,
        #[serde(default)]
        message: String,
        #[serde(default)]
        models: Vec<CatalogEntry>,
        #[serde(flatten)]
        status: StatusSnapshot,
    },
    #[serde(rename = "model")]
    Model {
        id: String,
        #[serde(default)]
        context_length: Option<i64>,
        #[serde(flatten)]
        status: StatusSnapshot,
    },
    #[serde(rename = "key")]
    Key {
        #[serde(default)]
        scope: String,
        #[serde(default)]
        saved: bool,
    },
    #[serde(rename = "listing")]
    Listing {
        #[serde(default)]
        scope: String,
        #[serde(default)]
        kind: String,
        #[serde(default)]
        names: Vec<String>,
    },
    #[serde(rename = "phase")]
    Phase {
        phase: String,
        #[serde(default)]
        name: Option<String>,
    },
    #[serde(rename = "removed")]
    Removed {
        #[serde(default)]
        scope: String,
        #[serde(default)]
        kind: String,
        #[serde(default)]
        name: String,
    },
}

/// Parse one protocol line. Blank lines yield `Ok(None)`.
pub fn parse_line(line: &str) -> Result<Option<ServerMessage>, serde_json::Error> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    match serde_json::from_str::<RawMessage>(trimmed) {
        Ok(raw) => Ok(Some(raw.into())),
        // Objects we do not model are tolerated; malformed JSON is not.
        Err(err) => match serde_json::from_str::<Value>(trimmed) {
            Ok(_) => Ok(Some(ServerMessage::Unknown)),
            Err(_) => Err(err),
        },
    }
}

impl From<RawMessage> for ServerMessage {
    fn from(raw: RawMessage) -> Self {
        match raw {
            RawMessage::Ready {
                tools,
                key_set,
                status,
            } => ServerMessage::Ready {
                tools,
                key_set,
                status,
            },
            RawMessage::Step {
                name,
                call_id,
                ok,
                text,
                error,
                error_kind,
                artifacts,
            } => ServerMessage::Step {
                name,
                call_id,
                ok,
                text,
                error,
                error_kind,
                artifacts,
            },
            RawMessage::Done {
                response,
                stopped_for_limit,
                status,
            } => ServerMessage::Done {
                response,
                stopped_for_limit,
                status,
            },
            RawMessage::Error {
                message,
                http_status,
                elapsed_seconds,
                error_body,
                status_line,
            } => ServerMessage::Error {
                message,
                http_status,
                elapsed_seconds,
                error_body,
                status_line,
            },
            RawMessage::Models {
                ok,
                message,
                models,
                status,
            } => ServerMessage::Models {
                ok,
                message,
                models,
                status,
            },
            RawMessage::Model {
                id,
                context_length,
                status,
            } => ServerMessage::ModelSelected {
                id,
                context_length,
                status,
            },
            RawMessage::Key { scope, saved } => ServerMessage::KeySet { scope, saved },
            RawMessage::Listing { scope, kind, names } => {
                ServerMessage::Listing { scope, kind, names }
            }
            RawMessage::Removed { scope, kind, name } => {
                ServerMessage::Removed { scope, kind, name }
            }
            RawMessage::Phase { phase, name } => ServerMessage::Phase { phase, name },
        }
    }
}

/// Status text used before the session sends a line, and as a fallback.
///
/// The words match `format_tui_status` for an empty session so the first
/// frame is never a blank status row.
pub fn default_status_line() -> String {
    "model unset  context 0/unknown  tokens 0  spend n/a  n/a tok/s".to_string()
}

pub fn status_text(status: &StatusSnapshot) -> String {
    if status.status_line.is_empty() {
        default_status_line()
    } else {
        status.status_line.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ready() {
        let line = r#"{"type":"ready","tools":[{"name":"list_images","description":"d","parameters":{}}]}"#;
        match parse_line(line).unwrap() {
            Some(ServerMessage::Ready { tools, .. }) => {
                assert_eq!(tools.len(), 1);
                assert_eq!(tools[0].name, "list_images");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parses_step() {
        let line = r#"{"type":"step","name":"inspect_image","call_id":"c1","arguments":{},"ok":true,"text":"20x10","error":null,"error_kind":null,"artifacts":["a.png"]}"#;
        match parse_line(line).unwrap() {
            Some(ServerMessage::Step {
                name,
                ok,
                text,
                artifacts,
                ..
            }) => {
                assert_eq!(name, "inspect_image");
                assert!(ok);
                assert_eq!(text.as_deref(), Some("20x10"));
                assert_eq!(artifacts, vec!["a.png".to_string()]);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parses_done() {
        let line = r#"{"type":"done","response":"hello","stopped_for_limit":false}"#;
        match parse_line(line).unwrap() {
            Some(ServerMessage::Done {
                response,
                stopped_for_limit,
                ..
            }) => {
                assert_eq!(response, "hello");
                assert!(!stopped_for_limit);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parses_done_status_fields() {
        let line = r#"{"type":"done","response":"hi","stopped_for_limit":false,"status_line":"context 12/128000  tokens 16 (prompt 12 completion 4)  spend $0.02  8.0 tok/s  http 200  latency 0.50s","context_used":12,"context_limit":128000,"prompt_tokens":12,"completion_tokens":4,"total_tokens":16,"spend":0.02,"tokens_per_second":8.0,"http_status":200,"elapsed_seconds":0.5}"#;
        match parse_line(line).unwrap() {
            Some(ServerMessage::Done { status, .. }) => {
                assert_eq!(status.context_used, 12);
                assert_eq!(status.context_limit, Some(128000));
                assert_eq!(status.prompt_tokens, 12);
                assert_eq!(status.completion_tokens, 4);
                assert_eq!(status.http_status, Some(200));
                let text = status_text(&status);
                assert!(text.contains("context 12/128000"));
                assert!(text.contains("prompt 12"));
                assert!(text.contains("completion 4"));
                assert!(text.contains("spend $0.02"));
                assert!(text.contains("tok/s"));
                assert!(text.contains("http 200"));
                assert!(text.contains("latency 0.50s"));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn default_status_is_not_blank() {
        let text = default_status_line();
        assert!(text.contains("context 0/unknown"));
        assert!(text.contains("tokens 0"));
        assert!(!text.contains("prompt"));
        assert!(!text.contains("completion"));
        assert!(text.contains("spend n/a"));
        assert!(text.contains("tok/s"));
        assert!(!text.contains("http"));
        assert!(!text.contains("latency"));
        assert!(!text.contains("ttft"));
        assert!(!text.contains("recent"));
    }

    #[test]
    fn parses_error() {
        let line = r#"{"type":"error","message":"OPENROUTER_API_KEY is not set"}"#;
        match parse_line(line).unwrap() {
            Some(ServerMessage::Error { message, .. }) => {
                assert_eq!(message, "OPENROUTER_API_KEY is not set");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parses_http_error() {
        let line = r#"{"type":"error","message":"OpenRouter HTTP 429: slow down","http_status":429,"elapsed_seconds":0.25,"error_body":"slow down","status_line":"http 429  latency 0.25s"}"#;
        match parse_line(line).unwrap() {
            Some(ServerMessage::Error {
                http_status,
                error_body,
                status_line,
                ..
            }) => {
                assert_eq!(http_status, Some(429));
                assert_eq!(error_body.as_deref(), Some("slow down"));
                assert!(status_line.contains("http 429"));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parses_model_catalog_failure_without_dropping_fields() {
        let line =
            r#"{"type":"models","ok":false,"message":"OpenRouter HTTP 500: down","models":[]}"#;
        match parse_line(line).unwrap() {
            Some(ServerMessage::Models {
                ok,
                message,
                models,
                ..
            }) => {
                assert!(!ok);
                assert!(message.contains("500"));
                assert!(models.is_empty());
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn ignores_blank_lines() {
        assert_eq!(parse_line("   ").unwrap(), None);
        assert_eq!(parse_line("").unwrap(), None);
    }

    #[test]
    fn rejects_invalid_json() {
        assert!(parse_line("not json at all").is_err());
    }
}
