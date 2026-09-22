//! Parsing for one line of the Atlas agent session protocol.

use serde::Deserialize;
use serde_json::Value;

/// A single server -> client message.
#[derive(Debug, Clone, PartialEq)]
pub enum ServerMessage {
    Ready {
        tools: Vec<ToolInfo>,
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
    },
    Error {
        message: String,
    },
    /// A line we do not understand yet.
    Unknown,
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
    Ready { tools: Vec<ToolInfo> },
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
    },
    #[serde(rename = "error")]
    Error { message: String },
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
            RawMessage::Ready { tools } => ServerMessage::Ready { tools },
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
            } => ServerMessage::Done {
                response,
                stopped_for_limit,
            },
            RawMessage::Error { message } => ServerMessage::Error { message },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ready() {
        let line = r#"{"type":"ready","tools":[{"name":"list_images","description":"d","parameters":{}}]}"#;
        match parse_line(line).unwrap() {
            Some(ServerMessage::Ready { tools }) => {
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
            }) => {
                assert_eq!(response, "hello");
                assert!(!stopped_for_limit);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parses_error() {
        let line = r#"{"type":"error","message":"OPENROUTER_API_KEY is not set"}"#;
        match parse_line(line).unwrap() {
            Some(ServerMessage::Error { message }) => {
                assert_eq!(message, "OPENROUTER_API_KEY is not set");
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
