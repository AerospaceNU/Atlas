use super::{dashboard_text, draw, is_finished_command, menu_options, slash_outcome, App};
use crate::protocol::ServerMessage;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

fn screen(app: &mut App, width: u16, height: u16) -> String {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    terminal.draw(|frame| draw(frame, app)).expect("draw");

    let buffer = terminal.backend().buffer();
    (0..height)
        .map(|y| {
            (0..width)
                .map(|x| buffer[(x, y)].symbol())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn yolo_status_is_visible_only_after_the_server_enables_it() {
    let mut app = App::new();
    assert!(!dashboard_text(&app).contains("YOLO on"));
    assert!(!screen(&mut app, 100, 28).contains("YOLO on"));

    app.apply(ServerMessage::Yolo { enabled: true });
    assert!(dashboard_text(&app).contains("YOLO on"));
    assert!(screen(&mut app, 100, 28).contains("YOLO on"));

    app.apply(ServerMessage::Yolo { enabled: false });
    assert!(!dashboard_text(&app).contains("YOLO on"));
    let painted = screen(&mut app, 100, 28);
    let status = painted
        .lines()
        .find(|line| line.trim_start().starts_with("Atlas  "))
        .unwrap();
    assert!(!status.contains("YOLO on"));
    assert!(painted.contains("YOLO off"));
}

#[test]
fn yolo_command_waits_for_server_state_instead_of_toggling_optimistically() {
    let mut app = App::new();
    let command = slash_outcome(&mut app, "/yolo");

    assert_eq!(command, Some(serde_json::json!({"type": "yolo"})));
    assert!(!app.yolo_enabled);
    assert!(!dashboard_text(&app).contains("YOLO on"));
}

#[test]
fn yolo_command_appears_in_menu_and_accepts_only_the_bare_command() {
    assert!(menu_options("/yo").iter().any(|item| item.token == "/yolo"));
    assert!(is_finished_command("/yolo"));
    assert!(slash_outcome(&mut App::new(), "/yolo extra").is_none());
}

#[test]
fn yolo_state_is_parsed_from_the_session_protocol() {
    let message = crate::protocol::parse_line(r#"{"type":"yolo","enabled":true}"#).unwrap();
    assert_eq!(message, Some(ServerMessage::Yolo { enabled: true }));
}

#[cfg(unix)]
#[test]
fn yolo_can_be_typed_at_a_pending_prompt_and_during_a_turn() {
    use super::{handle_key, ChildProcess, Status};
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    let mut process = Command::new("cat")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let stdin = process.stdin.take().unwrap();
    let stdout = process.stdout.take().unwrap();
    let mut child = ChildProcess {
        child: process,
        stdin,
    };
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let mut app = App::new();
    app.input = "keep my draft".to_string();
    app.apply(ServerMessage::PermissionRequest {
        id: "pending".to_string(),
        tool: "bash_tool".to_string(),
        summary: "run a command".to_string(),
        preview: "pwd".to_string(),
    });
    for ch in "/yolo".chars() {
        handle_key(
            &mut app,
            &mut child,
            KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE),
        )
        .unwrap();
    }
    assert!(screen(&mut app, 60, 22).contains("› /yolo"));
    handle_key(
        &mut app,
        &mut child,
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
    )
    .unwrap();
    let command: serde_json::Value =
        serde_json::from_str(&rx.recv_timeout(Duration::from_secs(1)).unwrap()).unwrap();
    assert_eq!(command, serde_json::json!({"type":"yolo"}));
    assert!(!app.yolo_enabled);
    assert!(app.pending_permission.is_some());
    assert_eq!(app.input, "keep my draft");
    app.apply(ServerMessage::Yolo { enabled: true });
    assert!(app.pending_permission.is_none());

    app.status = Status::Waiting;
    app.input = "/yolo".to_string();
    handle_key(
        &mut app,
        &mut child,
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
    )
    .unwrap();
    let command: serde_json::Value =
        serde_json::from_str(&rx.recv_timeout(Duration::from_secs(1)).unwrap()).unwrap();
    assert_eq!(command, serde_json::json!({"type":"yolo"}));
    app.apply(ServerMessage::Yolo { enabled: false });
    assert!(!app.yolo_enabled);
}
