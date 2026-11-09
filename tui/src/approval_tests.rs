use super::{draw, note_session_closed, slash_outcome, Activity, App, Status, SPINNER};
use crate::protocol::{ServerMessage, StatusSnapshot};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use ratatui::backend::TestBackend;
use ratatui::Terminal;

const CALL_ID: &str = "7f57e9d2-19bf-4f43-a7c0-6ffca84ab8b3";
const COMMAND: &str = "pwd && ls -a";

fn approval_app() -> App {
    let mut app = App::new();
    app.apply(ServerMessage::PermissionRequest {
        id: CALL_ID.to_string(),
        tool: "bash_tool".to_string(),
        summary: format!("run: {COMMAND}"),
        preview: COMMAND.to_string(),
    });
    app
}

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

fn normalized(screen: &str) -> String {
    screen.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[test]
fn approval_panel_shows_the_request_and_safe_choices() {
    let mut app = approval_app();
    let painted = screen(&mut app, 100, 28);
    let visible = normalized(&painted);
    let visible_lower = visible.to_lowercase();

    assert!(visible.contains("Approval required"), "{painted}");
    assert!(visible.contains("bash_tool"), "{painted}");
    assert!(visible.contains(COMMAND), "{painted}");
    assert_eq!(visible.matches(COMMAND).count(), 1, "{painted}");
    assert!(visible.contains("Allow once"), "{painted}");
    assert!(visible.contains("Deny"), "{painted}");
    assert!(visible_lower.contains("enter"), "{painted}");
    assert!(visible_lower.contains("esc"), "{painted}");
    assert!(!visible.contains(CALL_ID), "raw call id leaked: {painted}");
    assert!(!visible_lower.contains("/allow"), "{painted}");
    assert!(!visible_lower.contains("/deny"), "{painted}");
}

#[test]
fn short_approval_stays_visible_at_supported_terminal_sizes() {
    for (width, height) in [(100, 28), (60, 22), (40, 20)] {
        let mut app = approval_app();
        let painted = screen(&mut app, width, height);
        let visible = normalized(&painted);

        assert!(
            visible.contains("Approval required"),
            "{width}x{height}: {painted}"
        );
        assert!(visible.contains("bash_tool"), "{width}x{height}: {painted}");
        assert!(visible.contains(COMMAND), "{width}x{height}: {painted}");
        assert!(
            visible.contains("Allow once"),
            "{width}x{height}: {painted}"
        );
        assert!(visible.contains("Deny"), "{width}x{height}: {painted}");
    }
}

#[test]
fn waiting_for_approval_has_no_activity_spinner() {
    let mut app = approval_app();
    app.status = Status::Waiting;
    app.activity = Some(Activity::ToolExecuting("bash_tool".to_string()));

    let painted = screen(&mut app, 100, 28);
    let visible_lower = normalized(&painted).to_lowercase();

    assert!(visible_lower.contains("awaiting approval"), "{painted}");
    assert!(!visible_lower.contains("thinking"), "{painted}");
    assert!(!visible_lower.contains("tool executing"), "{painted}");
    assert!(
        !SPINNER.iter().any(|glyph| painted.contains(glyph)),
        "activity spinner was rendered: {painted}"
    );
}

#[test]
fn long_approval_previews_scroll_without_hiding_the_choices() {
    let mut app = App::new();
    app.apply(ServerMessage::PermissionRequest {
        id: CALL_ID.to_string(),
        tool: "write_file".to_string(),
        summary: "write_file report.txt".to_string(),
        preview: (0..40).map(|index| format!("line {index}\n")).collect(),
    });
    let first = screen(&mut app, 60, 22);
    assert!(first.contains("line 0"), "{first}");
    assert!(!first.contains("line 39"), "{first}");
    assert!(first.contains("PgUp/PgDn scroll"), "{first}");
    let prompt = app.pending_permission.as_mut().unwrap();
    assert_eq!(
        prompt.key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE)),
        None
    );
    let last = screen(&mut app, 60, 22);
    assert!(last.contains("line 39"), "{last}");
    assert!(last.contains("Allow once"), "{last}");
    assert!(last.contains("Deny"), "{last}");
    let prompt = app.pending_permission.as_mut().unwrap();
    prompt.key(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
    assert!(screen(&mut app, 60, 22).contains("line 0"));
}

#[test]
fn approval_choices_support_tab_arrows_and_mouse() {
    let mut app = approval_app();
    let painted = screen(&mut app, 60, 22);
    assert!(painted.contains("› Deny"), "{painted}");
    let prompt = app.pending_permission.as_mut().unwrap();
    for code in [
        KeyCode::Tab,
        KeyCode::Down,
        KeyCode::BackTab,
        KeyCode::Right,
    ] {
        assert_eq!(prompt.key(KeyEvent::new(code, KeyModifiers::NONE)), None);
    }
    assert_eq!(
        prompt.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        Some(false)
    );
    prompt.key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
    assert_eq!(
        prompt.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        Some(true)
    );
    // A modified key or repeat cannot confirm a focused action.
    assert_eq!(
        prompt.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::CONTROL)),
        None
    );
    let mut repeated = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
    repeated.kind = crossterm::event::KeyEventKind::Repeat;
    assert_eq!(prompt.key(repeated), None);

    for (label, allow) in [("Allow once", true), ("Deny", false)] {
        let row = painted
            .lines()
            .position(|line| line.contains(label))
            .unwrap();
        let column = painted.lines().nth(row).unwrap().find(label).unwrap();
        assert_eq!(
            prompt.mouse(MouseEvent {
                kind: MouseEventKind::Down(MouseButton::Left),
                column: column as u16,
                row: row as u16,
                modifiers: KeyModifiers::NONE,
            }),
            Some(allow)
        );
    }
    assert_eq!(
        prompt.mouse(MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 0,
            row: 0,
            modifiers: KeyModifiers::NONE,
        }),
        None
    );
}

#[test]
fn completed_or_failed_turns_remove_stale_approval_prompts() {
    let terminal_events = [
        ServerMessage::Done {
            response: String::new(),
            stopped_for_limit: false,
            stopped_for_budget: false,
            cancelled: true,
            compaction: Default::default(),
            status: StatusSnapshot::default(),
        },
        ServerMessage::Error {
            message: "session failed".to_string(),
            http_status: None,
            elapsed_seconds: None,
            error_body: None,
            status_line: String::new(),
        },
        ServerMessage::Step {
            name: "bash_tool".to_string(),
            call_id: CALL_ID.to_string(),
            ok: false,
            arguments: serde_json::json!({}),
            text: None,
            error: Some("approval timed out".to_string()),
            error_kind: None,
            artifacts: Vec::new(),
        },
    ];
    for event in terminal_events {
        let mut app = approval_app();
        app.apply(event);
        assert!(app.pending_permission.is_none());
        assert!(!screen(&mut app, 60, 22).contains("Approval required"));
    }
    let mut app = approval_app();
    note_session_closed(&mut app);
    assert!(app.pending_permission.is_none());
    let mut app = approval_app();
    slash_outcome(&mut app, "/new");
    assert!(app.pending_permission.is_none());
}

#[cfg(unix)]
#[test]
fn keyboard_decisions_reach_the_session_while_the_turn_is_waiting() {
    use super::{handle_key, ChildProcess};
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    for (keys, allow) in [
        (vec![KeyCode::Up, KeyCode::Enter], true),
        (vec![KeyCode::Enter], false),
        (vec![KeyCode::Up, KeyCode::Esc], false),
    ] {
        let mut process = Command::new("cat")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("protocol sink");
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
        let mut app = approval_app();
        app.input = "keep this draft".to_string();
        assert!(app.status == Status::Waiting);
        handle_key(
            &mut app,
            &mut child,
            KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE),
        )
        .unwrap();
        for code in keys {
            assert!(!handle_key(
                &mut app,
                &mut child,
                KeyEvent::new(code, KeyModifiers::NONE)
            )
            .unwrap());
        }
        let payload: serde_json::Value = serde_json::from_str(
            &rx.recv_timeout(Duration::from_secs(1))
                .expect("permission response"),
        )
        .unwrap();
        assert_eq!(
            payload,
            serde_json::json!({"type": "permission", "id": CALL_ID, "allow": allow})
        );
        assert_eq!(app.input, "keep this draft");
        assert!(app.session_open);
        assert!(app.pending_permission.is_none());
        assert!(app.status == Status::Waiting);
        // A second Enter cannot answer the same approval twice or submit the draft.
        handle_key(
            &mut app,
            &mut child,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
        )
        .unwrap();
        assert!(rx.recv_timeout(Duration::from_millis(30)).is_err());
    }
}
