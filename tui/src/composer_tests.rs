use super::{
    apply_paste, draw, handle_key, is_newline_key, show_slash_menu, App, ChildProcess, Status,
};
use crate::protocol::{ServerMessage, StatusSnapshot};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{backend::TestBackend, Terminal};

fn screen(app: &mut App, width: u16, height: u16) -> String {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    terminal.draw(|frame| draw(frame, app)).unwrap();
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

#[cfg(unix)]
fn sink() -> (ChildProcess, std::sync::mpsc::Receiver<serde_json::Value>) {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};
    let mut process = Command::new("cat")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let stdin = process.stdin.take().unwrap();
    let stdout = process.stdout.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if tx.send(serde_json::from_str(&line).unwrap()).is_err() {
                break;
            }
        }
    });
    (
        ChildProcess {
            child: process,
            stdin,
        },
        rx,
    )
}

fn key(app: &mut App, child: &mut ChildProcess, code: KeyCode, modifiers: KeyModifiers) {
    assert!(!handle_key(app, child, KeyEvent::new(code, modifiers)).unwrap());
}

fn done(app: &mut App) {
    app.apply(ServerMessage::Done {
        response: String::new(),
        stopped_for_limit: false,
        stopped_for_budget: false,
        cancelled: false,
        compaction: Default::default(),
        status: StatusSnapshot::default(),
    });
}

#[cfg(unix)]
#[test]
fn editing_and_paste_insert_at_the_cursor_and_send_the_edited_text() {
    let (mut child, rx) = sink();
    let mut app = App::new();
    apply_paste(&mut app, "ac");
    key(&mut app, &mut child, KeyCode::Left, KeyModifiers::NONE);
    key(&mut app, &mut child, KeyCode::Char('b'), KeyModifiers::NONE);
    apply_paste(&mut app, "👩‍🚀");
    assert_eq!(app.input, "ab👩‍🚀c");
    key(&mut app, &mut child, KeyCode::Backspace, KeyModifiers::NONE);
    key(&mut app, &mut child, KeyCode::Delete, KeyModifiers::NONE);
    assert_eq!(app.input, "ab");
    key(&mut app, &mut child, KeyCode::Home, KeyModifiers::NONE);
    key(&mut app, &mut child, KeyCode::Char('x'), KeyModifiers::NONE);
    key(&mut app, &mut child, KeyCode::End, KeyModifiers::NONE);
    key(&mut app, &mut child, KeyCode::Char('!'), KeyModifiers::NONE);
    key(&mut app, &mut child, KeyCode::Enter, KeyModifiers::NONE);
    assert_eq!(
        rx.recv_timeout(std::time::Duration::from_secs(1)).unwrap(),
        serde_json::json!({"type":"user","text":"xab!"})
    );
    assert!(app.input.is_empty());
}

#[cfg(unix)]
#[test]
fn prompt_history_restores_the_unsent_draft_and_excludes_slash_commands() {
    let (mut child, rx) = sink();
    let mut app = App::new();
    for text in ["first", "second"] {
        apply_paste(&mut app, text);
        key(&mut app, &mut child, KeyCode::Enter, KeyModifiers::NONE);
        rx.recv_timeout(std::time::Duration::from_secs(1)).unwrap();
        done(&mut app);
    }
    apply_paste(&mut app, "/goal test");
    key(&mut app, &mut child, KeyCode::Enter, KeyModifiers::NONE);
    assert_eq!(
        rx.recv_timeout(std::time::Duration::from_secs(1)).unwrap()["type"],
        "goal"
    );
    apply_paste(&mut app, "unfinished");
    for (code, expected) in [
        (KeyCode::Up, "second"),
        (KeyCode::Up, "first"),
        (KeyCode::Down, "second"),
        (KeyCode::Down, "unfinished"),
    ] {
        key(&mut app, &mut child, code, KeyModifiers::NONE);
        assert_eq!(app.input, expected);
    }
}

#[cfg(unix)]
#[test]
fn escape_and_enter_keep_a_working_draft_and_escape_keeps_an_idle_draft() {
    let (mut child, rx) = sink();
    let mut app = App::new();
    apply_paste(&mut app, "unfinished");
    app.status = Status::Waiting;
    key(&mut app, &mut child, KeyCode::Enter, KeyModifiers::NONE);
    assert_eq!(app.input, "unfinished");
    assert!(app.status_notice.unwrap().contains("draft kept"));
    assert!(rx.try_recv().is_err());
    key(&mut app, &mut child, KeyCode::Esc, KeyModifiers::NONE);
    assert_eq!(
        rx.recv_timeout(std::time::Duration::from_secs(1)).unwrap()["type"],
        "cancel"
    );
    done(&mut app);
    key(&mut app, &mut child, KeyCode::Esc, KeyModifiers::NONE);
    assert_eq!(app.input, "unfinished");
    assert!(app.session_open);
    assert!(child.child.try_wait().unwrap().is_none());
}

#[cfg(unix)]
#[test]
fn dismissing_completion_keeps_the_command_and_does_not_accept_a_hidden_suggestion() {
    let (mut child, rx) = sink();
    let mut app = App::new();
    apply_paste(&mut app, "/mo");
    assert!(show_slash_menu(&app));
    key(&mut app, &mut child, KeyCode::Esc, KeyModifiers::NONE);
    assert_eq!(app.input, "/mo");
    assert!(!show_slash_menu(&app));
    key(&mut app, &mut child, KeyCode::Enter, KeyModifiers::NONE);
    assert!(rx.try_recv().is_err());
}

#[cfg(unix)]
#[test]
fn alt_enter_inserts_a_newline_and_vertical_arrows_edit_multiline_drafts() {
    let (mut child, rx) = sink();
    let mut app = App::new();
    apply_paste(&mut app, "first");
    key(&mut app, &mut child, KeyCode::Enter, KeyModifiers::ALT);
    apply_paste(&mut app, "second");
    assert!(rx.try_recv().is_err());
    key(&mut app, &mut child, KeyCode::Up, KeyModifiers::NONE);
    key(&mut app, &mut child, KeyCode::Char('X'), KeyModifiers::NONE);
    assert_eq!(app.input, "firstX\nsecond");
    let painted = screen(&mut app, 60, 22);
    assert!(painted.contains("firstX▌"), "{painted}");
    assert!(painted.contains("second"), "{painted}");
}

#[test]
fn ctrl_j_is_a_newline_key_and_enter_alone_is_not() {
    assert!(is_newline_key(KeyEvent::new(
        KeyCode::Char('j'),
        KeyModifiers::CONTROL
    )));
    assert!(is_newline_key(KeyEvent::new(
        KeyCode::Char('J'),
        KeyModifiers::CONTROL
    )));
    assert!(is_newline_key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::ALT
    )));
    assert!(is_newline_key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::SHIFT
    )));
    assert!(!is_newline_key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::NONE
    )));
    assert!(!is_newline_key(KeyEvent::new(
        KeyCode::Char('j'),
        KeyModifiers::NONE
    )));
}

#[cfg(unix)]
#[test]
fn ctrl_j_inserts_a_newline_instead_of_sending_the_draft() {
    let (mut child, rx) = sink();
    let mut app = App::new();
    apply_paste(&mut app, "first");
    key(
        &mut app,
        &mut child,
        KeyCode::Char('j'),
        KeyModifiers::CONTROL,
    );
    apply_paste(&mut app, "second");
    assert!(rx.try_recv().is_err());
    // Ctrl+J is not text: the character arm drops control-modified keys, so
    // before this binding the chord silently did nothing at all.
    assert_eq!(app.input, "first\nsecond");
    let painted = screen(&mut app, 60, 22);
    assert!(painted.contains("first"), "{painted}");
    assert!(painted.contains("second"), "{painted}");
}

#[cfg(unix)]
#[test]
fn the_composer_viewport_follows_the_cursor_through_a_long_wrapped_draft() {
    let (mut child, _) = sink();
    let mut app = App::new();
    apply_paste(&mut app, &format!("start{}end", "x".repeat(350)));
    let tail = screen(&mut app, 60, 22);
    assert!(tail.contains("d▌"), "{tail}");
    assert!(!tail.contains("start"), "{tail}");
    key(&mut app, &mut child, KeyCode::Home, KeyModifiers::NONE);
    let head = screen(&mut app, 60, 22);
    assert!(head.contains("▌start"), "{head}");
    assert!(!head.contains("d▌"), "{head}");
}
