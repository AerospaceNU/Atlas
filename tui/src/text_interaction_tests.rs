use super::{
    apply_paste, draw, handle_key, is_copy_key, is_paste_key, selection_mouse, App, ChildProcess,
    Mode, SelectionEvent, Status, TranscriptLine,
};
use crate::protocol::ServerMessage;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::layout::Position;
use ratatui::Terminal;

fn screen(app: &mut App, width: u16, height: u16) -> Buffer {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    terminal.draw(|frame| draw(frame, app)).unwrap();
    terminal.backend().buffer().clone()
}

fn text(buffer: &Buffer) -> String {
    (0..buffer.area.height)
        .map(|y| {
            (0..buffer.area.width)
                .map(|x| buffer[(x, y)].symbol())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn find(buffer: &Buffer, needle: &str) -> Position {
    for (row, line) in text(buffer).lines().enumerate() {
        if let Some(byte) = line.find(needle) {
            let column = ratatui::text::Span::raw(&line[..byte]).width();
            return Position::new(column as u16, row as u16);
        }
    }
    panic!("missing {needle}:\n{}", text(buffer));
}

fn mouse(kind: MouseEventKind, position: Position) -> MouseEvent {
    MouseEvent {
        kind,
        column: position.x,
        row: position.y,
        modifiers: KeyModifiers::NONE,
    }
}

#[test]
fn dragging_copies_only_the_selected_visible_text_and_highlights_it() {
    let mut app = App::new();
    app.transcript
        .push(TranscriptLine::User("Alpha beta gamma".to_string()));
    let before = screen(&mut app, 80, 24);
    let start = find(&before, "beta");
    let end = Position::new(start.x + 3, start.y);
    assert_eq!(
        selection_mouse(
            &mut app,
            mouse(MouseEventKind::Down(MouseButton::Left), start)
        ),
        SelectionEvent::Handled
    );
    selection_mouse(
        &mut app,
        mouse(MouseEventKind::Drag(MouseButton::Left), end),
    );
    assert_eq!(
        selection_mouse(&mut app, mouse(MouseEventKind::Up(MouseButton::Left), end)),
        SelectionEvent::Copy("beta".to_string())
    );
    let highlighted = screen(&mut app, 80, 24);
    assert_ne!(
        highlighted[(start.x, start.y)].bg,
        before[(start.x, start.y)].bg
    );
    assert_eq!(app.text_selection.as_ref().unwrap().text(), "beta");
}

#[test]
fn selecting_keeps_a_stable_view_and_resize_clears_the_selection() {
    let mut app = App::new();
    app.transcript
        .push(TranscriptLine::User("original message".to_string()));
    let before = screen(&mut app, 80, 24);
    let start = find(&before, "original");
    selection_mouse(
        &mut app,
        mouse(MouseEventKind::Down(MouseButton::Left), start),
    );
    let end = Position::new(start.x + 7, start.y);
    selection_mouse(
        &mut app,
        mouse(MouseEventKind::Drag(MouseButton::Left), end),
    );
    selection_mouse(&mut app, mouse(MouseEventKind::Up(MouseButton::Left), end));
    app.transcript
        .push(TranscriptLine::User("new incoming message".to_string()));
    assert!(!text(&screen(&mut app, 80, 24)).contains("new incoming message"));
    assert!(text(&screen(&mut app, 60, 24)).contains("new incoming message"));
    assert!(app.text_selection.is_none());
}

#[test]
fn multiline_paste_is_composed_without_submitting_or_running_commands() {
    let mut app = App::new();
    app.input = "prefix ".to_string();
    apply_paste(&mut app, "first line\r\nsecond line\n/yolo\n/quit");
    assert_eq!(app.input, "prefix first line\nsecond line\n/yolo\n/quit");
    assert!(app.transcript.is_empty());
    assert!(app.status == Status::Ready);
    assert!(!app.yolo_enabled);
    let painted = text(&screen(&mut app, 80, 24));
    assert!(painted.contains("prefix first line"), "{painted}");
    assert!(painted.contains("second line"), "{painted}");
    assert!(painted.contains("/quit"), "{painted}");
}

#[test]
fn paste_targets_the_model_filter_or_hidden_key_field() {
    let mut app = App::new();
    app.mode = Mode::ModelPicker;
    apply_paste(&mut app, "example/model\n");
    assert_eq!(app.picker_filter, "example/model");
    assert!(app.input.is_empty());
    app.mode = Mode::Key {
        scope: "session".to_string(),
        name: String::new(),
    };
    apply_paste(&mut app, "dummy-value\n");
    assert_eq!(app.input, "dummy-value");
    let painted = text(&screen(&mut app, 80, 24));
    assert!(!painted.contains("dummy-value"));
    assert!(painted.contains("••••"));
}

#[test]
fn command_preview_selection_preserves_approval_button_clicks() {
    let mut app = App::new();
    app.apply(ServerMessage::PermissionRequest {
        id: "pending".to_string(),
        tool: "bash_tool".to_string(),
        summary: "run: pwd".to_string(),
        preview: "pwd".to_string(),
    });
    let painted = screen(&mut app, 80, 24);
    let button = find(&painted, "Allow once");
    let click = mouse(MouseEventKind::Down(MouseButton::Left), button);
    assert_eq!(selection_mouse(&mut app, click), SelectionEvent::Pass);
    assert!(app.text_selection.is_none());
    assert_eq!(
        app.pending_permission.as_mut().unwrap().mouse(click),
        Some(true)
    );
    let start = find(&painted, "pwd");
    selection_mouse(
        &mut app,
        mouse(MouseEventKind::Down(MouseButton::Left), start),
    );
    let end = Position::new(start.x + 2, start.y);
    assert_eq!(
        selection_mouse(&mut app, mouse(MouseEventKind::Up(MouseButton::Left), end)),
        SelectionEvent::Copy("pwd".to_string())
    );
    assert!(app.pending_permission.is_some());
}

#[test]
fn copy_and_paste_shortcuts_include_terminal_conventions() {
    for modifiers in [
        KeyModifiers::CONTROL,
        KeyModifiers::CONTROL | KeyModifiers::SHIFT,
        KeyModifiers::SUPER,
        KeyModifiers::META,
    ] {
        assert!(is_copy_key(KeyEvent::new(KeyCode::Char('c'), modifiers)));
        assert!(is_paste_key(KeyEvent::new(KeyCode::Char('v'), modifiers)));
    }
    assert!(is_copy_key(KeyEvent::new(
        KeyCode::Insert,
        KeyModifiers::CONTROL
    )));
    assert!(is_paste_key(KeyEvent::new(
        KeyCode::Insert,
        KeyModifiers::SHIFT
    )));
    assert!(!is_copy_key(KeyEvent::new(
        KeyCode::Char('c'),
        KeyModifiers::NONE
    )));
}

#[cfg(unix)]
#[test]
fn escape_clears_a_selection_without_quitting_or_cancelling_the_turn() {
    use std::process::{Command, Stdio};
    let mut process = Command::new("cat")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    let stdin = process.stdin.take().unwrap();
    let mut child = ChildProcess {
        child: process,
        stdin,
    };
    let mut app = App::new();
    app.status = Status::Waiting;
    app.transcript
        .push(TranscriptLine::User("select me".to_string()));
    let painted = screen(&mut app, 80, 24);
    let start = find(&painted, "select me");
    selection_mouse(
        &mut app,
        mouse(MouseEventKind::Down(MouseButton::Left), start),
    );
    assert!(!handle_key(
        &mut app,
        &mut child,
        KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)
    )
    .unwrap());
    assert!(app.text_selection.is_none());
    assert!(app.transcript.len() == 1);
    assert!(child.child.try_wait().unwrap().is_none());
}
