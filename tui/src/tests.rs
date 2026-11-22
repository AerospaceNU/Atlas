//! End-to-end TUI behavior, UX regression, and rendering tests.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use crossterm::event::{
    KeyCode as K, KeyEvent, KeyModifiers as Mod, MouseButton, MouseEvent, MouseEventKind as Mk,
};
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::layout::Position;
use ratatui::Terminal;
use serde_json::{json, Value};

use crate::app::{cancel_message, Activity, App, Mode, Status, TranscriptEntry, STILL_STOPPING};
use crate::history::PromptHistory;
use crate::input::{apply_paste, handle_key, handle_mouse};
use crate::process::ChildProcess;
use crate::protocol::parse_line;
use crate::render::draw;
use crate::render::overlays::help_lines;
use crate::render::status::dashboard_text;
use crate::theme::Theme;

fn test_app() -> App {
    App::new(PathBuf::from("/tmp/atlas-ws"), PromptHistory::default())
}

fn feed(app: &mut App, raw: &str) {
    app.apply(parse_line(raw).unwrap().unwrap());
}

fn render(app: &mut App, w: u16, h: u16) -> (Buffer, Option<Position>) {
    let theme = Theme::truecolor();
    let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
    term.draw(|f| draw(f, app, &theme)).unwrap();
    (
        term.backend().buffer().clone(),
        term.get_cursor_position().ok(),
    )
}

fn text_of(buf: &Buffer) -> String {
    (0..buf.area.height)
        .map(|y| {
            (0..buf.area.width)
                .map(|x| buf[(x, y)].symbol())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(unix)]
fn pipe_child() -> (ChildProcess, std::sync::mpsc::Receiver<Value>) {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};

    let mut p = Command::new("cat")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let (stdin, stdout) = (p.stdin.take().unwrap(), p.stdout.take().unwrap());
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if tx.send(serde_json::from_str(&line).unwrap()).is_err() {
                break;
            }
        }
    });
    (ChildProcess::from_parts(p, stdin), rx)
}

#[cfg(unix)]
fn key(app: &mut App, child: &mut ChildProcess, code: K, mods: Mod) -> bool {
    handle_key(app, child, KeyEvent::new(code, mods)).unwrap()
}

#[cfg(unix)]
fn mouse(app: &mut App, child: &mut ChildProcess, kind: Mk, column: u16, row: u16) {
    handle_mouse(
        app,
        child,
        MouseEvent {
            kind,
            column,
            row,
            modifiers: Mod::NONE,
        },
    )
    .unwrap();
}

#[cfg(unix)]
fn run_cmd(app: &mut App, child: &mut ChildProcess, cmd: &str) -> bool {
    app.clear_composer();
    apply_paste(app, cmd);
    key(app, child, K::Enter, Mod::NONE)
}

#[cfg(unix)]
#[test]
fn esc_bare_slash_help_and_ctrl_c_behave_safely() {
    let (mut child, rx) = pipe_child();
    let mut app = test_app();
    app.transcript.push(TranscriptEntry::User("keep me".into()));

    assert!(!key(&mut app, &mut child, K::Esc, Mod::NONE));
    key(&mut app, &mut child, K::Char('/'), Mod::NONE);
    assert!(!key(&mut app, &mut child, K::Enter, Mod::NONE));
    assert_eq!(app.transcript.len(), 1);
    assert!(rx.try_recv().is_err());

    assert!(!run_cmd(&mut app, &mut child, "/help"));
    assert!(matches!(app.mode, Mode::Help(_)));
    let help = help_lines(&app).join("\n");
    assert!(["/resume", "/undo", "/compact", "Ctrl+O"]
        .iter()
        .all(|s| help.contains(s)));
    key(&mut app, &mut child, K::Esc, Mod::NONE);
    assert!(key(&mut app, &mut child, K::Char('c'), Mod::CONTROL));
}

#[cfg(unix)]
#[test]
fn streaming_tool_click_diff_colors_and_nonblocking_selection_work() {
    let (mut child, _rx) = pipe_child();
    let mut app = test_app();
    app.status = Status::Waiting;
    feed(
        &mut app,
        r#"{"type":"thought_delta","text":"analyzing..."}"#,
    );
    feed(
        &mut app,
        r#"{"type":"content_delta","text":"Visible answer right now"}"#,
    );
    let screen = text_of(&render(&mut app, 100, 24).0);
    assert!(screen.contains("Visible answer right now") && screen.contains("thought"));

    feed(
        &mut app,
        r#"{"type":"tool_start","call_id":"c1","name":"edit_file","summary":"src/atlas/agent/runtime.py"}"#,
    );
    feed(
        &mut app,
        r#"{"type":"tool_output_delta","call_id":"c1","chunk":"patching line 42\n"}"#,
    );
    assert!(text_of(&render(&mut app, 100, 24).0).contains("patching line 42"));

    feed(
        &mut app,
        r#"{"type":"step","call_id":"c1","name":"edit_file","summary":"src/atlas/agent/runtime.py","ok":true,"duration_ms":12,"text":"edited","diff":"@@ -1 +1 @@\n-before\n+after"}"#,
    );
    let collapsed = text_of(&render(&mut app, 100, 24).0);
    assert!(
        collapsed.contains("◆ edit_file src/atlas/agent/runtime.py (12ms)")
            && !collapsed.contains("+after")
    );

    let y = app.tool_hits.last().unwrap().y;
    mouse(&mut app, &mut child, Mk::Down(MouseButton::Left), 6, y);
    mouse(&mut app, &mut child, Mk::Up(MouseButton::Left), 6, y);
    let (buf2, _) = render(&mut app, 100, 24);
    assert!(text_of(&buf2).contains("+after"));
    let t = Theme::truecolor();
    let has = |c| (0..24).any(|r| (0..100).any(|x| buf2[(x, r)].fg == c));
    assert!(has(t.green) && has(t.red) && has(t.cyan));

    mouse(&mut app, &mut child, Mk::Down(MouseButton::Left), 4, 1);
    mouse(&mut app, &mut child, Mk::Drag(MouseButton::Left), 10, 1);
    feed(
        &mut app,
        r#"{"type":"permission_request","id":"p1","tool":"bash_tool","summary":"run cargo test","preview":"cargo test","preview_kind":"command"}"#,
    );
    let modal = text_of(&render(&mut app, 100, 24).0);
    assert!(modal.contains("Approval required") && modal.contains("cargo test"));
}

#[cfg(unix)]
#[test]
fn resume_undo_model_picker_empty_state_and_unseen_badge_work() {
    let (mut child, rx) = pipe_child();
    let mut app = test_app();
    let recv = || rx.recv_timeout(Duration::from_secs(1)).unwrap();

    feed(
        &mut app,
        r#"{"type":"ready","session_id":"s1","model":"openai/gpt-5","workspace":"/repo/atlas","tools":["read_file"],"key_set":false,"yolo_enabled":true,"status":{"model":"openai/gpt-5","context_used":50000,"context_limit":200000,"branch":"feat","yolo_enabled":true}}"#,
    );
    apply_paste(&mut app, "hi");
    let (buf, cursor) = render(&mut app, 100, 28);
    let empty = text_of(&buf);
    assert!(
        empty.contains("API key not set")
            && empty.contains("/repo/atlas")
            && !empty.contains('▌')
            && cursor.is_some()
    );
    let dash = dashboard_text(&app, &Theme::truecolor());
    assert!(dash.contains("YOLO on") && dash.contains("25%") && dash.contains("(feat)"));

    run_cmd(&mut app, &mut child, "/resume sess-9");
    assert_eq!(recv(), json!({"type":"load_session","id":"sess-9"}));
    assert_eq!(app.status, Status::Waiting);
    feed(
        &mut app,
        r#"{"type":"session_loaded","session_id":"sess-9","model":"anthropic/claude-sonnet-4","messages":[{"role":"user","text":"old"}]}"#,
    );
    assert_eq!(app.status, Status::Ready);

    run_cmd(&mut app, &mut child, "/undo");
    assert_eq!(recv(), json!({"type":"undo"}));
    feed(
        &mut app,
        r#"{"type":"undo","ok":true,"path":"a.py","message":"restored a.py"}"#,
    );
    assert!(text_of(&render(&mut app, 90, 24).0).contains("restored a.py"));

    feed(
        &mut app,
        r#"{"type":"models","selected":"openai/gpt-4o","models":[{"id":"openai/gpt-4o","context_length":200000},{"id":"anthropic/claude-sonnet-4","context_length":200000,"prompt_price":"0.000003"}]}"#,
    );
    run_cmd(&mut app, &mut child, "/model clson");
    let _ = recv();
    assert!(text_of(&render(&mut app, 100, 24).0).contains("anthropic/claude-sonnet-4"));
    key(&mut app, &mut child, K::Enter, Mod::NONE);
    assert_eq!(
        recv(),
        json!({"type":"select_model","model":"anthropic/claude-sonnet-4"})
    );

    for i in 0..30 {
        app.transcript
            .push(TranscriptEntry::User(format!("msg {i}")));
    }
    let _ = render(&mut app, 80, 14);
    app.scroll_by(-10);
    feed(&mut app, r#"{"type":"content_delta","text":"brand new"}"#);
    assert!(text_of(&render(&mut app, 80, 14).0).contains("new output"));
}

#[test]
fn done_does_not_duplicate_streamed_thought_or_assistant() {
    use TranscriptEntry as E;

    let mut app = test_app();
    app.transcript.push(E::User("hi".into()));
    feed(&mut app, r#"{"type":"thought_delta","text":"plan "}"#);
    feed(&mut app, r#"{"type":"thought_delta","text":"step"}"#);
    feed(&mut app, r#"{"type":"content_delta","text":"Hel"}"#);
    feed(&mut app, r#"{"type":"content_delta","text":"lo"}"#);
    feed(
        &mut app,
        r#"{"type":"done","response":"Hello","thoughts":"plan step","status":{}}"#,
    );
    assert_eq!(app.transcript.len(), 3);
    assert!(
        matches!(&app.transcript[1], E::Thought { text, streaming: false, .. } if text == "plan step")
    );
    assert!(
        matches!(&app.transcript[2], E::Assistant { text, streaming: false, .. } if text == "Hello")
    );

    // Multi-step turn with tool call between thought and final answer.
    app.transcript.push(E::User("check file".into()));
    feed(&mut app, r#"{"type":"thought_delta","text":"read it"}"#);
    feed(
        &mut app,
        r#"{"type":"tool_start","call_id":"c1","name":"read_file","summary":"a.py"}"#,
    );
    feed(
        &mut app,
        r#"{"type":"step","call_id":"c1","name":"read_file","summary":"a.py","ok":true,"duration_ms":2,"text":"ok"}"#,
    );
    feed(&mut app, r#"{"type":"content_delta","text":"Done."}"#);
    feed(
        &mut app,
        r#"{"type":"done","response":"Done.","thoughts":"read it","status":{}}"#,
    );
    let tail = &app.transcript[3..];
    assert_eq!(tail.len(), 4);
    assert!(matches!(tail[0], E::User(_)));
    assert!(matches!(&tail[1], E::Thought { text, streaming: false, .. } if text == "read it"));
    assert!(matches!(tail[2], E::Tool(_)));
    assert!(matches!(&tail[3], E::Assistant { text, streaming: false, .. } if text == "Done."));
}

#[test]
fn missing_key_skips_startup_catalog_lookup_and_error_closes_loading_picker() {
    let mut app = test_app();
    feed(
        &mut app,
        r#"{"type":"ready","session_id":"s1","model":"openai/gpt-5","workspace":"/tmp","tools":[],"key_set":false,"status":{"model":"openai/gpt-5"}}"#,
    );
    assert!(!app.lookup_context);
    assert!(app.transcript.is_empty());

    feed(
        &mut app,
        r#"{"type":"key_set","provider":"openrouter","key_set":true}"#,
    );
    assert!(app.key_set && app.lookup_context);

    app.mode = Mode::ModelPicker(crate::app::ModelPickerState {
        loading: true,
        filter: Default::default(),
        index: 0,
    });
    assert!(matches!(app.mode, Mode::ModelPicker(ref s) if s.loading));
    feed(&mut app, r#"{"type":"error","message":"catalog failed"}"#);
    assert!(matches!(app.mode, Mode::Compose));
}

#[cfg(unix)]
#[test]
fn v017_tool_notes_cancel_clock_compaction_and_resume_work() {
    let (mut child, rx) = pipe_child();
    let recv = || rx.recv_timeout(Duration::from_secs(1)).unwrap();

    // 1. Tool rows: prefix stripping, notes/artifacts in expanded view, `bash` green exit_code.
    let mut app = test_app();
    feed(
        &mut app,
        r#"{"type":"step","call_id":"b1","name":"bash","summary":"bash: sleep 8","ok":true,"duration_ms":100,"text":"exit_code: 0\ndone","notes":["repeated result"],"artifacts":["report.md"]}"#,
    );
    let collapsed = text_of(&render(&mut app, 90, 24).0);
    assert!(
        collapsed.contains("◆ bash sleep 8")
            && !collapsed.contains("bash bash:")
            && !collapsed.contains("repeated result")
            && !collapsed.contains("report.md")
    );
    app.toggle_selected(Some(true));
    let (buf, _) = render(&mut app, 90, 24);
    let opened = text_of(&buf);
    assert!(
        opened.contains("notes")
            && opened.contains("repeated result")
            && opened.contains("artifacts")
            && opened.contains("report.md")
    );
    let green = Theme::truecolor().green;
    assert!((0..24).any(|y| (0..90).any(|x| buf[(x, y)].fg == green)));

    // 2. Tool-class cancel messages, two-stage Esc cancel, and elapsed clock (>= 5s).
    let run = |name: &str| cancel_message(&Some(Activity::ToolExecuting(name.into())));
    assert_eq!(run("bash"), "cancelling the running command...");
    assert!(run("read_file").contains("read_file is still running"));
    assert!(run("band_math").contains("next check"));
    assert_eq!(cancel_message(&Some(Activity::Thinking)), "cancelling...");
    assert!(cancel_message(&None).contains("next step"));

    app.status = Status::Waiting;
    app.set_activity(Some(Activity::ToolExecuting("bash".into())));
    let first_since = app.activity_since;
    app.set_activity(Some(Activity::ToolExecuting("bash".into())));
    assert!(first_since.is_some() && app.activity_since == first_since);
    app.activity_since = Instant::now().checked_sub(Duration::from_secs(72));
    assert!(text_of(&render(&mut app, 90, 20).0).contains("1m12s"));

    assert!(!key(&mut app, &mut child, K::Esc, Mod::NONE));
    assert_eq!(recv(), json!({"type": "cancel"}));
    assert!(!key(&mut app, &mut child, K::Esc, Mod::NONE));
    assert!(
        app.cancel_stage == 2
            && rx.try_recv().is_err()
            && app.status_notice.as_deref() == Some(STILL_STOPPING)
    );
    feed(
        &mut app,
        r#"{"type":"done","response":"","cancelled":true,"status":{}}"#,
    );
    assert!(app.cancel_stage == 0 && app.activity_since.is_none());

    // 3. `/compact`, `Compacted` events, and `Done` context/budget/compaction notices.
    feed(
        &mut app,
        r#"{"type":"ready","session_id":"s1","model":"openai/gpt-5","workspace":"/tmp","tools":[],"key_set":true,"status":{"model":"openai/gpt-5","context_used":5000,"context_limit":10000}}"#,
    );
    run_cmd(&mut app, &mut child, "/compact");
    assert_eq!(recv(), json!({"type": "compact"}));
    feed(
        &mut app,
        r#"{"type":"compacted","compaction":{"occurred":true,"tokens_reclaimed":900,"messages_dropped":2},"status":{"context_used":4100,"context_limit":10000}}"#,
    );
    assert_eq!(
        (
            app.status_snapshot.context_used,
            app.status_snapshot.model.as_str()
        ),
        (4100, "openai/gpt-5")
    );
    feed(&mut app, r#"{"type":"compacted"}"#);
    feed(
        &mut app,
        r#"{"type":"done","response":"","stopped_for_context":true,"stopped_for_budget":true,"compaction":{"occurred":true,"tokens_reclaimed":321},"status":{}}"#,
    );
    let compact_screen = text_of(&render(&mut app, 100, 32).0);
    assert!(
        compact_screen.contains("900 tokens reclaimed")
            && compact_screen.contains("nothing to compact")
            && compact_screen.contains("no longer fits the model context")
            && compact_screen.contains("budget")
            && compact_screen.contains("321 tokens reclaimed")
            && app.status == Status::Ready
    );

    // 4. Rich `/sessions` rows, bare `/resume` -> `"last"`, `omitted`, and `interrupted` -> `Stopped` + `Ctrl+R`.
    feed(
        &mut app,
        r#"{"type":"list","kind":"sessions","items":["s-new"],"sessions":[{"id":"s-new","saved":"2026-11-22T18:00","model":"m","messages":6,"prompt":"fix tests","interrupted":true,"stopped_for_limit":false}]}"#,
    );
    assert_eq!(app.session_names, vec!["s-new"]);
    let listed = text_of(&render(&mut app, 120, 32).0);
    assert!(
        listed.contains("s-new") && listed.contains("6 msgs") && listed.contains("[interrupted]")
    );

    run_cmd(&mut app, &mut child, "/resume");
    assert_eq!(recv(), json!({"type": "load_session", "id": "last"}));
    assert!(app.status == Status::Waiting && !app.transcript.is_empty());

    feed(
        &mut app,
        r#"{"type":"session_loaded","session_id":"s-new","model":"openai/gpt-5","interrupted":true,"omitted":5,"messages":[{"role":"user","text":"hello"}]}"#,
    );
    assert_eq!(app.status, Status::Stopped);
    let resumed = text_of(&render(&mut app, 100, 24).0);
    assert!(
        resumed.contains("5 earlier items not shown")
            && resumed.contains("hello")
            && resumed.contains("resumed s-new")
    );
    assert!(!key(&mut app, &mut child, K::Char('r'), Mod::CONTROL));
    assert_eq!(recv(), json!({"type": "resume"}));
    assert_eq!(app.status, Status::Waiting);
}
