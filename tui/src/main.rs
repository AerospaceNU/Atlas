//! Minimal terminal client for the Atlas agent session protocol.

mod logo;
mod markdown;
mod protocol;
mod theme;

use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers, MouseButton, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen,
};
use markdown::{render_fitted, Mark, Piece};
use protocol::{default_status_line, parse_line, status_text, CatalogEntry, ServerMessage};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Wrap};
use ratatui::Frame;
use serde_json::Value;
use theme::{
    BG_BASE, BG_LIGHT, FG, FG_SECONDARY, GRAY, GRAY_BRIGHT, MAGENTA, PROMPT_BORDER,
    PROMPT_BORDER_ACTIVE, RED, YELLOW,
};

const TICK: Duration = Duration::from_millis(50);

struct Config {
    workspace: PathBuf,
    repo: PathBuf,
}

fn parse_args() -> Result<Config> {
    let mut workspace: Option<PathBuf> = None;
    let mut repo: Option<PathBuf> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--workspace" => {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow!("--workspace needs a path"))?;
                workspace = Some(PathBuf::from(value));
            }
            "--repo" => {
                let value = args.next().ok_or_else(|| anyhow!("--repo needs a path"))?;
                repo = Some(PathBuf::from(value));
            }
            "--help" | "-h" => {
                println!("usage: atlas-tui [--workspace PATH] [--repo PATH]");
                std::process::exit(0);
            }
            other => return Err(anyhow!("unknown argument: {other}")),
        }
    }

    let cwd = std::env::current_dir().context("could not read the current directory")?;
    let workspace = workspace.unwrap_or_else(|| cwd.clone());
    let workspace = workspace.canonicalize().unwrap_or(workspace).to_path_buf();

    let repo = match repo {
        Some(path) => path,
        None => find_repo_root(&cwd)
            .ok_or_else(|| anyhow!("could not find the Atlas repo root; pass --repo PATH"))?,
    };

    Ok(Config { workspace, repo })
}

fn find_repo_root(start: &Path) -> Option<PathBuf> {
    for directory in start.ancestors() {
        let candidate = directory.join("pyproject.toml");
        if let Ok(text) = std::fs::read_to_string(&candidate) {
            if text.contains("name = \"atlas\"") {
                return Some(directory.to_path_buf());
            }
        }
    }
    None
}

struct ChildProcess {
    child: Child,
    stdin: ChildStdin,
}

impl ChildProcess {
    fn spawn(config: &Config) -> Result<Self> {
        let mut child = Command::new("uv")
            .args(["run", "python", "-m", "atlas.agent", "--workspace"])
            .arg(&config.workspace)
            .current_dir(&config.repo)
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("failed to start `uv run python -m atlas.agent`")?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("no child stdin"))?;
        Ok(Self { child, stdin })
    }

    fn send(&mut self, value: serde_json::Value) -> Result<()> {
        let line = serde_json::to_string(&value)?;
        self.stdin.write_all(line.as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        Ok(())
    }

    fn shutdown(&mut self) {
        let _ = self.send(serde_json::json!({"type": "quit"}));
        for _ in 0..20 {
            if let Ok(Some(_)) = self.child.try_wait() {
                return;
            }
            thread::sleep(Duration::from_millis(50));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for ChildProcess {
    fn drop(&mut self) {
        if !matches!(self.child.try_wait(), Ok(Some(_))) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

#[derive(PartialEq, Eq)]
enum Status {
    Ready,
    Waiting,
    Stopped,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Mode {
    Compose,
    ModelWait,
    ModelPicker,
    Key { scope: String },
}

struct App {
    tools: Vec<String>,
    transcript: Vec<TranscriptLine>,
    input: String,
    status: Status,
    /// The in-flight request is a resume, so a failure should stay on the limit.
    waiting_from_stopped: bool,
    session_open: bool,
    last_stderr: Option<String>,
    scroll: u16,
    follow: bool,
    status_line: String,
    model_id: String,
    mode: Mode,
    picker_filter: String,
    picker_index: usize,
    picker_models: Vec<CatalogEntry>,
    /// Highlighted row in the slash menu; reset when the filter token changes.
    menu_index: usize,
    /// `project` or `user`. Listing stays inside that root.
    scope_root: String,
    /// Ask the session for the catalog once so the dashboard can show the
    /// active model's context window without opening the picker.
    lookup_context: bool,
    /// Last wrapped row the transcript can scroll to. Updated while drawing.
    scroll_max: u16,
    ticks: u32,
    activity: Option<Activity>,
    /// Transcript index of the tool row Left and Right act on.
    selected: Option<usize>,
    /// Scroll the selected tool's header into view on the next draw.
    reveal_selected: bool,
    /// Tool rows painted in the last frame, in screen coordinates.
    tool_hits: Vec<ToolHit>,
    /// Last positive output rate from the session, in tokens per second.
    tokens_per_second: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Activity {
    Thinking,
    Working,
    ToolCalling(String),
    ToolExecuting(String),
}

struct ToolHit {
    y: u16,
    height: u16,
    index: usize,
}

struct ToolLine {
    label: String,
    /// Structured lines shown while the row is expanded. Empty rows stay one line.
    body: Vec<String>,
    expanded: bool,
}

struct ThoughtLine {
    stream: StreamText,
    /// User opened the finished text. Starts false.
    expanded: bool,
}

/// Pace used when a completion did not report a positive token rate.
const STREAM_FALLBACK_TPS: f64 = 24.0;

struct StreamText {
    text: String,
    /// Characters revealed so far.
    shown: usize,
    /// Output tokens revealed per second.
    tps: f64,
    /// Fractional token left over from the last tick.
    credit: f64,
}

impl StreamText {
    fn fresh(text: impl Into<String>, tps: f64) -> Self {
        let tps = if tps > 0.0 { tps } else { STREAM_FALLBACK_TPS };
        Self {
            text: text.into(),
            shown: 0,
            tps,
            credit: 0.0,
        }
    }

    fn settled(text: impl Into<String>) -> Self {
        let text = text.into();
        let shown = text.chars().count();
        Self {
            text,
            shown,
            tps: 0.0,
            credit: 0.0,
        }
    }

    fn done(&self) -> bool {
        self.shown >= self.text.chars().count()
    }

    fn revealed(&self) -> String {
        self.text.chars().take(self.shown).collect()
    }
}

enum TranscriptLine {
    User(String),
    Assistant(StreamText),
    Thought(ThoughtLine),
    Tool(ToolLine),
}

fn tool_note(label: impl Into<String>) -> TranscriptLine {
    TranscriptLine::Tool(ToolLine {
        label: label.into(),
        body: Vec::new(),
        expanded: false,
    })
}

fn tool_body(
    arguments: &Value,
    text: Option<&str>,
    error: Option<&str>,
    error_kind: Option<&str>,
    artifacts: &[String],
) -> Vec<String> {
    let mut lines = Vec::new();
    let args = format_arguments(arguments);
    if !args.is_empty() {
        lines.push("arguments".to_string());
        for line in args.lines() {
            lines.push(format!("  {line}"));
        }
    }
    if let Some(err) = error.filter(|item| !item.is_empty()) {
        lines.push("error".to_string());
        if let Some(kind) = error_kind.filter(|item| !item.is_empty()) {
            lines.push(format!("  kind: {kind}"));
        }
        for line in err.lines() {
            lines.push(format!("  {line}"));
        }
    } else {
        lines.push("result".to_string());
        let body = text.filter(|item| !item.is_empty()).unwrap_or("(empty)");
        for line in body.lines() {
            lines.push(format!("  {line}"));
        }
    }
    if !artifacts.is_empty() {
        lines.push("artifacts".to_string());
        for name in artifacts {
            lines.push(format!("  {name}"));
        }
    }
    lines
}

fn format_arguments(value: &Value) -> String {
    let Value::Object(map) = value else {
        if value.is_null() {
            return String::new();
        }
        return serde_json::to_string_pretty(value).unwrap_or_default();
    };
    if map.is_empty() {
        return String::new();
    }
    let mut lines = Vec::new();
    for (key, item) in map {
        match item {
            Value::String(text) => lines.push(format!("{key}: {text}")),
            Value::Null => lines.push(format!("{key}: null")),
            Value::Bool(flag) => lines.push(format!("{key}: {flag}")),
            Value::Number(number) => lines.push(format!("{key}: {number}")),
            other => {
                let pretty = serde_json::to_string_pretty(other).unwrap_or_default();
                if pretty.contains('\n') {
                    lines.push(format!("{key}:"));
                    for line in pretty.lines() {
                        lines.push(format!("  {line}"));
                    }
                } else {
                    lines.push(format!("{key}: {pretty}"));
                }
            }
        }
    }
    lines.join("\n")
}

impl App {
    fn new() -> Self {
        Self {
            tools: Vec::new(),
            transcript: Vec::new(),
            input: String::new(),
            status: Status::Ready,
            waiting_from_stopped: false,
            session_open: true,
            last_stderr: None,
            scroll: 0,
            follow: true,
            status_line: default_status_line(),
            model_id: String::new(),
            mode: Mode::Compose,
            picker_filter: String::new(),
            picker_index: 0,
            picker_models: Vec::new(),
            menu_index: 0,
            scope_root: "project".to_string(),
            lookup_context: false,
            scroll_max: 0,
            ticks: 0,
            activity: None,
            selected: None,
            reveal_selected: false,
            tool_hits: Vec::new(),
            tokens_per_second: 0.0,
        }
    }

    fn in_flight(&self) -> bool {
        matches!(self.status, Status::Waiting)
    }

    fn apply(&mut self, message: ServerMessage) {
        match message {
            ServerMessage::Ready { tools, status, .. } => {
                self.tools = tools.into_iter().map(|tool| tool.name).collect();
                self.adopt_status(&status);
                if status.context_limit.is_none() {
                    self.lookup_context = true;
                }
            }
            ServerMessage::Step {
                name,
                ok,
                arguments,
                text,
                error,
                error_kind,
                artifacts,
                ..
            } => {
                let label = if ok { name } else { format!("{name} failed") };
                let body = tool_body(
                    &arguments,
                    text.as_deref(),
                    error.as_deref(),
                    error_kind.as_deref(),
                    &artifacts,
                );
                self.transcript.push(TranscriptLine::Tool(ToolLine {
                    label,
                    body,
                    expanded: false,
                }));
                self.selected = Some(self.transcript.len() - 1);
            }
            ServerMessage::Thought {
                text,
                tokens_per_second,
            } => {
                let text = text.trim();
                if !text.is_empty() {
                    let tps = positive_tps(tokens_per_second).unwrap_or(self.tokens_per_second);
                    self.transcript.push(TranscriptLine::Thought(ThoughtLine {
                        stream: StreamText::fresh(text, tps),
                        expanded: false,
                    }));
                    self.selected = Some(self.transcript.len() - 1);
                }
            }
            ServerMessage::Phase { phase, name } => {
                self.activity = activity_from(&phase, name.as_deref());
            }
            ServerMessage::Done {
                response,
                stopped_for_limit,
                status,
            } => {
                self.adopt_status(&status);
                if !response.is_empty() {
                    let tps =
                        positive_tps(status.tokens_per_second).unwrap_or(self.tokens_per_second);
                    self.transcript
                        .push(TranscriptLine::Assistant(StreamText::fresh(response, tps)));
                }
                self.activity = None;
                self.waiting_from_stopped = false;
                self.status = if stopped_for_limit {
                    Status::Stopped
                } else {
                    Status::Ready
                };
            }
            ServerMessage::Error {
                message,
                status_line,
                ..
            } => {
                self.transcript
                    .push(TranscriptLine::Assistant(StreamText::settled(format!(
                        "error: {message}"
                    ))));
                if !status_line.is_empty() {
                    self.status_line = status_line;
                }
                self.activity = None;
                self.status = if self.waiting_from_stopped {
                    Status::Stopped
                } else {
                    Status::Ready
                };
                self.waiting_from_stopped = false;
                self.follow = true;
            }
            ServerMessage::Models {
                ok,
                message,
                models,
                status,
            } => {
                if !status.status_line.is_empty() {
                    self.adopt_status(&status);
                }
                let user_asked = matches!(self.mode, Mode::ModelWait | Mode::ModelPicker);
                if !ok {
                    if user_asked {
                        self.transcript
                            .push(tool_note(format!("model list failed: {message}")));
                        if self.mode == Mode::ModelWait {
                            self.mode = Mode::Compose;
                        }
                    }
                } else if user_asked {
                    self.picker_models = models;
                    self.picker_index = 0;
                    self.mode = Mode::ModelPicker;
                    let count = self.filtered_models().len();
                    self.transcript.push(tool_note(format!("models: {count}")));
                } else {
                    self.picker_models = models;
                }
            }
            ServerMessage::ModelSelected { id, status, .. } => {
                self.model_id = id.clone();
                self.adopt_status(&status);
                self.mode = Mode::Compose;
                self.transcript.push(tool_note(format!("model {id}")));
            }
            ServerMessage::KeySet { scope, saved } => {
                self.mode = Mode::Compose;
                self.input.clear();
                let label = if saved { "saved" } else { "not saved" };
                self.transcript
                    .push(tool_note(format!("{scope} key {label}")));
            }
            ServerMessage::Listing { scope, kind, names } => {
                let body = if names.is_empty() {
                    "(none)".to_string()
                } else {
                    names.join(", ")
                };
                self.transcript
                    .push(tool_note(format!("{scope} {kind}: {body}")));
            }
            ServerMessage::Removed { scope, kind, name } => {
                self.mode = Mode::Compose;
                self.transcript
                    .push(tool_note(format!("removed {scope} {kind} {name}")));
            }
            ServerMessage::Unknown => {}
        }
    }

    fn adopt_status(&mut self, status: &protocol::StatusSnapshot) {
        if !status.model.is_empty() {
            self.model_id = status.model.clone();
        }
        if let Some(speed) = positive_tps(status.tokens_per_second) {
            self.tokens_per_second = speed;
        }
        self.status_line = status_text(status);
    }

    fn filtered_models(&self) -> Vec<&CatalogEntry> {
        let query = self.picker_filter.to_lowercase();
        self.picker_models
            .iter()
            .filter(|model| query.is_empty() || model.id.to_lowercase().contains(&query))
            .collect()
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("atlas-tui: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let config = parse_args()?;
    let mut child = ChildProcess::spawn(&config)?;

    let stdout = child
        .child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("no child stdout"))?;
    let stderr = child
        .child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("no child stderr"))?;
    let (tx, rx) = mpsc::channel::<String>();
    let (erx, errx) = mpsc::channel::<String>();

    thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if erx.send(line).is_err() {
                break;
            }
        }
    });

    let mut app = App::new();

    // Consume the first line before entering the UI so a missing API key or a
    // crashed child shows as a plain message rather than a half-started session.
    let first = match recv_first_line(&mut child, &rx) {
        Ok(line) => line,
        Err(err) => {
            let stderr = join_stderr(&errx);
            child.shutdown();
            if stderr.is_empty() {
                return Err(err);
            }
            return Err(err.context(stderr));
        }
    };
    match parse_line(&first) {
        Ok(Some(ServerMessage::Error { message, .. })) => {
            child.shutdown();
            println!("atlas-tui: {message}");
            println!("Press enter to exit.");
            wait_for_any_key();
            return Ok(());
        }
        Ok(Some(message)) => app.apply(message),
        Ok(None) => return Err(anyhow!("agent session sent an empty first line")),
        Err(_) => return Err(anyhow!("agent session sent a non-json first line: {first}")),
    }

    let mut terminal = enter_ui()?;
    let ui = UiGuard;
    let previous_hook = Arc::new(std::panic::take_hook());
    let hook_for_panic = Arc::clone(&previous_hook);
    std::panic::set_hook(Box::new(move |info| {
        leave_ui();
        hook_for_panic(info);
    }));

    let outcome = event_loop(&mut terminal, &mut app, &mut child, &rx, &errx);

    drop(ui);
    child.shutdown();
    outcome
}

fn enter_ui() -> Result<ratatui::DefaultTerminal> {
    enable_raw_mode().context("atlas-tui needs an interactive terminal")?;
    let mut stdout = io::stdout();
    // The alternate screen plus mouse capture keeps wheel events inside Atlas.
    // Without capture, the wheel scrolls the shell that was on screen before launch.
    if let Err(err) = execute!(
        stdout,
        EnterAlternateScreen,
        Clear(ClearType::All),
        EnableMouseCapture
    ) {
        let _ = disable_raw_mode();
        return Err(err).context("atlas-tui needs an interactive terminal");
    }
    match ratatui::Terminal::new(CrosstermBackend::new(stdout)) {
        Ok(terminal) => Ok(terminal),
        Err(err) => {
            leave_ui();
            Err(anyhow!(err)).context("atlas-tui needs an interactive terminal")
        }
    }
}

fn leave_ui() {
    let _ = execute!(io::stdout(), DisableMouseCapture, LeaveAlternateScreen);
    let _ = disable_raw_mode();
}

struct UiGuard;

impl Drop for UiGuard {
    fn drop(&mut self) {
        leave_ui();
    }
}

fn event_loop(
    terminal: &mut ratatui::DefaultTerminal,
    app: &mut App,
    child: &mut ChildProcess,
    rx: &Receiver<String>,
    errx: &Receiver<String>,
) -> Result<()> {
    let mut streamed_at = Instant::now();
    loop {
        let now = Instant::now();
        let dt = now.saturating_duration_since(streamed_at).as_secs_f64();
        streamed_at = now;
        advance_streams(app, dt);
        terminal.draw(|frame| draw(frame, app))?;

        let mut stdout_closed = false;
        loop {
            match rx.try_recv() {
                Ok(line) => match parse_line(&line) {
                    Ok(Some(message)) => app.apply(message),
                    Ok(None) => {}
                    Err(_) => app
                        .transcript
                        .push(tool_note(format!("unparsable: {line}"))),
                },
                Err(mpsc::TryRecvError::Empty) => {
                    if app.lookup_context {
                        app.lookup_context = false;
                        child.send(serde_json::json!({"type": "models"}))?;
                    }
                    break;
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    stdout_closed = true;
                    break;
                }
            }
        }
        while let Ok(line) = errx.try_recv() {
            app.last_stderr = Some(line);
        }
        if stdout_closed {
            note_session_closed(app);
        }

        if event::poll(TICK)? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    if handle_key(app, child, key)? {
                        return Ok(());
                    }
                }
                Event::Mouse(mouse) => match mouse.kind {
                    MouseEventKind::ScrollUp => nudge_scroll(app, -3),
                    MouseEventKind::ScrollDown => nudge_scroll(app, 3),
                    MouseEventKind::Down(MouseButton::Left) => select_tool_at(app, mouse.row),
                    _ => {}
                },
                _ => {}
            }
        }
        app.ticks = app.ticks.wrapping_add(1);
    }
}

/// Returns `true` when the user asked to quit.
fn handle_key(app: &mut App, child: &mut ChildProcess, key: KeyEvent) -> Result<bool> {
    if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
        child.shutdown();
        return Ok(true);
    }
    if key.code == KeyCode::Esc && matches!(app.mode, Mode::Compose) && !app.input.starts_with('/')
    {
        child.shutdown();
        return Ok(true);
    }
    if key.code == KeyCode::Esc {
        app.mode = Mode::Compose;
        app.input.clear();
        return Ok(false);
    }
    let mode = app.mode.clone();
    match mode {
        Mode::ModelPicker => handle_picker_key(app, child, key)?,
        Mode::Key { scope } => {
            handle_secret_key(app, child, key, &scope)?;
        }
        Mode::ModelWait => {}
        Mode::Compose => {
            if handle_compose_key(app, child, key)? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn handle_compose_key(app: &mut App, child: &mut ChildProcess, key: KeyEvent) -> Result<bool> {
    let menu_open = !menu_options(&app.input).is_empty();
    match key.code {
        KeyCode::Char('r') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            if app.session_open && app.status == Status::Stopped && !app.in_flight() {
                child.send(serde_json::json!({"type": "resume"}))?;
                app.waiting_from_stopped = true;
                app.status = Status::Waiting;
            }
        }
        KeyCode::Up if menu_open && plain_arrow(key.modifiers) => {
            app.menu_index = app.menu_index.saturating_sub(1);
        }
        KeyCode::Down if menu_open && plain_arrow(key.modifiers) => {
            let count = menu_options(&app.input).len();
            if count > 0 && app.menu_index + 1 < count {
                app.menu_index += 1;
            }
        }
        KeyCode::Up if tool_select_modifier(key.modifiers) => select_tool(app, -1),
        KeyCode::Down if tool_select_modifier(key.modifiers) => select_tool(app, 1),
        KeyCode::Up => nudge_scroll(app, -1),
        KeyCode::Down => nudge_scroll(app, 1),
        KeyCode::Left => collapse_selected(app),
        KeyCode::Right => expand_selected(app),
        KeyCode::Tab if menu_open => {
            if let MenuAction::Complete(prefix, replacement) =
                menu_action(&app.input, app.menu_index, false)
            {
                apply_completion(app, &prefix, &replacement);
            }
        }
        KeyCode::Enter => {
            if app.session_open && !app.input.is_empty() && !app.in_flight() {
                if app.input.starts_with('/') {
                    let text = app.input.clone();
                    match menu_action(&text, app.menu_index, true) {
                        MenuAction::Run(run) => {
                            app.input.clear();
                            if is_quit_command(&run) {
                                child.shutdown();
                                return Ok(true);
                            }
                            if !run_local_command(app, &run) {
                                handle_slash(app, child, &run)?;
                            }
                        }
                        MenuAction::Complete(prefix, replacement) => {
                            apply_completion(app, &prefix, &replacement);
                        }
                        MenuAction::None => {
                            // No rows (unknown command or free text): keep the
                            // existing slash behavior.
                            app.input.clear();
                            if is_quit_command(&text) {
                                child.shutdown();
                                return Ok(true);
                            }
                            if !run_local_command(app, &text) {
                                handle_slash(app, child, &text)?;
                            }
                        }
                    }
                } else {
                    let text = std::mem::take(&mut app.input);
                    finish_streams(app);
                    child.send(serde_json::json!({"type": "user", "text": text}))?;
                    app.transcript.push(TranscriptLine::User(text));
                    app.waiting_from_stopped = false;
                    app.status = Status::Waiting;
                    app.follow = true;
                }
            }
        }
        KeyCode::Backspace => {
            rubout(&mut app.input, key.modifiers);
            app.menu_index = 0;
        }
        KeyCode::Char('w') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            delete_word_backward(&mut app.input);
            app.menu_index = 0;
        }
        KeyCode::PageUp => nudge_scroll(app, -5),
        KeyCode::PageDown => nudge_scroll(app, 5),
        KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.input.push(ch);
            app.menu_index = 0;
        }
        _ => {}
    }
    Ok(false)
}

fn is_quit_command(text: &str) -> bool {
    matches!(text.trim(), "/quit" | "/exit")
}

fn run_local_command(app: &mut App, text: &str) -> bool {
    if text.trim() != "/copy" {
        return false;
    }
    match last_reply(app) {
        Some(reply) => {
            let note = if copy_to_clipboard(reply) {
                "copied"
            } else {
                "copy failed"
            };
            app.transcript.push(tool_note(note.to_string()));
        }
        None => {
            app.transcript.push(tool_note("nothing to copy"));
        }
    }
    app.follow = true;
    true
}

fn last_reply(app: &App) -> Option<&str> {
    app.transcript.iter().rev().find_map(|line| match line {
        TranscriptLine::Assistant(stream) if !stream.text.is_empty() => Some(stream.text.as_str()),
        _ => None,
    })
}

fn copy_to_clipboard(text: &str) -> bool {
    for (program, args) in [
        ("pbcopy", &[][..]),
        ("wl-copy", &[]),
        ("xclip", &["-selection", "clipboard"]),
        ("clip", &[]),
    ] {
        let Ok(mut child) = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        else {
            continue;
        };
        let Some(mut stdin) = child.stdin.take() else {
            continue;
        };
        if stdin.write_all(text.as_bytes()).is_err() {
            continue;
        }
        drop(stdin);
        if child.wait().map(|status| status.success()).unwrap_or(false) {
            return true;
        }
    }
    false
}

fn plain_arrow(modifiers: KeyModifiers) -> bool {
    !modifiers.contains(KeyModifiers::SHIFT)
        && !modifiers.contains(KeyModifiers::ALT)
        && !modifiers.contains(KeyModifiers::CONTROL)
        && !modifiers.contains(KeyModifiers::META)
}

fn tool_select_modifier(modifiers: KeyModifiers) -> bool {
    modifiers.contains(KeyModifiers::SHIFT)
        || modifiers.contains(KeyModifiers::ALT)
        || modifiers.contains(KeyModifiers::META)
}

fn word_modifier(modifiers: KeyModifiers) -> bool {
    modifiers.contains(KeyModifiers::ALT) || modifiers.contains(KeyModifiers::META)
}

/// Delete one character, or the previous word on Option-Backspace.
fn rubout(text: &mut String, modifiers: KeyModifiers) {
    if word_modifier(modifiers) {
        delete_word_backward(text);
    } else {
        text.pop();
    }
}

/// Drop the word before the cursor. Trailing spaces go with that word.
fn delete_word_backward(text: &mut String) {
    let mut chars: Vec<char> = text.chars().collect();
    while chars.last().is_some_and(|ch| ch.is_whitespace()) {
        chars.pop();
    }
    while chars.last().is_some_and(|ch| !ch.is_whitespace()) {
        chars.pop();
    }
    *text = chars.into_iter().collect();
}

fn expandable_tools(app: &App) -> Vec<usize> {
    app.transcript
        .iter()
        .enumerate()
        .filter_map(|(index, line)| match line {
            TranscriptLine::Tool(tool) if !tool.body.is_empty() => Some(index),
            TranscriptLine::Thought(_) => Some(index),
            _ => None,
        })
        .collect()
}

fn select_tool(app: &mut App, delta: isize) {
    let tools = expandable_tools(app);
    if tools.is_empty() {
        return;
    }
    let next = match app
        .selected
        .and_then(|current| tools.iter().position(|index| *index == current))
    {
        Some(pos) => {
            let pos = if delta < 0 {
                pos.saturating_sub(1)
            } else {
                (pos + 1).min(tools.len() - 1)
            };
            tools[pos]
        }
        None if delta < 0 => tools[tools.len() - 1],
        None => tools[0],
    };
    app.selected = Some(next);
    app.reveal_selected = true;
}

fn select_tool_at(app: &mut App, row: u16) {
    if let Some(hit) = app
        .tool_hits
        .iter()
        .find(|hit| row >= hit.y && row < hit.y + hit.height)
    {
        app.selected = Some(hit.index);
    }
}

fn expand_selected(app: &mut App) {
    let Some(index) = app.selected else {
        return;
    };
    match app.transcript.get_mut(index) {
        Some(TranscriptLine::Tool(tool)) => {
            if tool.body.is_empty() || tool.expanded {
                return;
            }
            tool.expanded = true;
        }
        Some(TranscriptLine::Thought(thought)) => {
            if thought.expanded {
                return;
            }
            thought.expanded = true;
        }
        _ => return,
    }
    app.reveal_selected = true;
}

fn collapse_selected(app: &mut App) {
    let Some(index) = app.selected else {
        return;
    };
    match app.transcript.get_mut(index) {
        Some(TranscriptLine::Tool(tool)) => {
            if !tool.expanded {
                return;
            }
            tool.expanded = false;
        }
        Some(TranscriptLine::Thought(thought)) => {
            if !thought.expanded {
                return;
            }
            thought.expanded = false;
        }
        _ => return,
    }
    app.reveal_selected = true;
}

fn positive_tps(speed: Option<f64>) -> Option<f64> {
    speed.filter(|value| value.is_finite() && *value > 0.0)
}

/// End of the next display token at or after `shown`, including the spaces after it.
fn next_token_end(text: &str, shown: usize) -> Option<usize> {
    let chars: Vec<char> = text.chars().skip(shown).collect();
    if chars.is_empty() {
        return None;
    }
    let mut end = 0;
    if chars[0].is_whitespace() {
        while end < chars.len() && chars[end].is_whitespace() {
            end += 1;
        }
        return Some(shown + end);
    }
    end = 1;
    if chars[0].is_alphanumeric() {
        while end < chars.len()
            && (chars[end].is_alphanumeric() || matches!(chars[end], '\'' | '_' | '-'))
        {
            end += 1;
        }
    } else {
        let mark = chars[0];
        while end < chars.len() && chars[end] == mark {
            end += 1;
        }
    }
    while end < chars.len() && chars[end].is_whitespace() {
        end += 1;
    }
    Some(shown + end)
}

fn advance_streams(app: &mut App, dt: f64) {
    if dt <= 0.0 {
        return;
    }
    let mut remaining = dt;
    while remaining > 0.0 {
        let Some(stream) = app.transcript.iter_mut().find_map(|line| match line {
            TranscriptLine::Assistant(stream) if !stream.done() => Some(stream),
            TranscriptLine::Thought(thought) if !thought.stream.done() => Some(&mut thought.stream),
            _ => None,
        }) else {
            return;
        };
        let speed = stream.tps;
        if speed <= 0.0 {
            stream.shown = stream.text.chars().count();
            stream.credit = 0.0;
            continue;
        }
        let carried = stream.credit;
        stream.credit += speed * remaining;
        let mut revealed = 0.0;
        while stream.credit >= 1.0 && !stream.done() {
            let Some(end) = next_token_end(&stream.text, stream.shown) else {
                stream.shown = stream.text.chars().count();
                break;
            };
            if end <= stream.shown {
                stream.shown = stream.text.chars().count();
                break;
            }
            stream.shown = end;
            stream.credit -= 1.0;
            revealed += 1.0;
        }
        if !stream.done() {
            return;
        }
        let used = ((revealed - carried) / speed).clamp(0.0, remaining);
        stream.credit = 0.0;
        if used <= 1e-9 {
            return;
        }
        remaining -= used;
    }
}

fn finish_streams(app: &mut App) {
    for line in &mut app.transcript {
        match line {
            TranscriptLine::Assistant(stream) => {
                stream.shown = stream.text.chars().count();
                stream.credit = 0.0;
            }
            TranscriptLine::Thought(thought) => {
                thought.stream.shown = thought.stream.text.chars().count();
                thought.stream.credit = 0.0;
            }
            _ => {}
        }
    }
}

fn nudge_scroll(app: &mut App, delta: i32) {
    if delta < 0 {
        let step = delta.unsigned_abs().min(u32::from(u16::MAX)) as u16;
        app.follow = false;
        app.scroll = app.scroll.saturating_sub(step);
        return;
    }
    let step = u16::try_from(delta).unwrap_or(u16::MAX);
    let next = app.scroll.saturating_add(step);
    if next >= app.scroll_max {
        app.scroll = app.scroll_max;
        app.follow = true;
    } else {
        app.follow = false;
        app.scroll = next;
    }
}

fn activity_from(phase: &str, name: Option<&str>) -> Option<Activity> {
    let name = name.unwrap_or("").to_string();
    match phase {
        "thinking" => Some(Activity::Thinking),
        "working" => Some(Activity::Working),
        "tool_calling" => Some(Activity::ToolCalling(name)),
        "tool_executing" => Some(Activity::ToolExecuting(name)),
        _ => None,
    }
}

fn apply_completion(app: &mut App, prefix: &str, replacement: &str) {
    if let Some(rest) = app.input.strip_prefix(prefix) {
        app.input = format!("{replacement}{rest}");
    }
    app.menu_index = 0;
}

fn handle_picker_key(app: &mut App, child: &mut ChildProcess, key: KeyEvent) -> Result<()> {
    let count = app.filtered_models().len();
    match key.code {
        KeyCode::Up => {
            if count > 0 {
                app.picker_index = app.picker_index.saturating_sub(1);
            }
        }
        KeyCode::Down => {
            if count > 0 && app.picker_index + 1 < count {
                app.picker_index += 1;
            }
        }
        KeyCode::Backspace => {
            rubout(&mut app.picker_filter, key.modifiers);
            app.picker_index = 0;
        }
        KeyCode::Char('w') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            delete_word_backward(&mut app.picker_filter);
            app.picker_index = 0;
        }
        KeyCode::Enter => {
            if let Some(payload) = picker_selection(app) {
                child.send(payload)?;
            }
        }
        KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.picker_filter.push(ch);
            app.picker_index = 0;
        }
        _ => {}
    }
    Ok(())
}

fn handle_secret_key(
    app: &mut App,
    child: &mut ChildProcess,
    key: KeyEvent,
    scope: &str,
) -> Result<()> {
    match key.code {
        KeyCode::Backspace => {
            rubout(&mut app.input, key.modifiers);
        }
        KeyCode::Char('w') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            delete_word_backward(&mut app.input);
        }
        KeyCode::Enter => {
            if !app.input.is_empty() {
                let secret = std::mem::take(&mut app.input);
                child.send(serde_json::json!({
                    "type": "set_key",
                    "scope": scope,
                    "key": secret
                }))?;
                // The secret is not copied into the transcript.
            }
        }
        KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.input.push(ch);
        }
        _ => {}
    }
    Ok(())
}

fn handle_slash(app: &mut App, child: &mut ChildProcess, text: &str) -> Result<()> {
    if let Some(payload) = slash_outcome(app, text) {
        child.send(payload)?;
    }
    Ok(())
}

/// Apply a slash command and return the protocol line to send, if any.
///
/// `/model` only asks for the catalog.
fn slash_outcome(app: &mut App, text: &str) -> Option<serde_json::Value> {
    let mut parts = text.split_whitespace();
    let command = parts.next().unwrap_or("");
    match command {
        "/new" => {
            if parts.next().is_some() {
                app.transcript.push(tool_note("usage: /new"));
                None
            } else {
                app.transcript.clear();
                app.status = Status::Ready;
                app.waiting_from_stopped = false;
                app.activity = None;
                app.selected = None;
                app.reveal_selected = false;
                app.scroll = 0;
                app.follow = true;
                app.tokens_per_second = 0.0;
                Some(serde_json::json!({"type": "new"}))
            }
        }
        "/model" => {
            let query = text.trim().trim_start_matches("/model").trim();
            app.picker_filter = query.to_string();
            app.picker_index = 0;
            app.mode = Mode::ModelWait;
            let suffix = if query.is_empty() {
                String::new()
            } else {
                format!(" matching {query}")
            };
            app.transcript
                .push(tool_note(format!("fetching models{suffix}")));
            Some(serde_json::json!({"type": "models"}))
        }
        "/key" => {
            let scope = parts.next().unwrap_or("");
            if scope == "session" || scope == "user" {
                app.mode = Mode::Key {
                    scope: scope.to_string(),
                };
                app.transcript.push(tool_note(format!(
                    "Enter the OpenRouter key for this {scope}. It stays hidden."
                )));
            } else {
                app.transcript
                    .push(tool_note("usage: /key session | /key user"));
            }
            None
        }
        "/sessions" => Some(serde_json::json!({
            "type": "list",
            "scope": app.scope_root,
            "kind": "sessions"
        })),
        "/artifacts" => Some(serde_json::json!({
            "type": "list",
            "scope": app.scope_root,
            "kind": "artifacts"
        })),
        _ => {
            app.transcript
                .push(tool_note(format!("unknown command {command}")));
            None
        }
    }
}

const PICKER_PAGE: usize = 8;

/// Slice of the filtered catalog that stays on screen around `index`.
fn picker_window(len: usize, index: usize, page: usize) -> std::ops::Range<usize> {
    if len == 0 || page == 0 {
        return 0..0;
    }
    let page = page.min(len);
    let index = index.min(len - 1);
    let start = index.saturating_add(1).saturating_sub(page).min(len - page);
    start..(start + page)
}

const MENU_HEADER: &str = "↑↓ move  tab complete  enter run  esc close";

#[derive(Clone, Copy)]
struct MenuItem {
    token: &'static str,
    description: &'static str,
}

/// Command or fixed-argument options that match `text`, or an empty list when
/// the menu should stay hidden (free text after a finished command).
fn menu_options(text: &str) -> Vec<MenuItem> {
    const COMMANDS: [MenuItem; 8] = [
        MenuItem {
            token: "/new",
            description: "Start a fresh conversation",
        },
        MenuItem {
            token: "/model",
            description: "Choose a model",
        },
        MenuItem {
            token: "/copy",
            description: "Copy the last reply",
        },
        MenuItem {
            token: "/key",
            description: "Set an OpenRouter key",
        },
        MenuItem {
            token: "/sessions",
            description: "List sessions",
        },
        MenuItem {
            token: "/artifacts",
            description: "List artifacts",
        },
        MenuItem {
            token: "/quit",
            description: "Quit Atlas",
        },
        MenuItem {
            token: "/exit",
            description: "Quit Atlas",
        },
    ];
    const KEY_ARGS: [MenuItem; 2] = [
        MenuItem {
            token: "session",
            description: "Key for this session",
        },
        MenuItem {
            token: "user",
            description: "Saved for this user",
        },
    ];

    if !text.starts_with('/') {
        return Vec::new();
    }
    let mut words = text.split_whitespace();
    let command = words.next().unwrap_or("");
    let rest: Vec<&str> = words.collect();

    if !text.contains(char::is_whitespace) {
        // One command token: filter commands whose name contains it.
        let query = command.to_lowercase();
        return COMMANDS
            .into_iter()
            .filter(|item| item.token.contains(&query))
            .collect();
    }
    // Once a fixed set is exhausted and free text follows, show nothing.
    if !rest.is_empty() && text.ends_with(char::is_whitespace) {
        return Vec::new();
    }

    match command {
        "/model" => Vec::new(),
        "/sessions" | "/artifacts" => Vec::new(),
        "/key" => {
            if rest.len() > 1 {
                Vec::new()
            } else {
                filter_args(&KEY_ARGS, rest.first().copied().unwrap_or(""))
            }
        }
        _ => Vec::new(),
    }
}

fn filter_args(options: &[MenuItem], query: &str) -> Vec<MenuItem> {
    let query = query.to_lowercase();
    options
        .iter()
        .filter(|item| item.token.contains(&query))
        .copied()
        .collect()
}

/// Text the active token occupies, replaced wholesale on completion.
fn menu_prefix(text: &str) -> Option<String> {
    if !text.starts_with('/') {
        return None;
    }
    if !text.contains(char::is_whitespace) {
        return Some(text.to_string());
    }
    let (head, tail) = text.rsplit_once(' ')?;
    if tail.contains(char::is_whitespace) {
        return None;
    }
    Some(format!("{head} {tail}"))
}

/// The line to write when the highlighted `option` is accepted, or `None`.
fn menu_completion(text: &str, option: &str) -> Option<String> {
    if !text.starts_with('/') {
        return None;
    }
    if option.starts_with('/') {
        // Command options only apply while one command token is being typed.
        if text.contains(char::is_whitespace) {
            return None;
        }
        return Some(option.to_string());
    }
    let (head, _tail) = text.rsplit_once(' ')?;
    Some(format!("{head} {option}"))
}

/// What Enter or Tab should do while the slash menu is visible.
#[derive(Debug, PartialEq, Eq)]
enum MenuAction {
    /// Run `text` through `slash_outcome`.
    Run(String),
    /// Replace `prefix` with `replacement`, then leave the menu open.
    Complete(String, String),
    /// Do nothing: no completion is possible.
    None,
}

/// Decide Enter (`run`) or Tab for the highlighted menu row.
fn menu_action(text: &str, index: usize, run: bool) -> MenuAction {
    let options = menu_options(text);
    if options.is_empty() {
        return MenuAction::None;
    }
    if is_finished_command(text) {
        return if run {
            MenuAction::Run(text.to_string())
        } else {
            MenuAction::None
        };
    }

    let option = options[index.min(options.len() - 1)].token;
    let Some(prefix) = menu_prefix(text) else {
        return MenuAction::None;
    };
    let Some(replacement) = menu_completion(text, option) else {
        return MenuAction::None;
    };
    let finished = is_finished_command(&replacement);

    // Enter accepts the highlighted row once it is a command slash_outcome can run.
    if run && finished {
        return MenuAction::Run(replacement);
    }
    // Otherwise complete the token, leaving a space when more is expected.
    let replacement = if finished {
        replacement
    } else {
        format!("{replacement} ")
    };
    if replacement == text {
        return MenuAction::None;
    }
    MenuAction::Complete(prefix, replacement)
}

/// Commands `slash_outcome` accepts without showing a usage error.
fn is_finished_command(text: &str) -> bool {
    let trimmed = text.trim_end();
    if matches!(
        trimmed,
        "/new" | "/sessions" | "/artifacts" | "/quit" | "/exit" | "/copy"
    ) {
        return true;
    }
    let mut words = trimmed.split_whitespace();
    let command = words.next().unwrap_or("");
    match command {
        "/model" => text.starts_with("/model"),
        "/key" => matches!(words.next(), Some("session") | Some("user")),
        _ => false,
    }
}

fn menu_rows(app: &App) -> u16 {
    if app.mode != Mode::Compose {
        return 0;
    }
    let options = menu_options(&app.input);
    if options.is_empty() {
        return 0;
    }
    let count = options.len().clamp(1, PICKER_PAGE) as u16;
    count.saturating_add(1)
}

fn render_menu(frame: &mut Frame, app: &App, area: Rect) {
    let options = menu_options(&app.input);
    let mut lines = vec![Line::from(Span::styled(
        MENU_HEADER,
        Style::default().fg(GRAY).bg(BG_BASE),
    ))];
    let window = picker_window(options.len(), app.menu_index, PICKER_PAGE);
    for (index, option) in options
        .iter()
        .enumerate()
        .skip(window.start)
        .take(window.len())
    {
        let selected = index == app.menu_index;
        let mark = if selected { ">" } else { " " };
        let style = if selected {
            Style::default().fg(FG).bg(BG_LIGHT)
        } else {
            Style::default().fg(FG_SECONDARY).bg(BG_BASE)
        };
        let content = menu_row_text(mark, option, area.width as usize);
        lines.push(Line::from(Span::styled(content, style)));
    }
    frame.render_widget(Paragraph::new(lines), area);
}

/// One menu row: mark, option on the left, description toward the right.
fn menu_row_text(mark: &str, option: &MenuItem, width: usize) -> String {
    let left = format!("{mark} {}", option.token);
    let description = option.description;
    let gap = width.saturating_sub(left.chars().count() + description.chars().count());
    let text = if gap >= 2 {
        format!("{left}{}{description}", " ".repeat(gap))
    } else {
        format!("{left}  {description}")
    };
    text.chars().take(width).collect()
}

fn picker_selection(app: &App) -> Option<serde_json::Value> {
    app.filtered_models()
        .get(app.picker_index)
        .map(|model| serde_json::json!({"type": "select_model", "id": model.id}))
}

/// Convert a per-token USD decimal string to USD per 1M tokens.
///
/// The decimal point moves six places by shifting digit characters, so values
/// like `0.0000000198` stay exact; multiplying the parsed `f64` would round it.
/// Returns `None` for empty, negative, or non-decimal text.
fn price_per_million(token_price: &str) -> Option<String> {
    let text = token_price.trim();
    let text = text.strip_prefix('+').unwrap_or(text);
    let (int_part, frac_part) = match text.split_once('.') {
        Some((int_part, frac_part)) => (int_part, frac_part),
        None => (text, ""),
    };
    if (int_part.is_empty() && frac_part.is_empty())
        || !int_part.bytes().all(|byte| byte.is_ascii_digit())
        || !frac_part.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }

    // Six fractional digits move across the decimal point; shorter fractions
    // pad with zeros, longer ones stay fractional.
    let mut frac = frac_part.to_string();
    while frac.len() < 6 {
        frac.push('0');
    }
    let integer = format!("{int_part}{}", &frac[..6]);
    let integer = integer.trim_start_matches('0');
    let integer = if integer.is_empty() { "0" } else { integer };
    let fraction = frac[6..].trim_end_matches('0');
    let result = if fraction.is_empty() {
        integer.to_string()
    } else {
        format!("{integer}.{fraction}")
    };
    Some(result)
}

/// Token count in K, M, or B, rounded to three significant figures.
///
/// `1_048_576` is `1.05M`, `128_000` is `128K`, `1_500_000_000` is `1.50B`,
/// and values under 1,000 stay plain integers.
fn format_context(tokens: i64) -> String {
    if tokens <= 0 {
        return "0".to_string();
    }
    let rounded = round_sig3(tokens as u64);
    if rounded >= 1_000_000_000 {
        format!("{}B", format_sig_coeff(rounded, 1_000_000_000))
    } else if rounded >= 1_000_000 {
        format!("{}M", format_sig_coeff(rounded, 1_000_000))
    } else if rounded >= 1_000 {
        format!("{}K", format_sig_coeff(rounded, 1_000))
    } else {
        rounded.to_string()
    }
}

/// `rounded` is already on a three-significant-figure boundary, so the
/// remainder against `unit` (1_000 or 1_000_000) is the fractional digits.
fn format_sig_coeff(rounded: u64, unit: u64) -> String {
    let whole = rounded / unit;
    let frac = rounded % unit;
    if whole >= 100 {
        whole.to_string()
    } else if whole >= 10 {
        let digit = frac / (unit / 10);
        format!("{whole}.{digit}")
    } else {
        let digits = frac / (unit / 100);
        format!("{whole}.{digits:02}")
    }
}

fn round_sig3(n: u64) -> u64 {
    let digits = decimal_digits(n);
    if digits <= 3 {
        return n;
    }
    let shift = digits - 3;
    let factor = 10u64.pow(shift);
    let head = n / factor;
    let rem = n % factor;
    let mut rounded = head + u64::from(rem >= factor / 2);
    let mut result_shift = shift;
    if rounded >= 1_000 {
        rounded /= 10;
        result_shift += 1;
    }
    rounded * 10u64.pow(result_shift)
}

fn decimal_digits(n: u64) -> u32 {
    let mut digits = 1;
    let mut value = n;
    while value >= 10 {
        value /= 10;
        digits += 1;
    }
    digits
}

/// `input $2/M` or `input n/a` for one price side.
fn price_field(label: &str, price: Option<&str>) -> String {
    match price.and_then(price_per_million) {
        Some(amount) => format!("{label} ${amount}/M"),
        None => format!("{label} n/a"),
    }
}

fn wait_for_any_key() {
    let _ = crossterm::terminal::disable_raw_mode();
    let stdin = std::io::stdin();
    let mut buffer = String::new();
    let _ = stdin.read_line(&mut buffer);
}

fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    frame.render_widget(Block::default().style(Style::default().bg(BG_BASE)), area);

    let outer = inset(area, 1, 2);
    if outer.width < 8 || outer.height < 4 {
        render_status(frame, app, outer);
        return;
    }

    let composer_height = composer_rows(app, outer.width);
    let picker_height = picker_rows(app);
    let activity_height = u16::from(app.in_flight());
    let status_height = status_block_rows(&dashboard_text(app), outer.width);
    let [transcript_area, activity_area, picker_area, composer_area, status_area] =
        Layout::vertical([
            Constraint::Min(1),
            Constraint::Length(activity_height),
            Constraint::Length(picker_height),
            Constraint::Length(composer_height),
            Constraint::Length(status_height),
        ])
        .areas(outer);

    render_transcript(frame, app, transcript_area);
    if app.transcript.is_empty() {
        logo::paint_logo(frame, transcript_area);
    }
    if activity_height > 0 {
        render_activity(frame, app, activity_area);
    }
    if picker_height > 0 {
        render_picker(frame, app, picker_area);
    }
    render_composer(frame, app, composer_area);
    render_status(frame, app, status_area);
}

/// Wrapped dashboard line, plus the hint row.
fn status_block_rows(dashboard: &str, width: u16) -> u16 {
    let text_width = usize::from(width.max(1));
    let lines = wrap_text(dashboard, text_width).len().clamp(1, 6) as u16;
    lines.saturating_add(1)
}

fn picker_rows(app: &App) -> u16 {
    match app.mode {
        Mode::ModelWait => 1,
        Mode::ModelPicker => {
            let count = app.filtered_models().len().clamp(1, PICKER_PAGE) as u16;
            count.saturating_add(1)
        }
        Mode::Compose => menu_rows(app),
        _ => 0,
    }
}

fn inset(area: Rect, vertical: u16, horizontal: u16) -> Rect {
    let vertical = vertical.min(area.height / 2);
    let horizontal = horizontal.min(area.width / 2);
    Rect {
        x: area.x.saturating_add(horizontal),
        y: area.y.saturating_add(vertical),
        width: area.width.saturating_sub(horizontal.saturating_mul(2)),
        height: area.height.saturating_sub(vertical.saturating_mul(2)),
    }
}

fn composer_rows(app: &App, outer_width: u16) -> u16 {
    let text_width = outer_width.saturating_sub(4).max(1) as usize;
    let body = if app.input.is_empty() {
        1
    } else {
        wrap_text(&app.input, text_width).len().clamp(1, 5)
    };
    (body as u16).saturating_add(2)
}

fn render_transcript(frame: &mut Frame, app: &mut App, area: Rect) {
    let rows = transcript_rows(app, area.width as usize);
    let height = area.height as usize;
    let max_scroll = rows.len().saturating_sub(height) as u16;
    app.scroll_max = max_scroll;
    if app.follow {
        app.scroll = max_scroll;
    }
    if app.reveal_selected {
        if let Some(header) = rows.iter().position(|painted| {
            painted.header
                && app
                    .selected
                    .is_some_and(|index| painted.tool == Some(index))
        }) {
            let start = usize::from(app.scroll);
            if header < start {
                app.scroll = u16::try_from(header).unwrap_or(u16::MAX);
                app.follow = false;
            } else if height > 0 && header >= start + height {
                app.scroll = u16::try_from(header + 1 - height).unwrap_or(0);
                app.follow = false;
            }
        }
        app.reveal_selected = false;
    }
    app.scroll = app.scroll.min(max_scroll);
    let start = app.scroll as usize;
    app.tool_hits.clear();
    for (offset, painted) in rows.iter().skip(start).take(height).enumerate() {
        let y = area
            .y
            .saturating_add(u16::try_from(offset).unwrap_or(u16::MAX));
        if let Some(index) = painted.tool {
            note_tool_hit(&mut app.tool_hits, y, index);
        }
        let row_area = Rect {
            x: area.x,
            y,
            width: area.width,
            height: 1,
        };
        paint_row(frame, row_area, &painted.row);
    }
}

fn note_tool_hit(hits: &mut Vec<ToolHit>, y: u16, index: usize) {
    if let Some(last) = hits.last_mut() {
        if last.index == index && last.y + last.height == y {
            last.height = last.height.saturating_add(1);
            return;
        }
    }
    hits.push(ToolHit {
        y,
        height: 1,
        index,
    });
}

struct Painted {
    tool: Option<usize>,
    header: bool,
    row: PaintedRow,
}

enum PaintedRow {
    Gap,
    Text {
        kind: RowKind,
        first: bool,
        text: String,
    },
    Rich {
        pieces: Vec<Piece>,
    },
}

fn blank_row(row: PaintedRow) -> Painted {
    Painted {
        tool: None,
        header: false,
        row,
    }
}

fn stream_visible(stream: &StreamText, caret: bool) -> String {
    let mut text = stream.revealed();
    if caret {
        text.push('▌');
    }
    text
}

fn transcript_rows(app: &App, width: usize) -> Vec<Painted> {
    let content_width = width.saturating_sub(2).max(1);
    let active = app.transcript.iter().position(|line| match line {
        TranscriptLine::Assistant(stream) => !stream.done(),
        TranscriptLine::Thought(thought) => !thought.stream.done(),
        _ => false,
    });
    let mut rows = Vec::new();
    for (index, line) in app.transcript.iter().enumerate() {
        if let TranscriptLine::Assistant(stream) = line {
            if !stream.done() && active != Some(index) {
                continue;
            }
        }
        if let TranscriptLine::Thought(thought) = line {
            if !thought.stream.done() && active != Some(index) {
                continue;
            }
        }
        if !rows.is_empty() {
            rows.push(blank_row(PaintedRow::Gap));
        }
        match line {
            TranscriptLine::Assistant(stream) => {
                let visible = stream_visible(stream, !stream.done());
                for rich in render_fitted(&visible, content_width) {
                    for pieces in wrap_rich(&rich, content_width) {
                        rows.push(blank_row(PaintedRow::Rich { pieces }));
                    }
                }
            }
            TranscriptLine::Thought(thought) => {
                let selected = app.selected == Some(index);
                let kind = match (selected, thought.expanded) {
                    (true, true) => RowKind::ThoughtOpenSelected,
                    (true, false) => RowKind::ThoughtSelected,
                    (false, true) => RowKind::ThoughtOpen,
                    (false, false) => RowKind::Thought,
                };
                push_wrapped(
                    &mut rows,
                    "thinking",
                    content_width,
                    kind,
                    Some(index),
                    true,
                );
                if thought.expanded {
                    if thought.stream.done() {
                        push_wrapped(
                            &mut rows,
                            &thought.stream.text,
                            content_width,
                            RowKind::ThoughtBody,
                            Some(index),
                            false,
                        );
                    } else {
                        let visible = stream_visible(&thought.stream, true);
                        push_wrapped(
                            &mut rows,
                            &visible,
                            content_width,
                            RowKind::ThoughtBody,
                            Some(index),
                            false,
                        );
                    }
                }
            }
            TranscriptLine::User(text) => {
                push_wrapped(&mut rows, text, content_width, RowKind::User, None, false);
            }
            TranscriptLine::Tool(tool) => {
                let selected = app.selected == Some(index);
                let kind = match (selected, tool.expanded) {
                    (true, true) => RowKind::ToolOpenSelected,
                    (true, false) => RowKind::ToolSelected,
                    (false, true) => RowKind::ToolOpen,
                    (false, false) => RowKind::Tool,
                };
                let hit = if tool.body.is_empty() {
                    None
                } else {
                    Some(index)
                };
                push_wrapped(&mut rows, &tool.label, content_width, kind, hit, true);
                if tool.expanded {
                    for line in &tool.body {
                        let section = !line.starts_with("  ");
                        let kind = if section {
                            RowKind::ToolSection
                        } else {
                            RowKind::ToolBody
                        };
                        push_wrapped(&mut rows, line, content_width, kind, hit, false);
                    }
                }
            }
        }
    }
    rows
}

fn push_wrapped(
    rows: &mut Vec<Painted>,
    text: &str,
    width: usize,
    kind: RowKind,
    tool: Option<usize>,
    header: bool,
) {
    let wrapped = wrap_text(text, width);
    let wrapped = if wrapped.is_empty() {
        vec![String::new()]
    } else {
        wrapped
    };
    for (line_index, chunk) in wrapped.into_iter().enumerate() {
        rows.push(Painted {
            tool,
            header: header && line_index == 0,
            row: PaintedRow::Text {
                kind,
                first: line_index == 0,
                text: chunk,
            },
        });
    }
}

#[derive(Clone, Copy)]
enum RowKind {
    User,
    Tool,
    ToolSelected,
    ToolOpen,
    ToolOpenSelected,
    Thought,
    ThoughtSelected,
    ThoughtOpen,
    ThoughtOpenSelected,
    ThoughtBody,
    ToolSection,
    ToolBody,
}

fn paint_row(frame: &mut Frame, area: Rect, row: &PaintedRow) {
    if let PaintedRow::Rich { pieces, .. } = row {
        frame.render_widget(Block::default().style(Style::default().bg(BG_BASE)), area);
        let mut spans = vec![Span::styled("  ", Style::default().fg(FG).bg(BG_BASE))];
        for piece in pieces {
            spans.push(Span::styled(piece.text.as_str(), rich_style(piece.mark)));
        }
        frame.render_widget(Paragraph::new(Line::from(spans)), area);
        return;
    }
    let PaintedRow::Text { kind, first, text } = row else {
        return;
    };
    let (background, prefix, color, italic) = match kind {
        RowKind::User => (BG_LIGHT, if *first { "› " } else { "  " }, FG, false),
        RowKind::Tool => (
            BG_BASE,
            if *first { "◆ " } else { "  " },
            GRAY_BRIGHT,
            false,
        ),
        RowKind::ToolSelected => (BG_LIGHT, if *first { "▸ " } else { "  " }, FG, false),
        RowKind::ToolOpen => (
            BG_BASE,
            if *first { "▾ " } else { "  " },
            GRAY_BRIGHT,
            false,
        ),
        RowKind::ToolOpenSelected => (BG_LIGHT, if *first { "▾ " } else { "  " }, FG, false),
        RowKind::Thought => (
            BG_BASE,
            if *first { "· " } else { "  " },
            FG_SECONDARY,
            true,
        ),
        RowKind::ThoughtSelected => (BG_LIGHT, if *first { "▸ " } else { "  " }, FG, true),
        RowKind::ThoughtOpen => (
            BG_BASE,
            if *first { "▾ " } else { "  " },
            FG_SECONDARY,
            true,
        ),
        RowKind::ThoughtOpenSelected => (BG_LIGHT, if *first { "▾ " } else { "  " }, FG, true),
        RowKind::ThoughtBody => (BG_BASE, "  ", FG_SECONDARY, true),
        RowKind::ToolSection => (BG_BASE, "  ", GRAY_BRIGHT, false),
        RowKind::ToolBody => (BG_BASE, "  ", FG_SECONDARY, false),
    };
    frame.render_widget(
        Block::default().style(Style::default().bg(background)),
        area,
    );
    let mut style = Style::default().fg(color).bg(background);
    if italic {
        style = style.add_modifier(Modifier::ITALIC);
    }
    let line = Line::from(vec![
        Span::styled(prefix, Style::default().fg(color).bg(background)),
        Span::styled(text.as_str(), style),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

fn rich_style(mark: Mark) -> Style {
    let base = Style::default().fg(FG).bg(BG_BASE);
    match mark {
        Mark::Plain => base,
        Mark::Strong | Mark::Heading => base.add_modifier(Modifier::BOLD),
        Mark::Emphasis => base.add_modifier(Modifier::ITALIC),
        Mark::Code => base.fg(MAGENTA),
        Mark::Border => base.fg(GRAY_BRIGHT),
    }
}

fn wrap_rich(pieces: &[Piece], width: usize) -> Vec<Vec<Piece>> {
    if width == 0 {
        return vec![vec![]];
    }
    let mut lines: Vec<Vec<Piece>> = vec![Vec::new()];
    let mut column = 0;
    for piece in pieces {
        let mut rest = piece.text.as_str();
        while !rest.is_empty() {
            if column == width {
                lines.push(Vec::new());
                column = 0;
            }
            let room = width - column;
            let take: String = rest.chars().take(room).collect();
            let bytes = take.len();
            column += take.chars().count();
            rest = &rest[bytes..];
            lines.last_mut().expect("line").push(Piece {
                text: take,
                mark: piece.mark,
            });
        }
    }
    if lines.last().is_some_and(Vec::is_empty) {
        lines.pop();
    }
    if lines.is_empty() {
        lines.push(vec![]);
    }
    lines
}

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

fn render_activity(frame: &mut Frame, app: &App, area: Rect) {
    let activity = app.activity.clone().unwrap_or(Activity::Working);
    let frame_index = (app.ticks as usize / 2) % SPINNER.len();
    let label = match &activity {
        Activity::Thinking => "thinking".to_string(),
        Activity::Working => "model working".to_string(),
        Activity::ToolCalling(name) => format!("tool calling {name}"),
        Activity::ToolExecuting(name) => format!("tool executing {name}"),
    };
    let line = Line::from(vec![
        Span::styled(
            SPINNER[frame_index],
            Style::default().fg(MAGENTA).bg(BG_BASE),
        ),
        Span::styled(
            format!(" {label}"),
            Style::default().fg(FG_SECONDARY).bg(BG_BASE),
        ),
    ]);
    frame.render_widget(
        Paragraph::new(line).style(Style::default().bg(BG_BASE)),
        area,
    );
}

fn render_picker(frame: &mut Frame, app: &App, area: Rect) {
    if app.mode == Mode::Compose && !menu_options(&app.input).is_empty() {
        render_menu(frame, app, area);
        return;
    }
    let mut lines = Vec::new();
    if app.mode == Mode::ModelWait {
        lines.push(Line::from(Span::styled(
            "fetching models…",
            Style::default().fg(GRAY_BRIGHT).bg(BG_BASE),
        )));
    } else {
        let models = app.filtered_models();
        let header = if app.picker_filter.is_empty() {
            "models  type to filter  enter selects  esc cancels".to_string()
        } else {
            format!("filter {}  enter selects  esc cancels", app.picker_filter)
        };
        lines.push(Line::from(Span::styled(
            header,
            Style::default().fg(GRAY).bg(BG_BASE),
        )));
        if models.is_empty() {
            lines.push(Line::from(Span::styled(
                "no models",
                Style::default().fg(YELLOW).bg(BG_BASE),
            )));
        }
        let window = picker_window(models.len(), app.picker_index, PICKER_PAGE);
        for (index, model) in models
            .iter()
            .enumerate()
            .skip(window.start)
            .take(window.len())
        {
            let context = model
                .context_length
                .map(format_context)
                .unwrap_or_else(|| "unknown".to_string());
            let input = price_field("input", model.prompt_price.as_deref());
            let output = price_field("output", model.completion_price.as_deref());
            let mark = if index == app.picker_index { ">" } else { " " };
            let style = if index == app.picker_index {
                Style::default().fg(FG).bg(BG_LIGHT)
            } else {
                Style::default().fg(FG_SECONDARY).bg(BG_BASE)
            };
            lines.push(Line::from(Span::styled(
                format!("{mark} {}  ctx {context}  {input}  {output}", model.id),
                style,
            )));
        }
    }
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), area);
}

fn render_composer(frame: &mut Frame, app: &App, area: Rect) {
    let active = app.session_open
        && !app.in_flight()
        && matches!(app.mode, Mode::Compose | Mode::Key { .. });
    let border = if active {
        PROMPT_BORDER_ACTIVE
    } else {
        PROMPT_BORDER
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border))
        .style(Style::default().bg(BG_BASE).fg(FG));
    let mut spans = vec![Span::styled("› ", Style::default().fg(FG_SECONDARY))];
    match &app.mode {
        Mode::Key { .. } => {
            if app.input.is_empty() {
                spans.push(Span::styled("hidden key", Style::default().fg(GRAY)));
            } else {
                let hidden = "•".repeat(app.input.chars().count());
                spans.push(Span::styled(hidden, Style::default().fg(FG)));
                spans.push(Span::styled("▌", Style::default().fg(MAGENTA)));
            }
        }
        _ => {
            if app.input.is_empty() {
                spans.push(Span::styled("Message Atlas", Style::default().fg(GRAY)));
            } else {
                spans.push(Span::styled(app.input.as_str(), Style::default().fg(FG)));
                spans.push(Span::styled("▌", Style::default().fg(MAGENTA)));
            }
        }
    }
    let input = Paragraph::new(Line::from(spans))
        .block(block)
        .wrap(Wrap { trim: false });
    frame.render_widget(input, area);
}

fn render_status(frame: &mut Frame, app: &App, area: Rect) {
    let (label, color) = status_label(app);
    let model = model_name(app);
    let metrics = strip_model_field(&app.status_line);
    let mut spans = vec![
        Span::styled("Atlas", Style::default().fg(GRAY_BRIGHT).bg(BG_BASE)),
        Span::styled("  ", Style::default().bg(BG_BASE)),
        Span::styled(label, Style::default().fg(color).bg(BG_BASE)),
        Span::styled(format!("  {model}"), Style::default().fg(GRAY).bg(BG_BASE)),
    ];
    if !metrics.is_empty() {
        spans.push(Span::styled(
            format!("  {metrics}"),
            Style::default().fg(FG_SECONDARY).bg(BG_BASE),
        ));
    }
    let [dashboard_area, hint_area] =
        Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(area);
    frame.render_widget(
        Paragraph::new(Line::from(spans))
            .style(Style::default().bg(BG_BASE))
            .wrap(Wrap { trim: false }),
        dashboard_area,
    );
    frame.render_widget(
        Paragraph::new(Span::styled(
            status_hint(app),
            Style::default().fg(GRAY).bg(BG_BASE),
        )),
        hint_area,
    );
}

/// One dashboard line. The model is already named here, so the metrics field
/// that repeats it is left off. `scope` is not shown; it stays `project`.
fn dashboard_text(app: &App) -> String {
    let (label, _) = status_label(app);
    let metrics = strip_model_field(&app.status_line);
    if metrics.is_empty() {
        format!("Atlas  {label}  {}", model_name(app))
    } else {
        format!("Atlas  {label}  {}  {metrics}", model_name(app))
    }
}

fn model_name(app: &App) -> &str {
    if app.model_id.is_empty() {
        "unset"
    } else {
        app.model_id.as_str()
    }
}

fn strip_model_field(status_line: &str) -> String {
    status_line
        .split("  ")
        .filter(|part| !part.starts_with("model "))
        .collect::<Vec<_>>()
        .join("  ")
}

fn status_hint(app: &App) -> String {
    let mut hint = if !app.session_open {
        "ctrl-c quit".to_string()
    } else if app.status == Status::Stopped {
        "enter send  /new /model /key /quit  ctrl-r resume".to_string()
    } else {
        "enter send  /new /model /key /quit".to_string()
    };
    if !expandable_tools(app).is_empty() {
        hint.push_str("  ←→ tool");
    }
    hint
}

fn status_label(app: &App) -> (&'static str, Color) {
    if !app.session_open {
        return ("session ended", RED);
    }
    match app.status {
        Status::Ready => ("ready", GRAY_BRIGHT),
        Status::Waiting => ("working", MAGENTA),
        Status::Stopped => ("stopped for tool limit", YELLOW),
    }
}

fn wrap_text(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![String::new()];
    }
    let mut lines = Vec::new();
    for paragraph in text.split('\n') {
        if paragraph.is_empty() {
            lines.push(String::new());
            continue;
        }
        let mut rest = paragraph;
        while !rest.is_empty() {
            let mut end = rest.chars().take(width).map(char::len_utf8).sum::<usize>();
            let splits_a_word = end < rest.len() && !rest[end..].starts_with(char::is_whitespace);
            if splits_a_word {
                if let Some(space) = rest[..end].rfind(' ') {
                    if space > 0 {
                        end = space;
                    }
                }
            }
            let (chunk, next) = rest.split_at(end);
            lines.push(chunk.trim_end().to_string());
            rest = next.trim_start();
        }
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn recv_first_line(child: &mut ChildProcess, rx: &Receiver<String>) -> Result<String> {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(anyhow!("agent session did not start"));
        }
        let wait = remaining.min(Duration::from_millis(200));
        match rx.recv_timeout(wait) {
            Ok(line) => return Ok(line),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(anyhow!("agent session exited before it was ready"));
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if let Ok(Some(status)) = child.child.try_wait() {
                    return Err(anyhow!(
                        "agent session exited before it was ready ({status})"
                    ));
                }
            }
        }
    }
}

fn join_stderr(errx: &Receiver<String>) -> String {
    thread::sleep(Duration::from_millis(50));
    let mut lines = Vec::new();
    while let Ok(line) = errx.try_recv() {
        lines.push(line);
    }
    lines.join(" ")
}

fn note_session_closed(app: &mut App) {
    if !app.session_open {
        return;
    }
    app.session_open = false;
    let detail = app
        .last_stderr
        .clone()
        .unwrap_or_else(|| "session ended".to_string());
    app.transcript.push(tool_note(format!("error: {detail}")));
    app.status = Status::Ready;
    app.waiting_from_stopped = false;
}

#[cfg(test)]
mod tests {
    use super::{
        advance_streams, collapse_selected, delete_word_backward, draw, expand_selected,
        finish_streams, format_context, is_finished_command, menu_action, menu_options,
        nudge_scroll, picker_selection, price_per_million, select_tool, select_tool_at,
        slash_outcome, wrap_text, Activity, App, MenuAction, Mode, Status, StreamText, ToolHit,
        TranscriptLine,
    };
    use crate::logo::logo_size;
    use crate::protocol::{CatalogEntry, ServerMessage, StatusSnapshot};

    fn tool_text(app: &App) -> String {
        app.transcript
            .iter()
            .filter_map(|line| match line {
                TranscriptLine::Tool(tool) => Some(tool.label.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn wraps_on_word_boundaries() {
        assert_eq!(wrap_text("one two three", 7), vec!["one two", "three"]);
    }

    #[test]
    fn model_fetch_failure_keeps_the_selected_model() {
        let mut app = App::new();
        app.model_id = "keep-me".to_string();
        app.mode = Mode::ModelWait;
        app.apply(ServerMessage::Models {
            ok: false,
            message: "OpenRouter HTTP 500: down".to_string(),
            models: Vec::new(),
            status: crate::protocol::StatusSnapshot::default(),
        });

        assert_eq!(app.model_id, "keep-me");
        assert_eq!(app.mode, Mode::Compose);
        let text = tool_text(&app);
        assert!(text.contains("model list failed"));
        assert!(text.contains("500"));
    }

    #[test]
    fn selecting_a_model_updates_the_status_speed() {
        let mut app = App::new();
        app.model_id = "old".to_string();
        app.mode = Mode::ModelPicker;
        app.apply(ServerMessage::ModelSelected {
            id: "google/gemini".to_string(),
            context_length: Some(128_000),
            status: StatusSnapshot {
                status_line: "model google/gemini  context 12/128000  tokens 16 (prompt 12 completion 4)  spend $0.02  8.0 tok/s  http 200  latency 0.50s  ttft n/a  recent 200 0.50s".to_string(),
                model: "google/gemini".to_string(),
                context_used: 12,
                context_limit: Some(128_000),
                tokens_per_second: Some(8.0),
                ..StatusSnapshot::default()
            },
        });

        assert_eq!(app.model_id, "google/gemini");
        assert_eq!(app.mode, Mode::Compose);
        assert!(app.status_line.contains("context 12/128000"));
        assert!(app.status_line.contains("8.0 tok/s"));
    }

    #[test]
    fn model_command_fetches_without_changing_the_selection() {
        let mut app = App::new();
        app.model_id = "keep-me".to_string();
        let payload = slash_outcome(&mut app, "/model gemini").expect("models request");

        assert_eq!(payload["type"], "models");
        assert_eq!(app.mode, Mode::ModelWait);
        assert_eq!(app.picker_filter, "gemini");
        assert_eq!(app.model_id, "keep-me");
        assert!(tool_text(&app).contains("fetching models matching gemini"));
    }

    #[test]
    fn picker_filters_then_selects_the_visible_row() {
        let mut app = App::new();
        app.picker_filter = "gem".to_string();
        app.picker_models = vec![
            CatalogEntry {
                id: "other/model".to_string(),
                context_length: None,
                prompt_price: None,
                completion_price: None,
            },
            CatalogEntry {
                id: "google/gemini".to_string(),
                context_length: Some(128_000),
                prompt_price: Some("0.1".to_string()),
                completion_price: Some("0.2".to_string()),
            },
        ];

        let payload = picker_selection(&app).expect("filtered row");
        assert_eq!(payload["type"], "select_model");
        assert_eq!(payload["id"], "google/gemini");
    }

    #[test]
    fn context_uses_k_and_m_at_three_sigfigs() {
        assert_eq!(format_context(1_048_576), "1.05M");
        assert_eq!(format_context(1_000_000), "1.00M");
        assert_eq!(format_context(999_500), "1.00M");
        assert_eq!(format_context(128_000), "128K");
        assert_eq!(format_context(200_000), "200K");
        assert_eq!(format_context(32_768), "32.8K");
        assert_eq!(format_context(10_000), "10.0K");
        assert_eq!(format_context(8_192), "8.19K");
        assert_eq!(format_context(4_096), "4.10K");
        assert_eq!(format_context(1_500_000_000), "1.50B");
        assert_eq!(format_context(12_300_000_000), "12.3B");
        assert_eq!(format_context(512), "512");
        assert_eq!(format_context(0), "0");
    }

    #[test]
    fn price_per_million_shifts_the_decimal_point() {
        assert_eq!(price_per_million("0.000002").as_deref(), Some("2"));
        assert_eq!(price_per_million("0.00001").as_deref(), Some("10"));
        assert_eq!(price_per_million("0.0000000198").as_deref(), Some("0.0198"));
        assert_eq!(price_per_million("0.000000396").as_deref(), Some("0.396"));
        assert_eq!(price_per_million("0.00000015").as_deref(), Some("0.15"));
        assert_eq!(price_per_million("0.0000025").as_deref(), Some("2.5"));
        assert_eq!(price_per_million("2.").as_deref(), Some("2000000"));
        assert_eq!(price_per_million(".5").as_deref(), Some("500000"));
        assert_eq!(price_per_million("+0.000002").as_deref(), Some("2"));
        assert_eq!(price_per_million("0").as_deref(), Some("0"));
    }

    #[test]
    fn price_per_million_rejects_non_decimals() {
        for text in ["", "-1", "abc", "1e9", "n/a"] {
            assert_eq!(price_per_million(text), None, "accepted {text:?}");
        }
    }

    #[test]
    fn picker_paints_prices_per_million() {
        let mut app = App::new();
        app.mode = Mode::ModelPicker;
        app.picker_models = vec![CatalogEntry {
            id: "vendor/model".to_string(),
            context_length: Some(1_000_000),
            prompt_price: Some("0.000002".to_string()),
            completion_price: Some("0.00001".to_string()),
        }];
        let painted = screen(&mut app, 80, 24);
        assert!(painted.contains("ctx 1.00M"), "{painted}");
        assert!(painted.contains("input $2/M"), "{painted}");
        assert!(painted.contains("output $10/M"), "{painted}");
    }

    #[test]
    fn picker_paints_fractional_prices() {
        let mut app = App::new();
        app.mode = Mode::ModelPicker;
        app.picker_models = vec![CatalogEntry {
            id: "vendor/model".to_string(),
            context_length: Some(1_000_000),
            prompt_price: Some("0.0000000198".to_string()),
            completion_price: Some("0.000000396".to_string()),
        }];
        let painted = screen(&mut app, 80, 24);
        assert!(painted.contains("input $0.0198/M"), "{painted}");
        assert!(painted.contains("output $0.396/M"), "{painted}");
    }

    #[test]
    fn picker_paints_missing_prices_without_a_dollar_sign() {
        let mut app = App::new();
        app.mode = Mode::ModelPicker;
        app.picker_models = vec![CatalogEntry {
            id: "vendor/model".to_string(),
            context_length: None,
            prompt_price: None,
            completion_price: None,
        }];
        let painted = screen(&mut app, 80, 24);
        assert!(painted.contains("input n/a"), "{painted}");
        assert!(painted.contains("output n/a"), "{painted}");
        assert!(!painted.contains("input $"), "{painted}");
        assert!(!painted.contains("output $"), "{painted}");
    }

    #[test]
    fn removed_commands_fall_through_to_unknown() {
        let mut app = App::new();
        assert!(slash_outcome(&mut app, "/rm session s1").is_none());
        assert!(slash_outcome(&mut app, "/scope user").is_none());
        assert_eq!(app.mode, Mode::Compose);
        let text = tool_text(&app);
        assert!(text.contains("unknown command /rm"));
        assert!(text.contains("unknown command /scope"));
    }

    #[test]
    fn new_command_resets_the_conversation() {
        let mut app = App::new();
        app.transcript
            .push(TranscriptLine::Assistant(StreamText::settled("old reply")));
        app.status = Status::Stopped;
        app.waiting_from_stopped = true;
        app.activity = Some(Activity::Thinking);
        app.selected = Some(0);
        app.reveal_selected = true;
        app.scroll = 5;
        app.follow = false;
        app.tokens_per_second = 12.0;

        let payload = slash_outcome(&mut app, "/new").expect("new request");

        assert_eq!(payload["type"], "new");
        assert!(app.transcript.is_empty());
        assert!(app.status == Status::Ready);
        assert!(!app.waiting_from_stopped);
        assert!(app.activity.is_none());
        assert!(app.selected.is_none());
        assert!(!app.reveal_selected);
        assert_eq!(app.scroll, 0);
        assert!(app.follow);
        assert_eq!(app.tokens_per_second, 0.0);
    }

    #[test]
    fn new_with_extra_tokens_shows_usage() {
        let mut app = App::new();
        assert!(slash_outcome(&mut app, "/new extra").is_none());
        assert!(tool_text(&app).contains("usage: /new"));
    }

    #[test]
    fn new_is_a_finished_command() {
        assert!(is_finished_command("/new"));
        assert!(!is_finished_command("/new extra"));
    }

    #[test]
    fn slash_menu_lists_every_command() {
        let mut app = App::new();
        app.input = "/".to_string();
        let painted = screen(&mut app, 80, 24);
        for name in [
            "/new",
            "/model",
            "/copy",
            "/key",
            "/sessions",
            "/artifacts",
            "/quit",
            "/exit",
        ] {
            assert!(painted.contains(name), "missing {name} in:\n{painted}");
        }
        assert!(!painted.contains("/rm"), "{painted}");
        assert!(!painted.contains("/scope"), "{painted}");
        assert!(painted.contains("↑↓ move  tab complete  enter run  esc close"));
    }

    #[test]
    fn a_markdown_table_is_drawn_as_a_table() {
        let mut app = App::new();
        app.transcript
            .push(TranscriptLine::Assistant(StreamText::settled(
                "| Item | Count |\n| --- | --- |\n| `dist/` | 3 files |\n",
            )));
        let painted = screen(&mut app, 80, 24);
        assert!(painted.contains("Item"), "{painted}");
        assert!(painted.contains("3 files"), "{painted}");
        assert!(painted.contains("dist/"), "{painted}");
        assert!(painted.contains('┌'), "{painted}");
        assert!(!painted.contains("| --- |"), "{painted}");
    }

    #[test]
    fn assistant_markdown_is_not_painted_raw() {
        let mut app = App::new();
        app.transcript
            .push(TranscriptLine::Assistant(StreamText::settled(
                "**bold** and `code`",
            )));
        let painted = screen(&mut app, 80, 24);
        assert!(painted.contains("bold"), "{painted}");
        assert!(painted.contains("code"), "{painted}");
        assert!(!painted.contains("**"), "{painted}");
        assert!(!painted.contains('`'), "{painted}");
    }

    #[test]
    fn tool_step_hides_the_raw_result() {
        let mut app = App::new();
        app.apply(ServerMessage::Step {
            name: "read_file".to_string(),
            call_id: "1".to_string(),
            ok: true,
            arguments: serde_json::json!({"path": "notes.md"}),
            text: Some("SECRET_FILE_BODY".to_string()),
            error: None,
            error_kind: None,
            artifacts: vec!["notes.md".to_string()],
        });
        let painted = screen(&mut app, 80, 24);
        assert!(painted.contains("read_file"), "{painted}");
        assert!(!painted.contains("SECRET_FILE_BODY"), "{painted}");
        assert!(!painted.contains("notes.md"), "{painted}");
        expand_selected(&mut app);
        let opened = screen(&mut app, 80, 24);
        assert!(opened.contains("arguments"), "{opened}");
        assert!(opened.contains("path: notes.md"), "{opened}");
        assert!(opened.contains("result"), "{opened}");
        assert!(opened.contains("SECRET_FILE_BODY"), "{opened}");
        assert!(opened.contains("artifacts"), "{opened}");
        collapse_selected(&mut app);
        let closed = screen(&mut app, 80, 24);
        assert!(closed.contains("read_file"), "{closed}");
        assert!(!closed.contains("SECRET_FILE_BODY"), "{closed}");
    }

    #[test]
    fn option_delete_removes_the_previous_word() {
        let mut text = "hello world".to_string();
        delete_word_backward(&mut text);
        assert_eq!(text, "hello ");
        delete_word_backward(&mut text);
        assert_eq!(text, "");
        let mut spaced = "hello world  ".to_string();
        delete_word_backward(&mut spaced);
        assert_eq!(spaced, "hello ");
    }

    #[test]
    fn thoughts_are_painted_in_the_transcript() {
        let mut app = App::new();
        app.apply(ServerMessage::Thought {
            text: "check the coast first".to_string(),
            tokens_per_second: Some(1.0),
        });
        app.apply(ServerMessage::Thought {
            text: "   ".to_string(),
            tokens_per_second: None,
        });
        let early = screen(&mut app, 80, 24);
        assert!(early.contains("thinking"), "{early}");
        assert!(!early.contains("check"), "{early}");
        assert!(!early.contains("coast"), "{early}");
        assert!(!early.contains('▌'), "{early}");
        advance_streams(&mut app, 1.0);
        let mid = screen(&mut app, 80, 24);
        assert!(mid.contains("thinking"), "{mid}");
        assert!(!mid.contains("check"), "{mid}");
        assert!(!mid.contains('▌'), "{mid}");
        expand_selected(&mut app);
        let streaming = screen(&mut app, 80, 24);
        assert!(streaming.contains("check"), "{streaming}");
        assert!(!streaming.contains("coast"), "{streaming}");
        assert!(streaming.contains('▌'), "{streaming}");
        collapse_selected(&mut app);
        advance_streams(&mut app, 8.0);
        let painted = screen(&mut app, 80, 24);
        assert!(painted.contains("thinking"), "{painted}");
        assert!(!painted.contains("coast"), "{painted}");
        assert!(!painted.contains('▌'), "{painted}");
        expand_selected(&mut app);
        let opened = screen(&mut app, 80, 24);
        assert!(opened.contains("check the coast first"), "{opened}");
        assert!(!opened.contains('▌'), "{opened}");
        collapse_selected(&mut app);
        let closed = screen(&mut app, 80, 24);
        assert!(!closed.contains("coast"), "{closed}");
        assert!(closed.contains("thinking"), "{closed}");
        assert_eq!(
            app.transcript
                .iter()
                .filter(|line| matches!(line, TranscriptLine::Thought(_)))
                .count(),
            1
        );
        assert_eq!(app.selected, Some(0));
    }

    #[test]
    fn a_reply_streams_at_the_reported_token_rate_and_a_send_finishes_it() {
        let mut app = App::new();
        app.apply(ServerMessage::Done {
            response: "alpha beta gamma".to_string(),
            stopped_for_limit: false,
            status: StatusSnapshot {
                tokens_per_second: Some(2.0),
                ..StatusSnapshot::default()
            },
        });
        let early = screen(&mut app, 80, 24);
        assert!(!early.contains("gamma"), "{early}");
        advance_streams(&mut app, 0.5);
        let mid = screen(&mut app, 80, 24);
        assert!(mid.contains("alpha"), "{mid}");
        assert!(!mid.contains("gamma"), "{mid}");
        assert!(mid.contains('▌'), "{mid}");
        finish_streams(&mut app);
        let done = screen(&mut app, 80, 24);
        assert!(done.contains("gamma"), "{done}");
        assert!(!done.contains('▌'), "{done}");
    }

    #[test]
    fn the_next_stream_waits_until_the_current_one_finishes() {
        let mut app = App::new();
        app.apply(ServerMessage::Thought {
            text: "one two".to_string(),
            tokens_per_second: Some(1.0),
        });
        app.apply(ServerMessage::Thought {
            text: "three four".to_string(),
            tokens_per_second: Some(1.0),
        });
        advance_streams(&mut app, 1.0);
        let painted = screen(&mut app, 80, 24);
        assert!(painted.contains("thinking"), "{painted}");
        assert!(!painted.contains("one"), "{painted}");
        assert!(!painted.contains("three"), "{painted}");
        advance_streams(&mut app, 8.0);
        let finished = screen(&mut app, 80, 24);
        assert!(finished.contains("thinking"), "{finished}");
        assert!(!finished.contains("three"), "{finished}");
        assert!(!finished.contains("one"), "{finished}");
        assert_eq!(app.selected, Some(1));
        expand_selected(&mut app);
        let second = screen(&mut app, 80, 24);
        assert!(second.contains("three four"), "{second}");
        assert!(!second.contains("one two"), "{second}");
        select_tool(&mut app, -1);
        expand_selected(&mut app);
        let first = screen(&mut app, 80, 24);
        assert!(first.contains("one two"), "{first}");
    }

    #[test]
    fn a_thought_sits_in_the_tool_selection() {
        let mut app = App::new();
        app.apply(ServerMessage::Step {
            name: "read_file".to_string(),
            call_id: "1".to_string(),
            ok: true,
            arguments: serde_json::json!({"path": "notes.md"}),
            text: Some("secret".to_string()),
            error: None,
            error_kind: None,
            artifacts: Vec::new(),
        });
        app.apply(ServerMessage::Thought {
            text: "look around".to_string(),
            tokens_per_second: Some(1000.0),
        });
        advance_streams(&mut app, 5.0);
        assert_eq!(app.selected, Some(1));
        select_tool(&mut app, -1);
        assert_eq!(app.selected, Some(0));
        expand_selected(&mut app);
        let tool_view = screen(&mut app, 80, 24);
        assert!(tool_view.contains("secret"), "{tool_view}");
        assert!(!tool_view.contains("look around"), "{tool_view}");
        select_tool(&mut app, 1);
        expand_selected(&mut app);
        let thought_view = screen(&mut app, 80, 24);
        assert!(thought_view.contains("look around"), "{thought_view}");
        collapse_selected(&mut app);
        let collapsed = screen(&mut app, 80, 24);
        assert!(!collapsed.contains("look around"), "{collapsed}");
    }

    #[test]
    fn shift_up_selects_the_previous_tool_and_a_click_selects_a_row() {
        let mut app = App::new();
        app.apply(ServerMessage::Step {
            name: "read_file".to_string(),
            call_id: "1".to_string(),
            ok: true,
            arguments: serde_json::json!({}),
            text: Some("one".to_string()),
            error: None,
            error_kind: None,
            artifacts: Vec::new(),
        });
        app.apply(ServerMessage::Step {
            name: "list".to_string(),
            call_id: "2".to_string(),
            ok: false,
            arguments: serde_json::json!({"path": "missing"}),
            text: None,
            error: Some("missing".to_string()),
            error_kind: Some("tool_failure".to_string()),
            artifacts: Vec::new(),
        });
        assert_eq!(app.selected, Some(1));
        select_tool(&mut app, -1);
        assert_eq!(app.selected, Some(0));
        app.tool_hits = vec![ToolHit {
            y: 4,
            height: 1,
            index: 1,
        }];
        select_tool_at(&mut app, 4);
        assert_eq!(app.selected, Some(1));
        expand_selected(&mut app);
        let painted = screen(&mut app, 80, 24);
        assert!(painted.contains("error"), "{painted}");
        assert!(painted.contains("kind: tool_failure"), "{painted}");
        assert!(painted.contains("missing"), "{painted}");
        assert!(!painted.contains("one"), "{painted}");
    }

    #[test]
    fn tool_phase_paints_an_executing_spinner() {
        let mut app = App::new();
        app.status = Status::Waiting;
        app.apply(ServerMessage::Phase {
            phase: "tool_executing".to_string(),
            name: Some("read_file".to_string()),
        });
        let painted = screen(&mut app, 80, 24);
        assert!(painted.contains("tool executing read_file"), "{painted}");
    }

    #[test]
    fn scrolling_up_stops_following_the_tail() {
        let mut app = App::new();
        app.scroll_max = 20;
        app.scroll = 20;
        app.follow = true;
        nudge_scroll(&mut app, -3);
        assert!(!app.follow);
        assert_eq!(app.scroll, 17);
        nudge_scroll(&mut app, 100);
        assert!(app.follow);
        assert_eq!(app.scroll, 20);
    }

    #[test]
    fn slash_menu_filters_to_the_matching_command() {
        let mut app = App::new();
        app.input = "/se".to_string();
        let painted = screen(&mut app, 80, 24);
        assert!(painted.contains("/sessions"), "{painted}");
        assert!(!painted.contains("/artifacts"), "{painted}");
    }

    #[test]
    fn slash_menu_marks_the_highlighted_row() {
        let mut app = App::new();
        app.input = "/se".to_string();
        let painted = screen(&mut app, 80, 24);
        let highlighted: Vec<&str> = painted.lines().filter(|line| line.contains('>')).collect();
        assert!(
            highlighted.iter().any(|line| line.contains("/sessions")),
            "highlighted rows {highlighted:?}:\n{painted}"
        );
    }

    #[test]
    fn slash_menu_shows_key_arguments() {
        let mut app = App::new();
        app.input = "/key ".to_string();
        let painted = screen(&mut app, 80, 24);
        assert!(painted.contains("session"), "{painted}");
        assert!(painted.contains("user"), "{painted}");
    }

    #[test]
    fn partial_model_completes_before_running() {
        // Tab fills `/model`. Enter runs it, since the command needs nothing else.
        assert_eq!(
            menu_action("/mo", 0, false),
            MenuAction::Complete("/mo".to_string(), "/model".to_string())
        );
        assert_eq!(
            menu_action("/mo", 0, true),
            MenuAction::Run("/model".to_string())
        );
        assert_eq!(
            menu_action("/", 0, true),
            MenuAction::Run("/new".to_string())
        );
        assert_eq!(
            menu_action("/key", 0, true),
            MenuAction::Complete("/key".to_string(), "/key ".to_string())
        );

        // A finished command runs unchanged.
        assert_eq!(
            menu_action("/model", 0, true),
            MenuAction::Run("/model".to_string())
        );
    }

    #[test]
    fn choosing_a_final_argument_runs_it() {
        assert_eq!(
            menu_action("/key s", 0, true),
            MenuAction::Run("/key session".to_string())
        );
        // Tab still completes rather than running, and leaves a space so the
        // argument rows appear.
        assert_eq!(
            menu_action("/key", 0, false),
            MenuAction::Complete("/key".to_string(), "/key ".to_string())
        );
    }

    #[test]
    fn free_text_hides_the_menu() {
        assert!(menu_options("/model gemini").is_empty());
        assert!(menu_options("/key session x").is_empty());
        assert!(menu_options("/nope").is_empty());
        assert!(menu_options("/sessions extra").is_empty());
        // A single exact command still shows itself.
        assert_eq!(menu_options("/sessions").len(), 1);
    }

    fn screen(app: &mut App, width: u16, height: u16) -> String {
        let backend = ratatui::backend::TestBackend::new(width, height);
        let mut terminal = ratatui::Terminal::new(backend).expect("test terminal");
        terminal.draw(|frame| draw(frame, app)).expect("draw");
        let buffer = terminal.backend().buffer().clone();
        let mut lines = Vec::new();
        for y in 0..height {
            let mut line = String::new();
            for x in 0..width {
                line.push_str(buffer[(x, y)].symbol());
            }
            lines.push(line);
        }
        lines.join("\n")
    }

    fn logo_span(painted: &str) -> Option<(usize, usize, usize, usize)> {
        let mut bounds: Option<(usize, usize, usize, usize)> = None;
        for (y, line) in painted.lines().enumerate() {
            for (x, ch) in line.chars().enumerate() {
                if ch != '.' {
                    continue;
                }
                bounds = Some(match bounds {
                    None => (x, y, x, y),
                    Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
                });
            }
        }
        bounds
    }

    #[test]
    fn an_empty_screen_centers_the_logo_and_caps_its_size() {
        let mut app = App::new();
        let painted = screen(&mut app, 160, 50);
        let (x0, y0, x1, _) = logo_span(&painted).expect("logo");
        let atlas_row = painted
            .lines()
            .position(|line| line.contains("ATLAS"))
            .expect("word");
        let (max_cols, max_rows) = logo_size(1_000, 1_000).expect("max splash");
        let width = x1 - x0 + 1;
        let height = atlas_row - y0 + 1;
        assert!(width <= usize::from(max_cols), "{width} wide:\n{painted}");
        assert!(height <= usize::from(max_rows), "{height} tall:\n{painted}");
        let center = (x0 + x1) as f32 / 2.0;
        let frame_center = (160 - 1) as f32 / 2.0;
        assert!(
            (center - frame_center).abs() <= 1.5,
            "center {center} vs {frame_center}:\n{painted}"
        );
        assert!(painted.contains("ATLAS"), "{painted}");
        assert!(!painted.contains("PROJECT"), "{painted}");
    }

    #[test]
    fn the_logo_leaves_when_the_first_prompt_is_sent() {
        let mut app = App::new();
        let before = screen(&mut app, 100, 32);
        assert!(before.contains("ATLAS"), "{before}");
        app.transcript
            .push(TranscriptLine::User("first prompt".to_string()));
        let after = screen(&mut app, 100, 32);
        assert!(after.contains("first prompt"), "{after}");
        assert!(!after.contains("ATLAS"), "logo stayed up:\n{after}");
    }

    #[test]
    fn a_user_line_still_paints_its_text() {
        let mut app = App::new();
        app.transcript
            .push(TranscriptLine::User("hello star field".to_string()));
        let painted = screen(&mut app, 80, 24);
        assert!(painted.contains("hello star field"), "{painted}");
        let line = painted
            .lines()
            .find(|line| line.contains("hello star field"))
            .expect("user row");
        let text = line.replace(['·', '*', '+'], "");
        assert!(
            text.contains("hello star field"),
            "user row was only stars:\n{painted}"
        );
    }

    #[test]
    fn status_on_an_80_column_terminal_keeps_the_metrics() {
        let mut app = App::new();
        app.model_id = "example/model".to_string();
        app.status_line = "model example/model  context 12/128000  tokens 16 (prompt 12 completion 4)  spend $0.02  8.0 tok/s".to_string();

        let painted = screen(&mut app, 80, 24);
        let flat = painted.split_whitespace().collect::<Vec<_>>().join(" ");

        for field in ["spend $0.02", "8.0 tok/s"] {
            assert!(
                flat.contains(field),
                "missing {field} in status rect:\n{painted}"
            );
        }
        assert!(!flat.contains("http"));
        assert!(!flat.contains("latency"));
        assert!(!flat.contains("ttft"));
        assert!(!flat.contains("recent"));
        assert_eq!(flat.matches("example/model").count(), 1);
        assert!(!flat.contains("scope"));
    }

    #[test]
    fn picker_highlights_the_row_enter_would_select() {
        let mut app = App::new();
        app.mode = Mode::ModelPicker;
        app.picker_index = 10;
        app.picker_models = (0..12)
            .map(|index| CatalogEntry {
                id: format!("model-{index}"),
                context_length: Some(128_000),
                prompt_price: Some("0.1".to_string()),
                completion_price: Some("0.2".to_string()),
            })
            .collect();

        let selected = picker_selection(&app).expect("selected model");
        let painted = screen(&mut app, 80, 24);
        let highlighted: Vec<&str> = painted.lines().filter(|line| line.contains('>')).collect();

        assert_eq!(selected["id"], "model-10");
        assert!(
            highlighted.iter().any(|line| line.contains("model-10")),
            "highlighted rows were {highlighted:?}\n{painted}"
        );
        assert!(
            !painted.contains("model-0"),
            "hidden first row was painted:\n{painted}"
        );
    }
}
