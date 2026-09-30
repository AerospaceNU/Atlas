//! Minimal terminal client for the Atlas agent session protocol.

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
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use protocol::{default_status_line, parse_line, status_text, CatalogEntry, ServerMessage};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Wrap};
use ratatui::Frame;
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
    Key {
        scope: String,
    },
    ConfirmRemove {
        scope: String,
        kind: String,
        name: String,
    },
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
    /// `project` or `user`. Remove and list stay inside that root.
    scope_root: String,
}

enum TranscriptLine {
    User(String),
    Assistant(String),
    Tool(String),
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
            scope_root: "project".to_string(),
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
            }
            ServerMessage::Step {
                name,
                ok,
                text,
                error,
                ..
            } => {
                let detail = if ok {
                    text.unwrap_or_default()
                } else {
                    error.unwrap_or_else(|| "failed".to_string())
                };
                self.transcript
                    .push(TranscriptLine::Tool(format!("tool {name}: {detail}")));
            }
            ServerMessage::Done {
                response,
                stopped_for_limit,
                status,
            } => {
                if !response.is_empty() {
                    self.transcript.push(TranscriptLine::Assistant(response));
                }
                self.adopt_status(&status);
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
                    .push(TranscriptLine::Assistant(format!("error: {message}")));
                if !status_line.is_empty() {
                    self.status_line = status_line;
                }
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
            } => {
                if !ok {
                    self.transcript.push(TranscriptLine::Tool(format!(
                        "model list failed: {message}"
                    )));
                    if self.mode == Mode::ModelWait {
                        self.mode = Mode::Compose;
                    }
                } else if self.mode == Mode::ModelWait || self.mode == Mode::ModelPicker {
                    self.picker_models = models;
                    self.picker_index = 0;
                    self.mode = Mode::ModelPicker;
                    let count = self.filtered_models().len();
                    self.transcript
                        .push(TranscriptLine::Tool(format!("models: {count}")));
                }
            }
            ServerMessage::ModelSelected { id, status, .. } => {
                self.model_id = id.clone();
                self.adopt_status(&status);
                self.mode = Mode::Compose;
                self.transcript
                    .push(TranscriptLine::Tool(format!("model {id}")));
            }
            ServerMessage::KeySet { scope, saved } => {
                self.mode = Mode::Compose;
                self.input.clear();
                let label = if saved { "saved" } else { "not saved" };
                self.transcript
                    .push(TranscriptLine::Tool(format!("{scope} key {label}")));
            }
            ServerMessage::Listing { scope, kind, names } => {
                let body = if names.is_empty() {
                    "(none)".to_string()
                } else {
                    names.join(", ")
                };
                self.transcript
                    .push(TranscriptLine::Tool(format!("{scope} {kind}: {body}")));
            }
            ServerMessage::Removed { scope, kind, name } => {
                self.mode = Mode::Compose;
                self.transcript.push(TranscriptLine::Tool(format!(
                    "removed {scope} {kind} {name}"
                )));
            }
            ServerMessage::Unknown => {}
        }
    }

    fn adopt_status(&mut self, status: &protocol::StatusSnapshot) {
        if !status.model.is_empty() {
            self.model_id = status.model.clone();
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
    if let Err(err) = execute!(stdout, EnterAlternateScreen) {
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
    let _ = execute!(io::stdout(), LeaveAlternateScreen);
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
    loop {
        terminal.draw(|frame| draw(frame, app))?;

        let mut stdout_closed = false;
        loop {
            match rx.try_recv() {
                Ok(line) => match parse_line(&line) {
                    Ok(Some(message)) => app.apply(message),
                    Ok(None) => {}
                    Err(_) => app
                        .transcript
                        .push(TranscriptLine::Tool(format!("unparsable: {line}"))),
                },
                Err(mpsc::TryRecvError::Empty) => break,
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
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press && handle_key(app, child, key)? {
                    return Ok(());
                }
            }
        }
    }
}

/// Returns `true` when the user asked to quit.
fn handle_key(app: &mut App, child: &mut ChildProcess, key: KeyEvent) -> Result<bool> {
    if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
        child.shutdown();
        return Ok(true);
    }
    if key.code == KeyCode::Esc && matches!(app.mode, Mode::Compose) {
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
        Mode::ConfirmRemove { scope, kind, name } => {
            if key.code == KeyCode::Enter {
                child.send(confirmed_remove_message(&scope, &kind, &name))?;
                app.mode = Mode::Compose;
            }
        }
        Mode::ModelWait => {}
        Mode::Compose => handle_compose_key(app, child, key)?,
    }
    Ok(false)
}

fn handle_compose_key(app: &mut App, child: &mut ChildProcess, key: KeyEvent) -> Result<()> {
    match key.code {
        KeyCode::Char('r') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            if app.session_open && app.status == Status::Stopped && !app.in_flight() {
                child.send(serde_json::json!({"type": "resume"}))?;
                app.waiting_from_stopped = true;
                app.status = Status::Waiting;
            }
        }
        KeyCode::Enter => {
            if app.session_open && !app.input.is_empty() && !app.in_flight() {
                let text = std::mem::take(&mut app.input);
                if text.starts_with('/') {
                    handle_slash(app, child, &text)?;
                } else {
                    child.send(serde_json::json!({"type": "user", "text": text}))?;
                    app.transcript.push(TranscriptLine::User(text));
                    app.waiting_from_stopped = false;
                    app.status = Status::Waiting;
                    app.follow = true;
                }
            }
        }
        KeyCode::Backspace => {
            app.input.pop();
        }
        KeyCode::PageUp => {
            app.follow = false;
            app.scroll = app.scroll.saturating_sub(5);
        }
        KeyCode::PageDown => {
            app.scroll = app.scroll.saturating_add(5);
        }
        KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.input.push(ch);
        }
        _ => {}
    }
    Ok(())
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
            app.picker_filter.pop();
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
            app.input.pop();
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
/// `/model` only asks for the catalog. `/rm` without `--yes` waits for Enter
/// and does not send a removal.
fn slash_outcome(app: &mut App, text: &str) -> Option<serde_json::Value> {
    let mut parts = text.split_whitespace();
    let command = parts.next().unwrap_or("");
    match command {
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
                .push(TranscriptLine::Tool(format!("fetching models{suffix}")));
            Some(serde_json::json!({"type": "models"}))
        }
        "/key" => {
            let scope = parts.next().unwrap_or("");
            if scope == "session" || scope == "user" {
                app.mode = Mode::Key {
                    scope: scope.to_string(),
                };
                app.transcript.push(TranscriptLine::Tool(format!(
                    "Enter the OpenRouter key for this {scope}. It stays hidden."
                )));
            } else {
                app.transcript.push(TranscriptLine::Tool(
                    "usage: /key session | /key user".to_string(),
                ));
            }
            None
        }
        "/scope" => {
            let scope = parts.next().unwrap_or("");
            if scope == "project" || scope == "user" {
                app.scope_root = scope.to_string();
                app.transcript
                    .push(TranscriptLine::Tool(format!("scope {scope}")));
            } else {
                app.transcript.push(TranscriptLine::Tool(
                    "usage: /scope project | /scope user".to_string(),
                ));
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
        "/rm" => {
            let kind_word = parts.next().unwrap_or("");
            let name = parts.next().unwrap_or("");
            let flag = parts.next().unwrap_or("");
            let kind = match kind_word {
                "session" => "sessions",
                "artifact" => "artifacts",
                "key" => "keys",
                "weight" | "weights" => "weights",
                other => other,
            };
            if name.is_empty() || kind.is_empty() {
                app.transcript.push(TranscriptLine::Tool(
                    "usage: /rm session NAME [--yes]".to_string(),
                ));
                None
            } else if flag == "--yes" {
                Some(confirmed_remove_message(&app.scope_root, kind, name))
            } else {
                app.transcript.push(TranscriptLine::Tool(format!(
                    "Remove {kind} {name} from {}? Enter confirms, Esc cancels.",
                    app.scope_root
                )));
                app.mode = Mode::ConfirmRemove {
                    scope: app.scope_root.clone(),
                    kind: kind.to_string(),
                    name: name.to_string(),
                };
                None
            }
        }
        _ => {
            app.transcript
                .push(TranscriptLine::Tool(format!("unknown command {command}")));
            None
        }
    }
}

const PICKER_PAGE: usize = 6;

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

fn picker_selection(app: &App) -> Option<serde_json::Value> {
    app.filtered_models()
        .get(app.picker_index)
        .map(|model| serde_json::json!({"type": "select_model", "id": model.id}))
}

fn confirmed_remove_message(scope: &str, kind: &str, name: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "remove",
        "scope": scope,
        "kind": kind,
        "name": name,
        "confirm": true
    })
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
    let status_height = status_block_rows(&app.status_line, outer.width);
    let [transcript_area, picker_area, composer_area, status_area] = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(picker_height),
        Constraint::Length(composer_height),
        Constraint::Length(status_height),
    ])
    .areas(outer);

    render_transcript(frame, app, transcript_area);
    if picker_height > 0 {
        render_picker(frame, app, picker_area);
    }
    render_composer(frame, app, composer_area);
    render_status(frame, app, status_area);
}

/// State row, every wrapped metrics row, and the hint row.
///
/// A fixed 3-row block keeps only the first metrics line, so spend, speed,
/// HTTP, latency, time to first token, and recent calls fall outside an
/// 80-column status rect.
fn status_block_rows(status_line: &str, width: u16) -> u16 {
    let text_width = usize::from(width.max(1));
    let metrics = wrap_text(status_line, text_width).len().clamp(1, 8) as u16;
    metrics.saturating_add(2)
}

fn picker_rows(app: &App) -> u16 {
    match app.mode {
        Mode::ModelWait => 1,
        Mode::ModelPicker => {
            let count = app.filtered_models().len().clamp(1, PICKER_PAGE) as u16;
            count.saturating_add(1)
        }
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
    if app.follow {
        app.scroll = max_scroll;
    }
    app.scroll = app.scroll.min(max_scroll);
    let start = app.scroll as usize;
    for (offset, row) in rows.iter().skip(start).take(height).enumerate() {
        let row_area = Rect {
            x: area.x,
            y: area.y.saturating_add(offset as u16),
            width: area.width,
            height: 1,
        };
        paint_row(frame, row_area, row);
    }
}

enum PaintedRow {
    Gap,
    Text {
        kind: RowKind,
        first: bool,
        text: String,
    },
}

fn transcript_rows(app: &App, width: usize) -> Vec<PaintedRow> {
    let content_width = width.saturating_sub(2).max(1);
    let mut rows = Vec::new();
    for (index, line) in app.transcript.iter().enumerate() {
        if index > 0 {
            rows.push(PaintedRow::Gap);
        }
        let (kind, text) = match line {
            TranscriptLine::User(text) => (RowKind::User, text.as_str()),
            TranscriptLine::Assistant(text) => (RowKind::Assistant, text.as_str()),
            TranscriptLine::Tool(text) => (RowKind::Tool, text.as_str()),
        };
        let wrapped = wrap_text(text, content_width);
        let wrapped = if wrapped.is_empty() {
            vec![String::new()]
        } else {
            wrapped
        };
        for (line_index, chunk) in wrapped.into_iter().enumerate() {
            rows.push(PaintedRow::Text {
                kind,
                first: line_index == 0,
                text: chunk,
            });
        }
    }
    rows
}

#[derive(Clone, Copy)]
enum RowKind {
    User,
    Assistant,
    Tool,
}

fn paint_row(frame: &mut Frame, area: Rect, row: &PaintedRow) {
    let PaintedRow::Text { kind, first, text } = row else {
        return;
    };
    let (background, prefix, color) = match kind {
        RowKind::User => (BG_LIGHT, if *first { "› " } else { "  " }, FG),
        RowKind::Assistant => (BG_BASE, "  ", FG),
        RowKind::Tool => (BG_BASE, if *first { "◆ " } else { "  " }, GRAY_BRIGHT),
    };
    frame.render_widget(
        Block::default().style(Style::default().bg(background)),
        area,
    );
    let line = Line::from(vec![
        Span::styled(prefix, Style::default().fg(color).bg(background)),
        Span::styled(text.as_str(), Style::default().fg(color).bg(background)),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

fn render_picker(frame: &mut Frame, app: &App, area: Rect) {
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
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            let prompt = model.prompt_price.as_deref().unwrap_or("n/a");
            let completion = model.completion_price.as_deref().unwrap_or("n/a");
            let mark = if index == app.picker_index { ">" } else { " " };
            let style = if index == app.picker_index {
                Style::default().fg(FG).bg(BG_LIGHT)
            } else {
                Style::default().fg(FG_SECONDARY).bg(BG_BASE)
            };
            lines.push(Line::from(Span::styled(
                format!(
                    "{mark} {}  ctx {context}  ${prompt} / ${completion}",
                    model.id
                ),
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
    let hint = if app.session_open {
        "enter send  /model /key /rm  ctrl-r resume  ctrl-c quit"
    } else {
        "ctrl-c quit"
    };
    let state = Line::from(vec![
        Span::styled("Atlas", Style::default().fg(GRAY_BRIGHT).bg(BG_BASE)),
        Span::styled("  ", Style::default().bg(BG_BASE)),
        Span::styled(label, Style::default().fg(color).bg(BG_BASE)),
        Span::styled(
            format!("  {}  scope {}", app.model_id, app.scope_root),
            Style::default().fg(GRAY).bg(BG_BASE),
        ),
    ]);
    let metrics = Paragraph::new(app.status_line.as_str())
        .style(Style::default().fg(FG_SECONDARY).bg(BG_BASE))
        .wrap(Wrap { trim: false });
    let [state_area, metrics_area, hint_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(1),
        Constraint::Length(1),
    ])
    .areas(area);
    frame.render_widget(
        Paragraph::new(state).style(Style::default().bg(BG_BASE)),
        state_area,
    );
    frame.render_widget(metrics, metrics_area);
    frame.render_widget(
        Paragraph::new(Span::styled(hint, Style::default().fg(GRAY).bg(BG_BASE))),
        hint_area,
    );
}

fn status_label(app: &App) -> (&'static str, Color) {
    if !app.session_open {
        return ("session ended", RED);
    }
    match app.status {
        Status::Ready => ("ready", GRAY_BRIGHT),
        Status::Waiting => ("waiting", MAGENTA),
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
    app.transcript
        .push(TranscriptLine::Tool(format!("error: {detail}")));
    app.status = Status::Ready;
    app.waiting_from_stopped = false;
}

#[cfg(test)]
mod tests {
    use super::{
        confirmed_remove_message, draw, picker_selection, slash_outcome, wrap_text, App, Mode,
        TranscriptLine,
    };
    use crate::protocol::{CatalogEntry, ServerMessage, StatusSnapshot};

    fn tool_text(app: &App) -> String {
        app.transcript
            .iter()
            .filter_map(|line| match line {
                TranscriptLine::Tool(text) => Some(text.clone()),
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
    fn remove_waits_for_confirmation_and_stays_in_scope() {
        let mut app = App::new();
        app.scope_root = "project".to_string();
        assert!(slash_outcome(&mut app, "/rm session s1").is_none());
        assert_eq!(
            app.mode,
            Mode::ConfirmRemove {
                scope: "project".to_string(),
                kind: "sessions".to_string(),
                name: "s1".to_string(),
            }
        );

        let payload = confirmed_remove_message("project", "sessions", "s1");
        assert_eq!(payload["type"], "remove");
        assert_eq!(payload["scope"], "project");
        assert_eq!(payload["kind"], "sessions");
        assert_eq!(payload["name"], "s1");
        assert_eq!(payload["confirm"], true);

        let mut flagged = App::new();
        let sent = slash_outcome(&mut flagged, "/rm session s1 --yes").expect("flagged remove");
        assert_eq!(sent, payload);
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

    #[test]
    fn status_on_an_80_column_terminal_keeps_the_metrics() {
        let mut app = App::new();
        app.model_id = "example/model".to_string();
        app.status_line = "model example/model  context 12/128000  tokens 16 (prompt 12 completion 4)  spend $0.02  8.0 tok/s  http 200  latency 0.50s  ttft 0.10s  recent 200 0.50s".to_string();

        let painted = screen(&mut app, 80, 24);
        let flat = painted.split_whitespace().collect::<Vec<_>>().join(" ");

        for field in [
            "spend $0.02",
            "8.0 tok/s",
            "http 200",
            "latency 0.50s",
            "ttft 0.10s",
            "recent 200 0.50s",
        ] {
            assert!(
                flat.contains(field),
                "missing {field} in status rect:\n{painted}"
            );
        }
    }

    #[test]
    fn picker_highlights_the_row_enter_would_select() {
        let mut app = App::new();
        app.mode = Mode::ModelPicker;
        app.picker_index = 6;
        app.picker_models = (0..8)
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

        assert_eq!(selected["id"], "model-6");
        assert!(
            highlighted.iter().any(|line| line.contains("model-6")),
            "highlighted rows were {highlighted:?}\n{painted}"
        );
        assert!(
            !painted.contains("model-0"),
            "hidden first row was painted:\n{painted}"
        );
    }
}
