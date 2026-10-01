//! Minimal terminal client for the Atlas agent session protocol.

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
    KeyModifiers, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen,
};
use markdown::{render_lines, Mark, Piece};
use protocol::{default_status_line, parse_line, status_text, CatalogEntry, ServerMessage};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
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
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Activity {
    Thinking,
    Working,
    ToolCalling(String),
    ToolExecuting(String),
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
            menu_index: 0,
            scope_root: "project".to_string(),
            lookup_context: false,
            scroll_max: 0,
            ticks: 0,
            activity: None,
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
            ServerMessage::Step { name, ok, .. } => {
                let label = if ok { name } else { format!("{name} failed") };
                self.transcript.push(TranscriptLine::Tool(label));
            }
            ServerMessage::Phase { phase, name } => {
                self.activity = activity_from(&phase, name.as_deref());
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
                    .push(TranscriptLine::Assistant(format!("error: {message}")));
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
                        self.transcript.push(TranscriptLine::Tool(format!(
                            "model list failed: {message}"
                        )));
                        if self.mode == Mode::ModelWait {
                            self.mode = Mode::Compose;
                        }
                    }
                } else if user_asked {
                    self.picker_models = models;
                    self.picker_index = 0;
                    self.mode = Mode::ModelPicker;
                    let count = self.filtered_models().len();
                    self.transcript
                        .push(TranscriptLine::Tool(format!("models: {count}")));
                } else {
                    self.picker_models = models;
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
        KeyCode::Up if menu_open => {
            app.menu_index = app.menu_index.saturating_sub(1);
        }
        KeyCode::Down if menu_open => {
            let count = menu_options(&app.input).len();
            if count > 0 && app.menu_index + 1 < count {
                app.menu_index += 1;
            }
        }
        KeyCode::Up => nudge_scroll(app, -1),
        KeyCode::Down => nudge_scroll(app, 1),
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
            app.transcript.push(TranscriptLine::Tool(note.to_string()));
        }
        None => {
            app.transcript
                .push(TranscriptLine::Tool("nothing to copy".to_string()));
        }
    }
    app.follow = true;
    true
}

fn last_reply(app: &App) -> Option<&str> {
    app.transcript.iter().rev().find_map(|line| match line {
        TranscriptLine::Assistant(text) if !text.is_empty() => Some(text.as_str()),
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
/// `/model` only asks for the catalog.
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
                .push(TranscriptLine::Tool(format!("unknown command {command}")));
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
    const COMMANDS: [MenuItem; 7] = [
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
        "/sessions" | "/artifacts" | "/quit" | "/exit" | "/copy"
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
    let status_height = status_block_rows(&app.status_line, outer.width);
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
    if activity_height > 0 {
        render_activity(frame, app, activity_area);
    }
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
    Rich {
        pieces: Vec<Piece>,
    },
}

fn transcript_rows(app: &App, width: usize) -> Vec<PaintedRow> {
    let content_width = width.saturating_sub(2).max(1);
    let mut rows = Vec::new();
    for (index, line) in app.transcript.iter().enumerate() {
        if index > 0 {
            rows.push(PaintedRow::Gap);
        }
        match line {
            TranscriptLine::Assistant(text) => {
                for rich in render_lines(text) {
                    for pieces in wrap_rich(&rich, content_width) {
                        rows.push(PaintedRow::Rich { pieces });
                    }
                }
            }
            TranscriptLine::User(text) | TranscriptLine::Tool(text) => {
                let kind = match line {
                    TranscriptLine::User(_) => RowKind::User,
                    _ => RowKind::Tool,
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
        }
    }
    rows
}

#[derive(Clone, Copy)]
enum RowKind {
    User,
    Tool,
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
    let (background, prefix, color) = match kind {
        RowKind::User => (BG_LIGHT, if *first { "› " } else { "  " }, FG),
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

fn rich_style(mark: Mark) -> Style {
    let base = Style::default().fg(FG).bg(BG_BASE);
    match mark {
        Mark::Plain => base,
        Mark::Strong | Mark::Heading => base.add_modifier(Modifier::BOLD),
        Mark::Emphasis => base.add_modifier(Modifier::ITALIC),
        Mark::Code => base.fg(MAGENTA),
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
    let hint = if app.session_open {
        "enter send  /model /key /quit  ctrl-r resume  ctrl-c quit"
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
        Status::Waiting => match app.activity {
            Some(Activity::Thinking) => ("thinking", MAGENTA),
            Some(Activity::ToolCalling(_)) => ("tool calling", MAGENTA),
            Some(Activity::ToolExecuting(_)) => ("tool executing", MAGENTA),
            _ => ("model working", MAGENTA),
        },
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
        draw, format_context, menu_action, menu_options, nudge_scroll, picker_selection,
        price_per_million, slash_outcome, wrap_text, App, MenuAction, Mode, Status, TranscriptLine,
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
    fn slash_menu_lists_every_command() {
        let mut app = App::new();
        app.input = "/".to_string();
        let painted = screen(&mut app, 80, 24);
        for name in [
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
    fn assistant_markdown_is_not_painted_raw() {
        let mut app = App::new();
        app.transcript
            .push(TranscriptLine::Assistant("**bold** and `code`".to_string()));
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
            text: Some("SECRET_FILE_BODY".to_string()),
            error: None,
            error_kind: None,
            artifacts: Vec::new(),
        });
        let painted = screen(&mut app, 80, 24);
        assert!(painted.contains("read_file"), "{painted}");
        assert!(!painted.contains("SECRET_FILE_BODY"), "{painted}");
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
            MenuAction::Run("/model".to_string())
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
