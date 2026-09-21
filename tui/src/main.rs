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
use protocol::{parse_line, ServerMessage};
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
        }
    }

    fn in_flight(&self) -> bool {
        matches!(self.status, Status::Waiting)
    }

    fn apply(&mut self, message: ServerMessage) {
        match message {
            ServerMessage::Ready { tools } => {
                self.tools = tools.into_iter().map(|tool| tool.name).collect();
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
            } => {
                if !response.is_empty() {
                    self.transcript.push(TranscriptLine::Assistant(response));
                }
                self.waiting_from_stopped = false;
                self.status = if stopped_for_limit {
                    Status::Stopped
                } else {
                    Status::Ready
                };
            }
            ServerMessage::Error { message } => {
                self.transcript
                    .push(TranscriptLine::Assistant(format!("error: {message}")));
                self.status = if self.waiting_from_stopped {
                    Status::Stopped
                } else {
                    Status::Ready
                };
                self.waiting_from_stopped = false;
                self.follow = true;
            }
            ServerMessage::Unknown => {}
        }
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
        Ok(Some(ServerMessage::Error { message })) => {
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
    match key.code {
        KeyCode::Esc => {
            child.shutdown();
            return Ok(true);
        }
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            child.shutdown();
            return Ok(true);
        }
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
                child.send(serde_json::json!({"type": "user", "text": text}))?;
                app.transcript.push(TranscriptLine::User(text));
                app.waiting_from_stopped = false;
                app.status = Status::Waiting;
                app.follow = true;
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
    Ok(false)
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
    if outer.width < 8 || outer.height < 5 {
        return;
    }

    let composer_height = composer_rows(app, outer.width);
    let [transcript_area, composer_area, status_area] = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(composer_height),
        Constraint::Length(1),
    ])
    .areas(outer);

    render_transcript(frame, app, transcript_area);
    render_composer(frame, app, composer_area);
    render_status(frame, app, status_area);
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

fn render_composer(frame: &mut Frame, app: &App, area: Rect) {
    let active = app.session_open && !app.in_flight();
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
    if app.input.is_empty() {
        spans.push(Span::styled("Message Atlas", Style::default().fg(GRAY)));
    } else {
        spans.push(Span::styled(app.input.as_str(), Style::default().fg(FG)));
        spans.push(Span::styled("▌", Style::default().fg(MAGENTA)));
    }
    let input = Paragraph::new(Line::from(spans))
        .block(block)
        .wrap(Wrap { trim: false });
    frame.render_widget(input, area);
}

fn render_status(frame: &mut Frame, app: &App, area: Rect) {
    let (label, color) = status_label(app);
    let left = Line::from(vec![
        Span::styled("Atlas", Style::default().fg(GRAY_BRIGHT).bg(BG_BASE)),
        Span::styled("  ", Style::default().bg(BG_BASE)),
        Span::styled(label, Style::default().fg(color).bg(BG_BASE)),
    ]);
    let hint = if app.session_open {
        "enter send   ctrl-r resume   ctrl-c quit"
    } else {
        "ctrl-c quit"
    };
    let hint_width = hint.chars().count() as u16;
    let [left_area, right_area] = Layout::horizontal([
        Constraint::Min(1),
        Constraint::Length(hint_width.min(area.width.saturating_sub(1))),
    ])
    .areas(area);
    frame.render_widget(
        Paragraph::new(left).style(Style::default().bg(BG_BASE)),
        left_area,
    );
    if area.width > hint_width.saturating_add(8) {
        frame.render_widget(
            Paragraph::new(Span::styled(hint, Style::default().fg(GRAY).bg(BG_BASE))),
            right_area,
        );
    }
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
    use super::wrap_text;

    #[test]
    fn wraps_on_word_boundaries() {
        assert_eq!(wrap_text("one two three", 7), vec!["one two", "three"]);
    }
}
