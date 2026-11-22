//! Slim entrypoint and event-driven terminal loop for `atlas-tui`.

mod app;
mod approval;
mod clipboard;
mod commands;
mod history;
mod input;
mod markdown;
mod process;
mod protocol;
mod render;
mod selection;
mod theme;

#[cfg(test)]
mod tests;

use std::io::{self, BufRead, BufReader, Read, Write};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event, KeyEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, BeginSynchronizedUpdate, Clear, ClearType,
    EndSynchronizedUpdate, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;

use crate::app::App;
use crate::history::PromptHistory;
use crate::input::{apply_paste, handle_key, handle_mouse};
use crate::process::{ChildProcess, CliArgs, LaunchConfig};
use crate::protocol::{parse_line, ClientMessage, ServerMessage};
use crate::render::draw;
use crate::theme::Theme;

const BUSY_POLL: Duration = Duration::from_millis(50);
const IDLE_POLL: Duration = Duration::from_millis(120);

fn main() {
    if let Err(err) = run() {
        eprintln!("atlas-tui: {err:#}");
        std::process::exit(1);
    }
}

fn spawn_reader<R: Read + Send + 'static>(pipe: R) -> Receiver<String> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        for line in BufReader::new(pipe).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    rx
}

fn run() -> Result<()> {
    let config = LaunchConfig::from_cli(CliArgs::parse())?;
    let mut child = ChildProcess::spawn(&config)?;
    let rx = spawn_reader(
        child
            .child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("no child stdout"))?,
    );
    let errx = spawn_reader(
        child
            .child
            .stderr
            .take()
            .ok_or_else(|| anyhow!("no child stderr"))?,
    );

    let mut app = App::new(config.workspace.clone(), PromptHistory::load_default());
    let first = match recv_first_line(&mut child, &rx) {
        Ok(line) => line,
        Err(err) => {
            thread::sleep(Duration::from_millis(50));
            let msg: String = errx.try_iter().collect::<Vec<_>>().join(" ");
            child.shutdown();
            return Err(if msg.is_empty() {
                err
            } else {
                err.context(msg)
            });
        }
    };
    match parse_line(&first) {
        Ok(Some(ServerMessage::Error(err))) => {
            child.shutdown();
            println!("atlas-tui: {}\nPress enter to exit.", err.message);
            let _ = disable_raw_mode();
            let _ = io::stdin().read_line(&mut String::new());
            return Ok(());
        }
        Ok(Some(msg)) => app.apply(msg),
        Ok(None) => return Err(anyhow!("agent session sent an empty first line")),
        Err(_) => return Err(anyhow!("agent session sent a non-json first line: {first}")),
    }

    let mut terminal = enter_ui()?;
    struct UiGuard;
    impl Drop for UiGuard {
        fn drop(&mut self) {
            leave_ui();
        }
    }
    let ui = UiGuard;
    let prev = Arc::new(std::panic::take_hook());
    std::panic::set_hook(Box::new(move |info| {
        leave_ui();
        prev(info);
    }));
    let res = event_loop(
        &mut terminal,
        &mut app,
        &mut child,
        &rx,
        &errx,
        Theme::current(),
    );
    drop(ui);
    child.shutdown();
    res
}

fn event_loop(
    terminal: &mut ratatui::DefaultTerminal,
    app: &mut App,
    child: &mut ChildProcess,
    rx: &Receiver<String>,
    errx: &Receiver<String>,
    theme: &Theme,
) -> Result<()> {
    let mut last_tick = Instant::now();
    loop {
        let mut closed = false;
        loop {
            match rx.try_recv() {
                Ok(line) => match parse_line(&line) {
                    Ok(Some(msg)) => app.apply(msg),
                    Ok(None) => {}
                    Err(_) => app.push_notice(format!("unparsable: {line}"), true),
                },
                Err(mpsc::TryRecvError::Empty) => {
                    if std::mem::take(&mut app.lookup_context) {
                        child.send(&ClientMessage::Models {
                            refresh: Some(false),
                        })?;
                    }
                    break;
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    closed = true;
                    break;
                }
            }
        }
        for line in errx.try_iter() {
            app.last_stderr = Some(line);
        }
        if closed {
            app.note_session_closed();
        }

        if app.in_flight() && app.pending_permission.is_none() {
            if last_tick.elapsed() >= BUSY_POLL {
                app.ticks = app.ticks.wrapping_add(1);
                app.dirty = true;
                last_tick = Instant::now();
            }
        } else {
            last_tick = Instant::now();
        }

        if app.dirty {
            let _ = execute!(io::stdout(), BeginSynchronizedUpdate);
            let res = terminal.draw(|frame| draw(frame, app, theme));
            let _ = execute!(io::stdout(), EndSynchronizedUpdate);
            let _ = io::stdout().flush();
            res?;
            app.dirty = false;
        }

        if event::poll([IDLE_POLL, BUSY_POLL][usize::from(app.in_flight())])? {
            loop {
                match event::read()? {
                    Event::Key(k)
                        if k.kind == KeyEventKind::Press && handle_key(app, child, k)? =>
                    {
                        return Ok(())
                    }
                    Event::Mouse(m) => handle_mouse(app, child, m)?,
                    Event::Paste(t) => apply_paste(app, &t),
                    Event::Resize(_, _) => (app.text_selection, app.dirty) = (None, true),
                    _ => {}
                }
                if !event::poll(Duration::ZERO)? {
                    break;
                }
            }
        }
    }
}

fn enter_ui() -> Result<ratatui::DefaultTerminal> {
    let ctx = "atlas-tui needs an interactive terminal";
    enable_raw_mode().context(ctx)?;
    let mut stdout = io::stdout();
    execute!(
        stdout,
        EnterAlternateScreen,
        Clear(ClearType::All),
        EnableMouseCapture,
        EnableBracketedPaste
    )
    .inspect_err(|_| {
        let _ = disable_raw_mode();
    })
    .context(ctx)?;
    ratatui::Terminal::new(CrosstermBackend::new(stdout))
        .inspect_err(|_| leave_ui())
        .context(ctx)
}

fn leave_ui() {
    let _ = execute!(
        io::stdout(),
        EndSynchronizedUpdate,
        DisableBracketedPaste,
        DisableMouseCapture,
        LeaveAlternateScreen
    );
    let _ = disable_raw_mode();
}

fn recv_first_line(child: &mut ChildProcess, rx: &Receiver<String>) -> Result<String> {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let rem = deadline.saturating_duration_since(Instant::now());
        if rem.is_zero() {
            return Err(anyhow!("agent session did not start"));
        }
        match rx.recv_timeout(rem.min(Duration::from_millis(200))) {
            Ok(line) => return Ok(line),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(anyhow!("agent session exited before it was ready"))
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if let Ok(Some(st)) = child.child.try_wait() {
                    return Err(anyhow!("agent session exited before it was ready ({st})"));
                }
            }
        }
    }
}
