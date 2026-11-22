//! Overlays: activity spinner, slash/@file completion popup, fuzzy model picker, and `/help`.

use std::ops::Range;

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph};
use ratatui::Frame;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::app::{Activity, App, Mode};
use crate::commands::{CompletionList, COMMANDS};
use crate::markdown::wrap_plain;
use crate::protocol::{format_elapsed, format_tokens, price_per_million, CatalogEntry};
use crate::render::composer::render_textarea;
use crate::render::transcript::SPINNER;
use crate::theme::Theme;

const PICKER_PAGE: usize = 8;
const MENU_PAGE: usize = 10;

pub fn picker_rows(app: &App) -> u16 {
    match &app.mode {
        Mode::ModelPicker(s) if s.loading => 3,
        Mode::ModelPicker(s) => {
            (app.filtered_models(&s.filter.lines().join(" "))
                .len()
                .clamp(1, PICKER_PAGE) as u16)
                + 4
        }
        Mode::Compose => app
            .active_completions()
            .map_or(0, |l| (l.items.len().clamp(1, MENU_PAGE) as u16) + 1),
        _ => 0,
    }
}

pub fn render_activity(frame: &mut Frame<'_>, app: &App, area: Rect, theme: &Theme) {
    let glyph = SPINNER[(app.ticks as usize / 2) % SPINNER.len()];
    let label = match app.activity.as_ref().unwrap_or(&Activity::Working) {
        Activity::Thinking => "thinking…".into(),
        Activity::Working => "working…".into(),
        Activity::ToolCalling(n) if !n.is_empty() => format!("calling {n}…"),
        Activity::ToolCalling(_) => "calling tool…".into(),
        Activity::ToolExecuting(n) if !n.is_empty() => format!("running {n}…"),
        Activity::ToolExecuting(_) => "running tool…".into(),
    };
    let label = match app.activity_since.map(|since| since.elapsed().as_secs()) {
        Some(secs) if secs >= 5 => format!("{label} {}", format_elapsed(secs)),
        _ => label,
    };
    let line = Line::from(vec![
        Span::styled(glyph, theme.fg(theme.accent)),
        Span::styled(format!(" {label}"), theme.fg(theme.fg_secondary)),
    ]);
    frame.render_widget(Paragraph::new(line).style(theme.base()), area);
}

pub fn render_picker(frame: &mut Frame<'_>, app: &mut App, area: Rect, theme: &Theme) {
    if matches!(app.mode, Mode::ModelPicker(_)) {
        render_model_picker(frame, app, area, theme);
    } else if let Some(list) = app.active_completions() {
        render_completion_menu(frame, &list, app.menu_index, area, theme);
    }
}

fn row_style(sel: bool, t: &Theme) -> Style {
    [
        t.fg(t.fg_secondary),
        t.surface(t.fg).add_modifier(Modifier::BOLD),
    ][usize::from(sel)]
}

fn render_completion_menu(
    frame: &mut Frame<'_>,
    list: &CompletionList,
    idx: usize,
    area: Rect,
    theme: &Theme,
) {
    let mut lines = vec![Line::from(Span::styled(
        "↑↓ select · Tab complete · Enter run · Esc close",
        theme.muted_style(),
    ))];
    let idx = idx.min(list.items.len().saturating_sub(1));
    let win = scroll_window(list.items.len(), idx, MENU_PAGE);
    let w = usize::from(area.width);
    for (i, item) in list
        .items
        .iter()
        .enumerate()
        .skip(win.start)
        .take(win.len())
    {
        let sel = i == idx;
        let left = format!("{} {}", [" ", "›"][usize::from(sel)], item.label);
        let gap = w
            .saturating_sub(left.width() + item.description.width())
            .max(2);
        let row = format!("{left}{}{}", " ".repeat(gap), item.description);
        lines.push(Line::from(Span::styled(
            truncate_width(&row, w),
            row_style(sel, theme),
        )));
    }
    frame.render_widget(Paragraph::new(lines).style(theme.base()), area);
}

fn render_model_picker(frame: &mut Frame<'_>, app: &mut App, area: Rect, theme: &Theme) {
    let block = panel_block(" Models (fuzzy filter · Enter select · Esc close) ", theme);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height == 0 || inner.width == 0 {
        return;
    }
    let Mode::ModelPicker(ref s) = app.mode else {
        return;
    };
    if s.loading {
        frame.render_widget(
            Paragraph::new("Fetching model catalog…").style(theme.muted_style()),
            inner,
        );
        return;
    }
    let (query, idx) = (s.filter.lines().join(" "), s.index);
    let matches: Vec<CatalogEntry> = app.filtered_models(&query).into_iter().cloned().collect();
    let [filter_a, list_a] =
        Layout::vertical([Constraint::Length(1), Constraint::Min(1)]).areas(inner);

    let badge = format!(" {}/{}", matches.len(), app.cached_models.len());
    let bw = u16::try_from(badge.width()).unwrap_or(8);
    frame.render_widget(
        Paragraph::new("› ").style(theme.fg(theme.accent)),
        Rect::new(filter_a.x, filter_a.y, 2.min(filter_a.width), 1),
    );
    if let Mode::ModelPicker(s) = &mut app.mode {
        let r = Rect::new(
            filter_a.x + 2,
            filter_a.y,
            filter_a.width.saturating_sub(bw + 2),
            1,
        );
        render_textarea(
            frame,
            &mut s.filter,
            r,
            "Type to fuzzy-filter models…",
            true,
            theme,
        );
    }
    if filter_a.width > bw + 2 {
        let r = Rect::new(filter_a.right().saturating_sub(bw), filter_a.y, bw, 1);
        frame.render_widget(Paragraph::new(badge).style(theme.muted_style()), r);
    }
    if matches.is_empty() {
        frame.render_widget(
            Paragraph::new("  no matching models").style(theme.fg(theme.yellow)),
            list_a,
        );
        return;
    }

    let clamped = idx.min(matches.len() - 1);
    let win = scroll_window(
        matches.len(),
        clamped,
        usize::from(list_a.height).clamp(1, PICKER_PAGE),
    );
    let (w, cur) = (
        usize::from(list_a.width),
        app.status_snapshot.model.as_str(),
    );
    let price = |l: &str, p: Option<&str>| {
        p.and_then(price_per_million)
            .map_or_else(|| format!("{l} n/a"), |v| format!("{l} ${v}/M"))
    };
    let lines: Vec<Line> = matches
        .iter()
        .enumerate()
        .skip(win.start)
        .take(win.len())
        .map(|(i, m)| {
            let sel = i == clamped;
            let mark = match (sel, m.id == cur) {
                (true, _) => "›",
                (_, true) => "*",
                _ => " ",
            };
            let ctx = m.context_length.map_or_else(|| "?".into(), format_tokens);
            let nt = ["  [no tools]", ""][usize::from(m.supports_tools)];
            let row = format!(
                "{mark} {}  ctx {ctx}  {}  {}{nt}",
                m.id,
                price("in", m.prompt_price.as_deref()),
                price("out", m.completion_price.as_deref())
            );
            Line::from(Span::styled(truncate_width(&row, w), row_style(sel, theme)))
        })
        .collect();
    frame.render_widget(Paragraph::new(lines).style(theme.base()), list_a);
}

pub fn help_lines(app: &App) -> Vec<String> {
    let mut lines = vec!["Slash Commands".into()];
    lines.extend(COMMANDS.iter().map(|(tok, args, desc, _)| {
        let u = if args.is_empty() {
            (*tok).into()
        } else {
            format!("{tok} {args}")
        };
        format!("  {u:<22} {desc}")
    }));
    lines.extend([String::new(), "Keybindings".into()]);
    const KEYS: &[(&str, &str)] = &[
        ("Enter", "Send prompt, run slash command, or toggle tool"),
        ("Alt/Shift+Enter, Ctrl+J", "Insert newline in composer"),
        ("Tab", "Complete /cmd, /skill, /resume, /key, or @file"),
        ("↑ / ↓", "Navigate prompt history or completion menu"),
        (
            "Shift/Alt+↑ / ↓",
            "Select previous/next expandable tool/thought",
        ),
        (
            "← / → (empty), Ctrl+O",
            "Collapse/expand or toggle selected tool",
        ),
        ("Click tool row", "Toggle tool/thought details"),
        ("PgUp / PgDn", "Scroll transcript by one viewport"),
        (
            "Home / End (or Ctrl+)",
            "Jump to top / bottom of transcript",
        ),
        (
            "Esc",
            "Dismiss menu, cancel turn (2nd Esc warns), or clear selection",
        ),
        (
            "Ctrl+R",
            "Resume after a tool-limit stop or interrupted turn",
        ),
        ("Ctrl+Z / Ctrl+Y", "Undo / redo edits in composer"),
        ("Ctrl+C", "Copy active selection, or quit Atlas"),
        (
            "Approval: y/a/n/d",
            "Allow once / session / deny / deny + reason",
        ),
    ];
    lines.extend(KEYS.iter().map(|(k, d)| format!("  {k:<24} {d}")));
    lines.extend([String::new(), format!("Loaded Tools ({})", app.tools.len())]);
    if app.tools.is_empty() {
        lines.push("  (none registered yet)".into());
    } else {
        lines.extend(app.tools.iter().map(|t| format!("  {t}")));
    }
    lines
}

pub fn render_help(frame: &mut Frame<'_>, app: &mut App, area: Rect, theme: &Theme) {
    let block = panel_block(" Atlas Help ", theme);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.width < 8 || inner.height < 3 {
        return;
    }
    let wrapped: Vec<String> = help_lines(app)
        .iter()
        .flat_map(|l| wrap_plain(l, usize::from(inner.width)))
        .collect();
    let body_h = usize::from(inner.height.saturating_sub(1));
    let max_scroll = wrapped.len().saturating_sub(body_h);
    let scroll = match &mut app.mode {
        Mode::Help(s) => {
            *s = (*s).min(max_scroll);
            *s
        }
        _ => 0,
    };
    let [body_a, foot_a] =
        Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(inner);
    let shown: Vec<Line> = wrapped
        .iter()
        .skip(scroll)
        .take(body_h)
        .map(|s| Line::from(s.as_str()))
        .collect();
    frame.render_widget(
        Paragraph::new(shown).style(theme.fg(theme.fg_secondary)),
        body_a,
    );
    frame.render_widget(
        Paragraph::new("Esc / q close · ↑↓ / j k / PgUp / PgDn scroll").style(theme.muted_style()),
        foot_a,
    );
}

fn panel_block<'a>(title: &'a str, theme: &Theme) -> Block<'a> {
    Block::default()
        .title(Line::from(Span::styled(title, theme.bold(theme.accent))))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme.border_active))
        .style(theme.base())
}

fn scroll_window(len: usize, index: usize, page: usize) -> Range<usize> {
    if len == 0 || page == 0 {
        return 0..0;
    }
    let page = page.min(len);
    let start = index
        .min(len - 1)
        .saturating_add(1)
        .saturating_sub(page)
        .min(len - page);
    start..(start + page)
}

fn truncate_width(text: &str, max_width: usize) -> String {
    let mut used = 0;
    text.chars()
        .take_while(|ch| {
            used += ch.width().unwrap_or(0);
            used <= max_width
        })
        .collect()
}
