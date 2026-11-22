//! Transcript rendering: cached markdown, `◆ <name> <summary> (<duration>)` tool rows,
//! live tool output, colored unified diffs, `usize` scroll, scrollbar, and unseen-output badge.

use ratatui::layout::{Margin, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState};
use ratatui::Frame;
use serde_json::Value;
use unicode_width::UnicodeWidthStr;

use crate::app::{App, ToolEntry, ToolHit, TranscriptEntry};
use crate::markdown::{style_diff_lines, wrap_plain};
use crate::theme::Theme;

pub const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

struct Row {
    entry: Option<usize>,
    header: bool,
    bg: Color,
    line: Line<'static>,
}

struct Rows<'a> {
    list: Vec<Row>,
    theme: &'a Theme,
}

impl<'a> Rows<'a> {
    fn raw(&mut self, entry: Option<usize>, header: bool, bg: Color, line: Line<'static>) {
        self.list.push(Row {
            entry,
            header,
            bg,
            line,
        });
    }

    fn pair(&mut self, entry: Option<usize>, p: &str, text: impl Into<String>, st: Style) {
        let line = Line::from(vec![
            Span::styled(p.to_string(), self.theme.base()),
            Span::styled(text.into(), st),
        ]);
        self.raw(entry, false, self.theme.bg_base, line);
    }

    fn diff(&mut self, entry: Option<usize>, diff: &str, width: usize) {
        let indent = Span::styled("    ", self.theme.base());
        for dline in style_diff_lines(diff, width, self.theme) {
            let spans = std::iter::once(indent.clone())
                .chain(dline.spans)
                .collect::<Vec<_>>();
            self.raw(entry, false, self.theme.bg_base, Line::from(spans));
        }
    }
}

pub fn render_transcript(frame: &mut Frame<'_>, app: &mut App, area: Rect, theme: &Theme) {
    if area.width == 0 || area.height == 0 {
        app.tool_hits.clear();
        return;
    }
    app.viewport_height = usize::from(area.height);

    let show_bar = area.width >= 20;
    let text_area = Rect::new(
        area.x,
        area.y,
        area.width - u16::from(show_bar),
        area.height,
    );
    let rows = build_rows(
        app,
        usize::from(text_area.width).saturating_sub(2).max(1),
        theme,
    );
    let height = usize::from(text_area.height);
    let max_scroll = rows.len().saturating_sub(height);
    app.scroll_max = max_scroll;
    if app.follow {
        (app.scroll, app.unseen_output) = (max_scroll, false);
    }
    if std::mem::take(&mut app.reveal_selected) {
        if let Some(pos) = app
            .selected
            .and_then(|s| rows.iter().position(|r| r.header && r.entry == Some(s)))
        {
            if pos < app.scroll {
                (app.scroll, app.follow) = (pos, false);
            } else if height > 0 && pos >= app.scroll + height {
                (app.scroll, app.follow) = (pos + 1 - height, false);
            }
        }
    }
    app.scroll = app.scroll.min(max_scroll);
    app.unseen_output &= app.scroll < max_scroll;

    app.tool_hits.clear();
    for (offset, row) in rows.iter().skip(app.scroll).take(height).enumerate() {
        let y = text_area.y.saturating_add(offset as u16);
        if let Some(idx) = row.entry {
            if let Some(last) = app
                .tool_hits
                .last_mut()
                .filter(|h| h.index == idx && h.y + h.height == y)
            {
                last.height = last.height.saturating_add(1);
            } else {
                app.tool_hits.push(ToolHit {
                    y,
                    height: 1,
                    index: idx,
                });
            }
        }
        let rect = Rect::new(text_area.x, y, text_area.width, 1);
        frame.render_widget(Block::default().style(Style::default().bg(row.bg)), rect);
        frame.render_widget(Paragraph::new(row.line.clone()), rect);
    }

    if show_bar && max_scroll > 0 {
        let mut sb = ScrollbarState::new(max_scroll + 1).position(app.scroll);
        let bar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(None)
            .end_symbol(None)
            .track_symbol(Some("│"))
            .thumb_symbol("┃")
            .track_style(theme.fg(theme.border))
            .thumb_style(theme.fg(theme.accent));
        frame.render_stateful_widget(bar, area.inner(Margin::new(0, 0)), &mut sb);
    }

    if !app.follow && app.unseen_output && text_area.height >= 2 {
        let badge = " ↓ new output (End) ";
        let bw = u16::try_from(badge.width())
            .unwrap_or(20)
            .min(text_area.width);
        let rect = Rect::new(text_area.right() - bw, text_area.bottom() - 1, bw, 1);
        let st = Style::default()
            .fg(theme.bg_base)
            .bg(theme.accent)
            .add_modifier(Modifier::BOLD);
        frame.render_widget(Paragraph::new(badge).style(st), rect);
    }
}

fn build_rows(app: &mut App, width: usize, theme: &Theme) -> Vec<Row> {
    let selected = app.selected;
    let spinner = SPINNER[(app.ticks as usize / 2) % SPINNER.len()];
    let mut r = Rows {
        list: Vec::new(),
        theme,
    };

    for (idx, entry) in app.transcript.iter_mut().enumerate() {
        if !r.list.is_empty() {
            r.raw(None, false, theme.bg_base, Line::from(""));
        }
        match entry {
            TranscriptEntry::User(text) => {
                let lead = theme.surface(theme.accent).add_modifier(Modifier::BOLD);
                let body = theme.surface(theme.fg);
                for (i, line) in wrap_plain(text, width).into_iter().enumerate() {
                    let p = ["› ", "  "][usize::from(i > 0)];
                    r.raw(
                        None,
                        false,
                        theme.bg_surface,
                        Line::from(vec![Span::styled(p, lead), Span::styled(line, body)]),
                    );
                }
            }
            TranscriptEntry::Assistant { .. } => {
                let indent = Span::styled("  ", theme.base());
                for line in entry.assistant_lines(width, theme) {
                    let spans = std::iter::once(indent.clone())
                        .chain(line.spans.iter().cloned())
                        .collect::<Vec<_>>();
                    r.raw(None, false, theme.bg_base, Line::from(spans));
                }
            }
            TranscriptEntry::Thought {
                text,
                expanded,
                streaming,
            } => {
                let sel = selected == Some(idx);
                let bg = [theme.bg_base, theme.bg_surface][usize::from(sel)];
                let icon = match (*expanded, *streaming, sel) {
                    (true, _, _) => "▾ ".into(),
                    (false, true, _) => format!("{spinner} "),
                    (false, false, true) => "▸ ".into(),
                    _ => "· ".into(),
                };
                let label = if *streaming {
                    "thinking…".into()
                } else {
                    format!("thought ({} lines)", text.lines().count().max(1))
                };
                let fg = [theme.muted_bright, theme.fg][usize::from(sel)];
                let lst = Style::default()
                    .fg(fg)
                    .bg(bg)
                    .add_modifier(Modifier::ITALIC);
                let spans = vec![
                    Span::styled(icon, Style::default().fg(theme.accent).bg(bg)),
                    Span::styled(label, lst),
                ];
                r.raw(Some(idx), true, bg, Line::from(spans));
                if *expanded {
                    let st = theme.fg(theme.fg_secondary).add_modifier(Modifier::ITALIC);
                    for chunk in wrap_plain(text, width) {
                        r.pair(Some(idx), "  ", chunk, st);
                    }
                }
            }
            TranscriptEntry::Tool(tool) => {
                push_tool(&mut r, idx, tool, selected == Some(idx), spinner, width)
            }
            TranscriptEntry::Notice { text, is_error } => {
                let st = theme.fg([theme.muted_bright, theme.red][usize::from(*is_error)]);
                for (i, chunk) in wrap_plain(text, width).into_iter().enumerate() {
                    let p = ["· ", "  "][usize::from(i > 0)];
                    r.raw(
                        None,
                        false,
                        theme.bg_base,
                        Line::from(vec![Span::styled(p, st), Span::styled(chunk, st)]),
                    );
                }
            }
        }
    }
    r.list
}

fn push_tool(
    r: &mut Rows<'_>,
    idx: usize,
    tool: &ToolEntry,
    sel: bool,
    spinner: &str,
    width: usize,
) {
    let (t, d, hit) = (r.theme, &tool.data, tool.has_details().then_some(idx));
    let bg = [t.bg_base, t.bg_surface][usize::from(sel)];
    let (icon, icon_fg) = match (tool.running, d.ok, tool.expanded) {
        (true, _, _) => (format!("{spinner} "), t.accent),
        (false, false, true) => ("▾ ".into(), t.red),
        (false, false, false) => ("◆ ".into(), t.red),
        (false, true, true) => ("▾ ".into(), t.green),
        (false, true, false) => ("◆ ".into(), [t.muted_bright, t.accent][usize::from(sel)]),
    };
    let fg = match (sel, !tool.running && !d.ok) {
        (true, _) => t.fg,
        (false, true) => t.red,
        _ => t.muted_bright,
    };
    let lst = Style::default().fg(fg).bg(bg);

    for (i, chunk) in wrap_plain(&tool.collapsed_label(), width)
        .into_iter()
        .enumerate()
    {
        let lead = if i == 0 {
            Span::styled(icon.clone(), Style::default().fg(icon_fg).bg(bg))
        } else {
            Span::styled("  ", Style::default().bg(bg))
        };
        r.raw(
            hit,
            i == 0,
            bg,
            Line::from(vec![lead, Span::styled(chunk, lst)]),
        );
    }

    let bw = width.saturating_sub(2).max(1);
    if tool.running && !tool.live_output.is_empty() && !tool.expanded {
        let tail: Vec<_> = tool.live_output.lines().rev().take(5).collect();
        for raw in tail.into_iter().rev() {
            for chunk in wrap_plain(raw, bw) {
                let spans = vec![
                    Span::styled("  │ ", t.fg(t.border)),
                    Span::styled(chunk, t.fg(t.fg_secondary)),
                ];
                r.raw(hit, false, t.bg_base, Line::from(spans));
            }
        }
    }
    if !tool.expanded {
        return;
    }

    let (sec, body_st) = (t.bold(t.muted_bright), t.fg(t.fg_secondary));
    if let Some(args) = d
        .arguments
        .as_ref()
        .map(format_args)
        .filter(|s| !s.is_empty())
    {
        r.pair(hit, "  ", "arguments", sec);
        for line in wrap_plain(&args, bw) {
            r.pair(hit, "    ", line, body_st);
        }
    }
    if let Some(diff) = d.diff.as_deref().filter(|s| !s.is_empty()) {
        r.pair(hit, "  ", "diff", sec);
        r.diff(hit, diff, bw);
    }
    if let Some(err) = d.error.as_deref().filter(|s| !s.is_empty()) {
        r.pair(hit, "  ", "error", t.bold(t.red));
        for line in wrap_plain(err, bw) {
            r.pair(hit, "    ", line, t.fg(t.red));
        }
    } else {
        let out = [tool.live_output.as_str(), d.text.as_str()][usize::from(!d.text.is_empty())];
        if !out.is_empty() {
            r.pair(hit, "  ", "output", sec);
            if d.diff.is_none() && d.name == "git_diff" {
                r.diff(hit, out, bw);
            } else {
                for raw in out.lines() {
                    let st = exit_style(&d.name, raw, t);
                    for chunk in wrap_plain(raw, bw) {
                        r.pair(hit, "    ", chunk, st);
                    }
                }
            }
        }
    }
    if d.notes.iter().any(|n| !n.trim().is_empty()) {
        r.pair(hit, "  ", "notes", t.bold(t.yellow));
        for note in &d.notes {
            for line in wrap_plain(note, bw) {
                r.pair(hit, "    ", line, body_st);
            }
        }
    }
    if !d.artifacts.is_empty() {
        r.pair(hit, "  ", "artifacts", sec);
        for name in &d.artifacts {
            r.pair(hit, "    ", name.clone(), body_st);
        }
    }
    if let Some(omitted) = d.omitted.as_deref().filter(|s| !s.is_empty()) {
        r.pair(hit, "    ", format!("… {omitted}"), t.muted_style());
    }
}

fn exit_style(tool_name: &str, raw: &str, t: &Theme) -> Style {
    if matches!(tool_name, "bash" | "bash_tool" | "run_tests") {
        if let Some(rest) = raw.trim_start().strip_prefix("exit_code:") {
            return t.fg([t.red, t.green][usize::from(rest.trim() == "0")]);
        }
    }
    t.fg(t.fg_secondary)
}

fn format_args(value: &Value) -> String {
    let Some(map) = value.as_object() else {
        if value.is_null() {
            return String::new();
        }
        return serde_json::to_string_pretty(value).unwrap_or_default();
    };
    map.iter()
        .map(|(k, v)| match v {
            Value::String(s) => format!("{k}: {s}"),
            other => format!("{k}: {}", serde_json::to_string(other).unwrap_or_default()),
        })
        .collect::<Vec<_>>()
        .join("\n")
}
