//! Top-level layout, informative empty-state welcome view, and live selection overlay.

pub mod composer;
pub mod overlays;
pub mod status;
pub mod transcript;

use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph};
use ratatui::Frame;

use crate::app::{App, Mode};
use crate::clipboard::copy_to_clipboard;
use crate::render::composer::{composer_rows, render_composer};
use crate::render::overlays::{picker_rows, render_activity, render_help, render_picker};
use crate::render::status::{render_status, status_rows};
use crate::render::transcript::render_transcript;
use crate::theme::Theme;

pub fn draw(frame: &mut Frame<'_>, app: &mut App, theme: &Theme) {
    app.screen_area = frame.area();
    if app
        .text_selection
        .as_ref()
        .is_some_and(|s| s.area() != frame.area())
    {
        (app.text_selection, app.status_notice) = (None, None);
    }
    let status_area = draw_content(frame, app, theme);
    if std::mem::take(&mut app.copy_selection) {
        if let Some(sel) = app.text_selection.as_ref().filter(|s| s.moved()) {
            let text = sel.extract_text(frame.buffer_mut());
            if text.trim().is_empty() {
                app.text_selection = None;
            } else {
                let ok = copy_to_clipboard(&text);
                let msg = [
                    "Copy failed · Ctrl+C retry · Esc clear",
                    "Selection copied · Ctrl+C copy · Esc clear",
                ][usize::from(ok)];
                app.status_notice = Some(msg.into());
                if let Some(sa) = status_area {
                    render_status(frame, app, sa, theme);
                }
            }
        }
    }
    if let Some(sel) = app.text_selection.as_ref() {
        sel.paint(frame.buffer_mut(), theme);
    }
}

fn draw_content(frame: &mut Frame<'_>, app: &mut App, theme: &Theme) -> Option<Rect> {
    let area = frame.area();
    frame.render_widget(Block::default().style(theme.base()), area);
    let (v, h) = (1u16.min(area.height / 2), 2u16.min(area.width / 2));
    let outer = Rect::new(
        area.x + h,
        area.y + v,
        area.width - h * 2,
        area.height - v * 2,
    );
    if outer.width < 8 || outer.height < 4 {
        render_status(frame, app, outer, theme);
        return Some(outer);
    }
    if matches!(app.mode, Mode::Help(_)) {
        render_help(frame, app, outer, theme);
        return None;
    }

    let status_h = status_rows(outer.height);
    if let Some(prompt) = app.pending_permission.as_ref() {
        let prompt_h = prompt
            .rows(outer.width, theme)
            .min(outer.height.saturating_sub(status_h));
        let [t_area, p_area, s_area] = Layout::vertical([
            Constraint::Min(0),
            Constraint::Length(prompt_h),
            Constraint::Length(status_h),
        ])
        .areas(outer);
        render_transcript(frame, app, t_area, theme);
        if let Some(p) = app.pending_permission.as_mut() {
            p.render(frame, p_area, theme);
        }
        render_status(frame, app, s_area, theme);
        return Some(s_area);
    }

    let [t_area, act_area, pick_area, comp_area, stat_area] = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(u16::from(app.in_flight())),
        Constraint::Length(picker_rows(app)),
        Constraint::Length(composer_rows(app)),
        Constraint::Length(status_h),
    ])
    .areas(outer);

    if app.transcript.is_empty() {
        (app.viewport_height, app.tool_hits) = (usize::from(t_area.height), Vec::new());
        render_empty_state(frame, app, t_area, theme);
    } else {
        render_transcript(frame, app, t_area, theme);
    }
    if act_area.height > 0 {
        render_activity(frame, app, act_area, theme);
    }
    if pick_area.height > 0 {
        render_picker(frame, app, pick_area, theme);
    }
    render_composer(frame, app, comp_area, theme);
    render_status(frame, app, stat_area, theme);
    Some(stat_area)
}

fn render_empty_state(frame: &mut Frame<'_>, app: &App, area: Rect, theme: &Theme) {
    if area.height < 4 || area.width < 16 {
        return;
    }
    let snap = &app.status_snapshot;
    let branch = snap
        .branch
        .as_deref()
        .filter(|b| !b.is_empty())
        .map_or_else(String::new, |b| format!(" ({b})"));
    let model = [snap.model.as_str(), "(unset — run /model)"][usize::from(snap.model.is_empty())];
    let kv = |k: &'static str, v: String, st| {
        Line::from(vec![
            Span::styled(k, theme.muted_style()),
            Span::styled(v, st),
        ])
    };
    let mut lines = vec![
        Line::from(Span::styled(
            "· ·  A T L A S  · ·",
            theme.bold(theme.accent),
        )),
        Line::from(""),
        kv(
            "workspace  ",
            format!("{}{branch}", app.workspace.display()),
            theme.base(),
        ),
        Line::from(vec![
            Span::styled("model      ", theme.muted_style()),
            Span::styled(model.to_string(), theme.fg(theme.cyan)),
            Span::styled(
                format!("  · {} tools", app.tools.len()),
                theme.muted_style(),
            ),
        ]),
    ];
    if let Some(goal) = snap.goal.as_deref().filter(|g| !g.is_empty()) {
        lines.push(kv("goal       ", goal.into(), theme.fg(theme.accent)));
    }
    if !snap.active_skills.is_empty() {
        lines.push(kv(
            "skills     ",
            snap.active_skills.join(", "),
            theme.fg(theme.green),
        ));
    }
    if !app.key_set {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "! API key not set — run `/key user` or `/key session` before chatting",
            theme.bold(theme.yellow),
        )));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "Type a prompt, `/help` for commands, `/resume` for past sessions, or `@` to reference files",
        theme.muted_style(),
    )));

    let y = area.y + area.height.saturating_sub(lines.len() as u16) / 2;
    let rect = Rect::new(area.x, y, area.width, area.bottom().saturating_sub(y));
    frame.render_widget(
        Paragraph::new(lines)
            .alignment(Alignment::Center)
            .style(theme.base()),
        rect,
    );
}
