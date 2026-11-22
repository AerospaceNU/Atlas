//! Compact 1–2 line status bar with workspace, git branch, model, context % bar, spend, goal/skills.

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Color;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::app::{App, Status};
use crate::protocol::format_tokens;
use crate::theme::Theme;

pub fn status_rows(area_height: u16) -> u16 {
    if area_height <= 6 {
        1
    } else {
        2
    }
}

pub fn render_status(frame: &mut Frame<'_>, app: &App, area: Rect, theme: &Theme) {
    if area.height == 0 || area.width == 0 {
        return;
    }
    let line1 = build_dashboard_line(app, theme);
    if area.height == 1 {
        frame.render_widget(Paragraph::new(line1).style(theme.base()), area);
        return;
    }
    let [top, bottom] =
        Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).areas(area);
    frame.render_widget(Paragraph::new(line1).style(theme.base()), top);
    let st = if app.status_notice.is_some() {
        theme.base()
    } else {
        theme.muted_style()
    };
    frame.render_widget(Paragraph::new(status_hint(app)).style(st), bottom);
}

#[cfg(test)]
pub fn dashboard_text(app: &App, theme: &Theme) -> String {
    build_dashboard_line(app, theme)
        .spans
        .into_iter()
        .map(|s| s.content.into_owned())
        .collect()
}

fn build_dashboard_line(app: &App, theme: &Theme) -> Line<'static> {
    let snap = &app.status_snapshot;
    let ws = app
        .workspace
        .file_name()
        .and_then(|n| n.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("Atlas");
    let mut spans = vec![Span::styled(ws.to_string(), theme.bold(theme.fg))];
    if let Some(branch) = snap.branch.as_deref().filter(|b| !b.is_empty()) {
        spans.push(Span::styled(format!(" ({branch})"), theme.fg(theme.cyan)));
    }
    let mut add = |text: String, st| {
        spans.push(Span::styled("  ", theme.base()));
        spans.push(Span::styled(text, st));
    };
    let (badge, bcolor) = if !app.session_open {
        ("session ended", theme.red)
    } else if app.pending_permission.is_some() {
        ("awaiting approval", theme.yellow)
    } else {
        match app.status {
            Status::Ready => ("ready", theme.muted_bright),
            Status::Waiting => ("working", theme.accent),
            Status::Stopped => ("stopped", theme.yellow),
        }
    };
    add(format!("● {badge}"), theme.fg(bcolor));
    if snap.yolo_enabled {
        add("YOLO on".into(), theme.bold(theme.yellow));
    }
    let model = if snap.model.is_empty() {
        "model unset"
    } else {
        snap.model.as_str()
    };
    add(model.into(), theme.fg(theme.fg_secondary));
    let (ctx, ccolor) = context_meter(snap.context_used, snap.context_limit, theme);
    add(ctx, theme.fg(ccolor));
    if let Some(spend) = snap.spend.filter(|s| s.is_finite() && *s > 0.0) {
        add(format!("${spend:.3}"), theme.fg(theme.muted_bright));
    }
    if let Some(tps) = snap.tokens_per_second.filter(|t| t.is_finite() && *t > 0.0) {
        add(format!("{tps:.0} tok/s"), theme.fg(theme.muted_bright));
    }
    if let Some(goal) = snap.goal.as_deref().filter(|g| !g.is_empty()) {
        let short: String = goal.chars().take(24).collect();
        let dots = if goal.chars().count() > 24 { "…" } else { "" };
        add(format!("goal:{short}{dots}"), theme.fg(theme.accent));
    }
    if !snap.active_skills.is_empty() {
        add(
            format!("skills:{}", snap.active_skills.join(",")),
            theme.fg(theme.green),
        );
    }
    Line::from(spans)
}

fn context_meter(used: i64, limit: Option<i64>, theme: &Theme) -> (String, Color) {
    let used_s = format_tokens(used);
    let Some(lim) = limit.filter(|&l| l > 0) else {
        return (format!("ctx {used_s}"), theme.muted_bright);
    };
    let ratio = (used.max(0) as f64 / lim as f64).clamp(0.0, 1.0);
    let pct = (ratio * 100.0).round() as usize;
    let filled = (ratio * 6.0).round() as usize;
    let color = if pct >= 85 {
        theme.red
    } else if pct >= 65 {
        theme.yellow
    } else {
        theme.green
    };
    let bar = format!(
        "[{}{}] {pct}% {used_s}/{}",
        "█".repeat(filled),
        "░".repeat(6 - filled),
        format_tokens(lim)
    );
    (bar, color)
}

pub fn status_hint(app: &App) -> String {
    if let Some(notice) = &app.status_notice {
        return notice.clone();
    }
    if app.pending_permission.is_some() {
        return "Approval pending · y allow · a session · n deny · d reason".into();
    }
    if !app.session_open {
        return "Ctrl+C quit".into();
    }
    let mut base = if app.status == Status::Stopped {
        "Ctrl+R resume · Enter send · PgUp/PgDn scroll · / commands".to_string()
    } else if app.in_flight() {
        "Esc cancel turn · Draft kept · PgUp/PgDn scroll · /help".to_string()
    } else {
        "Enter send · Alt+Enter newline · ↑↓ history · / commands · @ files".to_string()
    };
    if !app.expandable_indices().is_empty() {
        base.push_str(" · Ctrl+O/click tool");
    }
    base
}
