//! Composer and hidden-key input rendering via `tui-textarea` + real terminal cursor.

use ratatui::layout::{Position, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph};
use ratatui::Frame;
use tui_textarea::TextArea;
use unicode_width::UnicodeWidthStr;

use crate::app::{App, Mode};
use crate::theme::Theme;

pub fn composer_rows(app: &App) -> u16 {
    let body = match &app.mode {
        Mode::KeyInput(_) => 1,
        _ => app.composer.lines().len().clamp(1, 5),
    };
    (body as u16).saturating_add(2)
}

pub fn render_composer(frame: &mut Frame<'_>, app: &mut App, area: Rect, theme: &Theme) {
    let active = app.session_open && matches!(app.mode, Mode::Compose | Mode::KeyInput(_));
    let bcol = [theme.border, theme.border_active][usize::from(active)];
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(bcol))
        .style(theme.base());
    if let Mode::KeyInput(s) = &app.mode {
        let title = Span::styled(
            format!(" {} ({}) key ", s.provider, s.scope),
            theme.bold(theme.yellow),
        );
        block = block.title(Line::from(title));
    }
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.width < 3 || inner.height == 0 {
        return;
    }

    let is_key = matches!(app.mode, Mode::KeyInput(_));
    let glyph = ["› ", "* "][usize::from(is_key)];
    let gcol = [theme.muted, theme.accent][usize::from(active)];
    frame.render_widget(
        Paragraph::new(glyph).style(theme.bold(gcol)),
        Rect::new(inner.x, inner.y, 2, inner.height),
    );
    let ed_rect = Rect::new(inner.x + 2, inner.y, inner.width - 2, inner.height);
    let in_flight = app.in_flight();
    let show = active && (is_key || app.pending_permission.is_none());
    let (ed, ph) = match &mut app.mode {
        Mode::KeyInput(s) => (&mut s.editor, "Paste or type key (hidden)"),
        _ => (
            &mut app.composer,
            [
                "Message Atlas  (/ commands · @ files)",
                "Draft while Atlas works… (Esc cancels turn)",
            ][usize::from(in_flight)],
        ),
    };
    render_textarea(frame, ed, ed_rect, ph, show, theme);
}

pub fn render_textarea(
    frame: &mut Frame<'_>,
    editor: &mut TextArea<'static>,
    area: Rect,
    placeholder: &str,
    show_cursor: bool,
    theme: &Theme,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    editor.set_style(theme.base());
    editor.set_cursor_line_style(theme.base());
    editor.set_cursor_style(Style::default());
    editor.set_placeholder_text("");

    if editor.is_empty() {
        frame.render_widget(Paragraph::new(placeholder).style(theme.muted_style()), area);
        if show_cursor {
            frame.set_cursor_position(Position::new(area.x, area.y));
        }
        return;
    }
    frame.render_widget(&*editor, area);
    if show_cursor {
        let (row, col) = editor.cursor();
        let line = editor.lines().get(row).map(String::as_str).unwrap_or("");
        let disp = if editor.mask_char().is_some() {
            col
        } else {
            line.chars().take(col).collect::<String>().width()
        };
        let top_r = row.saturating_sub(usize::from(area.height) - 1);
        let top_c = disp.saturating_sub(usize::from(area.width) - 1);
        let cx = area
            .x
            .saturating_add((disp - top_c) as u16)
            .min(area.right() - 1);
        let cy = area
            .y
            .saturating_add((row - top_r) as u16)
            .min(area.bottom() - 1);
        frame.set_cursor_position(Position::new(cx, cy));
    }
}
