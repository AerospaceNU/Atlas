//! A focused, per-call approval prompt. The Python session owns execution.

use crossterm::event::{
    KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Padding, Paragraph};
use ratatui::Frame;

use crate::theme::{BG_BASE, BG_LIGHT, FG, FG_SECONDARY, GRAY, MAGENTA, YELLOW};
use crate::wrap_text;

pub(super) struct PermissionPrompt {
    pub id: String,
    pub tool: String,
    pub command: String,
    summary: String,
    preview: String,
    /// Enter must never approve an action merely because the prompt appeared.
    allow: bool,
    scroll: usize,
    max_scroll: usize,
    choices: Vec<(Rect, bool)>,
}

impl PermissionPrompt {
    pub fn new(id: String, tool: String, summary: String, preview: String) -> Self {
        Self {
            id,
            tool,
            command: String::new(),
            summary,
            preview,
            allow: false,
            scroll: 0,
            max_scroll: 0,
            choices: Vec::new(),
        }
    }

    pub fn body(&self) -> String {
        if self.preview.is_empty() {
            self.summary.clone()
        } else if self.tool == "bash_tool" || self.summary.is_empty() {
            self.preview.clone()
        } else {
            format!("{}\n\n{}", self.summary, self.preview)
        }
    }

    pub fn rows(&self, width: u16) -> u16 {
        let body = wrap_text(&self.body(), usize::from(width.saturating_sub(4).max(1)));
        body.len().clamp(1, 8) as u16 + 8
    }

    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        self.choices.clear();
        let block = Block::default()
            .title(Line::from(Span::styled(
                " Approval required ",
                Style::default().fg(YELLOW),
            )))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(YELLOW))
            .padding(Padding::horizontal(1))
            .style(Style::default().bg(BG_BASE).fg(FG));
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let [tool, preview, _gap, choices, hints] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(2),
            Constraint::Length(2),
        ])
        .areas(inner);
        frame.render_widget(
            Paragraph::new(self.tool.as_str()).style(Style::default().fg(FG_SECONDARY)),
            tool,
        );

        let body = wrap_text(&self.body(), usize::from(preview.width.max(1)));
        self.max_scroll = body.len().saturating_sub(usize::from(preview.height));
        self.scroll = self.scroll.min(self.max_scroll);
        let lines: Vec<Line> = body
            .into_iter()
            .skip(self.scroll)
            .take(usize::from(preview.height))
            .map(Line::from)
            .collect();
        frame.render_widget(
            Paragraph::new(lines).style(Style::default().bg(BG_LIGHT).fg(FG)),
            preview,
        );

        for (index, (label, allow)) in [("Allow once", true), ("Deny", false)].iter().enumerate() {
            if index >= usize::from(choices.height) {
                break;
            }
            let selected = self.allow == *allow;
            let mark = if selected { "›" } else { " " };
            let row = Rect::new(choices.x, choices.y + index as u16, choices.width, 1);
            let style = if selected {
                Style::default().fg(MAGENTA).bg(BG_LIGHT)
            } else {
                Style::default().fg(FG_SECONDARY).bg(BG_BASE)
            };
            frame.render_widget(Paragraph::new(format!("{mark} {label}")).style(style), row);
            self.choices.push((row, *allow));
        }
        let scroll_hint = if self.max_scroll > 0 {
            "Esc deny  /yolo  PgUp/PgDn scroll"
        } else {
            "Esc deny  /yolo skip approvals"
        };
        let hints_text = if self.command.is_empty() {
            vec![
                Line::from("↑↓/Tab choose  Enter confirm"),
                Line::from(scroll_hint),
            ]
        } else {
            vec![
                Line::from("Enter toggle YOLO  Esc back"),
                Line::from(format!("› {}", self.command)),
            ]
        };
        frame.render_widget(
            Paragraph::new(hints_text).style(Style::default().fg(GRAY)),
            hints,
        );
    }

    /// Return a decision only for an explicit confirm or denial.
    pub fn key(&mut self, key: KeyEvent) -> Option<bool> {
        if key.kind != KeyEventKind::Press
            || key
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SUPER)
        {
            return None;
        }
        match key.code {
            KeyCode::Up | KeyCode::Left => self.allow = true,
            KeyCode::Down | KeyCode::Right => self.allow = false,
            KeyCode::Tab | KeyCode::BackTab => self.allow = !self.allow,
            KeyCode::Enter => return Some(self.allow),
            KeyCode::Esc => return Some(false),
            KeyCode::PageUp => self.scroll = self.scroll.saturating_sub(5),
            KeyCode::PageDown => self.scroll = self.scroll.saturating_add(5).min(self.max_scroll),
            KeyCode::Home => self.scroll = 0,
            KeyCode::End => self.scroll = self.max_scroll,
            _ => {}
        }
        None
    }

    pub fn mouse(&mut self, mouse: MouseEvent) -> Option<bool> {
        match mouse.kind {
            MouseEventKind::ScrollUp => self.scroll = self.scroll.saturating_sub(3),
            MouseEventKind::ScrollDown => {
                self.scroll = self.scroll.saturating_add(3).min(self.max_scroll);
            }
            MouseEventKind::Down(MouseButton::Left) => {
                return self.choice_at(mouse.column, mouse.row);
            }
            _ => {}
        }
        None
    }

    pub fn choice_at(&self, column: u16, row: u16) -> Option<bool> {
        self.choices.iter().find_map(|(rect, allow)| {
            (column >= rect.x && column < rect.right() && row >= rect.y && row < rect.bottom())
                .then_some(*allow)
        })
    }
}
