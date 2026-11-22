//! Focused per-call approval modal: `y`/`a`/`n`/`d`/`Esc`, `Ctrl+Y` YOLO,
//! deny-with-reason `TextArea`, and colored unified diff preview with line counter.

use crossterm::event::{
    KeyCode as K, KeyEvent, KeyEventKind, KeyModifiers as Mod, MouseButton, MouseEvent,
    MouseEventKind,
};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Padding, Paragraph};
use ratatui::Frame;
use tui_textarea::TextArea;
use unicode_width::UnicodeWidthStr;

use crate::markdown::{style_diff_lines, wrap_plain};
use crate::protocol::{PermissionDecision, PermissionPayload};
use crate::render::composer::render_textarea;
use crate::theme::Theme;

const CHOICES: [&str; 4] = [
    "[y] Allow once",
    "[a] Allow for session",
    "[n] Deny",
    "[d] Deny with reason",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalOutcome {
    Decide {
        decision: PermissionDecision,
        reason: Option<String>,
    },
    ToggleYolo,
}

pub struct PermissionPrompt {
    pub payload: PermissionPayload,
    selected: usize,
    scroll: usize,
    max_scroll: usize,
    page_height: usize,
    reason_input: Option<TextArea<'static>>,
    choices: Vec<(Rect, usize)>,
}

impl PermissionPrompt {
    pub fn new(payload: PermissionPayload) -> Self {
        Self {
            payload,
            selected: 2, // Safe default: Deny so bare Enter never approves accidentally.
            scroll: 0,
            max_scroll: 0,
            page_height: 8,
            reason_input: None,
            choices: Vec::new(),
        }
    }

    pub fn body_plain(&self) -> String {
        let p = &self.payload;
        match (
            p.preview.is_empty(),
            p.summary.is_empty()
                || matches!(p.tool.as_str(), "bash" | "bash_tool")
                || p.preview.contains(&p.summary),
        ) {
            (true, _) => p.summary.clone(),
            (false, true) => p.preview.clone(),
            (false, false) => format!("{}\n{}", p.summary, p.preview),
        }
    }

    fn preview_lines(&self, width: usize, theme: &Theme) -> Vec<Line<'static>> {
        let p = &self.payload;
        match p.preview_kind.as_str() {
            "diff" if !p.preview.is_empty() => {
                let head = (!p.summary.is_empty())
                    .then(|| Line::from(Span::styled(p.summary.clone(), theme.bold(theme.fg))));
                head.into_iter()
                    .chain(style_diff_lines(&p.preview, width, theme))
                    .collect()
            }
            "command" if !p.preview.is_empty() => wrap_plain(&p.preview, width)
                .into_iter()
                .enumerate()
                .map(|(i, t)| {
                    let lead = Span::styled(
                        ["$ ", "  "][usize::from(i > 0)],
                        theme.surface(theme.accent),
                    );
                    Line::from(vec![lead, Span::styled(t, theme.surface(theme.fg))])
                })
                .collect(),
            _ => wrap_plain(&self.body_plain(), width)
                .into_iter()
                .map(|l| Line::from(Span::styled(l, theme.surface(theme.fg))))
                .collect(),
        }
    }

    pub fn rows(&self, width: u16, theme: &Theme) -> u16 {
        let n = self
            .preview_lines(usize::from(width.saturating_sub(4).max(1)), theme)
            .len()
            .clamp(1, 12);
        (n as u16)
            + [4, 1][usize::from(width >= 72)]
            + u16::from(self.reason_input.is_some()) * 2
            + 5
    }

    pub fn render(&mut self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        self.choices.clear();
        let all = self.preview_lines(usize::from(area.width.saturating_sub(4).max(1)), theme);
        let total = all.len();
        let ch_h: u16 = [4, 1][usize::from(area.width >= 72)];
        let rsn_h: u16 = u16::from(self.reason_input.is_some()) * 2;
        self.page_height = usize::from(area.height.saturating_sub(ch_h + rsn_h + 4).max(1));
        self.max_scroll = total.saturating_sub(self.page_height);
        self.scroll = self.scroll.min(self.max_scroll);

        let title = Line::from(vec![
            Span::styled(" Approval required ", theme.bold(theme.yellow)),
            Span::styled(
                format!("· {} ", self.payload.tool),
                theme.fg(theme.fg_secondary),
            ),
        ]);
        let mut block = Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(theme.yellow))
            .padding(Padding::horizontal(1))
            .style(theme.base());
        if total > self.page_height {
            let range = format!(
                " {}-{} of {total} ",
                self.scroll + 1,
                (self.scroll + self.page_height).min(total)
            );
            block = block.title_top(
                Line::from(Span::styled(range, theme.fg(theme.muted_bright))).right_aligned(),
            );
        }
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let [prev_a, _, ch_a, rsn_a, hint_a] = Layout::vertical([
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(ch_h),
            Constraint::Length(rsn_h),
            Constraint::Length(1),
        ])
        .areas(inner);

        let bg =
            [theme.bg_surface, theme.bg_base][usize::from(self.payload.preview_kind == "diff")];
        let shown: Vec<Line> = all
            .into_iter()
            .skip(self.scroll)
            .take(usize::from(prev_a.height))
            .collect();
        frame.render_widget(
            Paragraph::new(shown).style(Style::default().bg(bg).fg(theme.fg)),
            prev_a,
        );

        let mut x = ch_a.x;
        for (i, label) in CHOICES.iter().enumerate() {
            let (rect, text) = if ch_h == 1 {
                if x >= ch_a.right() {
                    break;
                }
                let text = format!(" {label} ");
                let w = u16::try_from(text.width()).unwrap_or(12);
                let r = Rect::new(x, ch_a.y, w.min(ch_a.right() - x), 1);
                x = x.saturating_add(w + 1);
                (r, text)
            } else {
                if i >= usize::from(ch_a.height) {
                    break;
                }
                let m = [" ", "›"][usize::from(self.selected == i)];
                (
                    Rect::new(ch_a.x, ch_a.y + i as u16, ch_a.width, 1),
                    format!("{m} {label}"),
                )
            };
            let fg = [theme.red, theme.green][usize::from(i < 2)];
            let st = if self.selected == i {
                theme.surface(fg).add_modifier(Modifier::BOLD)
            } else {
                theme.fg(theme.fg_secondary)
            };
            frame.render_widget(Paragraph::new(text).style(st), rect);
            self.choices.push((rect, i));
        }

        if let Some(editor) = self.reason_input.as_mut() {
            if rsn_a.height >= 1 {
                let lbl = Paragraph::new("Deny reason (Enter submit, Esc back):")
                    .style(theme.fg(theme.yellow));
                frame.render_widget(lbl, Rect::new(rsn_a.x, rsn_a.y, rsn_a.width, 1));
            }
            if rsn_a.height >= 2 {
                let r = Rect::new(rsn_a.x, rsn_a.y + 1, rsn_a.width, 1);
                render_textarea(frame, editor, r, "Optional explanation…", true, theme);
            }
        }
        let scroll_hint = ["", " · PgUp/PgDn scroll"][usize::from(self.max_scroll > 0)];
        let hint = format!(
            "y allow · a session · n deny · d reason · Esc deny · Ctrl+Y yolo{scroll_hint}"
        );
        frame.render_widget(Paragraph::new(hint).style(theme.muted_style()), hint_a);
    }

    fn choose(&mut self, idx: usize) -> Option<ApprovalOutcome> {
        let dec = |decision| {
            Some(ApprovalOutcome::Decide {
                decision,
                reason: None,
            })
        };
        match idx {
            0 => dec(PermissionDecision::Allow),
            1 => dec(PermissionDecision::AllowSession),
            3 => {
                (self.reason_input, self.selected) = (Some(TextArea::default()), 3);
                None
            }
            _ => dec(PermissionDecision::Deny),
        }
    }

    pub fn key(&mut self, key: KeyEvent) -> Option<ApprovalOutcome> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        if key.code == K::Char('y') && key.modifiers.contains(Mod::CONTROL) {
            return Some(ApprovalOutcome::ToggleYolo);
        }
        if let Some(editor) = self.reason_input.as_mut() {
            return match key.code {
                K::Esc => {
                    self.reason_input = None;
                    None
                }
                K::Enter => {
                    let t = editor.lines().join(" ").trim().to_string();
                    Some(ApprovalOutcome::Decide {
                        decision: PermissionDecision::Deny,
                        reason: (!t.is_empty()).then_some(t),
                    })
                }
                _ => {
                    editor.input(key);
                    None
                }
            };
        }
        if key
            .modifiers
            .intersects(Mod::CONTROL | Mod::ALT | Mod::SUPER)
        {
            return None;
        }
        let step = self.page_height.saturating_sub(1).max(1);
        match key.code {
            K::Char('y') => return self.choose(0),
            K::Char('a') => return self.choose(1),
            K::Char('n') | K::Esc => return self.choose(2),
            K::Char('d') => return self.choose(3),
            K::Enter => return self.choose(self.selected),
            K::Up | K::Left | K::BackTab => self.selected = (self.selected + 3) % 4,
            K::Down | K::Right | K::Tab => self.selected = (self.selected + 1) % 4,
            K::PageUp => self.scroll = self.scroll.saturating_sub(step),
            K::PageDown => self.scroll = self.scroll.saturating_add(step).min(self.max_scroll),
            K::Home => self.scroll = 0,
            K::End => self.scroll = self.max_scroll,
            _ => {}
        }
        None
    }

    pub fn paste(&mut self, text: &str) {
        if let Some(editor) = self.reason_input.as_mut() {
            editor.insert_str(text.replace('\n', " "));
        }
    }

    pub fn mouse(&mut self, mouse: MouseEvent) -> Option<ApprovalOutcome> {
        match mouse.kind {
            MouseEventKind::ScrollUp => self.scroll = self.scroll.saturating_sub(3),
            MouseEventKind::ScrollDown => {
                self.scroll = self.scroll.saturating_add(3).min(self.max_scroll)
            }
            MouseEventKind::Down(MouseButton::Left) => {
                let idx = self.choice_at(mouse.column, mouse.row)?;
                self.selected = idx;
                return self.choose(idx);
            }
            _ => {}
        }
        None
    }

    pub fn choice_at(&self, col: u16, row: u16) -> Option<usize> {
        self.choices.iter().find_map(|(r, c)| {
            (col >= r.x && col < r.right() && row >= r.y && row < r.bottom()).then_some(*c)
        })
    }
}

#[cfg(test)]
mod tests {
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    use super::*;

    fn diff_prompt(preview: String) -> PermissionPrompt {
        PermissionPrompt::new(PermissionPayload {
            id: "c1".into(),
            tool: "edit_file".into(),
            summary: "upd".into(),
            preview,
            preview_kind: "diff".into(),
            arguments: None,
        })
    }

    #[test]
    fn shortcuts_deny_reason_and_colored_diff_counter_work() {
        let mut p = diff_prompt("+a\n-b".into());
        let dec = |d, reason| {
            Some(ApprovalOutcome::Decide {
                decision: d,
                reason,
            })
        };
        assert_eq!(
            p.key(KeyEvent::new(K::Enter, Mod::NONE)),
            dec(PermissionDecision::Deny, None)
        );
        assert_eq!(
            p.key(KeyEvent::new(K::Char('y'), Mod::NONE)),
            dec(PermissionDecision::Allow, None)
        );
        assert_eq!(
            p.key(KeyEvent::new(K::Char('a'), Mod::NONE)),
            dec(PermissionDecision::AllowSession, None)
        );
        assert_eq!(
            p.key(KeyEvent::new(K::Char('y'), Mod::CONTROL)),
            Some(ApprovalOutcome::ToggleYolo)
        );
        assert_eq!(p.key(KeyEvent::new(K::Char('d'), Mod::NONE)), None);
        for ch in "use fmt".chars() {
            p.key(KeyEvent::new(K::Char(ch), Mod::NONE));
        }
        assert_eq!(
            p.key(KeyEvent::new(K::Enter, Mod::NONE)),
            dec(PermissionDecision::Deny, Some("use fmt".into()))
        );

        let theme = Theme::truecolor();
        let mut dp = diff_prompt((0..40).map(|i| format!("+line {i}\n")).collect());
        let mut term = Terminal::new(TestBackend::new(90, 20)).unwrap();
        term.draw(|f| dp.render(f, Rect::new(0, 0, 90, 16), &theme))
            .unwrap();
        let buf = term.backend().buffer();
        let screen: String = (0..16)
            .map(|y| (0..90).map(|x| buf[(x, y)].symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            screen.contains("of 41") && screen.contains("[y] Allow once"),
            "{screen}"
        );
        assert!((0..16).any(|y| (0..90).any(|x| buf[(x, y)].fg == theme.green)));
    }
}
