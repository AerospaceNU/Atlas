//! Non-blocking row-major text selection painted over the live rendered buffer.

use ratatui::buffer::Buffer;
use ratatui::layout::{Position, Rect};
use unicode_width::UnicodeWidthStr;

use crate::theme::Theme;

#[derive(Debug, Clone)]
pub struct TextSelection {
    area: Rect,
    anchor: Position,
    end: Position,
    pub dragging: bool,
}

impl TextSelection {
    pub fn start(area: Rect, anchor: Position) -> Self {
        let anchor = clamp(area, anchor);
        Self {
            area,
            anchor,
            end: anchor,
            dragging: true,
        }
    }

    pub fn extend(&mut self, end: Position) {
        self.end = clamp(self.area, end);
    }

    pub fn finish(&mut self, end: Position) {
        self.extend(end);
        self.dragging = false;
    }

    pub fn moved(&self) -> bool {
        self.anchor != self.end
    }
    pub fn area(&self) -> Rect {
        self.area
    }

    pub fn extract_text(&self, buffer: &Buffer) -> String {
        if !self.moved() || *buffer.area() != self.area || self.area.is_empty() {
            return String::new();
        }
        let (first, last) = self.endpoints();
        (first.y..=last.y)
            .map(|y| {
                let (l, r) = self.row_bounds(y, first, last);
                let mut text = String::new();
                walk_row(buffer, y, l, r, |_, _, sym| text.push_str(sym));
                text.trim_end_matches(' ').to_string()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn paint(&self, target: &mut Buffer, theme: &Theme) {
        if !self.moved() || *target.area() != self.area || self.area.is_empty() {
            return;
        }
        let style = theme.selection_style();
        let (first, last) = self.endpoints();
        for y in first.y..=last.y {
            let (l, r) = self.row_bounds(y, first, last);
            let mut ranges = Vec::new();
            walk_row(target, y, l, r, |x, span_r, _| ranges.push((x, span_r)));
            for (x, span_r) in ranges {
                for sx in x..=span_r {
                    if let Some(cell) = target.cell_mut(Position::new(sx, y)) {
                        cell.set_style(style);
                    }
                }
            }
        }
    }

    fn endpoints(&self) -> (Position, Position) {
        if (self.anchor.y, self.anchor.x) <= (self.end.y, self.end.x) {
            (self.anchor, self.end)
        } else {
            (self.end, self.anchor)
        }
    }

    fn row_bounds(&self, y: u16, first: Position, last: Position) -> (u16, u16) {
        let left = if y == first.y { first.x } else { self.area.x };
        let right = if y == last.y {
            last.x
        } else {
            self.area.right().saturating_sub(1)
        };
        (left, right)
    }
}

fn walk_row(buf: &Buffer, y: u16, sel_l: u16, sel_r: u16, mut f: impl FnMut(u16, u16, &str)) {
    let area = *buf.area();
    let row_r = area.right().saturating_sub(1);
    let mut x = area.x;
    while x <= row_r {
        let Some(cell) = buf.cell(Position::new(x, y)) else {
            break;
        };
        let sym = cell.symbol();
        let w = sym.width().max(1);
        let span_r = x
            .saturating_add(u16::try_from(w - 1).unwrap_or(u16::MAX))
            .min(row_r);
        if x <= sel_r && span_r >= sel_l {
            f(x, span_r, sym);
        }
        let next = x.saturating_add(u16::try_from(w).unwrap_or(u16::MAX));
        if next <= x {
            break;
        }
        x = next;
    }
}

fn clamp(area: Rect, pos: Position) -> Position {
    if area.is_empty() {
        return Position::new(area.x, area.y);
    }
    Position::new(
        pos.x.clamp(area.x, area.right().saturating_sub(1)),
        pos.y.clamp(area.y, area.bottom().saturating_sub(1)),
    )
}

#[cfg(test)]
mod tests {
    use ratatui::style::{Modifier, Style};

    use super::*;

    #[test]
    fn selection_extracts_wide_chars_and_paints_without_double_reverse() {
        let buf = Buffer::with_lines(["abcde", "fghij", "klmno"]);
        let mut fwd = TextSelection::start(*buf.area(), Position::new(1, 0));
        fwd.finish(Position::new(2, 2));
        let mut rev = TextSelection::start(*buf.area(), Position::new(2, 2));
        rev.finish(Position::new(1, 0));
        assert_eq!(fwd.extract_text(&buf), "bcde\nfghij\nklm");
        assert_eq!(rev.extract_text(&buf), fwd.extract_text(&buf));

        let mut wide = Buffer::empty(Rect::new(0, 0, 7, 1));
        wide.set_string(0, 0, "Aé中e\u{301}Z", Style::default());
        let mut full = TextSelection::start(*wide.area(), Position::new(0, 0));
        full.finish(Position::new(5, 0));
        assert_eq!(full.extract_text(&wide), "Aé中e\u{301}Z");

        let theme = Theme::truecolor();
        let mut target = Buffer::with_lines(["hello"]);
        let mut sel = TextSelection::start(*target.area(), Position::new(1, 0));
        sel.finish(Position::new(3, 0));
        sel.paint(&mut target, &theme);
        let cell = target.cell((1, 0)).unwrap();
        assert_eq!((cell.bg, cell.fg), (theme.selection_bg, theme.selection_fg));
        assert!(!cell.modifier.contains(Modifier::REVERSED));
    }
}
