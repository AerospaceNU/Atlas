use ratatui::buffer::Buffer;
use ratatui::layout::{Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;

const SELECTION_BG: Color = Color::Rgb(72, 60, 100);
const SELECTION_FG: Color = Color::Rgb(225, 225, 225);

/// A row-major text selection over a rendered buffer snapshot.
pub(super) struct TextSelection {
    snapshot: Buffer,
    anchor: Position,
    end: Position,
    pub dragging: bool,
}

impl TextSelection {
    /// Starts a selection at `anchor`, clamped to the snapshot's area.
    pub fn new(snapshot: Buffer, anchor: Position) -> Self {
        let anchor = clamp_position(*snapshot.area(), anchor);
        Self {
            snapshot,
            anchor,
            end: anchor,
            dragging: true,
        }
    }

    /// Extends the selection to `end`, clamped to the snapshot's area.
    pub fn extend(&mut self, end: Position) {
        self.end = clamp_position(*self.snapshot.area(), end);
    }

    /// Extends the selection and ends the drag.
    pub fn finish(&mut self, end: Position) {
        self.extend(end);
        self.dragging = false;
    }

    /// Returns whether the selection spans more than the initial cell.
    pub fn moved(&self) -> bool {
        self.anchor != self.end
    }

    /// Returns the area represented by the saved buffer snapshot.
    pub fn area(&self) -> Rect {
        *self.snapshot.area()
    }

    /// Returns the visible symbols in the inclusive row-major selection.
    pub fn text(&self) -> String {
        if !self.moved() || self.snapshot.area().is_empty() {
            return String::new();
        }

        let (first, last) = self.ordered_endpoints();
        let area = *self.snapshot.area();
        let mut rows = Vec::with_capacity(usize::from(last.y - first.y) + 1);

        for y in first.y..=last.y {
            let (left, right) = self.row_bounds(y, first, last);
            rows.push(self.row_text(y, left, right, area));
        }

        rows.join("\n")
    }

    /// Copies the saved snapshot into `target` and paints the selected cells.
    ///
    /// Only coordinates present in both buffers are copied or styled.
    pub fn paint(&self, target: &mut Buffer) {
        let snapshot_area = *self.snapshot.area();
        let target_area = *target.area();
        let left = snapshot_area.x.max(target_area.x);
        let top = snapshot_area.y.max(target_area.y);
        let right = snapshot_area.right().min(target_area.right());
        let bottom = snapshot_area.bottom().min(target_area.bottom());

        if left >= right || top >= bottom {
            return;
        }

        for y in top..bottom {
            for x in left..right {
                let position = Position::new(x, y);
                if let (Some(source), Some(destination)) =
                    (self.snapshot.cell(position), target.cell_mut(position))
                {
                    *destination = source.clone();
                }
            }
        }

        if !self.moved() || snapshot_area.is_empty() {
            return;
        }

        let (first, last) = self.ordered_endpoints();
        for y in first.y..=last.y {
            let (selected_left, selected_right) = self.row_bounds(y, first, last);
            self.paint_row(target, y, selected_left, selected_right, snapshot_area);
        }
    }

    fn ordered_endpoints(&self) -> (Position, Position) {
        if (self.anchor.y, self.anchor.x) <= (self.end.y, self.end.x) {
            (self.anchor, self.end)
        } else {
            (self.end, self.anchor)
        }
    }

    fn row_bounds(&self, y: u16, first: Position, last: Position) -> (u16, u16) {
        let area = *self.snapshot.area();
        let left = if y == first.y { first.x } else { area.x };
        let right = if y == last.y {
            last.x
        } else {
            area.right().saturating_sub(1)
        };
        (left, right)
    }

    fn row_text(&self, y: u16, selected_left: u16, selected_right: u16, area: Rect) -> String {
        let row_right = area.right().saturating_sub(1);
        let mut x = area.x;
        let mut text = String::new();

        while x <= row_right {
            let position = Position::new(x, y);
            let Some(cell) = self.snapshot.cell(position) else {
                break;
            };
            let symbol = cell.symbol();
            let width = Span::raw(symbol).width();
            let display_width = width.max(1);
            let span_right = x
                .saturating_add(u16::try_from(display_width - 1).unwrap_or(u16::MAX))
                .min(row_right);

            if x <= selected_right && span_right >= selected_left {
                text.push_str(symbol);
            }

            let next_x = x.saturating_add(u16::try_from(display_width).unwrap_or(u16::MAX));
            if next_x <= x {
                break;
            }
            x = next_x;
        }

        text.trim_end_matches(' ').to_string()
    }

    fn paint_row(
        &self,
        target: &mut Buffer,
        y: u16,
        selected_left: u16,
        selected_right: u16,
        area: Rect,
    ) {
        let row_right = area.right().saturating_sub(1);
        let mut x = area.x;

        while x <= row_right {
            let position = Position::new(x, y);
            let Some(cell) = self.snapshot.cell(position) else {
                break;
            };
            let display_width = Span::raw(cell.symbol()).width().max(1);
            let span_right = x
                .saturating_add(u16::try_from(display_width - 1).unwrap_or(u16::MAX))
                .min(row_right);

            if x <= selected_right && span_right >= selected_left {
                let mut selected_x = x;
                while selected_x <= span_right {
                    if let Some(cell) = target.cell_mut(Position::new(selected_x, y)) {
                        cell.set_bg(SELECTION_BG)
                            .set_fg(SELECTION_FG)
                            .set_style(Style::default().add_modifier(Modifier::REVERSED));
                    }
                    if selected_x == u16::MAX {
                        break;
                    }
                    selected_x += 1;
                }
            }

            let next_x = x.saturating_add(u16::try_from(display_width).unwrap_or(u16::MAX));
            if next_x <= x {
                break;
            }
            x = next_x;
        }
    }
}

fn clamp_position(area: Rect, position: Position) -> Position {
    if area.is_empty() {
        return Position::new(area.x, area.y);
    }

    Position::new(
        position.x.clamp(area.x, area.right().saturating_sub(1)),
        position.y.clamp(area.y, area.bottom().saturating_sub(1)),
    )
}

#[cfg(test)]
mod tests {
    use ratatui::style::{Color, Style};

    use super::*;

    #[test]
    fn forward_and_reverse_multiline_selection_match() {
        let snapshot = Buffer::with_lines(["abcde", "fghij", "klmno"]);
        let mut forward = TextSelection::new(snapshot.clone(), Position::new(1, 0));
        forward.finish(Position::new(2, 2));

        let mut reverse = TextSelection::new(snapshot, Position::new(2, 2));
        reverse.finish(Position::new(1, 0));

        assert_eq!(forward.text(), "bcde\nfghij\nklm");
        assert_eq!(reverse.text(), forward.text());
        assert_eq!(forward.area(), Rect::new(0, 0, 5, 3));
    }

    #[test]
    fn text_trims_row_padding_and_preserves_blank_lines_and_indentation() {
        let snapshot = Buffer::with_lines(["  ind  ", "", "  next   "]);
        let mut selection = TextSelection::new(snapshot, Position::new(0, 0));
        selection.finish(Position::new(8, 2));

        assert_eq!(selection.text(), "  ind\n\n  next");
    }

    #[test]
    fn same_cell_is_empty_and_partial_row_is_inclusive() {
        let snapshot = Buffer::with_lines(["abcde"]);
        let mut click = TextSelection::new(snapshot.clone(), Position::new(2, 0));
        assert_eq!(click.text(), "");
        assert!(!click.moved());

        click.extend(Position::new(4, 0));
        assert_eq!(click.text(), "cde");
        assert!(click.moved());
    }

    #[test]
    fn wide_graphemes_and_combining_accents_are_copied_as_visible_symbols() {
        let mut snapshot = Buffer::empty(Rect::new(0, 0, 7, 1));
        snapshot.set_string(0, 0, "Aé中e\u{301}Z", Style::default());

        let mut full = TextSelection::new(snapshot.clone(), Position::new(0, 0));
        full.finish(Position::new(5, 0));
        assert_eq!(full.text(), "Aé中e\u{301}Z");

        let mut across_filler = TextSelection::new(snapshot, Position::new(3, 0));
        across_filler.finish(Position::new(4, 0));
        assert_eq!(across_filler.text(), "中e\u{301}");
    }

    #[test]
    fn paint_copies_snapshot_and_highlights_clipped_target_without_mutating_snapshot() {
        let snapshot = Buffer::with_lines(["hello"]);
        let mut selection = TextSelection::new(snapshot.clone(), Position::new(1, 0));
        selection.finish(Position::new(3, 0));
        let mut target = Buffer::empty(Rect::new(1, 0, 3, 1));

        selection.paint(&mut target);

        assert_eq!(target.cell((1, 0)).unwrap().symbol(), "e");
        assert_eq!(target.cell((3, 0)).unwrap().symbol(), "l");
        assert_eq!(target.cell((1, 0)).unwrap().bg, Color::Rgb(72, 60, 100));
        assert_eq!(target.cell((1, 0)).unwrap().fg, Color::Rgb(225, 225, 225));
        assert!(target
            .cell((1, 0))
            .unwrap()
            .modifier
            .contains(Modifier::REVERSED));
        assert_eq!(snapshot.cell((1, 0)).unwrap().symbol(), "e");
        assert_eq!(snapshot.cell((1, 0)).unwrap().bg, Color::Reset);
    }
}
