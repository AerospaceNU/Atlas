//! Splash shown in the transcript until the first prompt.

use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::theme::{BG_BASE, FG, GRAY};

/// Largest splash, in columns and rows. `logo_size(1_000, 1_000)` returns this.
const MAX_COLS: u16 = 64;
const MAX_ROWS: u16 = 9;

/// Smallest area that can still hold a mark and the word ATLAS.
const MIN_COLS: u16 = 8;
const MIN_ROWS: u16 = 3;

/// Columns and rows the splash occupies in an area of `cols` by `rows`.
///
/// The size grows with the area up to [`MAX_COLS`] by [`MAX_ROWS`]. `None`
/// when the area is too small to draw.
pub fn logo_size(cols: u16, rows: u16) -> Option<(u16, u16)> {
    if cols < MIN_COLS || rows < MIN_ROWS {
        return None;
    }
    Some((cols.min(MAX_COLS), rows.min(MAX_ROWS)))
}

/// Center the ATLAS splash in `area`. No-op when `area` is too small.
pub fn paint_logo(frame: &mut Frame<'_>, area: Rect) {
    let Some((cols, rows)) = logo_size(area.width, area.height) else {
        return;
    };
    let width = usize::from(cols);
    let height = usize::from(rows);
    let lines: Vec<Line> = (0..height)
        .map(|row| {
            let last = row + 1 == height;
            let text = if last {
                labeled_row(width)
            } else {
                mark_row(row, width)
            };
            let color = if last { FG } else { GRAY };
            Line::from(Span::styled(text, Style::default().fg(color).bg(BG_BASE)))
        })
        .collect();

    let x = area.x.saturating_add(area.width.saturating_sub(cols) / 2);
    let y = area.y.saturating_add(area.height.saturating_sub(rows) / 2);
    let rect = Rect {
        x,
        y,
        width: cols.min(area.width),
        height: rows.min(area.height),
    };
    frame.render_widget(Paragraph::new(lines), rect);
}

/// One row of the mark. Both edges are `.` so the measured span is `width`.
fn mark_row(row: usize, width: usize) -> String {
    let mut cells = vec![' '; width];
    if width == 0 {
        return String::new();
    }
    cells[0] = '.';
    cells[width - 1] = '.';

    let mid = width / 2;
    // A small diamond, plus a second offset spark so the mark is not a line.
    let reach = match row % 4 {
        0 => 0,
        1 => 2,
        2 => 4,
        _ => 2,
    };
    for offset in 0..=reach {
        place_dot(&mut cells, mid.saturating_sub(offset));
        place_dot(&mut cells, mid.saturating_add(offset));
    }
    if row % 2 == 0 {
        place_dot(&mut cells, mid.saturating_sub(8));
        place_dot(&mut cells, mid.saturating_add(8));
    }
    cells.into_iter().collect()
}

fn place_dot(cells: &mut [char], index: usize) {
    if let Some(cell) = cells.get_mut(index) {
        *cell = '.';
    }
}

fn labeled_row(width: usize) -> String {
    const LABEL: &str = "ATLAS";
    let mut cells = vec![' '; width];
    if width == 0 {
        return String::new();
    }
    cells[0] = '.';
    cells[width - 1] = '.';
    if LABEL.len() + 2 <= width {
        let start = (width - LABEL.len()) / 2;
        for (index, ch) in LABEL.chars().enumerate() {
            cells[start + index] = ch;
        }
    }
    cells.into_iter().collect()
}
