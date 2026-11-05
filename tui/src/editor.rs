use ratatui::text::Span;
use unicode_segmentation::UnicodeSegmentation;

/// Cursor and editing operations for the compose draft.
#[derive(Default)]
pub(super) struct ComposerEditor {
    /// `None` keeps the cursor at the end as the draft changes.
    cursor: Option<usize>,
    preferred_column: Option<usize>,
    /// Keeps a vertically reached soft-wrap end on its selected visual row.
    vertical_affinity: Option<(usize, usize, usize)>,
}

impl ComposerEditor {
    /// Returns the cursor byte offset, clamped to a grapheme boundary.
    pub fn position(&self, text: &str) -> usize {
        self.cursor
            .map_or(text.len(), |position| boundary_at_or_before(text, position))
    }

    /// Resets the cursor to the end of the current draft.
    pub fn reset(&mut self) {
        self.cursor = None;
        self.preferred_column = None;
        self.vertical_affinity = None;
    }

    /// Inserts text at the current cursor and leaves the cursor after it.
    pub fn insert(&mut self, text: &mut String, added: &str) {
        let position = self.position(text);
        text.insert_str(position, added);
        self.cursor = Some(boundary_at_or_after(text, position + added.len()));
        self.preferred_column = None;
        self.vertical_affinity = None;
    }

    /// Moves the cursor one grapheme to the left.
    pub fn move_left(&mut self, text: &str) {
        let position = self.position(text);
        let boundaries = grapheme_boundaries(text);
        let index = boundaries
            .binary_search(&position)
            .unwrap_or_else(|index| index);
        let target = boundaries[index.saturating_sub(1)];
        self.set_cursor(text, target);
    }

    /// Moves the cursor one grapheme to the right.
    pub fn move_right(&mut self, text: &str) {
        let position = self.position(text);
        let boundaries = grapheme_boundaries(text);
        let index = boundaries
            .binary_search(&position)
            .unwrap_or_else(|index| index);
        let target = boundaries.get(index + 1).copied().unwrap_or(text.len());
        self.set_cursor(text, target);
    }

    /// Moves the cursor to the start of its logical line.
    pub fn move_home(&mut self, text: &str) {
        let position = self.position(text);
        let (line_start, _) = logical_line_bounds(text, position);
        self.set_cursor(text, line_start);
    }

    /// Moves the cursor to the end of its logical line.
    pub fn move_end(&mut self, text: &str) {
        let position = self.position(text);
        let (_, line_end) = logical_line_bounds(text, position);
        self.set_cursor(text, line_end);
    }

    /// Moves left over whitespace and the preceding whitespace-delimited word.
    pub fn move_word_left(&mut self, text: &str) {
        let target = word_left_position(text, self.position(text));
        self.set_cursor(text, target);
    }

    /// Moves right over the current word and following whitespace.
    pub fn move_word_right(&mut self, text: &str) {
        let target = word_right_position(text, self.position(text));
        self.set_cursor(text, target);
    }

    /// Removes the grapheme immediately before the cursor.
    pub fn backspace(&mut self, text: &mut String) {
        let position = self.position(text);
        let boundaries = grapheme_boundaries(text);
        let index = boundaries
            .binary_search(&position)
            .unwrap_or_else(|index| index);
        let target = if index > 0 {
            let start = boundaries[index - 1];
            text.replace_range(start..position, "");
            start
        } else {
            position
        };
        self.set_cursor(text, target);
    }

    /// Removes the grapheme immediately after the cursor.
    pub fn delete(&mut self, text: &mut String) {
        let position = self.position(text);
        let boundaries = grapheme_boundaries(text);
        let index = boundaries
            .binary_search(&position)
            .unwrap_or_else(|index| index);
        if let Some(end) = boundaries.get(index + 1).copied() {
            text.replace_range(position..end, "");
        }
        self.set_cursor(text, position);
    }

    /// Removes the preceding word and any whitespace between it and the cursor.
    pub fn delete_word_before(&mut self, text: &mut String) {
        let position = self.position(text);
        let start = word_left_position(text, position);
        text.replace_range(start..position, "");
        self.set_cursor(text, start);
    }

    /// Removes text before the cursor on its current logical line.
    pub fn clear_before(&mut self, text: &mut String) {
        let position = self.position(text);
        let (line_start, _) = logical_line_bounds(text, position);
        text.replace_range(line_start..position, "");
        self.set_cursor(text, line_start);
    }

    /// Removes text after the cursor on its current logical line.
    pub fn clear_after(&mut self, text: &mut String) {
        let position = self.position(text);
        let (_, line_end) = logical_line_bounds(text, position);
        text.replace_range(position..line_end, "");
        self.set_cursor(text, position);
    }

    /// Returns the visual row containing the current cursor.
    pub fn cursor_row(&self, text: &str, width: usize) -> usize {
        let rows = layout(text, width);
        let position = self.position(text);

        if let Some((affinity_position, affinity_width, affinity_row)) = self.vertical_affinity {
            if affinity_position == position
                && affinity_width == width
                && rows
                    .get(affinity_row)
                    .is_some_and(|row| row.end == position)
                && rows
                    .get(affinity_row + 1)
                    .is_some_and(|next| next.start == position)
                && !is_hard_newline_at(text, position)
            {
                return affinity_row;
            }
        }

        for (index, row) in rows.iter().enumerate() {
            if position < row.end {
                return index;
            }
            if position == row.end {
                if is_hard_newline_at(text, row.end) {
                    return index;
                }
                if rows
                    .get(index + 1)
                    .is_some_and(|next| next.start == position)
                {
                    return index + 1;
                }
                return index;
            }
        }

        rows.len().saturating_sub(1)
    }

    /// Moves vertically by visual rows, retaining the original display column.
    pub fn move_vertical(&mut self, text: &str, width: usize, delta: isize) -> bool {
        let rows = layout(text, width);
        let current_row = self.cursor_row(text, width);
        let target_row = current_row as i128 + delta as i128;
        if target_row < 0 || target_row >= rows.len() as i128 {
            return false;
        }

        let position = self.position(text);
        let preferred_column = *self
            .preferred_column
            .get_or_insert_with(|| display_column(text, rows[current_row], position));
        let target_row = target_row as usize;
        let target = position_at_display_column(text, rows[target_row], preferred_column);
        self.cursor = Some(target);
        self.vertical_affinity = rows
            .get(target_row + 1)
            .filter(|next| next.start == target && rows[target_row].end == target)
            .map(|_| (target, width, target_row));
        true
    }

    fn set_cursor(&mut self, text: &str, position: usize) {
        self.cursor = Some(boundary_at_or_before(text, position));
        self.preferred_column = None;
        self.vertical_affinity = None;
    }
}

/// A visual row's byte range, excluding its hard newline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct EditorLine {
    pub start: usize,
    pub end: usize,
}

/// Hard-wraps text at display-cell width while preserving all source bytes.
pub(super) fn layout(text: &str, width: usize) -> Vec<EditorLine> {
    let width = width.max(1);
    let mut rows = Vec::new();
    let mut line_start = 0;
    let mut line_width: usize = 0;

    for (start, grapheme) in text.grapheme_indices(true) {
        if grapheme == "\n" || grapheme == "\r\n" {
            rows.push(EditorLine {
                start: line_start,
                end: start,
            });
            line_start = start + grapheme.len();
            line_width = 0;
            continue;
        }

        let grapheme_width = Span::raw(grapheme).width();
        if line_width.saturating_add(grapheme_width) > width && start > line_start {
            rows.push(EditorLine {
                start: line_start,
                end: start,
            });
            line_start = start;
            line_width = 0;
        }
        line_width = line_width.saturating_add(grapheme_width);
    }

    rows.push(EditorLine {
        start: line_start,
        end: text.len(),
    });
    rows
}

fn grapheme_boundaries(text: &str) -> Vec<usize> {
    let mut boundaries: Vec<usize> = text
        .grapheme_indices(true)
        .map(|(index, _)| index)
        .collect();
    boundaries.push(text.len());
    boundaries
}

fn boundary_at_or_before(text: &str, position: usize) -> usize {
    let position = position.min(text.len());
    let boundaries = grapheme_boundaries(text);
    match boundaries.binary_search(&position) {
        Ok(index) => boundaries[index],
        Err(index) => boundaries[index.saturating_sub(1)],
    }
}

fn boundary_at_or_after(text: &str, position: usize) -> usize {
    let position = position.min(text.len());
    let boundaries = grapheme_boundaries(text);
    match boundaries.binary_search(&position) {
        Ok(index) => boundaries[index],
        Err(index) => boundaries.get(index).copied().unwrap_or(text.len()),
    }
}

fn logical_line_bounds(text: &str, position: usize) -> (usize, usize) {
    let position = boundary_at_or_before(text, position);
    let line_start = text[..position]
        .rfind('\n')
        .map_or(0, |newline| newline + 1);
    let mut line_end = text[position..]
        .find('\n')
        .map_or(text.len(), |newline| position + newline);

    if line_end > line_start && text.as_bytes().get(line_end - 1) == Some(&b'\r') {
        line_end -= 1;
    }
    (line_start, line_end)
}

fn word_left_position(text: &str, position: usize) -> usize {
    let position = boundary_at_or_before(text, position);
    let graphemes: Vec<(usize, &str)> = text.grapheme_indices(true).collect();
    let mut index = graphemes.partition_point(|(start, _)| *start < position);

    while index > 0 && is_whitespace(graphemes[index - 1].1) {
        index -= 1;
    }
    while index > 0 && !is_whitespace(graphemes[index - 1].1) {
        index -= 1;
    }
    graphemes.get(index).map_or(text.len(), |(start, _)| *start)
}

fn word_right_position(text: &str, position: usize) -> usize {
    let position = boundary_at_or_before(text, position);
    let graphemes: Vec<(usize, &str)> = text.grapheme_indices(true).collect();
    let mut index = graphemes.partition_point(|(start, _)| *start < position);

    while index < graphemes.len() && !is_whitespace(graphemes[index].1) {
        index += 1;
    }
    while index < graphemes.len() && is_whitespace(graphemes[index].1) {
        index += 1;
    }
    graphemes.get(index).map_or(text.len(), |(start, _)| *start)
}

fn is_whitespace(grapheme: &str) -> bool {
    grapheme.chars().all(char::is_whitespace)
}

fn is_hard_newline_at(text: &str, position: usize) -> bool {
    text.get(position..)
        .is_some_and(|suffix| suffix.starts_with('\n') || suffix.starts_with("\r\n"))
}

fn display_column(text: &str, row: EditorLine, position: usize) -> usize {
    text[row.start..position.min(row.end)]
        .graphemes(true)
        .map(|grapheme| Span::raw(grapheme).width())
        .sum()
}

fn position_at_display_column(text: &str, row: EditorLine, column: usize) -> usize {
    let mut position = row.start;
    let mut current_column = 0;

    for grapheme in text[row.start..row.end].graphemes(true) {
        let next_column = current_column + Span::raw(grapheme).width();
        if next_column > column {
            if column - current_column > next_column - column {
                position += grapheme.len();
            }
            return position;
        }
        position += grapheme.len();
        current_column = next_column;
    }

    position
}

#[cfg(test)]
mod tests {
    use super::{layout, ComposerEditor, EditorLine};

    #[test]
    fn inserts_and_deletes_at_a_middle_cursor() {
        let mut editor = ComposerEditor::default();
        let mut text = "ab".to_owned();

        editor.move_left(&text);
        editor.insert(&mut text, "界");
        assert_eq!(text, "a界b");
        assert_eq!(editor.position(&text), 4);

        editor.delete(&mut text);
        assert_eq!(text, "a界");
        editor.backspace(&mut text);
        assert_eq!(text, "a");
    }

    #[test]
    fn backspace_removes_an_emoji_grapheme_as_a_unit() {
        let mut editor = ComposerEditor::default();
        let mut text = "a👩‍🚀b".to_owned();

        editor.move_left(&text);
        editor.backspace(&mut text);

        assert_eq!(text, "ab");
        assert_eq!(editor.position(&text), 1);
    }

    #[test]
    fn home_end_and_line_clears_respect_logical_newlines() {
        let mut editor = ComposerEditor::default();
        let mut text = "one\ntwo\nthree".to_owned();

        editor.move_home(&text);
        assert_eq!(editor.position(&text), 8);
        editor.move_end(&text);
        assert_eq!(editor.position(&text), text.len());
        editor.move_home(&text);
        editor.move_right(&text);
        editor.move_right(&text);
        editor.clear_before(&mut text);
        assert_eq!(text, "one\ntwo\nree");
        assert_eq!(editor.position(&text), 8);
        editor.clear_after(&mut text);
        assert_eq!(text, "one\ntwo\n");
    }

    #[test]
    fn word_deletion_removes_the_word_and_adjacent_whitespace() {
        let mut editor = ComposerEditor::default();
        let mut text = "one two  ".to_owned();

        editor.delete_word_before(&mut text);

        assert_eq!(text, "one ");
        assert_eq!(editor.position(&text), text.len());
    }

    #[test]
    fn layout_preserves_spaces_newlines_and_a_trailing_empty_line() {
        let rows = layout("ab cd\nxy\n", 3);

        assert_eq!(
            rows,
            vec![
                EditorLine { start: 0, end: 3 },
                EditorLine { start: 3, end: 5 },
                EditorLine { start: 6, end: 8 },
                EditorLine { start: 9, end: 9 },
            ]
        );
    }

    #[test]
    fn cursor_rows_assign_soft_wraps_forward_and_newlines_backward() {
        let mut editor = ComposerEditor::default();
        let text = "abcd\nef";

        editor.move_home(text);
        editor.move_left(text);
        editor.move_left(text);
        editor.move_left(text);
        assert_eq!(editor.cursor_row(text, 2), 1);

        editor.move_right(text);
        editor.move_right(text);
        assert_eq!(editor.position(text), 4);
        assert_eq!(editor.cursor_row(text, 2), 1);
        editor.move_right(text);
        assert_eq!(editor.position(text), 5);
        assert_eq!(editor.cursor_row(text, 2), 2);

        editor.move_end(text);
        assert_eq!(editor.cursor_row(text, 2), 2);
    }

    #[test]
    fn vertical_movement_keeps_column_across_wide_and_short_rows() {
        let mut editor = ComposerEditor::default();
        let text = "界ab\nx\nwxyz";

        assert!(editor.move_vertical(&text, 4, -1));
        assert_eq!(editor.position(&text), 7);
        assert!(editor.move_vertical(&text, 4, -1));
        assert_eq!(editor.position(&text), 5);
        assert!(editor.move_vertical(&text, 4, 1));
        assert_eq!(editor.position(&text), 7);
        assert!(editor.move_vertical(&text, 4, 1));
        assert_eq!(editor.position(&text), text.len());
        assert!(!editor.move_vertical(&text, 4, 1));

        let mut wrapped_editor = ComposerEditor::default();
        let wrapped_text = "abcdef";
        assert_eq!(wrapped_editor.cursor_row(wrapped_text, 3), 1);
        assert!(wrapped_editor.move_vertical(wrapped_text, 3, -1));
        assert_eq!(wrapped_editor.position(wrapped_text), 3);
        assert_eq!(wrapped_editor.cursor_row(wrapped_text, 3), 0);
        assert!(!wrapped_editor.move_vertical(wrapped_text, 3, -1));
        assert!(wrapped_editor.move_vertical(wrapped_text, 3, 1));
        assert_eq!(wrapped_editor.position(wrapped_text), wrapped_text.len());
    }

    #[test]
    fn cursor_position_clamps_to_grapheme_boundaries() {
        let mut editor = ComposerEditor::default();
        let text = "e\u{301}x";
        editor.cursor = Some(1);

        assert_eq!(editor.position(text), 0);
        editor.move_right(text);
        assert_eq!(editor.position(text), 3);
    }
}
