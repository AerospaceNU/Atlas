//! A small markdown renderer for assistant replies.
//!
//! It covers the marks models actually emit: headings, emphasis, inline and
//! fenced code, links, lists, quotes, and pipe tables. Anything it does not
//! understand is left as plain text without the marker characters it could parse.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mark {
    Plain,
    Strong,
    Emphasis,
    Code,
    Heading,
    /// Box-drawing rules around a table.
    Border,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Piece {
    pub text: String,
    pub mark: Mark,
}

/// One visual line per entry. Fenced code stays one piece per source line.
#[cfg(test)]
fn render_lines(source: &str) -> Vec<Vec<Piece>> {
    render_fitted(source, usize::MAX)
}

/// Like [`render_lines`], but a pipe table is fitted to `width` columns.
pub fn render_fitted(source: &str, width: usize) -> Vec<Vec<Piece>> {
    let raws: Vec<&str> = source.split('\n').collect();
    let mut lines = Vec::new();
    let mut index = 0;
    let mut in_code = false;
    while index < raws.len() {
        let line = raws[index].trim_end();
        if line.starts_with("```") {
            in_code = !in_code;
            index += 1;
            continue;
        }
        if in_code {
            lines.push(vec![piece(line, Mark::Code)]);
            index += 1;
            continue;
        }
        if let Some((consumed, rendered)) = try_table(&raws[index..], width) {
            lines.extend(rendered);
            index += consumed;
            continue;
        }
        push_text_line(&mut lines, line);
        index += 1;
    }
    if lines.is_empty() {
        lines.push(vec![piece("", Mark::Plain)]);
    }
    lines
}

fn push_text_line(lines: &mut Vec<Vec<Piece>>, line: &str) {
    if line.is_empty() {
        lines.push(vec![piece("", Mark::Plain)]);
        return;
    }
    if let Some(title) = heading(line) {
        lines.push(vec![piece(&title, Mark::Heading)]);
        return;
    }
    if let Some(body) = line.strip_prefix("- ").or_else(|| line.strip_prefix("* ")) {
        let mut row = vec![piece("• ", Mark::Plain)];
        row.extend(parse_inline(body));
        lines.push(row);
        return;
    }
    if let Some((marker, body)) = numbered(line) {
        let mut row = vec![piece(marker, Mark::Plain)];
        row.extend(parse_inline(body));
        lines.push(row);
        return;
    }
    if let Some(body) = line.strip_prefix("> ") {
        let mut row = vec![piece("│ ", Mark::Plain)];
        row.extend(parse_inline(body));
        lines.push(row);
        return;
    }
    lines.push(parse_inline(line));
}

fn piece(text: &str, mark: Mark) -> Piece {
    Piece {
        text: text.to_string(),
        mark,
    }
}

fn heading(line: &str) -> Option<String> {
    let hashes = line.chars().take_while(|ch| *ch == '#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let rest = line[hashes..].strip_prefix(' ')?;
    Some(rest.to_string())
}

fn numbered(line: &str) -> Option<(&str, &str)> {
    let digits = line.chars().take_while(|ch| ch.is_ascii_digit()).count();
    if digits == 0 {
        return None;
    }
    let rest = &line[digits..];
    let body = rest.strip_prefix(". ")?;
    Some((&line[..digits + 2], body))
}

fn parse_inline(input: &str) -> Vec<Piece> {
    let chars: Vec<char> = input.chars().collect();
    let mut pieces = Vec::new();
    let mut buf = String::new();
    let mut index = 0;
    while index < chars.len() {
        if chars[index] == '`' {
            if let Some(end) = chars[index + 1..].iter().position(|ch| *ch == '`') {
                push_buf(&mut pieces, &mut buf, Mark::Plain);
                let text: String = chars[index + 1..index + 1 + end].iter().collect();
                pieces.push(Piece {
                    text,
                    mark: Mark::Code,
                });
                index += end + 2;
                continue;
            }
        }
        if chars[index] == '*' && index + 1 < chars.len() && chars[index + 1] == '*' {
            if let Some(end) = find_pair(&chars, index + 2) {
                push_buf(&mut pieces, &mut buf, Mark::Plain);
                let text: String = chars[index + 2..end].iter().collect();
                pieces.push(Piece {
                    text,
                    mark: Mark::Strong,
                });
                index = end + 2;
                continue;
            }
        }
        if chars[index] == '*' {
            if let Some(end) = chars[index + 1..].iter().position(|ch| *ch == '*') {
                if end > 0 {
                    push_buf(&mut pieces, &mut buf, Mark::Plain);
                    let text: String = chars[index + 1..index + 1 + end].iter().collect();
                    pieces.push(Piece {
                        text,
                        mark: Mark::Emphasis,
                    });
                    index += end + 2;
                    continue;
                }
            }
        }
        if chars[index] == '[' {
            if let Some((label, next)) = link(&chars, index) {
                push_buf(&mut pieces, &mut buf, Mark::Plain);
                pieces.push(Piece {
                    text: label,
                    mark: Mark::Plain,
                });
                index = next;
                continue;
            }
        }
        buf.push(chars[index]);
        index += 1;
    }
    push_buf(&mut pieces, &mut buf, Mark::Plain);
    if pieces.is_empty() {
        pieces.push(piece("", Mark::Plain));
    }
    pieces
}

fn push_buf(pieces: &mut Vec<Piece>, buf: &mut String, mark: Mark) {
    if !buf.is_empty() {
        pieces.push(Piece {
            text: std::mem::take(buf),
            mark,
        });
    }
}

fn find_pair(chars: &[char], start: usize) -> Option<usize> {
    let mut index = start;
    while index + 1 < chars.len() {
        if chars[index] == '*' && chars[index + 1] == '*' {
            return Some(index);
        }
        index += 1;
    }
    None
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Align {
    Left,
    Right,
    Center,
}

fn try_table(raws: &[&str], width: usize) -> Option<(usize, Vec<Vec<Piece>>)> {
    if raws.len() < 2 {
        return None;
    }
    let header = split_cells(raws[0].trim_end())?;
    let aligns = split_separator(raws[1].trim_end())?;
    if header.len() < 2 || aligns.len() != header.len() {
        return None;
    }
    let columns = header.len();
    let mut rows = vec![header];
    let mut consumed = 2;
    for raw in raws.iter().skip(2) {
        let line = raw.trim_end();
        if line.is_empty() || split_separator(line).is_some() {
            break;
        }
        let cells = split_cells(line)?;
        rows.push(fit_cells(cells, columns));
        consumed += 1;
    }
    let parsed: Vec<Vec<Vec<Piece>>> = rows
        .iter()
        .enumerate()
        .map(|(row_index, row)| {
            row.iter()
                .map(|cell| {
                    let pieces = parse_inline(cell);
                    if row_index == 0 {
                        bold_header(pieces)
                    } else {
                        pieces
                    }
                })
                .collect()
        })
        .collect();
    Some((consumed, layout_table(&parsed, &aligns, width)))
}

fn split_cells(line: &str) -> Option<Vec<String>> {
    let line = line.trim();
    if !line.starts_with('|') {
        return None;
    }
    let mut inner = line.strip_prefix('|').unwrap_or(line);
    if let Some(rest) = inner.strip_suffix('|') {
        inner = rest;
    }
    let cells: Vec<String> = inner
        .split('|')
        .map(|cell| cell.trim().to_string())
        .collect();
    if cells.len() < 2 {
        return None;
    }
    Some(cells)
}

fn split_separator(line: &str) -> Option<Vec<Align>> {
    let cells = split_cells(line)?;
    let mut aligns = Vec::with_capacity(cells.len());
    for cell in cells {
        let token = cell.trim();
        if token.is_empty() || !token.contains('-') {
            return None;
        }
        if !token.chars().all(|ch| matches!(ch, '-' | ':')) {
            return None;
        }
        let left = token.starts_with(':');
        let right = token.ends_with(':');
        aligns.push(match (left, right) {
            (true, true) => Align::Center,
            (false, true) => Align::Right,
            _ => Align::Left,
        });
    }
    Some(aligns)
}

fn fit_cells(mut cells: Vec<String>, columns: usize) -> Vec<String> {
    if cells.len() > columns {
        let tail = cells.split_off(columns - 1).join(" | ");
        cells.push(tail);
    }
    cells.resize(columns, String::new());
    cells
}

fn bold_header(pieces: Vec<Piece>) -> Vec<Piece> {
    pieces
        .into_iter()
        .map(|mut piece| {
            if piece.mark != Mark::Code {
                piece.mark = Mark::Strong;
            }
            piece
        })
        .collect()
}

fn layout_table(rows: &[Vec<Vec<Piece>>], aligns: &[Align], width: usize) -> Vec<Vec<Piece>> {
    let widths = column_widths(rows, width);
    let mut lines = vec![rule(&widths, '┌', '┬', '┐')];
    for (index, row) in rows.iter().enumerate() {
        lines.extend(render_row(row, &widths, aligns));
        if index == 0 {
            lines.push(rule(&widths, '├', '┼', '┤'));
        }
    }
    lines.push(rule(&widths, '└', '┴', '┘'));
    lines
}

fn column_widths(rows: &[Vec<Vec<Piece>>], width: usize) -> Vec<usize> {
    let columns = rows.first().map_or(0, Vec::len);
    let mut widths = vec![1; columns];
    for row in rows {
        for (index, cell) in row.iter().enumerate() {
            widths[index] = widths[index].max(piece_span(cell).max(1));
        }
    }
    if width == usize::MAX || columns == 0 {
        return widths;
    }
    let chrome = 3 * columns + 1;
    if width <= chrome {
        return vec![1; columns];
    }
    let budget = width - chrome;
    while widths.iter().sum::<usize>() > budget {
        let Some(index) = widths
            .iter()
            .enumerate()
            .filter(|(_, cell)| **cell > 1)
            .max_by_key(|(_, cell)| *cell)
            .map(|(index, _)| index)
        else {
            break;
        };
        widths[index] -= 1;
    }
    widths
}

fn piece_span(pieces: &[Piece]) -> usize {
    pieces.iter().map(|item| item.text.chars().count()).sum()
}

fn rule(widths: &[usize], left: char, junction: char, right: char) -> Vec<Piece> {
    let mut text = String::new();
    text.push(left);
    for (index, width) in widths.iter().enumerate() {
        if index > 0 {
            text.push(junction);
        }
        text.extend(std::iter::repeat_n('─', width + 2));
    }
    text.push(right);
    vec![piece(&text, Mark::Border)]
}

fn render_row(cells: &[Vec<Piece>], widths: &[usize], aligns: &[Align]) -> Vec<Vec<Piece>> {
    let wrapped: Vec<Vec<Vec<Piece>>> = cells
        .iter()
        .zip(widths)
        .zip(aligns)
        .map(|((cell, width), align)| {
            wrap_cell(cell, *width)
                .into_iter()
                .map(|line| pad_cell(&line, *width, *align))
                .collect()
        })
        .collect();
    let height = wrapped.iter().map(Vec::len).max().unwrap_or(1).max(1);
    let mut lines = Vec::with_capacity(height);
    for line_index in 0..height {
        let row: Vec<Vec<Piece>> = wrapped
            .iter()
            .zip(widths)
            .zip(aligns)
            .map(|((cell, width), align)| {
                cell.get(line_index)
                    .cloned()
                    .unwrap_or_else(|| pad_cell(&[], *width, *align))
            })
            .collect();
        lines.push(content_line(&row));
    }
    lines
}

fn content_line(cells: &[Vec<Piece>]) -> Vec<Piece> {
    let mut pieces = Vec::new();
    for cell in cells {
        pieces.push(piece("│", Mark::Border));
        pieces.push(piece(" ", Mark::Plain));
        pieces.extend(cell.clone());
        pieces.push(piece(" ", Mark::Plain));
    }
    pieces.push(piece("│", Mark::Border));
    pieces
}

fn pad_cell(cell: &[Piece], width: usize, align: Align) -> Vec<Piece> {
    let used = piece_span(cell);
    let pad = width.saturating_sub(used);
    let (left, right) = match align {
        Align::Left => (0, pad),
        Align::Right => (pad, 0),
        Align::Center => (pad / 2, pad - pad / 2),
    };
    let mut pieces = Vec::new();
    if left > 0 {
        pieces.push(piece(&" ".repeat(left), Mark::Plain));
    }
    pieces.extend(cell.iter().cloned());
    if right > 0 {
        pieces.push(piece(&" ".repeat(right), Mark::Plain));
    }
    pieces
}

fn wrap_cell(pieces: &[Piece], width: usize) -> Vec<Vec<Piece>> {
    let width = width.max(1);
    let mut lines: Vec<Vec<Piece>> = vec![Vec::new()];
    let mut column = 0;
    for item in pieces {
        let mut rest = item.text.as_str();
        while !rest.is_empty() {
            if column >= width {
                lines.push(Vec::new());
                column = 0;
            }
            if column == 0 {
                rest = rest.trim_start();
                if rest.is_empty() {
                    break;
                }
            }
            let room = width - column;
            let break_at = wrap_point(rest, room);
            let take: String = rest.chars().take(break_at).collect();
            let bytes = take.len();
            rest = &rest[bytes..];
            if take.chars().all(char::is_whitespace) && column == 0 {
                continue;
            }
            column += take.chars().count();
            lines
                .last_mut()
                .expect("line")
                .push(piece(&take, item.mark));
            if !rest.is_empty() && column >= width {
                lines.push(Vec::new());
                column = 0;
            }
        }
    }
    if lines.last().is_some_and(Vec::is_empty) && lines.len() > 1 {
        lines.pop();
    }
    if lines.is_empty() {
        lines.push(Vec::new());
    }
    lines
}

fn wrap_point(text: &str, room: usize) -> usize {
    let count = text.chars().count();
    if count <= room {
        return count;
    }
    let head: String = text.chars().take(room).collect();
    if let Some(space) = head.rfind(' ') {
        if space > 0 {
            return head[..space].chars().count();
        }
    }
    room
}

fn link(chars: &[char], start: usize) -> Option<(String, usize)> {
    let label_end = chars[start + 1..].iter().position(|ch| *ch == ']')? + start + 1;
    if label_end + 1 >= chars.len() || chars[label_end + 1] != '(' {
        return None;
    }
    let url_end = chars[label_end + 2..].iter().position(|ch| *ch == ')')? + label_end + 2;
    let label: String = chars[start + 1..label_end].iter().collect();
    Some((label, url_end + 1))
}

#[cfg(test)]
mod tests {
    use super::{render_lines, Mark};

    fn flat(source: &str) -> String {
        render_lines(source)
            .into_iter()
            .map(|line| line.into_iter().map(|piece| piece.text).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn drops_raw_markers() {
        let text = flat("# Title\n**bold** and *lean* `code` [docs](https://example.com)");
        assert!(text.contains("Title"));
        assert!(text.contains("bold"));
        assert!(text.contains("lean"));
        assert!(text.contains("code"));
        assert!(text.contains("docs"));
        assert!(!text.contains("**"));
        assert!(!text.contains('`'));
        assert!(!text.contains("https://"));
        assert!(!text.contains('#'));
    }

    #[test]
    fn fenced_code_is_marked_and_the_fence_is_hidden() {
        let lines = render_lines("```\nkeep\n```");
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0][0].text, "keep");
        assert_eq!(lines[0][0].mark, Mark::Code);
    }

    #[test]
    fn lists_and_quotes_use_plain_markers() {
        let text = flat("- one\n> two");
        assert!(text.contains("• one"));
        assert!(text.contains("│ two"));
        assert!(!text.contains("- one"));
    }

    #[test]
    fn a_pipe_table_is_drawn_with_borders() {
        let source = "\
| Item | Count | Notes |
| --- | ---: | :--- |
| `dist/` | 3 files | Built wheel |
| caches | 116 dirs | regenerable |
";
        let lines = render_lines(source);
        let text = lines
            .iter()
            .map(|line| {
                line.iter()
                    .map(|item| item.text.as_str())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains('┌'), "{text}");
        assert!(text.contains('├'), "{text}");
        assert!(text.contains('└'), "{text}");
        assert!(text.contains("Item"));
        assert!(text.contains("dist/"));
        assert!(text.contains("116 dirs"));
        assert!(!text.contains("| --- |"), "{text}");
        assert!(lines.iter().any(|line| {
            line.iter()
                .any(|item| item.text.contains("Item") && item.mark == Mark::Strong)
        }));
        assert!(lines.iter().any(|line| {
            line.iter()
                .any(|item| item.text.contains("dist/") && item.mark == Mark::Code)
        }));
        let count_row = lines
            .iter()
            .find(|line| line.iter().any(|item| item.text.contains("3 files")))
            .expect("count");
        let flat_count: String = count_row.iter().map(|item| item.text.as_str()).collect();
        let header = lines
            .iter()
            .find(|line| line.iter().any(|item| item.text.contains("Count")))
            .expect("header");
        let flat_header: String = header.iter().map(|item| item.text.as_str()).collect();
        let files_at = flat_count.find("3 files").expect("files");
        let count_at = flat_header.find("Count").expect("count header");
        let files_end = files_at + "3 files".len();
        let count_end = count_at + "Count".len();
        assert_eq!(files_end, count_end, "right edge:\n{text}");
        assert!(
            count_at > files_at,
            "count header is right-aligned:\n{text}"
        );
    }

    #[test]
    fn a_narrow_table_wraps_inside_the_cells() {
        let source = "\
| Name | Notes |
| --- | --- |
| atlas | a long note that should wrap inside the cell |
";
        let lines = super::render_fitted(source, 36);
        assert!(lines.len() > 4, "expected a wrapped body row");
        for line in &lines {
            let text: String = line.iter().map(|item| item.text.as_str()).collect();
            if text.is_empty() {
                continue;
            }
            assert!(text.chars().count() <= 36, "{text}");
            assert!(
                text.starts_with('┌')
                    || text.starts_with('├')
                    || text.starts_with('└')
                    || text.starts_with('│'),
                "{text}"
            );
        }
        let body: String = lines
            .iter()
            .map(|line| {
                line.iter()
                    .map(|item| item.text.as_str())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(body.contains("wrap"));
        assert!(!body.contains("| --- |"));
    }

    #[test]
    fn a_single_pipe_line_is_not_a_table() {
        let text = flat("use a | b in prose\n| only one cell");
        assert!(text.contains("use a | b"));
        assert!(!text.contains('┌'));
    }
}
