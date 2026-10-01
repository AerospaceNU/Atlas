//! A small markdown renderer for assistant replies.
//!
//! It covers the marks models actually emit: headings, emphasis, inline and
//! fenced code, links, lists, and quotes. Anything it does not understand is
//! left as plain text without the marker characters it could parse.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mark {
    Plain,
    Strong,
    Emphasis,
    Code,
    Heading,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Piece {
    pub text: String,
    pub mark: Mark,
}

/// One visual line per entry. Fenced code stays one piece per source line.
pub fn render_lines(source: &str) -> Vec<Vec<Piece>> {
    let mut lines = Vec::new();
    let mut in_code = false;
    for raw in source.split('\n') {
        let line = raw.trim_end();
        if line.starts_with("```") {
            in_code = !in_code;
            continue;
        }
        if in_code {
            lines.push(vec![piece(line, Mark::Code)]);
            continue;
        }
        if line.is_empty() {
            lines.push(vec![piece("", Mark::Plain)]);
            continue;
        }
        if let Some(title) = heading(line) {
            lines.push(vec![piece(&title, Mark::Heading)]);
            continue;
        }
        if let Some(body) = line.strip_prefix("- ").or_else(|| line.strip_prefix("* ")) {
            let mut row = vec![piece("• ", Mark::Plain)];
            row.extend(parse_inline(body));
            lines.push(row);
            continue;
        }
        if let Some((marker, body)) = numbered(line) {
            let mut row = vec![piece(marker, Mark::Plain)];
            row.extend(parse_inline(body));
            lines.push(row);
            continue;
        }
        if let Some(body) = line.strip_prefix("> ") {
            let mut row = vec![piece("│ ", Mark::Plain)];
            row.extend(parse_inline(body));
            lines.push(row);
            continue;
        }
        lines.push(parse_inline(line));
    }
    if lines.is_empty() {
        lines.push(vec![piece("", Mark::Plain)]);
    }
    lines
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
}
