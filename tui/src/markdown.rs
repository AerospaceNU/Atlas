//! CommonMark + GFM markdown (`pulldown-cmark`), syntax-highlighted code blocks (`syntect`),
//! Unicode-aware word wrapping (`textwrap` + `unicode-width`), and colored unified diffs.

use std::sync::OnceLock;

use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use syntect::easy::HighlightLines;
use syntect::highlighting::{FontStyle, ThemeSet};
use syntect::parsing::SyntaxSet;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::theme::Theme;

static SYNTAX: OnceLock<(SyntaxSet, ThemeSet)> = OnceLock::new();

fn sp(text: impl Into<String>, style: Style) -> Span<'static> {
    Span::styled(text.into(), style)
}

pub fn wrap_plain(text: &str, width: usize) -> Vec<String> {
    let opts = textwrap::Options::new(width.max(1)).break_words(true);
    text.split('\n')
        .flat_map(|p| textwrap::wrap(p, &opts))
        .map(|c| c.into_owned())
        .collect()
}

fn wrap_spans(spans: Vec<Span<'static>>, width: usize) -> Vec<Vec<Span<'static>>> {
    let width = width.max(1);
    let (mut lines, mut cur, mut col) = (Vec::new(), Vec::new(), 0usize);
    for span in spans {
        let mut rest = span.content.as_ref();
        while !rest.is_empty() {
            let ws = rest.chars().next().is_some_and(char::is_whitespace);
            let at = rest
                .char_indices()
                .find(|&(_, c)| c.is_whitespace() != ws)
                .map_or(rest.len(), |(i, _)| i);
            let (mut tok, next) = rest.split_at(at);
            rest = next;
            if ws && col == 0 {
                continue;
            }
            if col > 0 && col + tok.width() > width {
                lines.push(std::mem::take(&mut cur));
                col = 0;
                if ws {
                    continue;
                }
            }
            while !tok.is_empty() {
                let avail = width.saturating_sub(col).max(1);
                let (chunk, tail) = split_width(tok, avail);
                push_styled(&mut cur, chunk, span.style);
                col += chunk.width();
                tok = tail;
                if !tok.is_empty() {
                    lines.push(std::mem::take(&mut cur));
                    col = 0;
                }
            }
        }
    }
    if !cur.is_empty() || lines.is_empty() {
        lines.push(cur);
    }
    lines
}

fn split_width(s: &str, max_w: usize) -> (&str, &str) {
    if s.width() <= max_w {
        return (s, "");
    }
    let (mut used, mut end) = (0, 0);
    for (idx, ch) in s.char_indices() {
        let w = ch.width().unwrap_or(0);
        if end > 0 && used + w > max_w {
            break;
        }
        (used, end) = (used + w, idx + ch.len_utf8());
        if used >= max_w {
            break;
        }
    }
    s.split_at(end.max(s.chars().next().map_or(0, char::len_utf8)))
}

fn push_styled(spans: &mut Vec<Span<'static>>, text: &str, style: Style) {
    if text.is_empty() {
        return;
    }
    if let Some(last) = spans.last_mut().filter(|l| l.style == style) {
        last.content.to_mut().push_str(text);
    } else {
        spans.push(sp(text, style));
    }
}

pub fn style_diff_lines(diff: &str, width: usize, theme: &Theme) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    for raw in diff.lines() {
        let st = if raw.starts_with("@@") {
            theme.bold(theme.cyan)
        } else if ["+++", "---", "diff --git", "index "]
            .iter()
            .any(|p| raw.starts_with(p))
        {
            theme.bold(theme.muted_bright)
        } else if raw.starts_with('+') {
            theme.fg(theme.green)
        } else if raw.starts_with('-') {
            theme.fg(theme.red)
        } else {
            theme.fg(theme.fg_secondary)
        };
        out.extend(
            wrap_plain(raw, width.max(1))
                .into_iter()
                .map(|w| Line::from(sp(w, st))),
        );
    }
    if out.is_empty() {
        out.push(Line::from(""));
    }
    out
}

pub fn render_markdown(source: &str, width: usize, theme: &Theme) -> Vec<Line<'static>> {
    let opts = Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TASKLISTS;
    let mut r = MdRenderer {
        width: width.max(1),
        theme,
        styles: vec![theme.base()],
        lines: Vec::new(),
        inline: Vec::new(),
        lists: Vec::new(),
        bq: 0,
        code: None,
        link: None,
        table: None,
        item_first: false,
    };
    for ev in Parser::new_ext(source, opts) {
        r.push(ev);
    }
    if let Some((lang, code)) = r.code.take() {
        r.render_code_block(&lang, &code);
    }
    if let Some(table) = r.table.take() {
        r.render_table(table);
    }
    r.flush_inline();
    if r.lines.is_empty() {
        r.lines.push(Line::from(""));
    }
    r.lines
}

#[derive(Default)]
struct TableBuf {
    rows: Vec<Vec<String>>,
    header_rows: usize,
    row: Vec<String>,
    cell: String,
}

struct MdRenderer<'a> {
    width: usize,
    theme: &'a Theme,
    styles: Vec<Style>,
    lines: Vec<Line<'static>>,
    inline: Vec<Span<'static>>,
    lists: Vec<Option<u64>>,
    bq: usize,
    code: Option<(String, String)>,
    link: Option<(String, String)>,
    table: Option<TableBuf>,
    item_first: bool,
}

impl<'a> MdRenderer<'a> {
    fn style(&self) -> Style {
        *self.styles.last().unwrap_or(&self.theme.base())
    }

    fn blank_sep(&mut self) {
        if self.lines.last().is_some_and(|l| l.width() > 0) {
            self.lines.push(Line::from(""));
        }
    }

    fn flush_sep(&mut self) {
        self.flush_inline();
        self.blank_sep();
    }

    fn flush_inline(&mut self) {
        if self.inline.is_empty() {
            return;
        }
        let spans = std::mem::take(&mut self.inline);
        let bq = sp("│ ", self.theme.fg(self.theme.accent));
        let (mut first, mut cont) = (vec![bq.clone(); self.bq], vec![bq; self.bq]);
        let depth = self.lists.len();
        if let Some(last_list) = self.lists.last_mut() {
            if depth > 1 {
                let ind = sp("  ".repeat(depth - 1), self.theme.base());
                first.push(ind.clone());
                cont.push(ind);
            }
            let (m, ms) = if std::mem::take(&mut self.item_first) {
                let marker = last_list.as_mut().map_or_else(
                    || "• ".into(),
                    |n| {
                        let s = format!("{n}. ");
                        *n += 1;
                        s
                    },
                );
                (marker, self.theme.fg(self.theme.accent))
            } else {
                ("  ".into(), self.theme.base())
            };
            cont.push(sp(" ".repeat(m.width()), self.theme.base()));
            first.push(sp(m, ms));
        }
        let pw: usize = first.iter().map(|s| s.content.width()).sum();
        let avail = self.width.saturating_sub(pw).max(1);
        for (i, body) in wrap_spans(spans, avail).into_iter().enumerate() {
            let mut row = [&cont, &first][usize::from(i == 0)].clone();
            row.extend(body);
            self.lines.push(Line::from(row));
        }
    }

    fn push_table(&mut self, event: Event<'_>) {
        let Some(t) = self.table.as_mut() else {
            return;
        };
        match event {
            Event::End(TagEnd::TableHead) => {
                if !t.row.is_empty() {
                    t.rows.push(std::mem::take(&mut t.row));
                }
                t.header_rows = t.rows.len();
            }
            Event::Start(Tag::TableRow) => t.row.clear(),
            Event::End(TagEnd::TableRow) => t.rows.push(std::mem::take(&mut t.row)),
            Event::Start(Tag::TableCell) => t.cell.clear(),
            Event::End(TagEnd::TableCell) => t.row.push(std::mem::take(&mut t.cell)),
            Event::Text(s) | Event::Code(s) => t.cell.push_str(&s),
            Event::SoftBreak | Event::HardBreak => t.cell.push(' '),
            Event::End(TagEnd::Table) => {
                let tbl = self.table.take().unwrap();
                self.render_table(tbl);
            }
            _ => {}
        }
    }

    fn push(&mut self, event: Event<'_>) {
        if self.table.is_some() {
            return self.push_table(event);
        }
        if let Some((_, buf)) = self.code.as_mut() {
            match event {
                Event::Text(t) => buf.push_str(&t),
                Event::End(TagEnd::CodeBlock) => {
                    let (l, c) = self.code.take().unwrap();
                    self.render_code_block(&l, &c);
                }
                _ => {}
            }
            return;
        }
        let t = self.theme;
        match event {
            Event::Start(Tag::Paragraph) if self.lists.is_empty() => self.blank_sep(),
            Event::End(TagEnd::Paragraph | TagEnd::Item) | Event::HardBreak => self.flush_inline(),
            Event::Start(Tag::Heading { level, .. }) => {
                self.blank_sep();
                let c = match level {
                    HeadingLevel::H1 => t.accent,
                    HeadingLevel::H2 => t.cyan,
                    HeadingLevel::H3 => t.fg,
                    _ => t.fg_secondary,
                };
                self.styles.push(t.bold(c));
            }
            Event::End(TagEnd::Heading(lvl)) => {
                self.flush_inline();
                self.styles.pop();
                if lvl == HeadingLevel::H1 {
                    let rule = "─".repeat(self.width.min(40));
                    self.lines.push(Line::from(sp(rule, t.fg(t.border))));
                }
            }
            Event::Start(Tag::BlockQuote(_)) => {
                self.blank_sep();
                self.bq += 1;
                self.styles
                    .push(t.fg(t.fg_secondary).add_modifier(Modifier::ITALIC));
            }
            Event::End(TagEnd::BlockQuote(_)) => {
                self.flush_inline();
                self.bq = self.bq.saturating_sub(1);
                self.styles.pop();
            }
            Event::Start(Tag::CodeBlock(kind)) => {
                self.flush_sep();
                let lang = match kind {
                    CodeBlockKind::Fenced(i) => i.split_whitespace().next().unwrap_or("").into(),
                    CodeBlockKind::Indented => String::new(),
                };
                self.code = Some((lang, String::new()));
            }
            Event::Start(Tag::List(s)) => {
                if self.lists.is_empty() {
                    self.blank_sep();
                }
                self.lists.push(s);
            }
            Event::End(TagEnd::List(_)) => {
                self.flush_inline();
                self.lists.pop();
            }
            Event::Start(Tag::Item) => {
                self.flush_inline();
                self.item_first = true;
            }
            Event::Start(Tag::Table(_)) => {
                self.flush_sep();
                self.table = Some(TableBuf::default());
            }
            Event::Start(tag @ (Tag::Strong | Tag::Emphasis | Tag::Strikethrough)) => {
                let m = match tag {
                    Tag::Strong => Modifier::BOLD,
                    Tag::Emphasis => Modifier::ITALIC,
                    _ => Modifier::CROSSED_OUT,
                };
                self.styles.push(self.style().add_modifier(m));
            }
            Event::End(TagEnd::Strong | TagEnd::Emphasis | TagEnd::Strikethrough) => {
                self.styles.pop();
            }
            Event::Start(Tag::Link { dest_url, .. }) => {
                self.link = Some((dest_url.to_string(), String::new()));
                self.styles
                    .push(self.style().fg(t.cyan).add_modifier(Modifier::UNDERLINED));
            }
            Event::End(TagEnd::Link) => {
                self.styles.pop();
                let url = self
                    .link
                    .take()
                    .filter(|(u, l)| !u.is_empty() && u != l.trim());
                if let Some((u, _)) = url {
                    push_styled(&mut self.inline, &format!(" ({u})"), t.muted_style());
                }
            }
            Event::Text(ref text) | Event::Code(ref text) => {
                if let Some((_, l)) = self.link.as_mut() {
                    l.push_str(text);
                }
                if matches!(event, Event::Code(_)) {
                    push_styled(&mut self.inline, &format!(" {text} "), t.code(t.accent));
                } else {
                    let s = self.style();
                    push_styled(&mut self.inline, text, s);
                }
            }
            Event::TaskListMarker(ok) => {
                let (m, c) = [("[ ] ", t.muted_bright), ("[x] ", t.green)][usize::from(ok)];
                push_styled(&mut self.inline, m, t.fg(c));
            }
            Event::SoftBreak => {
                let s = self.style();
                push_styled(&mut self.inline, " ", s);
            }
            Event::Rule => {
                self.flush_sep();
                self.lines
                    .push(Line::from(sp("─".repeat(self.width), t.fg(t.border))));
            }
            _ => {}
        }
    }

    fn render_code_block(&mut self, lang: &str, code: &str) {
        let t = self.theme;
        let border = t.code(t.border_active);
        let label = [lang, "code"][usize::from(lang.is_empty())];
        let fill = self.width.saturating_sub(format!("┌─ {label} ").width());
        self.lines.push(Line::from(vec![
            sp("┌─ ", border),
            sp(label, t.code(t.accent).add_modifier(Modifier::BOLD)),
            sp(format!(" {}", "─".repeat(fill)), border),
        ]));

        let (ss, ts) = SYNTAX.get_or_init(|| {
            (
                SyntaxSet::load_defaults_newlines(),
                ThemeSet::load_defaults(),
            )
        });
        let syn = (!lang.is_empty())
            .then(|| ss.find_syntax_by_token(lang))
            .flatten()
            .unwrap_or_else(|| ss.find_syntax_plain_text());
        let st = ts
            .themes
            .get("base16-ocean.dark")
            .or_else(|| ts.themes.values().next());
        let mut hl = st.map(|th| HighlightLines::new(syn, th));
        let body_w = self.width.saturating_sub(2).max(1);

        for raw in code.strip_suffix('\n').unwrap_or(code).split('\n') {
            let mut spans = Vec::new();
            let with_nl = format!("{raw}\n");
            if let Some(Ok(regs)) = hl.as_mut().map(|h| h.highlight_line(&with_nl, ss)) {
                for (st, seg) in regs {
                    let text = seg.trim_end_matches('\n');
                    if text.is_empty() {
                        continue;
                    }
                    let mut m = Modifier::empty();
                    if st.font_style.contains(FontStyle::BOLD) {
                        m |= Modifier::BOLD;
                    }
                    if st.font_style.contains(FontStyle::ITALIC) {
                        m |= Modifier::ITALIC;
                    }
                    let fg = t.map_rgb(st.foreground.r, st.foreground.g, st.foreground.b);
                    push_styled(&mut spans, text, t.code(fg).add_modifier(m));
                }
            }
            if spans.is_empty() {
                spans.push(sp(raw, t.code(t.fg)));
            }
            for row in wrap_spans(spans, body_w) {
                let mut line = Line::from(
                    std::iter::once(sp("│ ", border))
                        .chain(row)
                        .collect::<Vec<_>>(),
                );
                let pad = self.width.saturating_sub(line.width());
                if pad > 0 {
                    line.spans
                        .push(sp(" ".repeat(pad), Style::default().bg(t.bg_code)));
                }
                self.lines.push(line);
            }
        }
        let foot = format!("└{}", "─".repeat(self.width.saturating_sub(1)));
        self.lines.push(Line::from(sp(foot, border)));
    }

    fn render_table(&mut self, table: TableBuf) {
        let cols = table.rows.iter().map(Vec::len).max().unwrap_or(0);
        if cols == 0 {
            return;
        }
        let mut widths = vec![1usize; cols];
        for row in &table.rows {
            for (c, cell) in row.iter().enumerate() {
                widths[c] = widths[c].max(cell.trim().width());
            }
        }
        let budget = self.width.saturating_sub(cols * 3 + 1).max(cols);
        while widths.iter().sum::<usize>() > budget {
            let Some((i, &w)) = widths.iter().enumerate().max_by_key(|(_, &w)| w) else {
                break;
            };
            if w <= 1 {
                break;
            }
            widths[i] -= 1;
        }
        let border = self.theme.fg(self.theme.muted_bright);
        let rule = |l: char, m: char, r: char| {
            let segs: Vec<_> = widths.iter().map(|&w| "─".repeat(w + 2)).collect();
            Line::from(sp(format!("{l}{}{r}", segs.join(&m.to_string())), border))
        };
        self.lines.push(rule('┌', '┬', '┐'));
        for (ri, row) in table.rows.iter().enumerate() {
            let head = ri < table.header_rows;
            let st = [self.theme.base(), self.theme.bold(self.theme.fg)][usize::from(head)];
            let cells: Vec<_> = (0..cols)
                .map(|c| wrap_plain(row.get(c).map_or("", |s| s.trim()), widths[c]))
                .collect();
            for li in 0..cells.iter().map(Vec::len).max().unwrap_or(1) {
                let mut spans = vec![sp("│", border)];
                for (c, cl) in cells.iter().enumerate() {
                    let txt = cl.get(li).map_or("", String::as_str);
                    let pad = " ".repeat(widths[c].saturating_sub(txt.width()));
                    spans.push(sp(format!(" {txt}{pad} "), st));
                    spans.push(sp("│", border));
                }
                self.lines.push(Line::from(spans));
            }
            if head && ri + 1 == table.header_rows && table.rows.len() > table.header_rows {
                self.lines.push(rule('├', '┼', '┤'));
            }
        }
        self.lines.push(rule('└', '┴', '┘'));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_words_and_cjk_and_renders_gfm_and_diffs() {
        assert_eq!(wrap_plain("one two three", 7), vec!["one two", "three"]);
        assert_eq!(wrap_plain("你好世界", 4), vec!["你好", "世界"]);
        let theme = Theme::truecolor();
        let md = "# Title\n\nSee [Atlas](https://example.com) & 2 * 3 * 4.\n\n| A | B |\n|---|---|\n| x | 1 |\n\n```rust\nfn main() {}\n```";
        let text: String = render_markdown(md, 60, &theme)
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            text.contains("Title")
                && text.contains("Atlas (https://example.com)")
                && text.contains("2 * 3 * 4")
                && text.contains("┌")
                && text.contains("fn main() {}")
        );
        let d = style_diff_lines("--- a\n+++ a\n@@ -1 +1 @@\n-old\n+new", 60, &theme);
        assert_eq!(
            (
                d[2].spans[0].style.fg,
                d[3].spans[0].style.fg,
                d[4].spans[0].style.fg
            ),
            (Some(theme.cyan), Some(theme.red), Some(theme.green))
        );
    }
}
