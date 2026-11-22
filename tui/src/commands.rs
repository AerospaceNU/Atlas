//! Single source of truth for slash commands, fuzzy matching (`nucleo-matcher`),
//! and tab completion (`/cmd`, `/key`, `/skill`, `/resume`, `@file`).

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use nucleo_matcher::pattern::{CaseMatching, Normalization, Pattern};
use nucleo_matcher::{Config, Matcher, Utf32Str};

use crate::protocol::ClientMessage as M;

/// `(token, args_hint, description, standalone)`
pub const COMMANDS: &[(&str, &str, &str, bool)] = &[
    ("/help", "", "Show commands, keys, and loaded tools", true),
    ("/new", "", "Start a fresh conversation", true),
    (
        "/resume",
        "[id]",
        "Load a saved session (default: newest)",
        false,
    ),
    ("/undo", "", "Restore the newest file checkpoint", true),
    ("/compact", "", "Compact conversation context", true),
    ("/model", "[filter]", "Choose a model", true),
    (
        "/goal",
        "[text|clear]",
        "Show, set, or clear session goal",
        true,
    ),
    ("/yolo", "", "Toggle automatic tool approval", true),
    ("/copy", "", "Copy the last assistant reply", true),
    (
        "/key",
        "[prov] scope",
        "Configure or delete API keys",
        false,
    ),
    ("/skill", "[name]", "List or activate a skill", true),
    ("/sessions", "", "List saved sessions", true),
    ("/artifacts", "", "List workspace artifacts", true),
    (
        "/save-skill",
        "[name]",
        "Save workflow as a skill draft",
        true,
    ),
    (
        "/enable-skill",
        "<name>",
        "Preview or enable a skill draft",
        false,
    ),
    ("/quit", "", "Quit Atlas", true),
    ("/exit", "", "Quit Atlas", true),
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionItem {
    pub label: String,
    pub insert: String,
    pub description: String,
    pub standalone: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionList {
    pub replace_start: usize,
    pub replace_end: usize,
    pub items: Vec<CompletionItem>,
}

pub fn fuzzy_filter<'a, T>(items: &'a [T], query: &str, key: impl Fn(&T) -> &str) -> Vec<&'a T> {
    let query = query.trim();
    if query.is_empty() {
        return items.iter().collect();
    }
    let mut matcher = Matcher::new(Config::DEFAULT);
    let pat = Pattern::parse(query, CaseMatching::Ignore, Normalization::Smart);
    let mut buf = Vec::new();
    let mut scored: Vec<_> = items
        .iter()
        .enumerate()
        .filter_map(|(i, it)| {
            let s = pat.score(Utf32Str::new(key(it), &mut buf), &mut matcher)?;
            Some((s, i, it))
        })
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    scored.into_iter().map(|(_, _, it)| it).collect()
}

fn finish_list(s: usize, e: usize, items: Vec<CompletionItem>) -> Option<CompletionList> {
    (!items.is_empty()).then_some(CompletionList {
        replace_start: s,
        replace_end: e,
        items,
    })
}

fn item(label: String, insert: String, desc: &str, standalone: bool) -> CompletionItem {
    CompletionItem {
        label,
        insert,
        description: desc.into(),
        standalone,
    }
}

pub fn completions_for(
    input: &str,
    cursor_byte: usize,
    workspace: &Path,
    known_skills: &[String],
    known_sessions: &[String],
) -> Option<CompletionList> {
    let cursor = cursor_byte.min(input.len());
    let before = &input[..cursor];
    if let Some(at) = before.rfind('@').filter(|&p| {
        (p == 0 || before[..p].ends_with(char::is_whitespace))
            && !before[p + 1..].contains(char::is_whitespace)
    }) {
        let files = collect_workspace_files(workspace, 400);
        let items = fuzzy_filter(&files, &before[at + 1..], String::as_str)
            .into_iter()
            .take(12)
            .map(|p| item(format!("@{p}"), format!("@{p}"), "file", false))
            .collect();
        return finish_list(at, cursor, items);
    }
    if !before.starts_with('/') || before.contains('\n') {
        return None;
    }
    if !before.contains(char::is_whitespace) {
        let q = before.trim_start_matches('/');
        let items = fuzzy_filter(COMMANDS, q, |(t, _, _, _)| t.trim_start_matches('/'))
            .into_iter()
            .map(|(t, _, d, st)| {
                let ins = [*t, &format!("{t} ")][usize::from(!*st)].into();
                item((*t).into(), ins, d, *st)
            })
            .collect();
        return finish_list(0, cursor, items);
    }
    let (cmd, rest_raw) = before.split_once(char::is_whitespace)?;
    let rest = rest_raw.trim_start();
    let arg_start = before.len() - rest.len();
    match cmd {
        "/key" => complete_key_args(before, rest),
        "/skill" | "/resume" if !rest.contains(char::is_whitespace) => {
            let is_skill = cmd == "/skill";
            let mut dirs =
                vec![workspace.join([".atlas/sessions", ".atlas/skills"][usize::from(is_skill)])];
            if is_skill {
                if let Some(h) = std::env::var_os("HOME") {
                    dirs.push(Path::new(&h).join(".atlas/skills"));
                }
            }
            let (known, desc) = if is_skill {
                (known_skills, "skill")
            } else {
                (known_sessions, "session")
            };
            let cands = collect_dir_names(&dirs, known, is_skill, !is_skill);
            let items = fuzzy_filter(&cands, rest, String::as_str)
                .into_iter()
                .take(12)
                .map(|s| item(s.clone(), s.clone(), desc, true))
                .collect();
            finish_list(arg_start, cursor, items)
        }
        _ => None,
    }
}

fn complete_key_args(before: &str, rest: &str) -> Option<CompletionList> {
    const FIRST: &[(&str, &str, bool)] = &[
        ("session", "OpenRouter key for session", true),
        ("user", "OpenRouter key for user", true),
        ("delete", "Delete OpenRouter key", true),
        ("earthdata", "NASA Earthdata token", false),
        ("firms", "NASA FIRMS map key", false),
    ];
    const SCOPES: &[(&str, &str, bool)] = &[
        ("session", "Save for this session", true),
        ("user", "Save in user config", true),
        ("delete", "Delete saved key", true),
    ];
    let words: Vec<&str> = rest.split_whitespace().collect();
    let ws = before.ends_with(char::is_whitespace);
    let (pool, q, start) = match (words.as_slice(), ws) {
        ([], _) => (FIRST, "", before.len()),
        ([w], false) => (FIRST, *w, before.len() - w.len()),
        ([p], true) if matches!(*p, "earthdata" | "firms") => (SCOPES, "", before.len()),
        ([p, q], false) if matches!(*p, "earthdata" | "firms") => {
            (SCOPES, *q, before.len() - q.len())
        }
        _ => return None,
    };
    let items = fuzzy_filter(pool, q, |(t, _, _)| t)
        .into_iter()
        .map(|(t, d, st)| {
            let ins = [*t, &format!("{t} ")][usize::from(!*st)].into();
            item((*t).into(), ins, d, *st)
        })
        .collect();
    finish_list(start, before.len(), items)
}

fn collect_dir_names(dirs: &[PathBuf], known: &[String], strip_md: bool, rev: bool) -> Vec<String> {
    let mut set: BTreeSet<String> = known.iter().cloned().collect();
    for dir in dirs {
        if let Ok(entries) = fs::read_dir(dir) {
            for name in entries
                .flatten()
                .filter_map(|e| e.file_name().into_string().ok())
            {
                if !name.starts_with('.') {
                    let s = if strip_md {
                        name.trim_end_matches(".md")
                    } else {
                        &name
                    };
                    set.insert(s.into());
                }
            }
        }
    }
    let mut out: Vec<String> = set.into_iter().collect();
    if rev {
        out.reverse();
    }
    out
}

fn collect_workspace_files(root: &Path, limit: usize) -> Vec<String> {
    const SKIP: &[&str] = &[
        ".git",
        ".venv",
        ".atlas",
        "node_modules",
        "target",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
    ];
    let (mut out, mut stack) = (Vec::new(), vec![root.to_path_buf()]);
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        let mut children: Vec<_> = entries.flatten().collect();
        children.sort_by_key(|e| e.file_name());
        for entry in children {
            let (Ok(name), Ok(ft)) = (entry.file_name().into_string(), entry.file_type()) else {
                continue;
            };
            if ft.is_dir() && !SKIP.contains(&name.as_str()) {
                stack.push(entry.path());
            } else if ft.is_file() {
                if let Ok(rel) = entry.path().strip_prefix(root) {
                    out.push(rel.to_string_lossy().replace('\\', "/"));
                    if out.len() >= limit {
                        return out;
                    }
                }
            }
        }
    }
    out
}

#[derive(Debug, PartialEq, Eq)]
pub enum CommandOutcome {
    Send(M),
    OpenHelp,
    OpenModelPicker { query: String },
    PromptKey { provider: String, scope: String },
    CopyLastReply,
    NewSession,
    LoadSession(String),
    Compact,
    Quit,
    Notice(String),
}

pub fn parse_slash_command(text: &str, preview_sha: Option<&(String, String)>) -> CommandOutcome {
    use CommandOutcome as O;
    let trimmed = text.trim();
    if trimmed == "/" {
        return O::Notice("Type a command name (e.g. /help) or Tab to complete".into());
    }
    let mut parts = trimmed.split_whitespace();
    let cmd = parts.next().unwrap_or("");
    let no_args = parts.clone().next().is_none();
    let rest = trimmed.strip_prefix(cmd).unwrap_or("").trim();
    match cmd {
        "/quit" | "/exit" => O::Quit,
        "/copy" => O::CopyLastReply,
        "/help" if no_args => O::OpenHelp,
        "/new" if no_args => O::NewSession,
        "/undo" if no_args => O::Send(M::Undo),
        "/compact" if no_args => O::Compact,
        "/yolo" if no_args => O::Send(M::Yolo { enabled: None }),
        "/help" | "/new" | "/undo" | "/compact" | "/yolo" => O::Notice(format!("usage: {cmd}")),
        "/model" => O::OpenModelPicker { query: rest.into() },
        "/goal" => match rest {
            "" => O::Send(M::Goal { text: None }),
            "clear" => O::Send(M::Goal {
                text: Some(String::new()),
            }),
            r if r.split_whitespace().next() == Some("clear") => {
                O::Notice("usage: /goal [text|clear]".into())
            }
            r => O::Send(M::Goal {
                text: Some(r.into()),
            }),
        },
        "/resume" => match (parts.next(), parts.next()) {
            (None, _) => O::LoadSession("last".into()),
            (Some(id), None) => O::LoadSession(id.into()),
            _ => O::Notice("usage: /resume [ID]".into()),
        },
        "/sessions" => O::Send(M::List {
            kind: "sessions".into(),
        }),
        "/artifacts" => O::Send(M::List {
            kind: "artifacts".into(),
        }),
        "/skill" if rest.is_empty() => O::Send(M::List {
            kind: "skills".into(),
        }),
        "/skill" => O::Send(M::Skill { name: rest.into() }),
        "/save-skill" => match (parts.next(), parts.next()) {
            (n, None) => O::Send(M::SaveSkill {
                name: n.map(str::to_string),
            }),
            _ => O::Notice("usage: /save-skill [name]".into()),
        },
        "/enable-skill" => match (parts.next(), parts.next(), parts.next(), parts.next()) {
            (Some(n), None, _, _) => O::Send(M::EnableSkill {
                name: n.into(),
                sha256: None,
            }),
            (Some(n), Some("confirm"), Some(sha), None) | (Some(n), Some(sha), None, None)
                if sha.len() >= 8 && sha.bytes().all(|b| b.is_ascii_hexdigit()) =>
            {
                let full = preview_sha
                    .filter(|(pn, f)| pn == n && f.starts_with(sha))
                    .map_or_else(|| sha.into(), |(_, f)| f.clone());
                O::Send(M::EnableSkill {
                    name: n.into(),
                    sha256: Some(full),
                })
            }
            _ => O::Notice("usage: /enable-skill <name> [confirm <sha>]".into()),
        },
        "/key" => match (parts.next().unwrap_or(""), parts.next(), parts.next()) {
            (s @ ("session" | "user"), None, None) => O::PromptKey {
                provider: "openrouter".into(),
                scope: s.into(),
            },
            ("delete", None, None) => O::Send(M::DeleteKey {
                provider: Some("openrouter".into()),
                scope: None,
            }),
            (p @ ("earthdata" | "firms"), Some(s @ ("session" | "user")), None) => O::PromptKey {
                provider: p.into(),
                scope: s.into(),
            },
            (p @ ("earthdata" | "firms"), Some("delete"), None) => O::Send(M::DeleteKey {
                provider: Some(p.into()),
                scope: None,
            }),
            _ => O::Notice("usage: /key [earthdata|firms] session|user|delete".into()),
        },
        _ => O::Notice(format!("unknown command: {cmd}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_commands_fuzzy_filters_and_completes_paths_skills_sessions() {
        assert!(matches!(
            parse_slash_command("/", None),
            CommandOutcome::Notice(_)
        ));
        assert_eq!(parse_slash_command("/help", None), CommandOutcome::OpenHelp);
        assert_eq!(
            parse_slash_command("/undo", None),
            CommandOutcome::Send(M::Undo)
        );
        assert_eq!(
            parse_slash_command("/compact", None),
            CommandOutcome::Compact
        );
        assert_eq!(
            parse_slash_command("/resume", None),
            CommandOutcome::LoadSession("last".into())
        );
        assert_eq!(
            parse_slash_command("/resume s-42", None),
            CommandOutcome::LoadSession("s-42".into())
        );
        assert!(matches!(
            parse_slash_command("/resume a b", None),
            CommandOutcome::Notice(_)
        ));
        assert_eq!(
            fuzzy_filter(COMMANDS, "cmp", |(t, _, _, _)| t)
                .first()
                .map(|c| c.0),
            Some("/compact")
        );

        let dir = std::env::temp_dir().join(format!("atlas-comp-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(dir.join("src/runtime.py"), "").unwrap();
        let top = |s, pos, sk: &[String], ss: &[String]| {
            completions_for(s, pos, &dir, sk, ss).unwrap().items[0]
                .insert
                .clone()
        };
        assert_eq!(top("check @run", 10, &[], &[]), "@src/runtime.py");
        assert_eq!(top("/skill fl", 9, &["flood-map".into()], &[]), "flood-map");
        assert_eq!(top("/resume s2", 10, &[], &["s2026".into()]), "s2026");
        let _ = fs::remove_dir_all(&dir);
    }
}
