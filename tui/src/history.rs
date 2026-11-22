//! Prompt history with JSON-lines persistence across TUI sessions.

use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};

const MAX_PROMPTS: usize = 200;

#[derive(Default)]
pub struct PromptHistory {
    entries: VecDeque<String>,
    browsing: Option<(usize, String)>,
    path: Option<PathBuf>,
}

impl PromptHistory {
    pub fn load_default() -> Self {
        std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map(|h| Self::load_from(PathBuf::from(h).join(".atlas/tui_history")))
            .unwrap_or_default()
    }

    pub fn load_from(path: PathBuf) -> Self {
        let mut entries: VecDeque<String> = fs::read_to_string(&path)
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str::<String>(l).ok())
            .filter(|s| !s.trim().is_empty())
            .collect();
        while entries.len() > MAX_PROMPTS {
            entries.pop_front();
        }
        Self {
            entries,
            browsing: None,
            path: Some(path),
        }
    }

    pub fn record(&mut self, text: &str) {
        if text.trim().is_empty() {
            return;
        }
        self.browsing = None;
        if self.entries.back().is_some_and(|p| p == text) {
            return;
        }
        self.entries.push_back(text.to_string());
        if self.entries.len() > MAX_PROMPTS {
            self.entries.pop_front();
        }
        if let Some(path) = &self.path {
            let _ = save(path, &self.entries);
        }
    }

    pub fn older(&mut self, current: &str) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }
        let idx = match &mut self.browsing {
            Some((i, _)) => {
                *i = i.saturating_sub(1);
                *i
            }
            None => {
                let i = self.entries.len() - 1;
                self.browsing = Some((i, current.to_string()));
                i
            }
        };
        self.entries.get(idx).cloned()
    }

    pub fn newer(&mut self) -> Option<String> {
        let (idx, _) = self.browsing.as_mut()?;
        if *idx + 1 < self.entries.len() {
            *idx += 1;
            return self.entries.get(*idx).cloned();
        }
        self.browsing.take().map(|(_, draft)| draft)
    }

    pub fn is_browsing(&self) -> bool {
        self.browsing.is_some()
    }
    pub fn edited(&mut self) {
        self.browsing = None;
    }
}

fn save(path: &Path, entries: &VecDeque<String>) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let out: String = entries
        .iter()
        .filter_map(|e| serde_json::to_string(e).ok())
        .map(|l| format!("{l}\n"))
        .collect();
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, out)?;
    fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn navigates_dedups_and_persists_multiline_history() {
        let dir = std::env::temp_dir().join(format!("atlas-hist-{}", std::process::id()));
        let path = dir.join("tui_history");
        let _ = fs::remove_dir_all(&dir);

        let mut h = PromptHistory::load_from(path.clone());
        h.record("  ");
        h.record("line1\nline2");
        h.record("line1\nline2");
        h.record("second");
        assert_eq!(h.older("draft"), Some("second".into()));
        assert_eq!(h.older("ignored"), Some("line1\nline2".into()));
        assert_eq!(h.newer(), Some("second".into()));
        assert_eq!(h.newer(), Some("draft".into()));

        let mut reloaded = PromptHistory::load_from(path);
        assert_eq!(reloaded.older(""), Some("second".into()));
        assert_eq!(reloaded.older(""), Some("line1\nline2".into()));
        let _ = fs::remove_dir_all(&dir);
    }
}
