use std::collections::VecDeque;

const MAX_PROMPTS: usize = 100;

#[derive(Default)]
pub(super) struct PromptHistory {
    entries: VecDeque<String>,
    browsing: Option<BrowseState>,
}

struct BrowseState {
    index: usize,
    draft: String,
}

impl PromptHistory {
    pub(super) fn record(&mut self, text: &str) {
        if text.trim().is_empty() {
            return;
        }
        self.browsing = None;
        if self.entries.back().is_some_and(|previous| previous == text) {
            return;
        }

        self.entries.push_back(text.to_string());
        if self.entries.len() > MAX_PROMPTS {
            self.entries.pop_front();
        }
    }

    pub(super) fn older(&mut self, current: &str) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }

        let index = match &mut self.browsing {
            Some(state) => {
                state.index = state.index.saturating_sub(1);
                state.index
            }
            None => {
                let index = self.entries.len() - 1;
                self.browsing = Some(BrowseState {
                    index,
                    draft: current.to_string(),
                });
                index
            }
        };
        self.entries.get(index).cloned()
    }

    pub(super) fn newer(&mut self) -> Option<String> {
        let state = self.browsing.as_mut()?;
        if state.index + 1 < self.entries.len() {
            state.index += 1;
            return self.entries.get(state.index).cloned();
        }

        self.browsing.take().map(|state| state.draft)
    }

    pub(super) fn is_browsing(&self) -> bool {
        self.browsing.is_some()
    }

    pub(super) fn edited(&mut self) {
        self.browsing = None;
    }
}

#[cfg(test)]
mod tests {
    use super::{PromptHistory, MAX_PROMPTS};

    #[test]
    fn navigation_orders_prompts_and_restores_the_original_draft() {
        let mut history = PromptHistory::default();
        history.record("first prompt");
        history.record("second prompt");

        assert_eq!(history.older("unsent draft"), Some("second prompt".into()));
        assert!(history.is_browsing());
        assert_eq!(
            history.older("ignored while browsing"),
            Some("first prompt".into())
        );
        assert_eq!(
            history.older("ignored while browsing"),
            Some("first prompt".into())
        );
        assert_eq!(history.newer(), Some("second prompt".into()));
        assert_eq!(history.newer(), Some("unsent draft".into()));
        assert!(!history.is_browsing());
        assert_eq!(history.newer(), None);
    }

    #[test]
    fn blank_and_consecutive_duplicate_prompts_are_skipped() {
        let mut history = PromptHistory::default();
        history.record(" \n\t ");
        assert_eq!(history.older("draft"), None);

        history.record("repeat");
        history.record("repeat");
        history.record("different");
        history.record("repeat");

        assert_eq!(history.older("draft"), Some("repeat".into()));
        assert_eq!(history.older("draft"), Some("different".into()));
        assert_eq!(history.older("draft"), Some("repeat".into()));
    }

    #[test]
    fn edited_prompt_starts_a_fresh_browsing_draft() {
        let mut history = PromptHistory::default();
        history.record("sent prompt");
        assert_eq!(history.older("original draft"), Some("sent prompt".into()));

        history.edited();
        assert!(!history.is_browsing());
        assert_eq!(history.older("edited prompt"), Some("sent prompt".into()));
        assert_eq!(history.newer(), Some("edited prompt".into()));
    }

    #[test]
    fn capacity_retains_only_the_latest_prompts() {
        let mut history = PromptHistory::default();
        for number in 0..=MAX_PROMPTS {
            history.record(&format!("prompt {number}"));
        }

        assert_eq!(
            history.older("draft"),
            Some(format!("prompt {MAX_PROMPTS}"))
        );
        for number in (1..MAX_PROMPTS).rev() {
            assert_eq!(history.older("draft"), Some(format!("prompt {number}")));
        }
        assert_eq!(history.older("draft"), Some("prompt 1".into()));
    }
}
