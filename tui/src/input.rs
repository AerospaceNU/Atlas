//! Keyboard, mouse, and paste input dispatch.

use anyhow::Result;
use crossterm::event::{
    KeyCode as K, KeyEvent, KeyEventKind, KeyModifiers as Mod, MouseButton, MouseEvent,
    MouseEventKind,
};
use ratatui::layout::Position;
use tui_textarea::{CursorMove, TextArea};

use crate::app::{
    cancel_message, App, KeyInputState, Mode, ModelPickerState, Status, ToolEntry, TranscriptEntry,
    STILL_STOPPING,
};
use crate::approval::ApprovalOutcome;
use crate::clipboard::{copy_to_clipboard, read_from_clipboard};
use crate::commands::{parse_slash_command, CommandOutcome, CompletionList};
use crate::process::ChildProcess;
use crate::protocol::{ClientMessage, PermissionDecision, StepPayload};
use crate::selection::TextSelection;

pub fn handle_key(app: &mut App, child: &mut ChildProcess, key: KeyEvent) -> Result<bool> {
    if key.kind != KeyEventKind::Press {
        return Ok(false);
    }
    app.dirty = true;
    let cmd = key
        .modifiers
        .intersects(Mod::CONTROL | Mod::SUPER | Mod::META);
    let copy_key = (matches!(key.code, K::Char('c' | 'C')) && cmd)
        || (key.code == K::Insert && key.modifiers.contains(Mod::CONTROL));
    if copy_key {
        if app
            .text_selection
            .as_ref()
            .is_some_and(TextSelection::moved)
        {
            app.copy_selection = true;
            return Ok(false);
        }
        if key.modifiers.contains(Mod::SHIFT)
            || !key.modifiers.contains(Mod::CONTROL)
            || key.code == K::Insert
        {
            return Ok(false);
        }
    }
    if (matches!(key.code, K::Char('v' | 'V')) && cmd)
        || (key.code == K::Insert && key.modifiers.contains(Mod::SHIFT))
    {
        match read_from_clipboard() {
            Some(text) => apply_paste(app, &text),
            None => app.status_notice = Some("Clipboard unavailable; use terminal paste".into()),
        }
        return Ok(false);
    }
    if app.text_selection.take().is_some() && key.code == K::Esc {
        app.status_notice = None;
        return Ok(false);
    }
    app.status_notice = None;
    if key.code == K::Char('c') && key.modifiers.contains(Mod::CONTROL) {
        child.shutdown();
        return Ok(true);
    }
    if let Some(prompt) = app.pending_permission.as_mut() {
        if let Some(outcome) = prompt.key(key) {
            handle_approval(app, child, outcome)?;
        }
        return Ok(false);
    }

    let page = app.viewport_height.saturating_sub(2).max(1);
    match &mut app.mode {
        Mode::Help(scroll) => match key.code {
            K::Esc | K::Char('q') => app.mode = Mode::Compose,
            K::Up | K::Char('k') => *scroll = scroll.saturating_sub(1),
            K::Down | K::Char('j') => *scroll = scroll.saturating_add(1),
            K::PageUp => *scroll = scroll.saturating_sub(page),
            K::PageDown => *scroll = scroll.saturating_add(page),
            K::Home => *scroll = 0,
            K::End => *scroll = usize::MAX / 2,
            _ => {}
        },
        Mode::KeyInput(s) => match key.code {
            K::Esc => app.mode = Mode::Compose,
            K::Enter => {
                let secret = s.editor.lines().join("");
                if !secret.trim().is_empty() {
                    let msg = ClientMessage::SetKey {
                        key: secret,
                        provider: Some(s.provider.clone()),
                        scope: Some(s.scope.clone()),
                    };
                    child.send(&msg)?;
                    app.mode = Mode::Compose;
                }
            }
            _ => {
                s.editor.input(key);
            }
        },
        Mode::ModelPicker(_) => handle_picker_key(app, child, key)?,
        Mode::Compose => return handle_compose_key(app, child, key),
    }
    Ok(false)
}

fn handle_approval(app: &mut App, child: &mut ChildProcess, out: ApprovalOutcome) -> Result<()> {
    match out {
        ApprovalOutcome::ToggleYolo => child.send(&ClientMessage::Yolo { enabled: None })?,
        ApprovalOutcome::Decide { decision, reason } => {
            let Some(p) = app.pending_permission.take() else {
                return Ok(());
            };
            let msg = ClientMessage::Permission {
                id: p.payload.id.clone(),
                decision,
                reason: reason.clone(),
            };
            child.send(&msg)?;
            let base = match decision {
                PermissionDecision::Allow => "allowed once",
                PermissionDecision::AllowSession => "allowed for session",
                PermissionDecision::Deny => "denied",
            };
            let summary = reason.map_or_else(|| base.into(), |r| format!("{base}: {r}"));
            let data = StepPayload {
                name: p.payload.tool.clone(),
                summary,
                ok: decision != PermissionDecision::Deny,
                text: p.body_plain(),
                diff: (p.payload.preview_kind == "diff").then_some(p.payload.preview.clone()),
                ..Default::default()
            };
            app.transcript
                .push(TranscriptEntry::Tool(Box::new(ToolEntry {
                    data,
                    ..Default::default()
                })));
            (app.follow, app.unseen_output) = (true, false);
        }
    }
    Ok(())
}

fn handle_picker_key(app: &mut App, child: &mut ChildProcess, key: KeyEvent) -> Result<()> {
    let Mode::ModelPicker(ref s) = app.mode else {
        return Ok(());
    };
    let (query, cur_idx) = (s.filter.lines().join(" "), s.index);
    let count = app.filtered_models(&query).len();
    match key.code {
        K::Esc => app.mode = Mode::Compose,
        K::Enter => {
            if let Some(model) = app
                .filtered_models(&query)
                .get(cur_idx)
                .map(|m| m.id.clone())
            {
                child.send(&ClientMessage::SelectModel { model })?;
                app.mode = Mode::Compose;
            }
        }
        code => {
            if let Mode::ModelPicker(s) = &mut app.mode {
                match code {
                    K::Up => s.index = s.index.saturating_sub(1),
                    K::Down if s.index + 1 < count => s.index += 1,
                    K::PageUp => s.index = s.index.saturating_sub(8),
                    K::PageDown if count > 0 => s.index = (s.index + 8).min(count - 1),
                    _ if s.filter.input(key) => s.index = 0,
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

fn handle_compose_key(app: &mut App, child: &mut ChildProcess, key: KeyEvent) -> Result<bool> {
    let completions = app.active_completions();
    let ctrl = key.modifiers.contains(Mod::CONTROL);
    let nav = key.modifiers.intersects(Mod::SHIFT | Mod::ALT | Mod::META);
    let empty = app.composer.is_empty();
    let page = app.viewport_height.saturating_sub(2).max(1) as isize;

    if key.code == K::Esc {
        if completions.is_some() {
            app.menu_dismissed = true;
        } else if app.in_flight() {
            match app.cancel_stage {
                0 => {
                    child.send(&ClientMessage::Cancel)?;
                    app.cancel_stage = 1;
                    app.push_notice(cancel_message(&app.activity), false);
                }
                1 => {
                    app.cancel_stage = 2;
                    app.status_notice = Some(STILL_STOPPING.into());
                    app.push_notice(STILL_STOPPING, false);
                }
                _ => {}
            }
        } else {
            let msg = [
                "Draft kept · Ctrl+C or /exit to quit",
                "Ctrl+C or /exit to quit",
            ][usize::from(empty)];
            app.status_notice = Some(msg.into());
        }
        return Ok(false);
    }

    if let Some(list) = completions {
        match key.code {
            K::Up | K::Down if key.modifiers.is_empty() => {
                let last = list.items.len().saturating_sub(1);
                app.menu_index = [
                    app.menu_index.saturating_add(1).min(last),
                    app.menu_index.saturating_sub(1),
                ][usize::from(key.code == K::Up)];
                app.menu_interacted = true;
                return Ok(false);
            }
            K::Tab => {
                apply_completion(app, &list);
                return Ok(false);
            }
            K::Enter => {
                let text = app.composer_text();
                let trimmed = text.trim();
                if trimmed == "/" && !app.menu_interacted {
                    app.status_notice =
                        Some("Use ↑↓ + Enter or type a command (e.g. /help)".into());
                    return Ok(false);
                }
                if app.menu_interacted {
                    let standalone =
                        list.items[app.menu_index.min(list.items.len() - 1)].standalone;
                    apply_completion(app, &list);
                    if !trimmed.starts_with('/') || !standalone {
                        return Ok(false);
                    }
                }
            }
            _ => {}
        }
    }

    let is_nl = (key.code == K::Enter && key.modifiers.intersects(Mod::SHIFT | Mod::ALT))
        || (matches!(key.code, K::Char('j' | 'J')) && ctrl);
    if is_nl {
        app.composer.insert_newline();
        draft_edited(app);
        return Ok(false);
    }

    match key.code {
        K::Char('r')
            if ctrl && app.session_open && app.status == Status::Stopped && !app.in_flight() =>
        {
            child.send(&ClientMessage::Resume)?;
            (app.waiting_from_stopped, app.cancel_stage, app.status) = (true, 0, Status::Waiting);
        }
        K::Char('o') if ctrl => app.toggle_selected(None),
        K::Up if nav => app.select_expandable(-1),
        K::Down if nav => app.select_expandable(1),
        K::Left if nav && !empty => app.composer.move_cursor(CursorMove::WordBack),
        K::Right if nav && !empty => app.composer.move_cursor(CursorMove::WordForward),
        K::Left if empty && (nav || key.modifiers.is_empty()) => app.toggle_selected(Some(false)),
        K::Right if empty && (nav || key.modifiers.is_empty()) => app.toggle_selected(Some(true)),
        K::PageUp => app.scroll_by(-page),
        K::PageDown => app.scroll_by(page),
        K::Home if ctrl || empty => app.jump_scroll(false),
        K::End if ctrl || empty => app.jump_scroll(true),
        K::Enter if empty && key.modifiers.is_empty() => app.toggle_selected(None),
        K::Enter => return submit_composer(app, child),
        K::Up | K::Down if key.modifiers.is_empty() && try_history(app, key.code) => {}
        _ if app.composer.input(key) => draft_edited(app),
        _ => {}
    }
    Ok(false)
}

fn try_history(app: &mut App, code: K) -> bool {
    let (row, last) = (
        app.composer.cursor().0,
        app.composer.lines().len().saturating_sub(1),
    );
    let at_edge = (code == K::Up && row == 0) || (code == K::Down && row == last);
    if !app.prompt_history.is_browsing() && !at_edge {
        return false;
    }
    let recalled = if code == K::Up {
        app.prompt_history.older(&app.composer_text())
    } else {
        app.prompt_history.newer()
    };
    recalled.map(|t| app.set_composer_text(&t)).is_some()
}

fn submit_composer(app: &mut App, child: &mut ChildProcess) -> Result<bool> {
    let text = app.composer_text();
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(false);
    }
    let local_ok = matches!(trimmed, "/help" | "/yolo" | "/quit" | "/exit" | "/copy");
    if app.in_flight() && !local_ok {
        app.status_notice = Some("Still working · draft kept · Esc cancels turn".into());
        return Ok(false);
    }
    if !app.session_open && !matches!(trimmed, "/help" | "/quit" | "/exit" | "/copy") {
        return Ok(false);
    }
    app.clear_composer();
    if trimmed.starts_with('/') {
        app.prompt_history.edited();
        return execute_slash(app, child, trimmed);
    }
    app.prompt_history.record(&text);
    app.transcript.push(TranscriptEntry::User(text.clone()));
    child.send(&ClientMessage::User { text })?;
    (
        app.waiting_from_stopped,
        app.cancel_stage,
        app.status,
        app.follow,
        app.unseen_output,
    ) = (false, 0, Status::Waiting, true, false);
    Ok(false)
}

fn execute_slash(app: &mut App, child: &mut ChildProcess, text: &str) -> Result<bool> {
    use CommandOutcome as O;
    match parse_slash_command(text, app.preview_sha.as_ref()) {
        O::Quit => {
            child.shutdown();
            return Ok(true);
        }
        O::OpenHelp => app.mode = Mode::Help(0),
        O::OpenModelPicker { query } => {
            let mut filter = TextArea::default();
            if !query.is_empty() {
                filter.insert_str(&query);
            }
            app.mode = Mode::ModelPicker(ModelPickerState {
                loading: app.cached_models.is_empty(),
                filter,
                index: 0,
            });
            child.send(&ClientMessage::Models {
                refresh: Some(false),
            })?;
        }
        O::PromptKey { provider, scope } => {
            app.push_notice(
                format!("enter {provider} key for {scope} (input stays hidden)"),
                false,
            );
            let mut editor = TextArea::default();
            editor.set_mask_char('•');
            app.mode = Mode::KeyInput(KeyInputState {
                provider,
                scope,
                editor,
            });
        }
        O::CopyLastReply => match app.last_reply().map(str::to_string) {
            Some(reply) => {
                let ok = copy_to_clipboard(&reply);
                app.push_notice(["copy failed", "copied last reply"][usize::from(ok)], !ok);
            }
            None => app.push_notice("nothing to copy", false),
        },
        O::NewSession => {
            app.reset_conversation();
            child.send(&ClientMessage::New)?;
        }
        O::LoadSession(id) => {
            app.waiting_from_stopped = app.status == Status::Stopped;
            app.status = Status::Waiting;
            child.send(&ClientMessage::LoadSession { id })?;
        }
        O::Compact => {
            app.compacting = true;
            app.push_notice("compacting the conversation context", false);
            child.send(&ClientMessage::Compact)?;
        }
        O::Send(msg) => child.send(&msg)?,
        O::Notice(msg) => app.push_notice(msg, false),
    }
    Ok(false)
}

fn apply_completion(app: &mut App, list: &CompletionList) {
    let Some(item) = list.items.get(app.menu_index.min(list.items.len() - 1)) else {
        return;
    };
    let text = app.composer_text();
    let (s, e) = (
        list.replace_start.min(text.len()),
        list.replace_end.min(text.len()),
    );
    app.set_composer_text(&format!("{}{}{}", &text[..s], item.insert, &text[e..]));
    app.set_composer_cursor_byte(s + item.insert.len());
    app.prompt_history.edited();
}

pub fn apply_paste(app: &mut App, raw: &str) {
    (app.text_selection, app.status_notice, app.dirty) = (None, None, true);
    let cleaned: String = raw
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .chars()
        .filter(|ch| !ch.is_control() || matches!(ch, '\n' | '\t'))
        .collect();
    if let Some(p) = app.pending_permission.as_mut() {
        p.paste(&cleaned);
        return;
    }
    match &mut app.mode {
        Mode::Compose => {
            app.composer.insert_str(&cleaned);
            draft_edited(app);
        }
        Mode::KeyInput(s) => {
            s.editor.insert_str(cleaned.trim_end_matches('\n'));
        }
        Mode::ModelPicker(s) => {
            s.filter.insert_str(cleaned.trim());
            s.index = 0;
        }
        Mode::Help(_) => {}
    }
}

pub fn handle_mouse(app: &mut App, child: &mut ChildProcess, mouse: MouseEvent) -> Result<()> {
    use MouseEventKind as M;
    app.dirty = true;
    let pos = Position::new(mouse.column, mouse.row);
    if let Mode::Help(scroll) = &mut app.mode {
        match mouse.kind {
            M::ScrollUp => *scroll = scroll.saturating_sub(3),
            M::ScrollDown => *scroll = scroll.saturating_add(3),
            _ => {}
        }
        return Ok(());
    }
    if let Some(p) = app.pending_permission.as_mut() {
        let on_choice = matches!(mouse.kind, M::Down(MouseButton::Left))
            && p.choice_at(mouse.column, mouse.row).is_some();
        if on_choice || matches!(mouse.kind, M::ScrollUp | M::ScrollDown) {
            if let Some(outcome) = p.mouse(mouse) {
                handle_approval(app, child, outcome)?;
            }
            return Ok(());
        }
    }
    match mouse.kind {
        M::Down(MouseButton::Left) => {
            (app.text_selection, app.status_notice) = (None, None);
            app.text_selection = app
                .screen_area
                .contains(pos)
                .then(|| TextSelection::start(app.screen_area, pos));
        }
        M::Drag(MouseButton::Left) => {
            if let Some(sel) = app.text_selection.as_mut().filter(|s| s.dragging) {
                sel.extend(pos);
            }
        }
        M::Up(MouseButton::Left) => {
            let moved = app
                .text_selection
                .as_mut()
                .filter(|s| s.dragging)
                .is_some_and(|s| {
                    s.finish(pos);
                    s.moved()
                });
            if moved {
                app.copy_selection = true;
            } else {
                app.text_selection = None;
                let hit = app
                    .pending_permission
                    .is_none()
                    .then(|| {
                        app.tool_hits
                            .iter()
                            .find(|h| mouse.row >= h.y && mouse.row < h.y + h.height)
                    })
                    .flatten()
                    .map(|h| h.index);
                if let Some(idx) = hit {
                    app.set_expanded(idx, None);
                }
            }
        }
        M::ScrollUp | M::ScrollDown => {
            app.text_selection = None;
            app.scroll_by([3, -3][usize::from(matches!(mouse.kind, M::ScrollUp))]);
        }
        _ => {}
    }
    Ok(())
}

fn draft_edited(app: &mut App) {
    (
        app.menu_index,
        app.menu_dismissed,
        app.menu_interacted,
        app.dirty,
    ) = (0, false, false, true);
    app.prompt_history.edited();
}
