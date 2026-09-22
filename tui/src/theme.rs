//! Colors from Grok Build's default GrokNight theme.
//!
//! Values match `crates/codegen/xai-grok-pager-render/src/theme/groknight.rs`
//! in <https://github.com/xai-org/grok-build>. Only the slots this screen uses
//! are copied.

use ratatui::style::Color;

pub const BG_BASE: Color = Color::Rgb(20, 20, 20);
pub const BG_LIGHT: Color = Color::Rgb(36, 36, 36);
pub const FG: Color = Color::Rgb(225, 225, 225);
pub const FG_SECONDARY: Color = Color::Rgb(200, 200, 200);
pub const GRAY: Color = Color::Rgb(108, 108, 108);
pub const GRAY_BRIGHT: Color = Color::Rgb(120, 120, 120);
pub const MAGENTA: Color = Color::Rgb(187, 154, 247);
pub const YELLOW: Color = Color::Rgb(224, 175, 104);
pub const RED: Color = Color::Rgb(247, 118, 142);
pub const PROMPT_BORDER: Color = Color::Rgb(50, 50, 55);
pub const PROMPT_BORDER_ACTIVE: Color = Color::Rgb(80, 80, 88);
