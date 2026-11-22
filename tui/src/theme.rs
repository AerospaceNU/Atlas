//! Adaptive color palette with truecolor, 256-color, and `NO_COLOR` support.

use std::sync::OnceLock;

use ratatui::style::{Color, Modifier, Style};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMode {
    TrueColor,
    Ansi256,
    NoColor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Theme {
    pub mode: ColorMode,
    pub bg_base: Color,
    pub bg_surface: Color,
    pub bg_code: Color,
    pub fg: Color,
    pub fg_secondary: Color,
    pub muted: Color,
    pub muted_bright: Color,
    pub accent: Color,
    pub cyan: Color,
    pub yellow: Color,
    pub green: Color,
    pub red: Color,
    pub border: Color,
    pub border_active: Color,
    pub selection_bg: Color,
    pub selection_fg: Color,
}

static THEME: OnceLock<Theme> = OnceLock::new();

const fn from_array(mode: ColorMode, c: [Color; 16]) -> Theme {
    Theme {
        mode,
        bg_base: c[0],
        bg_surface: c[1],
        bg_code: c[2],
        fg: c[3],
        fg_secondary: c[4],
        muted: c[5],
        muted_bright: c[6],
        accent: c[7],
        cyan: c[8],
        yellow: c[9],
        green: c[10],
        red: c[11],
        border: c[12],
        border_active: c[13],
        selection_bg: c[14],
        selection_fg: c[15],
    }
}

impl Theme {
    pub fn current() -> &'static Theme {
        THEME.get_or_init(Self::detect)
    }

    pub fn detect() -> Self {
        let e = |k| std::env::var(k).unwrap_or_default();
        let nc = std::env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty());
        Self::from_env_values(nc, &e("COLORTERM"), &e("TERM_PROGRAM"), &e("TERM"))
    }

    pub fn from_env_values(no_color: bool, ct: &str, tp: &str, term: &str) -> Self {
        if no_color || term.eq_ignore_ascii_case("dumb") {
            return Self::no_color();
        }
        let (ct, tp) = (ct.to_ascii_lowercase(), tp.to_ascii_lowercase());
        let tc = ct.contains("truecolor")
            || ct.contains("24bit")
            || matches!(
                tp.as_str(),
                "iterm.app" | "wezterm" | "ghostty" | "vscode" | "alacritty" | "kitty" | "hyper"
            )
            || ["direct", "kitty", "ghostty"]
                .iter()
                .any(|s| term.contains(s));
        if tc {
            Self::truecolor()
        } else {
            Self::ansi256()
        }
    }

    pub const fn truecolor() -> Self {
        const HEX: [u32; 16] = [
            0x141416, 0x222228, 0x1c1c22, 0xe4e4e8, 0xc4c6d0, 0x70727e, 0x8c909e, 0xbb9af7,
            0x7dcfff, 0xe0af68, 0x9ece6a, 0xf7768e, 0x363842, 0x606478, 0x483c68, 0xececf4,
        ];
        let mut c = [Color::Reset; 16];
        let mut i = 0;
        while i < 16 {
            let h = HEX[i];
            c[i] = Color::Rgb((h >> 16) as u8, (h >> 8) as u8, h as u8);
            i += 1;
        }
        from_array(ColorMode::TrueColor, c)
    }

    pub const fn ansi256() -> Self {
        const IDX: [u8; 16] = [
            0, 237, 236, 0, 250, 244, 247, 141, 81, 215, 114, 204, 240, 141, 60, 255,
        ];
        let mut c = [Color::Reset; 16];
        let mut i = 0;
        while i < 16 {
            c[i] = match IDX[i] {
                0 => Color::Reset,
                n => Color::Indexed(n),
            };
            i += 1;
        }
        from_array(ColorMode::Ansi256, c)
    }

    pub const fn no_color() -> Self {
        from_array(ColorMode::NoColor, [Color::Reset; 16])
    }

    pub fn base(&self) -> Style {
        Style::default().fg(self.fg).bg(self.bg_base)
    }
    pub fn fg(&self, fg: Color) -> Style {
        Style::default().fg(fg).bg(self.bg_base)
    }
    pub fn bold(&self, fg: Color) -> Style {
        self.fg(fg).add_modifier(Modifier::BOLD)
    }
    pub fn surface(&self, fg: Color) -> Style {
        Style::default().fg(fg).bg(self.bg_surface)
    }
    pub fn code(&self, fg: Color) -> Style {
        Style::default().fg(fg).bg(self.bg_code)
    }
    pub fn muted_style(&self) -> Style {
        match self.mode {
            ColorMode::NoColor => self.fg(self.muted).add_modifier(Modifier::DIM),
            _ => self.fg(self.muted),
        }
    }
    pub fn selection_style(&self) -> Style {
        match self.mode {
            ColorMode::NoColor => Style::default().add_modifier(Modifier::REVERSED),
            _ => Style::default().fg(self.selection_fg).bg(self.selection_bg),
        }
    }
    pub fn map_rgb(&self, r: u8, g: u8, b: u8) -> Color {
        match self.mode {
            ColorMode::TrueColor => Color::Rgb(r, g, b),
            ColorMode::Ansi256 => {
                let c = |v: u8| (u16::from(v) * 5 / 255) as u8;
                Color::Indexed(16 + 36 * c(r) + 6 * c(g) + c(b))
            }
            ColorMode::NoColor => Color::Reset,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_no_color_truecolor_and_256_modes() {
        let nc = Theme::from_env_values(true, "truecolor", "iTerm.app", "xterm-256color");
        assert_eq!(
            (nc.mode, nc.map_rgb(9, 9, 9)),
            (ColorMode::NoColor, Color::Reset)
        );
        assert!(nc
            .selection_style()
            .add_modifier
            .contains(Modifier::REVERSED));
        let tc = Theme::from_env_values(false, "truecolor", "", "xterm-256color");
        assert_eq!(tc.mode, ColorMode::TrueColor);
        assert!(!tc
            .selection_style()
            .add_modifier
            .contains(Modifier::REVERSED));
        let c256 = Theme::from_env_values(false, "", "", "xterm-256color");
        assert_eq!(
            (c256.mode, c256.bg_base),
            (ColorMode::Ansi256, Color::Reset)
        );
        assert!(matches!(c256.map_rgb(255, 0, 0), Color::Indexed(_)));
    }
}
