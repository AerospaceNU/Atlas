//! Cross-platform clipboard via `arboard` with an OSC 52 fallback for SSH/tmux.

use std::io::{self, IsTerminal, Write};

use base64::engine::general_purpose::STANDARD;
use base64::Engine;

pub fn copy_to_clipboard(text: &str) -> bool {
    if let Ok(mut clipboard) = arboard::Clipboard::new() {
        if clipboard.set_text(text.to_owned()).is_ok() {
            return true;
        }
    }
    let mut stdout = io::stdout();
    if !stdout.is_terminal() {
        return false;
    }
    let in_tmux = std::env::var_os("TMUX").is_some();
    write_osc52(&mut stdout, text, in_tmux).is_ok()
}

pub fn read_from_clipboard() -> Option<String> {
    let mut clipboard = arboard::Clipboard::new().ok()?;
    clipboard.get_text().ok()
}

pub fn write_osc52(writer: &mut impl Write, text: &str, in_tmux: bool) -> io::Result<()> {
    let encoded = STANDARD.encode(text.as_bytes());
    if in_tmux {
        write!(writer, "\x1bPtmux;\x1b\x1b]52;c;{encoded}\x07\x1b\\")?;
    } else {
        write!(writer, "\x1b]52;c;{encoded}\x07")?;
    }
    writer.flush()
}

#[cfg(test)]
mod tests {
    use super::write_osc52;

    #[test]
    fn osc52_encodes_plain_and_tmux_sequences() {
        let mut plain = Vec::new();
        write_osc52(&mut plain, "hello", false).unwrap();
        assert_eq!(String::from_utf8(plain).unwrap(), "\x1b]52;c;aGVsbG8=\x07");

        let mut tmux = Vec::new();
        write_osc52(&mut tmux, "hello", true).unwrap();
        assert_eq!(
            String::from_utf8(tmux).unwrap(),
            "\x1bPtmux;\x1b\x1b]52;c;aGVsbG8=\x07\x1b\\"
        );
    }
}
