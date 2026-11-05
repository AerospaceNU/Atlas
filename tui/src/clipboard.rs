//! Platform clipboard commands used by the terminal client.

use std::io::Write;
use std::process::{Child, Command, Stdio};

#[derive(Clone, Copy)]
struct CommandSpec<'a> {
    program: &'a str,
    args: &'a [&'a str],
}

const WRITE_COMMANDS: &[CommandSpec<'static>] = &[
    CommandSpec {
        program: "pbcopy",
        args: &[],
    },
    CommandSpec {
        program: "wl-copy",
        args: &[],
    },
    CommandSpec {
        program: "xclip",
        args: &["-selection", "clipboard"],
    },
    CommandSpec {
        program: "xsel",
        args: &["--clipboard", "--input"],
    },
    CommandSpec {
        program: "clip",
        args: &[],
    },
];

const READ_COMMANDS: &[CommandSpec<'static>] = &[
    CommandSpec {
        program: "pbpaste",
        args: &[],
    },
    CommandSpec {
        program: "wl-paste",
        args: &["--no-newline"],
    },
    CommandSpec {
        program: "xclip",
        args: &["-selection", "clipboard", "-o"],
    },
    CommandSpec {
        program: "xsel",
        args: &["--clipboard", "--output"],
    },
    CommandSpec {
        program: "powershell",
        args: &[
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); [Console]::Out.Write((Get-Clipboard -Raw))",
        ],
    },
];

pub(super) fn copy_to_clipboard(text: &str) -> bool {
    copy_with_commands(text, WRITE_COMMANDS)
}

pub(super) fn read_from_clipboard() -> Option<String> {
    read_with_commands(READ_COMMANDS)
}

fn copy_with_commands(text: &str, commands: &[CommandSpec<'_>]) -> bool {
    commands
        .iter()
        .any(|command| write_to_command(command.program, command.args, text))
}

fn write_to_command(program: &str, args: &[&str], text: &str) -> bool {
    let Ok(mut child) = Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    else {
        return false;
    };

    let Some(mut stdin) = child.stdin.take() else {
        reap_after_failure(&mut child);
        return false;
    };

    let write_succeeded = stdin.write_all(text.as_bytes()).is_ok();
    drop(stdin);
    if !write_succeeded {
        reap_after_failure(&mut child);
        return false;
    }

    match child.wait() {
        Ok(status) => status.success(),
        Err(_) => {
            reap_after_failure(&mut child);
            false
        }
    }
}

fn reap_after_failure(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn read_with_commands(commands: &[CommandSpec<'_>]) -> Option<String> {
    commands
        .iter()
        .find_map(|command| read_from_command(command.program, command.args))
}

fn read_from_command(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    String::from_utf8(output.stdout).ok()
}

#[cfg(all(test, unix))]
mod tests {
    use super::{
        copy_with_commands, read_from_command, read_with_commands, write_to_command, CommandSpec,
    };
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_FILE: AtomicUsize = AtomicUsize::new(0);

    fn temp_file() -> std::path::PathBuf {
        let index = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "atlas-clipboard-test-{}-{index}",
            std::process::id()
        ))
    }

    #[test]
    fn write_helper_passes_input_bytes_unchanged() {
        let path = temp_file();
        let path_arg = path.to_string_lossy();
        let args = ["-c", "cat > \"$1\"", "clipboard-test", path_arg.as_ref()];
        let text = "line one\n🛰️ café\n";

        assert!(write_to_command("sh", &args, text));
        assert_eq!(fs::read(&path).unwrap(), text.as_bytes());

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn write_helper_falls_back_after_a_failed_backend() {
        let path = temp_file();
        let path_arg = path.to_string_lossy();
        let args = ["-c", "cat > \"$1\"", "clipboard-test", path_arg.as_ref()];
        let commands = [
            CommandSpec {
                program: "false",
                args: &[],
            },
            CommandSpec {
                program: "sh",
                args: &args,
            },
        ];
        let text = "fallback\n🛰️\n";

        assert!(copy_with_commands(text, &commands));
        assert_eq!(fs::read(&path).unwrap(), text.as_bytes());

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn read_helper_preserves_unicode_and_newlines() {
        let expected = "line one\n雪と🛰️\nline three\n";
        let args = ["-c", "printf '%s' \"$1\"", "clipboard-test", expected];

        assert_eq!(read_from_command("sh", &args).as_deref(), Some(expected));
    }

    #[test]
    fn read_helper_falls_back_after_invalid_utf8() {
        let expected = "fallback\nnaïve 🛰️\n";
        let invalid_args = ["-c", "printf '\\377'"];
        let valid_args = ["-c", "printf '%s' \"$1\"", "clipboard-test", expected];
        let commands = [
            CommandSpec {
                program: "sh",
                args: &invalid_args,
            },
            CommandSpec {
                program: "sh",
                args: &valid_args,
            },
        ];

        assert_eq!(read_with_commands(&commands).as_deref(), Some(expected));
    }
}
