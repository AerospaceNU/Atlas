//! CLI argument parsing (`clap`) and Python `atlas.agent` subprocess lifecycle.

use std::ffi::OsStr;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use clap::Parser;

use crate::protocol::ClientMessage;

#[derive(Debug, Clone, Parser)]
#[command(
    name = "atlas-tui",
    about = "Interactive terminal client for the Atlas agent session"
)]
pub struct CliArgs {
    #[arg(long)]
    pub workspace: Option<PathBuf>,
    #[arg(long)]
    pub repo: Option<PathBuf>,
    #[arg(long)]
    pub python: Option<PathBuf>,
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long)]
    pub resume: Option<String>,
    #[arg(long)]
    pub approval: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LaunchConfig {
    pub workspace: PathBuf,
    pub repo: Option<PathBuf>,
    pub python: Option<PathBuf>,
    pub model: Option<String>,
    pub resume: Option<String>,
    pub approval: Option<String>,
}

fn py_cmd(bin: impl AsRef<OsStr>) -> Command {
    let mut c = Command::new(bin);
    c.args(["-P", "-m", "atlas.agent"]);
    c
}

impl LaunchConfig {
    pub fn from_cli(args: CliArgs) -> Result<Self> {
        let cwd = std::env::current_dir().context("could not read the current directory")?;
        let raw_ws = args.workspace.unwrap_or_else(|| cwd.clone());
        let workspace = raw_ws.canonicalize().unwrap_or(raw_ws);
        let python = args
            .python
            .or_else(|| std::env::var_os("ATLAS_PYTHON").map(PathBuf::from))
            .filter(|p| !p.as_os_str().is_empty());
        let repo = args.repo.or_else(|| find_repo_root(&cwd)).or_else(|| {
            std::env::current_exe()
                .ok()
                .and_then(|e| find_repo_root(&e))
        });
        Ok(Self {
            workspace,
            repo,
            python,
            model: args.model,
            resume: args.resume,
            approval: args.approval,
        })
    }

    pub fn build_command(&self) -> Command {
        let py = self
            .python
            .clone()
            .or_else(|| self.repo.as_deref().and_then(venv_python));
        let mut cmd = match (py, &self.repo) {
            (Some(py), _) => py_cmd(py),
            (None, Some(_)) => {
                let mut c = Command::new("uv");
                c.args(["run", "python", "-P", "-m", "atlas.agent"]);
                c
            }
            (None, None) => py_cmd("python3"),
        };

        let approval = self.approval.as_deref().unwrap_or("ask");
        cmd.args(["--approval", approval])
            .arg("--workspace")
            .arg(&self.workspace);
        for (flag, val) in [("--model", &self.model), ("--resume", &self.resume)] {
            if let Some(v) = val {
                cmd.args([flag, v]);
            }
        }
        cmd.current_dir(self.repo.as_deref().unwrap_or(&self.workspace));
        if let Some(src) = self
            .repo
            .as_ref()
            .map(|r| r.join("src"))
            .filter(|s| s.join("atlas").is_dir() && cmd.get_program() != "uv")
        {
            let cur = std::env::var_os("PYTHONPATH");
            let paths = std::iter::once(src).chain(cur.iter().flat_map(std::env::split_paths));
            if let Ok(joined) = std::env::join_paths(paths) {
                cmd.env("PYTHONPATH", joined);
            }
        }
        cmd.env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        cmd
    }
}

pub fn find_repo_root(start: &Path) -> Option<PathBuf> {
    let base = if start.is_file() {
        start.parent()?
    } else {
        start
    };
    base.ancestors()
        .find(|d| {
            std::fs::read_to_string(d.join("pyproject.toml"))
                .is_ok_and(|t| t.contains("name = \"atlas\""))
        })
        .map(Path::to_path_buf)
}

fn venv_python(repo: &Path) -> Option<PathBuf> {
    [
        repo.join(".venv/bin/python"),
        repo.join(".venv/Scripts/python.exe"),
    ]
    .into_iter()
    .find(|p| p.is_file())
}

pub struct ChildProcess {
    pub child: Child,
    stdin: ChildStdin,
}

impl ChildProcess {
    pub fn spawn(config: &LaunchConfig) -> Result<Self> {
        let mut cmd = config.build_command();
        let prog = cmd.get_program().to_string_lossy().into_owned();
        let mut child = cmd
            .spawn()
            .with_context(|| format!("failed to start `{prog}` for atlas.agent"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("no child stdin"))?;
        Ok(Self { child, stdin })
    }

    #[cfg(test)]
    pub fn from_parts(child: Child, stdin: ChildStdin) -> Self {
        Self { child, stdin }
    }

    pub fn send(&mut self, msg: &ClientMessage) -> Result<()> {
        writeln!(self.stdin, "{}", serde_json::to_string(msg)?)?;
        self.stdin.flush()?;
        Ok(())
    }

    pub fn shutdown(&mut self) {
        let _ = self.send(&ClientMessage::Quit);
        for _ in 0..20 {
            if matches!(self.child.try_wait(), Ok(Some(_))) {
                return;
            }
            thread::sleep(Duration::from_millis(50));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for ChildProcess {
    fn drop(&mut self) {
        if !matches!(self.child.try_wait(), Ok(Some(_))) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_uses_safe_path_defaults_to_ask_approval_and_allows_override() {
        let args = |extra: &[&str]| {
            let base = [
                "atlas-tui",
                "--workspace",
                "/tmp/ws",
                "--python",
                "/usr/bin/python3",
                "--model",
                "m",
                "--resume",
                "s1",
            ];
            let cmd = LaunchConfig::from_cli(CliArgs::parse_from(
                base.into_iter().chain(extra.iter().copied()),
            ))
            .unwrap()
            .build_command();
            cmd.get_args()
                .map(|a| a.to_string_lossy().into_owned())
                .collect::<Vec<_>>()
        };
        let def = args(&[]);
        assert!(
            def.first().map(String::as_str) == Some("-P")
                && def.windows(2).any(|p| p == ["--approval", "ask"])
                && def.contains(&"--model".into())
                && def.contains(&"--resume".into())
        );
        assert!(args(&["--approval", "allow"])
            .windows(2)
            .any(|p| p == ["--approval", "allow"]));
    }
}
