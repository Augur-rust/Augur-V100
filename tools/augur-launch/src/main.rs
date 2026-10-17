//! 启动服务.exe: double-click entry point placed at the package root.
//!
//! Starts the bundled, isolated Python launcher (`source/scripts/augur_portable.py configure
//! --autostart`) without a console window. The launcher opens the local setup page in the browser
//! and, when a configuration was saved before, loads the engine with it. A second start finds the
//! running launcher and only reopens its page. Nothing is installed and no system Python, PATH
//! entry or registry key is used.
#![windows_subsystem = "windows"]

use std::io::Read;
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const CREATE_NO_WINDOW: u32 = 0x0800_0000;
const MB_ICONERROR: u32 = 0x10;
/// Long enough for package verification and the setup server to come up; a launcher that is
/// still running after this is healthy and owns its own errors through the setup page.
const STARTUP_WATCH: Duration = Duration::from_secs(30);

#[link(name = "user32")]
extern "system" {
    fn MessageBoxW(hwnd: *mut core::ffi::c_void, text: *const u16, caption: *const u16, kind: u32) -> i32;
}

fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(Some(0)).collect()
}

fn fail(message: &str) -> ! {
    let (text, caption) = (wide(message), wide("Augur"));
    // SAFETY: both buffers are NUL-terminated UTF-16 and outlive the call; a null owner is allowed.
    unsafe { MessageBoxW(core::ptr::null_mut(), text.as_ptr(), caption.as_ptr(), MB_ICONERROR) };
    std::process::exit(2)
}

fn package_root() -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_else(|_| fail("无法定位启动程序。"));
    exe.parent().map(Path::to_path_buf).unwrap_or_else(|| fail("无法定位 Augur 文件夹。"))
}

/// Same private directory the launcher uses, so the package folder may stay read-only.
fn data_dir(root: &Path) -> PathBuf {
    std::env::var_os("LOCALAPPDATA").map(|d| PathBuf::from(d).join("Augur")).unwrap_or_else(|| root.join("user-data"))
}

fn tail(path: &Path) -> String {
    let mut text = String::new();
    if let Ok(mut file) = std::fs::File::open(path) {
        let _ = file.read_to_string(&mut text);
    }
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    lines[lines.len().saturating_sub(6)..].join("\n")
}

fn main() {
    let root = package_root();
    let python = root.join("runtime").join("python").join("python.exe");
    let entry = root.join("source").join("scripts").join("augur_portable.py");
    if !python.is_file() || !entry.is_file() {
        fail("缺少内置运行环境。请完整解压压缩包，然后在解压后的文件夹里双击“启动服务.exe”。");
    }
    let data = data_dir(&root);
    let _ = std::fs::create_dir_all(&data);
    let log = data.join("launcher-start.log");
    let stderr = std::fs::File::create(&log).map(Stdio::from).unwrap_or_else(|_| Stdio::null());
    let mut command = Command::new(&python);
    command
        .args(["-I", "-X", "utf8", "-B"])
        .arg(&entry)
        .args(["configure", "--autostart"])
        .current_dir(&root)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(stderr)
        .creation_flags(CREATE_NO_WINDOW);
    let mut child = command.spawn().unwrap_or_else(|error| fail(&format!("Augur 无法启动内置运行环境：{error}")));
    // Without a console an early failure would be silent; report it instead of vanishing.
    let started = Instant::now();
    while started.elapsed() < STARTUP_WATCH {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => return, // reopened an already running launcher
            Ok(Some(_)) => fail(&format!("Augur 启动失败。\n\n{}\n\n完整日志：{}", tail(&log), log.display())),
            Ok(None) => std::thread::sleep(Duration::from_millis(200)),
            Err(_) => return,
        }
    }
}
