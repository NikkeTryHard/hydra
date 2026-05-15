use std::env;
use std::fs;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

#[test]
#[cfg(target_os = "linux")]
fn probe_child_guard_kills_child_on_drop() {
    if let Some(pid) = env::var_os("HYDRA_PROBE_CHILD_GUARD_CHECK_PID") {
        let pid = pid
            .to_string_lossy()
            .parse::<u32>()
            .expect("pid should parse");
        for _ in 0..100 {
            if !process_is_running(pid) {
                return;
            }
            thread::sleep(Duration::from_millis(10));
        }
        panic!("child pid {pid} still alive after guard drop");
    }

    let current_exe = env::current_exe().expect("current test executable path");
    if env::var_os("HYDRA_PROBE_CHILD_GUARD_SPAWN_AND_DROP").is_some() {
        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg("sleep 30 & wait");
        let pid = hydra_train_exec::probe_process::spawn_guarded_child_for_test(&mut cmd)
            .expect("guarded child should spawn");
        println!("{pid}");
        return;
    }

    let helper_start = Instant::now();
    let output = Command::new(&current_exe)
        .env("HYDRA_PROBE_CHILD_GUARD_SPAWN_AND_DROP", "1")
        .arg("probe_child_guard_kills_child_on_drop")
        .arg("--exact")
        .arg("--nocapture")
        .output()
        .expect("spawn helper test process");
    assert!(
        output.status.success(),
        "helper failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let helper_elapsed = helper_start.elapsed();
    let stdout = String::from_utf8(output.stdout).expect("helper stdout should be utf8");
    let pid = stdout
        .lines()
        .find_map(|line| line.trim().parse::<u32>().ok())
        .expect("helper should print guarded child pid");

    let status = Command::new(current_exe)
        .env("HYDRA_PROBE_CHILD_GUARD_CHECK_PID", pid.to_string())
        .arg("probe_child_guard_kills_child_on_drop")
        .arg("--exact")
        .arg("--nocapture")
        .status()
        .expect("spawn checker test process");
    assert!(status.success(), "guarded child pid {pid} survived");
    assert!(
        helper_elapsed >= Duration::from_millis(50),
        "guard helper returned before termination grace elapsed; child/process group kill was not awaited (elapsed={helper_elapsed:?})"
    );
}

fn process_is_running(pid: u32) -> bool {
    let Ok(status) = fs::read_to_string(format!("/proc/{pid}/status")) else {
        return false;
    };
    !status.lines().any(|line| line == "State:\tZ (zombie)")
}
