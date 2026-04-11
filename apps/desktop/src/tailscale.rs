//! Tailscale integration: detect Tailscale, get machine info, generate HTTPS certs.

use serde::Deserialize;
use std::path::Path;
use std::process::Command;

/// Info about the running Tailscale instance.
#[derive(Debug, Clone)]
pub struct TailscaleInfo {
    pub dns_name: String,
    pub cert_domain: String,
    pub tailscale_ips: Vec<String>,
}

// --- JSON structs for `tailscale status --json` ---

#[derive(Deserialize)]
struct TsStatus {
    #[serde(rename = "BackendState")]
    backend_state: String,
    #[serde(rename = "Self")]
    self_node: Option<TsSelfNode>,
    #[serde(rename = "CertDomains")]
    cert_domains: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct TsSelfNode {
    #[serde(rename = "DNSName")]
    dns_name: String,
    #[serde(rename = "TailscaleIPs")]
    tailscale_ips: Option<Vec<String>>,
}

/// Result of checking Tailscale status.
pub enum DetectResult {
    /// Tailscale is running and HTTPS is enabled.
    Ok(TailscaleInfo),
    /// Tailscale binary not found on the system.
    NotInstalled,
    /// Tailscale is installed but not running/connected.
    NotRunning(String),
    /// Tailscale is running but HTTPS certificates are not enabled.
    NoHttps,
}

/// Detect whether Tailscale is installed, running, and has HTTPS enabled.
pub fn detect() -> DetectResult {
    eprintln!("Checking for Tailscale...");

    let stdout = match run_tailscale(&["status", "--json"]) {
        Some(s) => s,
        None => return DetectResult::NotInstalled,
    };

    let status: TsStatus = match serde_json::from_slice(&stdout) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to parse tailscale status JSON: {e}");
            return DetectResult::NotInstalled;
        }
    };

    if status.backend_state != "Running" {
        eprintln!("Tailscale state is '{}', not Running", status.backend_state);
        return DetectResult::NotRunning(status.backend_state);
    }

    let self_node = match status.self_node {
        Some(n) => n,
        None => return DetectResult::NotRunning("No self node".into()),
    };

    let cert_domains = status.cert_domains.unwrap_or_default();
    if cert_domains.is_empty() {
        eprintln!("Tailscale is running but HTTPS (CertDomains) is not enabled");
        return DetectResult::NoHttps;
    }

    let cert_domain = cert_domains.first().unwrap().clone();

    let info = TailscaleInfo {
        dns_name: self_node.dns_name.trim_end_matches('.').to_string(),
        cert_domain,
        tailscale_ips: self_node.tailscale_ips.unwrap_or_default(),
    };

    eprintln!(
        "Tailscale detected: domain={}, ips={:?}",
        info.cert_domain, info.tailscale_ips
    );
    DetectResult::Ok(info)
}

/// Generate a TLS certificate via `tailscale cert`. Returns (cert_pem, key_pem) bytes.
pub fn generate_cert(
    domain: &str,
    cert_path: &Path,
    key_path: &Path,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let cert_arg = format!("--cert-file={}", cert_path.display());
    let key_arg = format!("--key-file={}", key_path.display());

    eprintln!("Running: tailscale cert {cert_arg} {key_arg} {domain}");

    // Try to run the cert command
    let output = try_command("tailscale", &["cert", &cert_arg, &key_arg, domain])
        .or_else(|| {
            try_command(
                r"C:\Program Files\Tailscale\tailscale.exe",
                &["cert", &cert_arg, &key_arg, domain],
            )
        })
        .ok_or_else(|| "Could not run tailscale cert (not on PATH?)".to_string())?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("tailscale cert failed: {stderr}"));
    }

    let cert_pem =
        std::fs::read(cert_path).map_err(|e| format!("Failed to read generated cert: {e}"))?;
    let key_pem =
        std::fs::read(key_path).map_err(|e| format!("Failed to read generated key: {e}"))?;

    eprintln!("Tailscale cert generated successfully for {domain}");
    Ok((cert_pem, key_pem))
}

/// Try running `tailscale status --json` with common executable locations.
fn run_tailscale(args: &[&str]) -> Option<Vec<u8>> {
    // Try PATH first
    if let Some(output) = try_command("tailscale", args) {
        if output.status.success() {
            return Some(output.stdout);
        }
    }

    // Try default Windows install location
    if let Some(output) = try_command(r"C:\Program Files\Tailscale\tailscale.exe", args) {
        if output.status.success() {
            return Some(output.stdout);
        }
    }

    eprintln!("Tailscale not found on PATH or at default install location");
    None
}

/// Attempt to spawn a command, returning None if the binary isn't found.
fn try_command(program: &str, args: &[&str]) -> Option<std::process::Output> {
    Command::new(program)
        .args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .ok()
}
