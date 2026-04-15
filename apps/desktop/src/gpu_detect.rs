//! Pre-flight Vulkan GPU detection.
//!
//! whisper.cpp and llama.cpp happily accept any Vulkan physical device the
//! system enumerates, including software renderers (Microsoft Basic Render
//! Driver, llvmpipe, SwiftShader) and adapters with insufficient VRAM. On
//! those adapters the model "loads" fine but actual compute hangs or crashes
//! later — leaving the app in a bad state and triggering the crash-marker
//! recovery path unnecessarily.
//!
//! This module does its own enumeration first using `ash`, filters out
//! adapters that are either software renderers or obviously underpowered,
//! and only allows the whisper/llama GPU path when a real usable GPU is
//! present. Result is cached for the life of the process — hardware doesn't
//! change mid-session.
//!
//! Only compiled when the `gpu` feature is enabled.

use std::ffi::CStr;
use std::sync::OnceLock;

/// Minimum device-local memory (in MB) to consider a GPU usable. This is
/// intentionally permissive — it catches the Microsoft Basic Render Driver
/// (~0 MB VRAM) and the most underpowered adapters, but doesn't gate out
/// legitimate integrated GPUs (Iris Xe, AMD iGPU, etc.) that can still run
/// Whisper. If an LLM model is too big for the available VRAM, llama.cpp
/// will fail its own load softly and we'll fall back to CPU via the existing
/// Err path in `Formatter::new`.
const MIN_VRAM_MB: u64 = 512;

/// Substrings in the Vulkan device name that indicate a software renderer.
/// All checks are case-insensitive.
const SOFTWARE_RENDERER_MARKERS: &[&str] = &[
    "llvmpipe",      // Mesa software rasterizer
    "swiftshader",   // Google's software Vulkan
    "basic render",  // "Microsoft Basic Render Driver"
    "software",      // generic
];

/// Result of pre-flight GPU detection.
#[derive(Debug, Clone)]
pub enum GpuDetection {
    /// A real GPU passed all our filters. Safe to proceed with Vulkan.
    Usable {
        name: String,
        device_type: &'static str,
        vram_mb: u64,
    },
    /// Some reason we should stay on CPU. The string is a human-readable
    /// explanation safe to log and (briefly) surface to the UI.
    NotUsable(String),
}

impl GpuDetection {
    /// Suitable for Whisper: any legit GPU passed all our filters. Whisper
    /// models are ~150 MB and run fine even on minimal integrated GPUs like
    /// Intel UHD — those pass.
    pub fn is_usable(&self) -> bool {
        matches!(self, GpuDetection::Usable { .. })
    }

    /// Suitable for LLM: stricter than `is_usable()`. LLM models are 1.5–4 GB
    /// quantized — big enough that integrated GPUs either fail to fit the
    /// model, or worse, "accept" the load via memory overcommit and then
    /// either HANG or take forever on the first inference pass (observed on
    /// Intel UHD on both Windows and Linux).
    ///
    /// Rule: require a discrete GPU. Integrated GPUs on Windows typically
    /// report the shared system RAM as DEVICE_LOCAL in Vulkan heaps, so a
    /// simple VRAM-threshold check can't tell them apart from real dedicated
    /// VRAM. Discrete-only is the reliable signal.
    ///
    /// Tradeoff: this rejects capable iGPUs (Intel Iris Xe, some AMD APUs)
    /// that could technically run an LLM on GPU. Most modern CPUs with AVX2
    /// run small LLMs fast enough in CPU mode that this isn't noticeable,
    /// and it's a big reliability win — no more hangs on Intel UHD.
    pub fn is_suitable_for_llm(&self) -> bool {
        match self {
            GpuDetection::Usable { device_type, .. } => {
                *device_type == "discrete GPU"
            }
            _ => false,
        }
    }

    /// Short human-readable summary for logs and the UI status row.
    pub fn summary(&self) -> String {
        match self {
            GpuDetection::Usable { name, device_type, vram_mb } => {
                format!("{name} ({device_type}, {vram_mb} MB VRAM)")
            }
            GpuDetection::NotUsable(r) => r.clone(),
        }
    }
}

static DETECTION: OnceLock<GpuDetection> = OnceLock::new();

/// Cached pre-flight GPU detection. Safe to call repeatedly — the actual
/// enumeration only runs once per process.
pub fn detect() -> GpuDetection {
    DETECTION.get_or_init(run_detection).clone()
}

fn run_detection() -> GpuDetection {
    // Wrap the whole thing in catch_unwind. A broken Vulkan loader (rare but
    // possible after a failed driver update) could panic inside ash FFI
    // rather than returning a clean error. We'd rather report "no usable
    // GPU" than abort the app during startup.
    let result = std::panic::catch_unwind(|| unsafe { enumerate_vulkan() });

    let detection = match result {
        Ok(Ok(d)) => d,
        Ok(Err(e)) => GpuDetection::NotUsable(e),
        Err(_) => GpuDetection::NotUsable(
            "Vulkan enumeration panicked — assuming no usable GPU".to_string(),
        ),
    };

    match &detection {
        GpuDetection::Usable { .. } => {
            eprintln!("gpu_detect: usable GPU — {}", detection.summary());
        }
        GpuDetection::NotUsable(reason) => {
            eprintln!("gpu_detect: no usable GPU — {reason}");
        }
    }

    detection
}

unsafe fn enumerate_vulkan() -> Result<GpuDetection, String> {
    // `Entry::load()` dynamically opens vulkan-1.dll (or the Linux/Mac
    // equivalent). If Vulkan isn't installed at all, this errors cleanly
    // without any side effects.
    let entry = unsafe { ash::Entry::load() }
        .map_err(|e| format!("Vulkan runtime not available: {e}"))?;

    let app_name = CStr::from_bytes_with_nul(b"Dictator\0").unwrap();
    let app_info = ash::vk::ApplicationInfo::default()
        .application_name(app_name)
        .api_version(ash::vk::API_VERSION_1_0);

    let create_info = ash::vk::InstanceCreateInfo::default()
        .application_info(&app_info);

    let instance = unsafe { entry.create_instance(&create_info, None) }
        .map_err(|e| format!("Vulkan instance creation failed: {e}"))?;

    // Always destroy the instance, even on error paths.
    let result = unsafe { evaluate_instance(&instance) };
    unsafe { instance.destroy_instance(None) };
    result
}

unsafe fn evaluate_instance(instance: &ash::Instance) -> Result<GpuDetection, String> {
    let devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|e| format!("Vulkan device enumeration failed: {e}"))?;

    if devices.is_empty() {
        return Err("No Vulkan physical devices present".to_string());
    }

    let mut best: Option<GpuDetection> = None;
    let mut rejected: Vec<String> = Vec::new();

    for device in devices {
        let props = unsafe { instance.get_physical_device_properties(device) };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(device) };

        let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        // Filter 1: device_type must be a real GPU.
        let (device_type, is_real_gpu): (&'static str, bool) = match props.device_type {
            ash::vk::PhysicalDeviceType::DISCRETE_GPU => ("discrete GPU", true),
            ash::vk::PhysicalDeviceType::INTEGRATED_GPU => ("integrated GPU", true),
            ash::vk::PhysicalDeviceType::VIRTUAL_GPU => ("virtual GPU", false),
            ash::vk::PhysicalDeviceType::CPU => ("CPU-software", false),
            _ => ("other", false),
        };
        if !is_real_gpu {
            rejected.push(format!("{name} is {device_type}"));
            continue;
        }

        // Filter 2: name must not match a known software-renderer marker.
        // Some software renderers claim to be a GPU to support games that
        // demand one — we don't want to offload LLM work to them.
        let name_lower = name.to_lowercase();
        if let Some(marker) = SOFTWARE_RENDERER_MARKERS
            .iter()
            .find(|m| name_lower.contains(*m))
        {
            rejected.push(format!("{name} matches software-renderer marker '{marker}'"));
            continue;
        }

        // Filter 3: must have at least MIN_VRAM_MB of device-local memory.
        // Integrated GPUs report a small dedicated heap + a larger host-visible
        // one; we only count device-local because that's what model weights
        // actually live in.
        let heap_count = mem_props.memory_heap_count as usize;
        let vram_bytes: u64 = mem_props.memory_heaps[..heap_count]
            .iter()
            .filter(|h| h.flags.contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|h| h.size)
            .sum();
        let vram_mb = vram_bytes / 1024 / 1024;

        if vram_mb < MIN_VRAM_MB {
            rejected.push(format!(
                "{name} has only {vram_mb} MB device-local memory (< {MIN_VRAM_MB} MB minimum)"
            ));
            continue;
        }

        // Passed all filters. Keep track of the best candidate: prefer discrete
        // over integrated when both are present (multi-GPU laptops).
        let candidate = GpuDetection::Usable {
            name: name.clone(),
            device_type,
            vram_mb,
        };

        match &best {
            None => best = Some(candidate),
            Some(GpuDetection::Usable { device_type: best_type, .. }) => {
                if *best_type == "integrated GPU" && device_type == "discrete GPU" {
                    best = Some(candidate);
                }
            }
            _ => {}
        }
    }

    match best {
        Some(d) => Ok(d),
        None => {
            let detail = if rejected.is_empty() {
                "no candidates".to_string()
            } else {
                rejected.join("; ")
            };
            Err(format!("No usable Vulkan GPU found ({detail})"))
        }
    }
}
