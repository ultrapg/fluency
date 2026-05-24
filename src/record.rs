use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

fn is_junk_device(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.contains("monitor")
        || lower.contains("loopback")
        || lower.contains("echo")
        || lower.contains("system")
        || lower == "default"
        || lower.starts_with("pulseaudio")
        || lower.contains("sink")
        || lower.contains("output")
        || lower.starts_with("dummy")
}

fn clean_device_name(name: &str) -> String {
    name.trim()
        .trim_start_matches("Front Left: ")
        .trim_start_matches("Front Right: ")
        .trim_start_matches("alsa_input.")
        .trim_start_matches("alsa_output.")
        .replace('_', " ")
        .trim()
        .to_string()
}

pub fn list_input_devices() -> Vec<(String, String)> {
    let host = cpal::default_host();
    let mut seen = std::collections::HashSet::new();
    let mut devices = Vec::new();
    if let Ok(input_devices) = host.input_devices() {
        for device in input_devices {
            if let Ok(name) = device.name() {
                let cleaned = clean_device_name(&name);
                if is_junk_device(&cleaned) || cleaned.is_empty() || seen.contains(&cleaned) {
                    continue;
                }
                seen.insert(cleaned.clone());
                devices.push((cleaned, name));
            }
        }
    }
    devices
}

fn select_device(name: Option<&str>) -> anyhow::Result<cpal::Device> {
    let host = cpal::default_host();
    match name {
        Some(name) if !name.is_empty() => {
            for device in host.input_devices()? {
                if device.name().map(|n| n == name).unwrap_or(false) {
                    return Ok(device);
                }
            }
            anyhow::bail!("Input device '{name}' not found")
        }
        _ => host
            .default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No audio input device available")),
    }
}

fn record_common(
    device_name: Option<&str>,
    stop: Option<Arc<AtomicBool>>,
) -> anyhow::Result<Vec<f32>> {
    let device = select_device(device_name)?;
    let supported = device.default_input_config()?;
    let sample_format = supported.sample_format();
    let channels = supported.channels();
    let sample_rate = supported.sample_rate().0;
    let config: cpal::StreamConfig = supported.into();

    let recorded = Arc::new(Mutex::new(Vec::new()));
    let recorded_clone = recorded.clone();

    let err_fn = move |err: cpal::StreamError| {
        eprintln!("Audio capture error: {err}");
    };

    let stream = match sample_format {
        cpal::SampleFormat::F32 => {
            device.build_input_stream(
                &config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    recorded_clone.lock().unwrap().extend_from_slice(data);
                },
                err_fn,
                None,
            )?
        }
        cpal::SampleFormat::I16 => {
            device.build_input_stream(
                &config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let mut buf = recorded_clone.lock().unwrap();
                    for &s in data {
                        buf.push(s as f32 / 32768.0);
                    }
                },
                err_fn,
                None,
            )?
        }
        cpal::SampleFormat::U16 => {
            device.build_input_stream(
                &config,
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    let mut buf = recorded_clone.lock().unwrap();
                    for &s in data {
                        buf.push((s as f32 - 32768.0) / 32768.0);
                    }
                },
                err_fn,
                None,
            )?
        }
        _ => anyhow::bail!("Unsupported audio sample format"),
    };

    stream.play()?;

    if let Some(stop_flag) = stop {
        while !stop_flag.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    } else {
        eprintln!("Recording... Press Enter to stop.");
        let mut _input = String::new();
        std::io::stdin().read_line(&mut _input)?;
    }

    drop(stream);

    let raw = recorded.lock().unwrap().clone();
    if raw.is_empty() {
        anyhow::bail!("No audio captured");
    }

    let mono = if channels == 2 {
        raw.chunks(2).map(|ch| (ch[0] + ch[1]) * 0.5).collect()
    } else {
        raw
    };

    if sample_rate != 16000 {
        Ok(resample(&mono, sample_rate, 16000))
    } else {
        Ok(mono)
    }
}

pub fn record_from_mic() -> anyhow::Result<Vec<f32>> {
    record_common(None, None)
}

pub fn record_from_mic_with_device(device_name: Option<&str>) -> anyhow::Result<Vec<f32>> {
    record_common(device_name, None)
}

pub fn record_from_mic_with_stop(stop: Arc<AtomicBool>) -> anyhow::Result<Vec<f32>> {
    record_common(None, Some(stop))
}

pub fn record_from_mic_with_stop_and_device(
    stop: Arc<AtomicBool>,
    device_name: Option<&str>,
) -> anyhow::Result<Vec<f32>> {
    record_common(device_name, Some(stop))
}

pub fn load_audio_file(path: &str) -> anyhow::Result<Vec<f32>> {
    let path = Path::new(path);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "wav" => load_wav(path),
        "mp3" => load_mp3(path),
        _ => anyhow::bail!("Unsupported audio format: .{ext} (supported: .wav, .mp3)"),
    }
}

fn load_wav(path: &Path) -> anyhow::Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;
    let samples: Vec<i16> = reader.samples::<i16>().collect::<Result<Vec<_>, _>>()?;
    process_samples_i16(&samples, sample_rate, channels)
}

fn load_mp3(path: &Path) -> anyhow::Result<Vec<f32>> {
    use std::fs::File;
    let file = File::open(path)?;
    let mut decoder = minimp3::Decoder::new(file);
    let mut all_samples = Vec::new();
    let mut sample_rate = 0u32;
    let mut channels = 0usize;

    loop {
        match decoder.next_frame() {
            Ok(frame) => {
                sample_rate = frame.sample_rate as u32;
                channels = frame.channels;
                all_samples.extend_from_slice(&frame.data);
            }
            Err(minimp3::Error::Eof) => break,
            Err(e) => anyhow::bail!("MP3 decode error: {e}"),
        }
    }

    if all_samples.is_empty() {
        anyhow::bail!("No audio data found");
    }

    process_samples_i16(&all_samples, sample_rate, channels)
}

fn process_samples_i16(samples: &[i16], sample_rate: u32, channels: usize) -> anyhow::Result<Vec<f32>> {
    use whisper_rs::{convert_integer_to_float_audio, convert_stereo_to_mono_audio};

    let n = samples.len();
    let mut float_samples = vec![0.0f32; n];
    convert_integer_to_float_audio(samples, &mut float_samples)?;

    if channels == 2 {
        let n_mono = float_samples.len() / 2;
        let mut mono = vec![0.0f32; n_mono];
        convert_stereo_to_mono_audio(&float_samples, &mut mono)?;
        float_samples = mono;
    }

    if sample_rate != 16000 {
        float_samples = resample(&float_samples, sample_rate, 16000);
    }

    Ok(float_samples)
}

fn resample(input: &[f32], input_rate: u32, output_rate: u32) -> Vec<f32> {
    if input_rate == output_rate || input.is_empty() {
        return input.to_vec();
    }

    let ratio = output_rate as f64 / input_rate as f64;
    let output_len = (input.len() as f64 * ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 / ratio;
        let src_idx = src_pos as usize;
        let frac = src_pos - src_idx as f64;

        let sample = if src_idx + 1 < input.len() {
            input[src_idx] as f64 * (1.0 - frac) + input[src_idx + 1] as f64 * frac
        } else {
            input[input.len() - 1] as f64
        };

        output.push(sample as f32);
    }

    output
}
