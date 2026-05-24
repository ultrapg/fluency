use crate::settings::AudioSettings;

pub fn preprocess(samples: &[f32], settings: &AudioSettings) -> Vec<f32> {
    if samples.is_empty() {
        return samples.to_vec();
    }

    let mut out = samples.to_vec();

    if settings.normalize {
        normalize(&mut out);
    }

    if settings.highpass_enabled {
        highpass(&mut out, settings.highpass_cutoff, 16000.0);
    }

    if settings.noise_gate_enabled {
        noise_gate(&mut out, settings.noise_gate_threshold);
    }

    if settings.preemphasis_enabled {
        preemphasis(&mut out, settings.preemphasis_coefficient);
    }

    out
}

fn normalize(samples: &mut [f32]) {
    let peak = samples
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, f32::max);

    if peak > 0.0 && peak < 1.0 {
        let gain = 0.95 / peak;
        for s in samples.iter_mut() {
            *s *= gain;
        }
    }
}

fn highpass(samples: &mut [f32], cutoff_hz: f32, sample_rate: f32) {
    if cutoff_hz <= 0.0 {
        return;
    }

    let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
    let dt = 1.0 / sample_rate;
    let alpha = dt / (rc + dt);

    let mut y = samples[0];
    for s in samples.iter_mut() {
        y += alpha * (*s - y);
    }

    let mut y = samples[0];
    for s in samples.iter_mut() {
        let y_prev = y;
        y = *s - y_prev + alpha * y_prev;
        *s = y;
    }
}

fn noise_gate(samples: &mut [f32], threshold: f32) {
    let window = 256;
    for chunk in samples.chunks_mut(window) {
        let rms = (chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32).sqrt();
        if rms < threshold {
            for s in chunk.iter_mut() {
                *s = 0.0;
            }
        }
    }
}

fn preemphasis(samples: &mut [f32], coeff: f32) {
    if coeff <= 0.0 {
        return;
    }
    let mut prev = samples[0];
    for i in 1..samples.len() {
        let cur = samples[i];
        samples[i] = cur - coeff * prev;
        prev = cur;
    }
}
