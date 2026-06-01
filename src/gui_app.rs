use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::Instant;

use eframe::egui::{self, Button, CentralPanel, Checkbox, ProgressBar, ScrollArea, Slider, TextEdit, TopBottomPanel};
use rfd::FileDialog;

use crate::{
    filter, format, model, record,
    settings::{ModelSize, Settings},
    transcribe,
};

fn lock_shared(shared: &Arc<Mutex<SharedState>>) -> std::sync::MutexGuard<'_, SharedState> {
    match shared.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            eprintln!("Mutex was poisoned, recovering");
            poisoned.into_inner()
        }
    }
}

struct SharedState {
    audio_to_transcribe: Option<Vec<f32>>,
    transcription_result: Option<String>,
    error: Option<String>,
    progress: f32,
}

struct HistoryEntry {
    path: PathBuf,
    timestamp: String,
    preview: String,
}

fn timestamp_for_filename() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let s = secs % 60;
    let m = (secs / 60) % 60;
    let h = (secs / 3600) % 24;
    let d = secs / 86400;
    format!("{d}-{h:02}-{m:02}-{s:02}")
}

fn transcriptions_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("fluency")
        .join("transcriptions")
}

fn scan_history() -> Vec<HistoryEntry> {
    let dir = transcriptions_dir();
    if !dir.exists() {
        return Vec::new();
    }
    let mut entries = Vec::new();
    if let Ok(read) = std::fs::read_dir(&dir) {
        for entry in read.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("txt") {
                continue;
            }
            let name = path.file_stem().and_then(|n| n.to_str()).unwrap_or("").to_string();
            let timestamp = name.strip_prefix("transcription_").unwrap_or(&name).replace('_', " ").replace('-', ":");
            let preview = std::fs::read_to_string(&path)
                .ok()
                .and_then(|s| s.lines().next().map(|l| l.chars().take(80).collect()))
                .unwrap_or_default();
            entries.push(HistoryEntry { path, timestamp, preview });
        }
    }
    entries.sort_by(|a, b| b.path.cmp(&a.path));
    entries
}

pub struct FluencyGui {
    shared: Arc<Mutex<SharedState>>,
    stop_recording: Arc<AtomicBool>,
    is_recording: bool,
    is_transcribing: bool,
    transcribed_text: String,
    settings: Settings,
    show_settings: bool,
    show_history: bool,
    settings_tab: usize,
    status: String,
    available_mics: Vec<(String, String)>,
    recording_start: Option<Instant>,
    history: Vec<HistoryEntry>,
}

impl FluencyGui {
    pub fn new() -> Self {
        let settings = Settings::load();
        let mics = record::list_input_devices();
        let history = scan_history();

        Self {
            shared: Arc::new(Mutex::new(SharedState {
                audio_to_transcribe: None,
                transcription_result: None,
                error: None,
                progress: 0.0,
            })),
            stop_recording: Arc::new(AtomicBool::new(false)),
            is_recording: false,
            is_transcribing: false,
            transcribed_text: String::new(),
            settings,
            show_settings: false,
            show_history: false,
            settings_tab: 0,
            status: "Ready".to_string(),
            available_mics: mics,
            recording_start: None,
            history,
        }
    }

    fn refresh_mic_list(&mut self) {
        self.available_mics = record::list_input_devices();
    }

    fn start_recording(&mut self) {
        self.stop_recording.store(false, Ordering::SeqCst);
        self.is_recording = true;
        self.recording_start = Some(Instant::now());
        self.status = "Recording...".to_string();

        let stop = self.stop_recording.clone();
        let shared = self.shared.clone();
        let audio_settings = self.settings.audio.clone();
        let device_name = self.settings.input_device_name.clone();

        thread::spawn(move || {
            let result = record::record_from_mic_with_stop_and_device(stop, device_name.as_deref())
                .map(|s| {
                    if audio_settings.highpass_enabled
                        || audio_settings.noise_gate_enabled
                        || audio_settings.normalize
                        || audio_settings.preemphasis_enabled
                    {
                        filter::preprocess(&s, &audio_settings)
                    } else {
                        s
                    }
                });
            let mut s = lock_shared(&shared);
            match result {
                Ok(audio) => s.audio_to_transcribe = Some(audio),
                Err(e) => s.error = Some(format!("Recording error: {e}")),
            }
        });
    }

    fn transcribe_audio(&mut self, audio: Vec<f32>) {
        self.is_transcribing = true;
        self.status = "Transcribing...".to_string();
        {
            let mut s = lock_shared(&self.shared);
            s.progress = 0.0;
        }

        let model_path = self.settings.resolved_model_path();
        let use_model_size = self.settings.model_size;
        let language = self.settings.language.clone();
        let processing = self.settings.processing.clone();
        let shared = self.shared.clone();
        let format_settings = self.settings.format.clone();
        let do_auto_save = self.settings.auto_save;

        thread::spawn(move || {
            let resolved = if model_path.exists() {
                Ok(model_path)
            } else {
                model::ensure_downloaded_by_size(use_model_size)
            };

            let result = resolved.and_then(|path| {
                let lang = if language == "auto" {
                    None
                } else {
                    Some(language.as_str())
                };
                let prog_shared = shared.clone();
                let cb = move |p: i32| {
                    let mut s = lock_shared(&prog_shared);
                    s.progress = p as f32 / 100.0;
                };
                transcribe::transcribe_with_progress(&path.to_string_lossy(), &audio, lang, &processing, cb)
            });

            let text = match result {
                Ok(t) => {
                    let f = format::Formatter::new()
                        .with_settings(&format_settings);
                    f.format(&t)
                }
                Err(e) => {
                    let mut s = lock_shared(&shared);
                    s.error = Some(format!("Error: {e}"));
                    return;
                }
            };

            if do_auto_save {
                let dir = transcriptions_dir();
                let _ = std::fs::create_dir_all(&dir);
                let now = timestamp_for_filename();
                let path = dir.join(format!("transcription_{now}.txt"));
                let _ = std::fs::write(&path, &text);
            }

            let mut s = lock_shared(&shared);
            s.transcription_result = Some(text);
        });
    }

    fn load_and_transcribe_file(&mut self, path: String) {
        self.is_transcribing = true;
        self.status = "Loading file...".to_string();

        let shared = self.shared.clone();
        let audio_settings = self.settings.audio.clone();

        thread::spawn(move || {
            let result = record::load_audio_file(&path).map(|s| {
                if audio_settings.highpass_enabled
                    || audio_settings.noise_gate_enabled
                    || audio_settings.normalize
                    || audio_settings.preemphasis_enabled
                {
                    filter::preprocess(&s, &audio_settings)
                } else {
                    s
                }
            });
            let mut s = lock_shared(&shared);
            match result {
                Ok(audio) => s.audio_to_transcribe = Some(audio),
                Err(e) => s.error = Some(format!("Error loading file: {e}")),
            }
        });
    }

    fn word_count(&self) -> usize {
        self.transcribed_text.split_whitespace().count()
    }

    fn char_count(&self) -> usize {
        self.transcribed_text.chars().count()
    }

    fn show_settings_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_settings;
        egui::Window::new("\u{2699}  Settings")
            .open(&mut open)
            .resizable(true)
            .constrain(true)
            .default_size([420.0, 540.0])
            .show(ctx, |ui| {
                egui::TopBottomPanel::top("settings_tabs")
                    .min_height(0.0)
                    .show_inside(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.selectable_value(&mut self.settings_tab, 0, "Audio");
                            ui.selectable_value(&mut self.settings_tab, 1, "Formatting");
                            ui.selectable_value(&mut self.settings_tab, 2, "Model");
                            ui.selectable_value(&mut self.settings_tab, 3, "Processing");
                        });
                    });

                egui::ScrollArea::vertical().show(ui, |ui| {
                    match self.settings_tab {
                        0 => self.show_audio_settings(ui),
                        1 => self.show_format_settings(ui),
                        2 => self.show_model_settings(ui),
                        3 => self.show_processing_settings(ui),
                        _ => {}
                    }
                });
            });
        self.show_settings = open;
    }

    fn show_history_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_history;
        egui::Window::new("\u{1F4C4}  History")
            .open(&mut open)
            .resizable(true)
            .constrain(true)
            .default_size([350.0, 450.0])
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Transcriptions");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Clear All").on_hover_text("Delete all saved transcriptions").clicked() {
                            let dir = transcriptions_dir();
                            let _ = std::fs::remove_dir_all(&dir);
                            self.history.clear();
                        }
                    });
                });
                ui.separator();

                if self.history.is_empty() {
                    ui.add_space(20.0);
                    ui.label("No saved transcriptions yet.");
                    ui.label("Auto-save is on \u{2014} each transcription will appear here.");
                }

                let mut to_delete: Option<usize> = None;
                let mut load_path: Option<PathBuf> = None;

                egui::ScrollArea::vertical().show(ui, |ui| {
                    for (i, entry) in self.history.iter().enumerate() {
                        if i > 0 {
                            ui.separator();
                        }
                        ui.horizontal(|ui| {
                            ui.set_min_height(32.0);
                            let label = format!("{}  {}", entry.timestamp, entry.preview);
                            if ui.selectable_label(false, label).clicked() {
                                load_path = Some(entry.path.clone());
                            }
                            if ui.button("\u{2716}").on_hover_text("Delete this entry").clicked() {
                                to_delete = Some(i);
                            }
                        });
                    }
                });

                if let Some(i) = to_delete {
                    if let Some(entry) = self.history.get(i) {
                        let _ = std::fs::remove_file(&entry.path);
                    }
                    self.history.remove(i);
                }

                if let Some(path) = load_path {
                    if let Ok(content) = std::fs::read_to_string(&path) {
                        self.transcribed_text = content;
                        self.status = format!("Loaded {}", path.file_name().and_then(|n| n.to_str()).unwrap_or(""));
                        self.show_history = false;
                    }
                }
            });
        self.show_history = open;
    }

    fn show_processing_settings(&mut self, ui: &mut egui::Ui) {
        let p = &mut self.settings.processing;

        ui.heading("Sampling Strategy");
        ui.separator();
        ui.horizontal(|ui| {
            ui.selectable_value(&mut p.sampling_strategy, "greedy".to_string(), "Greedy")
                .on_hover_text("Fast mode: always picks the most likely word (requires less thinking). Good enough for simple speech.");
            ui.selectable_value(&mut p.sampling_strategy, "beam_search".to_string(), "Beam Search")
                .on_hover_text("Careful mode: considers several possible word sequences before deciding. Slower but more accurate. (Default)");
        });
        if p.sampling_strategy == "beam_search" {
            ui.horizontal(|ui| {
                ui.add(Slider::new(&mut p.beam_size, 1..=10).text("Beam size"))
                    .on_hover_text("How many possible sentences to consider at once. Higher numbers give better results but take much longer. Default: 5.");
            });
            ui.horizontal(|ui| {
                ui.add(Slider::new(&mut p.patience, -1.0..=1.0).text("Patience"))
                    .on_hover_text("How long to keep searching for better sentences. Leave at -1.0 for automatic. Default: -1.0.");
            });
        } else {
            ui.horizontal(|ui| {
                ui.add(Slider::new(&mut p.best_of, 1..=10).text("Best of"))
                    .on_hover_text("Number of candidate words to try before picking the best one. Higher = better but slower. Default: 5.");
            });
        }

        ui.add_space(12.0);
        ui.heading("Word Choice Behavior");
        ui.separator();

        ui.horizontal(|ui| {
            ui.add(Slider::new(&mut p.temperature, 0.0..=2.0).text("Creativity"))
                .on_hover_text("How predictable vs creative the transcription should be. 0.0 = strict, follow the rules (recommended). Higher values can make unusual word choices.");
        });
        ui.horizontal(|ui| {
            ui.add(Slider::new(&mut p.temperature_inc, 0.0..=1.0).text("Fallback creativity"))
                .on_hover_text("If Whisper is unsure about a word, it can try harder with more creativity. 0.0 means 'give up', higher means 'keep trying'. Default: 0.2.");
        });

        ui.add(Checkbox::new(&mut p.suppress_blank, "Skip silence at start"))
            .on_hover_text("Prevents the transcription from starting with empty space. Leave this on.");
        ui.add(Checkbox::new(&mut p.suppress_nst, "Ignore sounds that aren't speech"))
            .on_hover_text("Filters out non-speech sounds like music thumping, door creaks, or coughs. Leave this on.");

        ui.horizontal(|ui| {
            ui.add(Slider::new(&mut p.no_speech_thold, 0.0..=1.0).text("Silence sensitivity"))
                .on_hover_text("How aggressively to skip sections that sound like silence or noise. Higher = more skipping. Default: 0.6.");
        });
        ui.horizontal(|ui| {
            ui.add(Slider::new(&mut p.length_penalty, -2.0..=2.0).text("Sentence length bias"))
                .on_hover_text("Prefer shorter or longer sentences. Negative = longer sentences, positive = shorter. Default: -1.0 (slightly longer).");
        });
        ui.horizontal(|ui| {
            ui.add(Slider::new(&mut p.entropy_thold, 0.0..=5.0).text("Uncertainty trigger"))
                .on_hover_text("When Whisper becomes very uncertain, it retries with more creativity. Lower value = retries more often. Default: 2.4.");
        });
        ui.horizontal(|ui| {
            ui.add(Slider::new(&mut p.logprob_thold, -5.0..=0.0).text("Confidence floor"))
                .on_hover_text("Minimum confidence level for words. If average confidence drops below this, Whisper retries. Default: -1.0.");
        });

        ui.add_space(12.0);
        ui.heading("Context & Prompt");
        ui.separator();

        ui.label("Writing style example (guides punctuation & tone):");
        ui.add_sized(
            [ui.available_width(), 80.0],
            TextEdit::multiline(&mut p.initial_prompt).desired_rows(5),
        ).on_hover_text("A short example text that teaches Whisper your preferred writing style. Helps it know where to put periods, commas, and question marks. The default is a good starting point.");

        ui.add(Checkbox::new(&mut p.no_context, "Fresh start each time"))
            .on_hover_text("Don't use the previous sentence as context. Turn this on if you notice Whisper repeating old words. Default: off.");
        ui.horizontal(|ui| {
            ui.add(Slider::new(&mut p.n_max_text_ctx, 256..=16384).text("Context memory"))
                .on_hover_text("How much of the previous conversation Whisper remembers. Higher = more context, but uses more memory. Default: 16384.");
        });

        ui.add_space(12.0);
        ui.heading("Processing Speed");
        ui.separator();
        ui.horizontal(|ui| {
            ui.add(Slider::new(&mut p.n_threads, 0..=16).text("CPU cores"))
                .on_hover_text("How many CPU cores to use for transcription. 0 = automatic (recommended). More cores = faster but uses more power.");
        });
        if p.n_threads == 0 {
            ui.label("0 = automatic (let the system decide)");
        }

        ui.add_space(12.0);
        ui.heading("Smart Silence Skipping");
        ui.separator();
        ui.add(Checkbox::new(&mut p.vad_enabled, "Auto-skip silence"))
            .on_hover_text("Automatically detect and skip silence in your recording. Saves time by not transcribing quiet sections.");
        if p.vad_enabled {
            ui.horizontal(|ui| {
                ui.add(Slider::new(&mut p.vad_threshold, 0.0..=1.0).text("Sensitivity"))
                    .on_hover_text("How quiet is 'quiet enough' to skip. Higher = needs more silence before skipping. Default: 0.5.");
            });
            ui.horizontal(|ui| {
                ui.add(Slider::new(&mut p.vad_min_speech_duration_ms, 50..=2000).text("Min talk time (ms)"))
                    .on_hover_text("Shortest bit of talking to keep (in milliseconds). Shorter sounds are treated as noise. Default: 250ms (quarter second).");
            });
            ui.horizontal(|ui| {
                ui.add(Slider::new(&mut p.vad_min_silence_duration_ms, 50..=2000).text("Min pause (ms)"))
                    .on_hover_text("How long a pause must be before splitting into a new sentence. Default: 100ms.");
            });
            ui.horizontal(|ui| {
                ui.add(Slider::new(&mut p.vad_max_speech_duration_s, 1.0..=60.0).text("Max talk time (s)"))
                    .on_hover_text("Longest allowed talking segment in seconds. Longer segments get split up. Default: unlimited.");
            });
            ui.horizontal(|ui| {
                ui.add(Slider::new(&mut p.vad_speech_pad_ms, 0..=500).text("Padding (ms)"))
                    .on_hover_text("Extra audio kept at the start and end of each speech chunk to avoid cutting words off. Default: 30ms.");
            });
            ui.horizontal(|ui| {
                ui.add(Slider::new(&mut p.vad_samples_overlap, 0.0..=1.0).text("Overlap"))
                    .on_hover_text("How much consecutive speech segments should overlap to avoid gaps. Default: 0.1.");
            });
        }

        ui.add_space(12.0);
        ui.heading("Experimental Features");
        ui.separator();
        ui.add(Checkbox::new(&mut p.single_segment, "One-block mode"))
            .on_hover_text("Treat the entire audio as one block. Only useful for very short recordings like voice commands.");
        ui.add(Checkbox::new(&mut p.debug_mode, "Debug output"))
            .on_hover_text("Print technical debug information. Only useful if something isn't working right.");
        ui.horizontal(|ui| {
            ui.add(Slider::new(&mut p.audio_ctx, 0..=2048).text("Audio chunk size"))
                .on_hover_text("Size of audio chunks to process at once. 0 = automatic. Lower = less memory but might miss context.");
        });
        ui.horizontal(|ui| {
            ui.add(Slider::new(&mut p.max_tokens, 0..=64).text("Max words per block"))
                .on_hover_text("Maximum words per output block. 0 = no limit. Lower = shorter blocks.");
        });

        ui.add_space(12.0);
        if ui.button("Reset processing to defaults").on_hover_text("Restore all processing settings to their factory defaults").clicked() {
            self.settings.processing = Default::default();
        }
    }

    fn show_audio_settings(&mut self, ui: &mut egui::Ui) {
        ui.heading("Input Device");
        ui.separator();

        ui.horizontal(|ui| {
            let current = self.settings.input_device_name.clone().unwrap_or_default();
            egui::ComboBox::from_id_salt("mic_device")
                .selected_text(if current.is_empty() { "Default device" } else { &current })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.settings.input_device_name, None, "Default device")
                        .on_hover_text("Use system default microphone");
                    for (name, _) in &self.available_mics {
                        let name_clone = Some(name.clone());
                        ui.selectable_value(&mut self.settings.input_device_name, name_clone, name)
                            .on_hover_text("Select this input device");
                    }
                });
            if ui.button("\u{1F504}").on_hover_text("Refresh device list").clicked() {
                self.refresh_mic_list();
            }
        });

        ui.add_space(12.0);
        ui.heading("Noise Filter");
        ui.separator();

        let a = &mut self.settings.audio;

        ui.add(Checkbox::new(&mut a.normalize, "Normalize volume"))
            .on_hover_text("Boosts quiet audio to a consistent peak level. Helps when speaking at varying distances from the mic.");
        ui.add_space(4.0);

        ui.add(Checkbox::new(&mut a.highpass_enabled, "High-pass filter"))
            .on_hover_text("Removes low-frequency rumble from fans, motors, AC units, and handling noise.");
        if a.highpass_enabled {
            ui.horizontal(|ui| {
                ui.label("Cutoff:");
                ui.add(Slider::new(&mut a.highpass_cutoff, 20.0..=400.0).text("Hz"))
                    .on_hover_text("Frequency cutoff: 20 Hz (minimal) to 400 Hz (aggressive). Start at 80 Hz.");
            });
        }

        ui.add_space(4.0);
        ui.add(Checkbox::new(&mut a.noise_gate_enabled, "Noise gate"))
            .on_hover_text("Silences quiet sections between speech.");
        if a.noise_gate_enabled {
            ui.horizontal(|ui| {
                ui.label("Threshold:");
                ui.add(Slider::new(&mut a.noise_gate_threshold, 0.0..=0.05).text("RMS"))
                    .on_hover_text("Gate sensitivity: 0.001 (very quiet) to 0.05 (loud). Default 0.005.");
            });
        }

        ui.add_space(4.0);
        ui.add(Checkbox::new(&mut a.preemphasis_enabled, "Pre-emphasis (treble boost)"))
            .on_hover_text("Boosts high frequencies to make consonants clearer.");
        if a.preemphasis_enabled {
            ui.horizontal(|ui| {
                ui.label("Amount:");
                ui.add(Slider::new(&mut a.preemphasis_coefficient, 0.8..=0.99).text("coeff"))
                    .on_hover_text("Boost: 0.8 (subtle) to 0.99 (strong). Default 0.97.");
            });
        }

        ui.add_space(12.0);
        if ui.button("Reset audio to defaults").on_hover_text("Restore audio settings to defaults").clicked() {
            self.settings.audio = Default::default();
        }
    }

    fn show_format_settings(&mut self, ui: &mut egui::Ui) {
        ui.heading("Text Formatting");
        ui.separator();

        let f = &mut self.settings.format;

        ui.add(Checkbox::new(&mut f.auto_capitalize, "Auto-capitalize"))
            .on_hover_text("Capitalizes first letter of sentences. Converts 'i' to 'I'.");
        ui.add_space(4.0);

        ui.add(Checkbox::new(&mut f.auto_punctuate, "Auto-punctuate"))
            .on_hover_text("Adds periods and question marks at sentence ends.");
        ui.add_space(4.0);

        ui.add(Checkbox::new(&mut f.remove_fillers, "Remove filler words"))
            .on_hover_text("Strips filler words (um, uh, like) for cleaner output.");
        ui.add_space(4.0);

        if f.remove_fillers {
            ui.label("Filler words (one per line):");
            let mut fillers_text = f.fillers.join("\n");
            let r = ui.add_sized(
                [ui.available_width(), 60.0],
                TextEdit::multiline(&mut fillers_text).desired_rows(4),
            ).on_hover_text("Edit the list of words to strip. One per line.");
            if r.changed() {
                f.fillers = fillers_text.lines()
                    .map(|l: &str| l.trim().to_string())
                    .filter(|l: &String| !l.is_empty())
                    .collect();
            }
            ui.add_space(4.0);
        }

        ui.add(Checkbox::new(&mut f.fix_corrections, "Fix self-corrections"))
            .on_hover_text("Detects '30 or no 50' → '50' patterns.");
        ui.add_space(4.0);

        if f.fix_corrections {
            ui.label("Correction markers (one per line):");
            let mut markers_text = f.correction_markers.join("\n");
            let r = ui.add_sized(
                [ui.available_width(), 60.0],
                TextEdit::multiline(&mut markers_text).desired_rows(4),
            ).on_hover_text("Edit the list of self-correction markers. Text before marker is removed.");
            if r.changed() {
                f.correction_markers = markers_text.lines()
                    .map(|l: &str| l.trim().to_string())
                    .filter(|l: &String| !l.is_empty())
                    .collect();
            }
            ui.add_space(4.0);
        }

        ui.add(Checkbox::new(&mut f.lm_correction, "Bigram LM correction"))
            .on_hover_text("Fixes common homophone errors (their/there, to/too) using a statistical model.");
        ui.add_space(12.0);

        if ui.button("Reset formatting to defaults").on_hover_text("Restore formatting settings to defaults").clicked() {
            self.settings.format = Default::default();
        }
    }

    fn show_model_settings(&mut self, ui: &mut egui::Ui) {
        ui.heading("Whisper Model");
        ui.separator();

        ui.add(Checkbox::new(&mut self.settings.auto_save, "Auto-save transcriptions"))
            .on_hover_text("Automatically saves each transcription to a timestamped file and makes it available in the history panel.");
        ui.add_space(8.0);

        ui.label("Model size:");
        for &size in ModelSize::all() {
            let selected = self.settings.model_size == size;
            let btn = egui::SelectableLabel::new(selected, format!("{} — {}", size.name(), size.description()));
            if ui.add(btn).on_hover_text(format!("Download size: {}. Click to select and auto-download.", size.size_mb())).clicked() {
                self.settings.model_size = size;
                self.settings.model_path.clear();
                let path = model::path_for_size(size);
                if !path.exists() {
                    self.status = format!("Downloading {} model...", size.name());
                }
                thread::spawn(move || {
                    if let Err(e) = model::ensure_downloaded_by_size(size) {
                        eprintln!("Failed to download model: {e}");
                    }
                });
            }
        }

        ui.add_space(8.0);
        ui.separator();
        ui.label("Or use a custom model file:");
        ui.horizontal(|ui| {
            let path_display = if self.settings.model_path.is_empty() {
                "(none, using selected size above)"
            } else {
                &self.settings.model_path
            };
            ui.label(path_display);
            if ui.button("Browse...").on_hover_text("Select a custom GGML Whisper model file (.bin)").clicked() {
                if let Some(path) = FileDialog::new()
                    .add_filter("Whisper model", &["bin"])
                    .pick_file()
                {
                    self.settings.model_path = path.display().to_string();
                }
            }
        });

        ui.add_space(8.0);
        ui.separator();
        ui.heading("Language");
        ui.horizontal(|ui| {
            ui.label("Recognition language:");
            egui::ComboBox::from_id_salt("settings_lang")
                .selected_text(&self.settings.language)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.settings.language, "auto".to_owned(), "Auto detect")
                        .on_hover_text("Let Whisper automatically detect the language");
                    ui.selectable_value(&mut self.settings.language, "en".to_owned(), "English");
                    ui.selectable_value(&mut self.settings.language, "de".to_owned(), "German");
                    ui.selectable_value(&mut self.settings.language, "fr".to_owned(), "French");
                    ui.selectable_value(&mut self.settings.language, "es".to_owned(), "Spanish");
                    ui.selectable_value(&mut self.settings.language, "it".to_owned(), "Italian");
                    ui.selectable_value(&mut self.settings.language, "pt".to_owned(), "Portuguese");
                    ui.selectable_value(&mut self.settings.language, "ja".to_owned(), "Japanese");
                    ui.selectable_value(&mut self.settings.language, "zh".to_owned(), "Chinese");
                    ui.selectable_value(&mut self.settings.language, "ru".to_owned(), "Russian");
                });
        });
    }
}

impl eframe::App for FluencyGui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.is_recording || self.is_transcribing {
            ctx.request_repaint();
        }

        // Keyboard shortcuts
        let (ctrl, shift, enter_pressed, c_pressed, s_pressed, esc_pressed) = ctx.input(|i| {
            (i.modifiers.ctrl, i.modifiers.shift,
             i.key_pressed(egui::Key::Enter),
             i.key_pressed(egui::Key::C),
             i.key_pressed(egui::Key::S),
             i.key_pressed(egui::Key::Escape))
        });
        if enter_pressed && ctrl {
            if self.is_recording {
                self.stop_recording.store(true, Ordering::SeqCst);
            } else if !self.is_transcribing {
                self.start_recording();
            }
        }
        if c_pressed && ctrl && shift {
            self.transcribed_text.clear();
            self.status = "Cleared".to_string();
        } else if c_pressed && ctrl {
            if let Ok(mut clip) = arboard::Clipboard::new() {
                if clip.set_text(&self.transcribed_text).is_ok() {
                    self.status = "Copied".to_string();
                }
            }
        }
        if s_pressed && ctrl {
            if !self.transcribed_text.is_empty() {
                if let Some(path) = FileDialog::new()
                    .add_filter("Text", &["txt"])
                    .set_file_name("transcription.txt")
                    .save_file()
                {
                    let _ = std::fs::write(&path, &self.transcribed_text);
                    self.status = format!("Saved to {}", path.display());
                }
            }
        }
        if esc_pressed {
            if self.is_recording {
                self.stop_recording.store(true, Ordering::SeqCst);
            } else if self.show_settings {
                self.show_settings = false;
            } else if self.show_history {
                self.show_history = false;
            } else if self.show_history {
                self.show_history = false;
            }
        }

        // Floating windows
        if self.show_settings {
            self.show_settings_window(ctx);
        }
        if self.show_history {
            self.show_history_window(ctx);
        }
        let audio_to_transcribe = {
            let mut s = lock_shared(&self.shared);
            let audio = s.audio_to_transcribe.take();
            let text = s.transcription_result.take();
            let err = s.error.take();

            if let Some(text) = text {
                self.transcribed_text = text;
                self.is_transcribing = false;
                self.status = "Done".to_string();
                s.progress = 0.0;
            }
            if let Some(err) = err {
                self.status = err;
                self.is_recording = false;
                self.is_transcribing = false;
                s.progress = 0.0;
            }

            audio
        };
        if let Some(audio) = audio_to_transcribe {
            self.is_recording = false;
            self.recording_start = None;
            self.transcribe_audio(audio);
        }

        TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let lang_display = if self.settings.language == "auto" {
                    "Auto"
                } else {
                    &self.settings.language
                };
                ui.label(format!("Lang: {lang_display}"))
                    .on_hover_text("Recognition language setting");

                ui.separator();

                let noise_on = self.settings.audio.highpass_enabled
                    || self.settings.audio.noise_gate_enabled;
                let mut noise_check = noise_on;
                let r = ui.add_enabled(true, Checkbox::new(&mut noise_check, "Noise filter"));
                if r.changed() {
                    self.settings.audio.highpass_enabled = noise_check;
                    self.settings.audio.noise_gate_enabled = noise_check;
                }
                ui.add(Checkbox::new(&mut self.settings.format.lm_correction, "LM"))
                    .on_hover_text("Bigram LM correction for homophones");

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("\u{1F4C4}  History").on_hover_text("View transcription history").clicked() {
                        self.history = scan_history();
                        self.show_history = !self.show_history;
                    }
                    if ui.button("\u{2699}  Settings").on_hover_text("Open settings").clicked() {
                        self.show_settings = !self.show_settings;
                    }
                });

                ui.menu_button("\u{1F4C1}  File", |ui| {
                    if ui.button("\u{1F3B5}  Open audio file...")
                        .on_hover_text("Transcribe an audio file (WAV/MP3)")
                        .clicked()
                    {
                        if let Some(path) = FileDialog::new()
                            .add_filter("Audio", &["wav", "mp3"])
                            .pick_file()
                        {
                            self.load_and_transcribe_file(path.to_string_lossy().to_string());
                        }
                        ui.close_menu();
                    }
                    if ui.button("\u{1F4BE}  Save transcription...")
                        .on_hover_text("Save to a text file (Ctrl+S)")
                        .clicked()
                    {
                        if let Some(path) = FileDialog::new()
                            .add_filter("Text", &["txt"])
                            .set_file_name("transcription.txt")
                            .save_file()
                        {
                            let _ = std::fs::write(&path, &self.transcribed_text);
                            self.status = format!("Saved to {}", path.display());
                        }
                        ui.close_menu();
                    }
                });
            });
        });

        CentralPanel::default().show(ctx, |ui| {
            ScrollArea::vertical().show(ui, |ui| {
                let resp = ui.add_sized(
                    ui.available_size(),
                    TextEdit::multiline(&mut self.transcribed_text)
                        .hint_text("Transcribed text will appear here...")
                        .desired_rows(20)
                        .font(egui::TextStyle::Monospace),
                );
                if resp.changed() && !self.transcribed_text.is_empty() {
                    self.status = format!(
                        "{} words, {} characters",
                        self.word_count(),
                        self.char_count()
                    );
                }
            });
        });

        TopBottomPanel::bottom("controls").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if self.is_recording {
                    if ui.add(Button::new("\u{23F9}  Stop")).clicked() {
                        self.stop_recording.store(true, Ordering::SeqCst);
                    }
                } else {
                    let btn = Button::new("\u{1F3A4}  Record");
                    if ui.add_enabled(!self.is_transcribing, btn)
                        .on_hover_text("Start recording (Ctrl+Enter)")
                        .clicked()
                    {
                        self.start_recording();
                    }
                }

                if ui
                    .add_enabled(!self.transcribed_text.is_empty() && !self.is_recording, Button::new("\u{1F4CB}  Copy"))
                    .on_hover_text("Copy to clipboard (Ctrl+C)")
                    .clicked()
                {
                    if let Ok(mut clip) = arboard::Clipboard::new() {
                        if clip.set_text(&self.transcribed_text).is_ok() {
                            self.status = "Copied to clipboard".to_string();
                        }
                    }
                }



                if ui.button("Clear")
                    .on_hover_text("Clear transcription (Ctrl+Shift+C)")
                    .clicked()
                {
                    self.transcribed_text.clear();
                    self.status = "Cleared".to_string();
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.is_transcribing {
                        let progress = lock_shared(&self.shared).progress;
                        if progress > 0.0 {
                            ui.add_sized([120.0, 0.0], ProgressBar::new(progress).show_percentage());
                        } else {
                            ui.spinner();
                        }
                    } else if self.is_recording {
                        let elapsed = self.recording_start
                            .map(|t| {
                                let d = t.elapsed();
                                format!("{:02}:{:02}", d.as_secs() / 60, d.as_secs() % 60)
                            })
                            .unwrap_or_default();
                        ui.label(elapsed);
                        ui.spinner();
                    }
                    if !self.transcribed_text.is_empty() && !self.is_recording && !self.is_transcribing {
                        ui.label(format!("{}w {}ch", self.word_count(), self.char_count()));
                    }
                    ui.label(&self.status);
                });
            });
        });

        if !self.show_settings {
            self.settings.save();
        }
    }
}
