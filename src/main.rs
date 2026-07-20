use clap::{Parser, Subcommand};
use fluency::gui_app::FluencyGui;
use fluency::{filter, format, model, record, settings::Settings, transcribe};

#[derive(Parser)]
#[command(name = "fluency", about = "Local speech-to-text dictation powered by Whisper")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Dictate {
        #[arg(short = 'm', long, help = "Whisper model path (defaults to auto-downloaded ggml-tiny.bin)")]
        model: Option<String>,

        #[arg(short = 'l', long, default_value = "auto")]
        language: String,

        #[arg(short = 'c', long)]
        clipboard: bool,

        #[arg(long, default_value = "true", help = "Apply text formatting (punctuation, caps)")]
        format: bool,

        #[arg(long, default_value = "false", help = "Apply bigram LM correction")]
        lm: bool,
    },
    Transcribe {
        input: String,

        #[arg(short = 'm', long, help = "Whisper model path (defaults to auto-downloaded ggml-tiny.bin)")]
        model: Option<String>,

        #[arg(short = 'l', long, default_value = "auto")]
        language: String,

        #[arg(long, default_value = "true", help = "Apply text formatting (punctuation, caps)")]
        format: bool,

        #[arg(long, default_value = "false", help = "Apply bigram LM correction")]
        lm: bool,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        None => run_gui(),
        Some(cmd) => run_cli(cmd),
    }
}

fn run_gui() -> ! {
    use eframe::egui;

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_min_inner_size([400.0, 300.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Fluency - Speech to Text",
        options,
        Box::new(|_cc| Ok(Box::new(FluencyGui::new()))),
    )
    .ok();
    std::process::exit(0);
}

fn run_cli(cmd: Commands) -> anyhow::Result<()> {
    let settings = Settings::load();

    match cmd {
        Commands::Dictate { model, language, clipboard, format: do_format, lm: do_lm } => {
            let model_path = resolve_model(model, &settings)?;
            let lang = if language == "auto" { None } else { Some(language.as_str()) };
            let audio = record::record_from_mic_with_device(settings.input_device_name.as_deref())?;
            let audio = filter::preprocess(&audio, &settings.audio);
            eprintln!("Transcribing...");
            let text = transcribe::transcribe(&model_path, &audio, lang, &settings.processing)?;
            let text = if do_format || do_lm {
                let f = format::Formatter::new()
                    .with_settings(&settings.format)
                    .with_lm_correction(do_lm)
                    .with_llm(&settings.llm);
                f.format(&text)
            } else {
                text
            };
            println!("{}", text);
            if clipboard {
                copy_to_clipboard(&text);
            }
        }
        Commands::Transcribe { input, model, language, format: do_format, lm: do_lm } => {
            let model_path = resolve_model(model, &settings)?;
            let lang = if language == "auto" { None } else { Some(language.as_str()) };
            let audio = record::load_audio_file(&input)?;
            let audio = filter::preprocess(&audio, &settings.audio);
            eprintln!("Transcribing...");
            let text = transcribe::transcribe(&model_path, &audio, lang, &settings.processing)?;
            let text = if do_format || do_lm {
                let f = format::Formatter::new()
                    .with_settings(&settings.format)
                    .with_lm_correction(do_lm)
                    .with_llm(&settings.llm);
                f.format(&text)
            } else {
                text
            };
            println!("{}", text);
        }
    }

    Ok(())
}

fn resolve_model(cli_model: Option<String>, settings: &Settings) -> anyhow::Result<String> {
    let model_path = match cli_model {
        Some(path) => std::path::PathBuf::from(path),
        None => settings.resolved_model_path(),
    };
    model::ensure_downloaded(&model_path)?;
    Ok(model_path.to_string_lossy().into_owned())
}

fn copy_to_clipboard(text: &str) {
    match arboard::Clipboard::new() {
        Ok(mut ctx) => {
            if let Err(e) = ctx.set_text(text) {
                eprintln!("Warning: failed to set clipboard text: {e}");
            } else {
                eprintln!("Copied to clipboard");
            }
        }
        Err(e) => {
            eprintln!("Warning: clipboard not available: {e}");
        }
    }
}
