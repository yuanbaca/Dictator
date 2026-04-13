use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormatType {
    None,
    LightCleanUp,
    CleanUp,
    StrictCleanUp,
    CasualEmail,
    ProfessionalEmail,
    FormalLetter,
    BulletSummary,
    MeetingNotes,
    Documentation,
    Message,
}

/// Chat template format — detected from the model filename.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatFormat {
    /// Phi-3: <|system|>...<|end|> <|user|>...<|end|> <|assistant|>
    Phi3,
    /// Llama 3: <|start_header_id|>system<|end_header_id|>\n\n...<|eot_id|>
    Llama3,
    /// ChatML: <|im_start|>system\n...<|im_end|> — Mistral, Qwen, Gemma, many others
    ChatML,
}

impl ChatFormat {
    /// Guess the chat format from the model filename.
    pub fn detect(model_path: &str) -> Self {
        let lower = model_path.to_lowercase();
        if lower.contains("llama") {
            ChatFormat::Llama3
        } else if lower.contains("mistral")
            || lower.contains("qwen")
            || lower.contains("gemma")
            || lower.contains("chatml")
        {
            ChatFormat::ChatML
        } else {
            // Default to Phi-3 format
            ChatFormat::Phi3
        }
    }

    /// Stable string key matching the serde serialization and frontend dropdown values.
    pub fn key(&self) -> &'static str {
        match self {
            ChatFormat::Phi3 => "phi3",
            ChatFormat::Llama3 => "llama3",
            ChatFormat::ChatML => "chat_ml",
        }
    }

    /// Human-readable label for this format.
    pub fn label(&self) -> &'static str {
        match self {
            ChatFormat::Phi3 => "Phi-3",
            ChatFormat::Llama3 => "Llama 3",
            ChatFormat::ChatML => "ChatML (Mistral/Qwen)",
        }
    }
}

impl FormatType {
    pub fn all() -> &'static [FormatType] {
        &[
            FormatType::None,
            FormatType::LightCleanUp,
            FormatType::CleanUp,
            FormatType::StrictCleanUp,
            FormatType::CasualEmail,
            FormatType::ProfessionalEmail,
            FormatType::FormalLetter,
            FormatType::BulletSummary,
            FormatType::MeetingNotes,
            FormatType::Documentation,
            FormatType::Message,
        ]
    }

    pub fn label(&self) -> &'static str {
        match self {
            FormatType::None => "None (raw text)",
            FormatType::LightCleanUp => "Light cleanup",
            FormatType::CleanUp => "Clean up",
            FormatType::StrictCleanUp => "Strict cleanup",
            FormatType::CasualEmail => "Casual email",
            FormatType::ProfessionalEmail => "Professional email",
            FormatType::FormalLetter => "Formal letter",
            FormatType::BulletSummary => "Bullet summary",
            FormatType::MeetingNotes => "Meeting notes",
            FormatType::Documentation => "Documentation",
            FormatType::Message => "Message",
        }
    }

    /// Returns the default instruction for this format type.
    ///
    /// Each prompt ends with a strong anti-commentary block because small
    /// local models (Phi-3, etc.) tend to append changelogs or explanations
    /// unless explicitly and repeatedly told not to.
    pub fn default_instruction(&self) -> &'static str {
        match self {
            FormatType::None => "",
            FormatType::LightCleanUp => {
                "Fix only obvious typos, punctuation errors, and capitalization in the \
                 following dictated text. Keep the speaker's exact wording, phrasing, \
                 and style completely intact. Do not restructure sentences, remove filler \
                 words, or change vocabulary.\n\n\
                 IMPORTANT: Your entire response must be the corrected text and nothing else. \
                 Do not list changes you made. Do not explain your edits. \
                 Do not add notes, commentary, or bullet points after the text."
            }
            FormatType::CleanUp => {
                "Rewrite the following dictated text with correct grammar, punctuation, \
                 and capitalization. Remove filler words (um, uh, like, you know). \
                 Keep the same meaning, tone, and length. Do not add information.\n\n\
                 IMPORTANT: Your entire response must be the rewritten text and nothing else. \
                 Do not list changes you made. Do not explain your edits. \
                 Do not add notes, commentary, or bullet points after the text."
            }
            FormatType::StrictCleanUp => {
                "Thoroughly rewrite the following dictated text for maximum clarity and \
                 coherence. Fix all grammar, punctuation, and capitalization. Remove filler \
                 words and false starts. If the speaker started a thought, changed their mind, \
                 and restarted, reconstruct the intended meaning into clean sentences. Assess \
                 the logical flow and reorder or restructure as needed. Preserve the original \
                 meaning and all key details but prioritize readability.\n\n\
                 IMPORTANT: Your entire response must be the rewritten text and nothing else. \
                 Do not list changes you made. Do not explain your edits. \
                 Do not add notes, commentary, or bullet points after the text."
            }
            FormatType::CasualEmail => {
                "Rewrite the following dictated text as a casual, friendly email. \
                 Include a greeting and sign-off but keep the tone relaxed and conversational. \
                 Fix grammar and remove filler words. Keep all original details. \
                 Do not add information that wasn't in the original.\n\n\
                 IMPORTANT: Your entire response must be the email and nothing else. \
                 Do not list changes you made. Do not explain your edits. \
                 Do not add notes, commentary, or bullet points after the email."
            }
            FormatType::ProfessionalEmail => {
                "Rewrite the following dictated text as a professional email. \
                 Include a clear subject line on the first line as 'Subject: ...', \
                 then a formal greeting, well-organized body, and professional sign-off. \
                 Keep all original details. Fix grammar and punctuation. \
                 Do not add information that wasn't in the original.\n\n\
                 IMPORTANT: Your entire response must be the email and nothing else. \
                 Do not list changes you made. Do not explain your edits. \
                 Do not add notes, commentary, or bullet points after the email."
            }
            FormatType::FormalLetter => {
                "Rewrite the following dictated text as a formal letter. \
                 Include the date, a formal salutation, well-structured body paragraphs, \
                 and a formal closing. Use formal language and tone throughout. \
                 Include recipient address block if those details are provided. \
                 Keep all original details. Fix grammar and punctuation. \
                 Do not add information that wasn't in the original.\n\n\
                 IMPORTANT: Your entire response must be the letter and nothing else. \
                 Do not list changes you made. Do not explain your edits. \
                 Do not add notes, commentary, or bullet points after the letter."
            }
            FormatType::BulletSummary => {
                "Summarize the following dictated text as a concise bulleted list. \
                 Extract the key points, decisions, and important details. Use clear, \
                 brief bullet points. Group related items if appropriate. Preserve all \
                 important information from the original.\n\n\
                 IMPORTANT: Your entire response must be the bullet-point summary and nothing else. \
                 Do not explain your edits. Do not add meta-commentary after the summary."
            }
            FormatType::MeetingNotes => {
                "Rewrite the following dictated text as structured meeting notes. \
                 Use bullet points. Extract key decisions, action items, and discussion points. \
                 Group related items together. Keep all details from the original.\n\n\
                 IMPORTANT: Your entire response must be the meeting notes and nothing else. \
                 Do not explain your edits. Do not add meta-commentary after the notes."
            }
            FormatType::Documentation => {
                "Rewrite the following dictated text as clear documentation. \
                 Use sections with headers where appropriate. Use clear, concise \
                 technical language. Organize logically. Keep all information from the original.\n\n\
                 IMPORTANT: Your entire response must be the documentation and nothing else. \
                 Do not explain your edits. Do not add meta-commentary after the documentation."
            }
            FormatType::Message => {
                "Rewrite the following dictated text as a casual chat message. \
                 Fix grammar and remove filler words, but keep it conversational and natural. \
                 Keep it brief. Do not add information.\n\n\
                 IMPORTANT: Your entire response must be the message and nothing else. \
                 Do not list changes you made. Do not explain your edits. \
                 Do not add notes, commentary, or bullet points after the message."
            }
        }
    }

    /// Build the full prompt using the appropriate chat template format.
    /// If `custom_instruction` is provided, it overrides the default.
    pub fn build_prompt(&self, raw_text: &str, format: ChatFormat) -> String {
        self.build_prompt_custom(raw_text, format, None)
    }

    /// Build prompt with an optional custom instruction override.
    pub fn build_prompt_custom(
        &self,
        raw_text: &str,
        format: ChatFormat,
        custom_instruction: Option<&str>,
    ) -> String {
        if *self == FormatType::None {
            return raw_text.to_string();
        }

        let instruction = custom_instruction.unwrap_or_else(|| self.default_instruction());

        match format {
            ChatFormat::Phi3 => {
                format!(
                    "<|system|>\n{instruction}\n<|end|>\n\
                     <|user|>\n{raw_text}\n<|end|>\n\
                     <|assistant|>\n"
                )
            }
            ChatFormat::Llama3 => {
                format!(
                    "<|begin_of_text|>\
                     <|start_header_id|>system<|end_header_id|>\n\n\
                     {instruction}<|eot_id|>\
                     <|start_header_id|>user<|end_header_id|>\n\n\
                     {raw_text}<|eot_id|>\
                     <|start_header_id|>assistant<|end_header_id|>\n\n"
                )
            }
            ChatFormat::ChatML => {
                format!(
                    "<|im_start|>system\n{instruction}<|im_end|>\n\
                     <|im_start|>user\n{raw_text}<|im_end|>\n\
                     <|im_start|>assistant\n"
                )
            }
        }
    }

    /// Convert to the snake_case key used for storage (matches serde serialization).
    pub fn key(&self) -> &'static str {
        match self {
            FormatType::None => "none",
            FormatType::LightCleanUp => "light_clean_up",
            FormatType::CleanUp => "clean_up",
            FormatType::StrictCleanUp => "strict_clean_up",
            FormatType::CasualEmail => "casual_email",
            FormatType::ProfessionalEmail => "professional_email",
            FormatType::FormalLetter => "formal_letter",
            FormatType::BulletSummary => "bullet_summary",
            FormatType::MeetingNotes => "meeting_notes",
            FormatType::Documentation => "documentation",
            FormatType::Message => "message",
        }
    }
}

// ── Custom template persistence ────────────────────────────────────────

use std::collections::HashMap;
use std::path::PathBuf;

fn custom_templates_path() -> PathBuf {
    crate::model_manager::models_dir().join("custom_templates.json")
}

pub fn load_custom_templates() -> HashMap<String, String> {
    let path = custom_templates_path();
    if let Ok(data) = std::fs::read_to_string(&path) {
        serde_json::from_str(&data).unwrap_or_default()
    } else {
        HashMap::new()
    }
}

pub fn save_custom_templates(templates: &HashMap<String, String>) {
    let path = custom_templates_path();
    if let Ok(json) = serde_json::to_string_pretty(templates) {
        let _ = std::fs::write(&path, json);
    }
}
