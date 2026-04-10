use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormatType {
    None,
    CleanUp,
    Email,
    MeetingNotes,
    Documentation,
    Message,
}

/// Chat template format — detected from the model filename.
#[derive(Debug, Clone, Copy)]
pub enum ChatFormat {
    /// Phi-3: <|system|>...<|end|> <|user|>...<|end|> <|assistant|>
    Phi3,
    /// Llama 3: <|start_header_id|>system<|end_header_id|>\n\n...<|eot_id|>
    Llama3,
}

impl ChatFormat {
    /// Guess the chat format from the model filename.
    pub fn detect(model_path: &str) -> Self {
        let lower = model_path.to_lowercase();
        if lower.contains("llama") {
            ChatFormat::Llama3
        } else {
            // Default to Phi-3 format (also works for many other models)
            ChatFormat::Phi3
        }
    }
}

impl FormatType {
    pub fn all() -> &'static [FormatType] {
        &[
            FormatType::None,
            FormatType::CleanUp,
            FormatType::Email,
            FormatType::MeetingNotes,
            FormatType::Documentation,
            FormatType::Message,
        ]
    }

    pub fn label(&self) -> &'static str {
        match self {
            FormatType::None => "None (raw text)",
            FormatType::CleanUp => "Clean up",
            FormatType::Email => "Email",
            FormatType::MeetingNotes => "Meeting notes",
            FormatType::Documentation => "Documentation",
            FormatType::Message => "Message",
        }
    }

    fn instruction(&self) -> &'static str {
        match self {
            FormatType::None => "",
            FormatType::CleanUp => {
                "Rewrite the following dictated text with correct grammar, punctuation, \
                 and capitalization. Remove filler words (um, uh, like, you know). \
                 Keep the same meaning, tone, and length. Do not add information."
            }
            FormatType::Email => {
                "Rewrite the following dictated text as a professional email. \
                 Include a clear subject line on the first line as 'Subject: ...', \
                 then a greeting, organized body, and sign-off. \
                 Keep all original details. Fix grammar and punctuation. \
                 Do not add information that wasn't in the original."
            }
            FormatType::MeetingNotes => {
                "Rewrite the following dictated text as structured meeting notes. \
                 Use bullet points. Extract key decisions, action items, and discussion points. \
                 Group related items together. Keep all details from the original."
            }
            FormatType::Documentation => {
                "Rewrite the following dictated text as clear documentation. \
                 Use sections with headers where appropriate. Use clear, concise \
                 technical language. Organize logically. Keep all information from the original."
            }
            FormatType::Message => {
                "Rewrite the following dictated text as a casual chat message. \
                 Fix grammar and remove filler words, but keep it conversational and natural. \
                 Keep it brief. Do not add information."
            }
        }
    }

    /// Build the full prompt using the appropriate chat template format.
    pub fn build_prompt(&self, raw_text: &str, format: ChatFormat) -> String {
        if *self == FormatType::None {
            return raw_text.to_string();
        }

        let instruction = self.instruction();

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
        }
    }
}
