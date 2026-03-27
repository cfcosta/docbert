use serde::{Deserialize, Deserializer, Serialize};

use crate::error::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub title: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub messages: Vec<ChatMessage>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub id: String,
    pub role: ChatRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actor: Option<ChatActor>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parts: Vec<ChatPart>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sources: Option<Vec<ChatSource>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatActor {
    Parent,
    Subagent {
        id: String,
        collection: String,
        path: String,
        status: ChatSubagentStatus,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChatSubagentStatus {
    Queued,
    Running,
    Done,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatPart {
    Text {
        text: String,
    },
    Thinking {
        text: String,
    },
    ToolCall {
        name: String,
        args: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        result: Option<String>,
        #[serde(default)]
        is_error: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatSource {
    pub collection: String,
    pub path: String,
    pub title: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyChatToolCall {
    name: String,
    args: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<String>,
    #[serde(default)]
    is_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum LegacyChatContentPart {
    Text { text: String },
    Thinking { text: String },
}

#[derive(Debug, Deserialize)]
struct RawChatMessage {
    id: String,
    role: ChatRole,
    #[serde(default)]
    actor: Option<ChatActor>,
    #[serde(default)]
    parts: Option<Vec<ChatPart>>,
    #[serde(default)]
    content: String,
    #[serde(default)]
    sources: Option<Vec<ChatSource>>,
    #[serde(default)]
    tool_calls: Option<Vec<LegacyChatToolCall>>,
    #[serde(default)]
    content_parts: Option<Vec<LegacyChatContentPart>>,
}

impl<'de> Deserialize<'de> for ChatMessage {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawChatMessage::deserialize(deserializer)?;
        Ok(Self {
            id: raw.id,
            role: raw.role,
            actor: Some(raw.actor.unwrap_or(ChatActor::Parent)),
            parts: raw.parts.unwrap_or_else(|| {
                legacy_parts(
                    &raw.content,
                    raw.content_parts.unwrap_or_default(),
                    raw.tool_calls.unwrap_or_default(),
                )
            }),
            sources: raw.sources,
        })
    }
}

fn legacy_parts(
    content: &str,
    content_parts: Vec<LegacyChatContentPart>,
    tool_calls: Vec<LegacyChatToolCall>,
) -> Vec<ChatPart> {
    let mut parts = Vec::new();

    if !content_parts.is_empty() {
        parts.extend(content_parts.into_iter().map(|part| match part {
            LegacyChatContentPart::Text { text } => ChatPart::Text { text },
            LegacyChatContentPart::Thinking { text } => {
                ChatPart::Thinking { text }
            }
        }));
    } else if !content.is_empty() {
        parts.push(ChatPart::Text {
            text: content.to_string(),
        });
    }

    parts.extend(tool_calls.into_iter().map(|call| ChatPart::ToolCall {
        name: call.name,
        args: call.args,
        result: call.result,
        is_error: call.is_error,
    }));

    parts
}

impl Conversation {
    pub fn serialize(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(self)?)
    }

    pub fn deserialize(data: &[u8]) -> Result<Self> {
        Ok(serde_json::from_slice(data)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_parent_message_roundtrips() {
        let conv = Conversation {
            id: "conv-1".to_string(),
            title: "Chat".to_string(),
            created_at: 1,
            updated_at: 2,
            messages: vec![ChatMessage {
                id: "msg-1".to_string(),
                role: ChatRole::Assistant,
                actor: Some(ChatActor::Parent),
                parts: vec![
                    ChatPart::Thinking {
                        text: "Let me think".to_string(),
                    },
                    ChatPart::Text {
                        text: "Answer".to_string(),
                    },
                    ChatPart::ToolCall {
                        name: "search_hybrid".to_string(),
                        args: serde_json::json!({ "query": "rust" }),
                        result: Some("[]".to_string()),
                        is_error: false,
                    },
                ],
                sources: Some(vec![ChatSource {
                    collection: "notes".to_string(),
                    path: "rust.md".to_string(),
                    title: "Rust".to_string(),
                }]),
            }],
        };

        let bytes = conv.serialize().unwrap();
        let decoded = Conversation::deserialize(&bytes).unwrap();
        let message = &decoded.messages[0];

        assert_eq!(message.actor, Some(ChatActor::Parent));
        assert_eq!(message.parts, conv.messages[0].parts);
        assert_eq!(message.sources, conv.messages[0].sources);
    }

    #[test]
    fn new_subagent_message_roundtrips() {
        let conv = Conversation {
            id: "conv-1".to_string(),
            title: "Chat".to_string(),
            created_at: 1,
            updated_at: 2,
            messages: vec![ChatMessage {
                id: "msg-1".to_string(),
                role: ChatRole::Assistant,
                actor: Some(ChatActor::Subagent {
                    id: "sub-1".to_string(),
                    collection: "notes".to_string(),
                    path: "rust.md".to_string(),
                    status: ChatSubagentStatus::Running,
                }),
                parts: vec![ChatPart::Text {
                    text: "Inspecting file".to_string(),
                }],
                sources: None,
            }],
        };

        let bytes = conv.serialize().unwrap();
        let decoded = Conversation::deserialize(&bytes).unwrap();
        let message = &decoded.messages[0];

        assert_eq!(message.actor, conv.messages[0].actor);
        assert_eq!(message.parts, conv.messages[0].parts);
    }

    #[test]
    fn legacy_payload_migrates_to_parts_and_parent_actor() {
        let data = serde_json::json!({
            "id": "conv-1",
            "title": "Legacy",
            "created_at": 1,
            "updated_at": 2,
            "messages": [
                {
                    "id": "msg-1",
                    "role": "assistant",
                    "content": "",
                    "content_parts": [
                        { "type": "thinking", "text": "Planning" },
                        { "type": "text", "text": "Answer" }
                    ],
                    "tool_calls": [
                        {
                            "name": "search_hybrid",
                            "args": { "query": "rust" },
                            "result": "[]",
                            "is_error": false
                        }
                    ],
                    "sources": [
                        {
                            "collection": "notes",
                            "path": "rust.md",
                            "title": "Rust"
                        }
                    ]
                }
            ]
        });

        let decoded = Conversation::deserialize(
            serde_json::to_string(&data).unwrap().as_bytes(),
        )
        .unwrap();
        let message = &decoded.messages[0];

        assert_eq!(message.actor, Some(ChatActor::Parent));
        assert_eq!(
            message.parts,
            vec![
                ChatPart::Thinking {
                    text: "Planning".to_string(),
                },
                ChatPart::Text {
                    text: "Answer".to_string(),
                },
                ChatPart::ToolCall {
                    name: "search_hybrid".to_string(),
                    args: serde_json::json!({ "query": "rust" }),
                    result: Some("[]".to_string()),
                    is_error: false,
                },
            ]
        );
        assert_eq!(
            message.sources,
            Some(vec![ChatSource {
                collection: "notes".to_string(),
                path: "rust.md".to_string(),
                title: "Rust".to_string(),
            }])
        );
    }
}
