use serde::{Deserialize, Serialize};

use crate::{
    error::Result,
    storage_codec::{decode_bytes, encode_bytes},
    stored_json::StoredJsonValue,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub title: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub messages: Vec<ChatMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub id: String,
    pub role: ChatRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actor: Option<ChatActor>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parts: Vec<ChatPart>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
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

#[derive(
    Debug, Clone, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
struct StoredConversation {
    id: String,
    title: String,
    created_at: u64,
    updated_at: u64,
    messages: Vec<StoredChatMessage>,
}

#[derive(
    Debug, Clone, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
struct StoredChatMessage {
    id: String,
    role: StoredChatRole,
    actor: Option<StoredChatActor>,
    parts: Vec<StoredChatPart>,
    sources: Option<Vec<StoredChatSource>>,
}

#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
enum StoredChatRole {
    User,
    Assistant,
}

#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
enum StoredChatActor {
    Parent,
    Subagent {
        id: String,
        collection: String,
        path: String,
        status: StoredChatSubagentStatus,
    },
}

#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
enum StoredChatSubagentStatus {
    Queued,
    Running,
    Done,
    Error,
}

#[derive(
    Debug, Clone, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
enum StoredChatPart {
    Text {
        text: String,
    },
    Thinking {
        text: String,
    },
    ToolCall {
        name: String,
        args: StoredJsonValue,
        result: Option<String>,
        is_error: bool,
    },
}

#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
struct StoredChatSource {
    collection: String,
    path: String,
    title: String,
}

impl Conversation {
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let stored = StoredConversation::from(self);
        encode_bytes(&stored)
    }

    /// Decode a stored conversation record. Records are rkyv-encoded;
    /// bytes in any other format fail to decode.
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        Ok(Self::from(decode_bytes::<StoredConversation>(data)?))
    }
}

impl From<&Conversation> for StoredConversation {
    fn from(value: &Conversation) -> Self {
        Self {
            id: value.id.clone(),
            title: value.title.clone(),
            created_at: value.created_at,
            updated_at: value.updated_at,
            messages: value
                .messages
                .iter()
                .map(StoredChatMessage::from)
                .collect(),
        }
    }
}

impl From<StoredConversation> for Conversation {
    fn from(value: StoredConversation) -> Self {
        Self {
            id: value.id,
            title: value.title,
            created_at: value.created_at,
            updated_at: value.updated_at,
            messages: value
                .messages
                .into_iter()
                .map(ChatMessage::from)
                .collect(),
        }
    }
}

impl From<&ChatMessage> for StoredChatMessage {
    fn from(value: &ChatMessage) -> Self {
        Self {
            id: value.id.clone(),
            role: StoredChatRole::from(&value.role),
            actor: value.actor.as_ref().map(StoredChatActor::from),
            parts: value.parts.iter().map(StoredChatPart::from).collect(),
            sources: value.sources.as_ref().map(|sources| {
                sources.iter().map(StoredChatSource::from).collect()
            }),
        }
    }
}

impl From<StoredChatMessage> for ChatMessage {
    fn from(value: StoredChatMessage) -> Self {
        Self {
            id: value.id,
            role: ChatRole::from(value.role),
            actor: value.actor.map(ChatActor::from),
            parts: value.parts.into_iter().map(ChatPart::from).collect(),
            sources: value.sources.map(|sources| {
                sources.into_iter().map(ChatSource::from).collect()
            }),
        }
    }
}

impl From<&ChatRole> for StoredChatRole {
    fn from(value: &ChatRole) -> Self {
        match value {
            ChatRole::User => Self::User,
            ChatRole::Assistant => Self::Assistant,
        }
    }
}

impl From<StoredChatRole> for ChatRole {
    fn from(value: StoredChatRole) -> Self {
        match value {
            StoredChatRole::User => Self::User,
            StoredChatRole::Assistant => Self::Assistant,
        }
    }
}

impl From<&ChatActor> for StoredChatActor {
    fn from(value: &ChatActor) -> Self {
        match value {
            ChatActor::Parent => Self::Parent,
            ChatActor::Subagent {
                id,
                collection,
                path,
                status,
            } => Self::Subagent {
                id: id.clone(),
                collection: collection.clone(),
                path: path.clone(),
                status: StoredChatSubagentStatus::from(status),
            },
        }
    }
}

impl From<StoredChatActor> for ChatActor {
    fn from(value: StoredChatActor) -> Self {
        match value {
            StoredChatActor::Parent => Self::Parent,
            StoredChatActor::Subagent {
                id,
                collection,
                path,
                status,
            } => Self::Subagent {
                id,
                collection,
                path,
                status: ChatSubagentStatus::from(status),
            },
        }
    }
}

impl From<&ChatSubagentStatus> for StoredChatSubagentStatus {
    fn from(value: &ChatSubagentStatus) -> Self {
        match value {
            ChatSubagentStatus::Queued => Self::Queued,
            ChatSubagentStatus::Running => Self::Running,
            ChatSubagentStatus::Done => Self::Done,
            ChatSubagentStatus::Error => Self::Error,
        }
    }
}

impl From<StoredChatSubagentStatus> for ChatSubagentStatus {
    fn from(value: StoredChatSubagentStatus) -> Self {
        match value {
            StoredChatSubagentStatus::Queued => Self::Queued,
            StoredChatSubagentStatus::Running => Self::Running,
            StoredChatSubagentStatus::Done => Self::Done,
            StoredChatSubagentStatus::Error => Self::Error,
        }
    }
}

impl From<&ChatPart> for StoredChatPart {
    fn from(value: &ChatPart) -> Self {
        match value {
            ChatPart::Text { text } => Self::Text { text: text.clone() },
            ChatPart::Thinking { text } => {
                Self::Thinking { text: text.clone() }
            }
            ChatPart::ToolCall {
                name,
                args,
                result,
                is_error,
            } => Self::ToolCall {
                name: name.clone(),
                args: StoredJsonValue::from(args.clone()),
                result: result.clone(),
                is_error: *is_error,
            },
        }
    }
}

impl From<StoredChatPart> for ChatPart {
    fn from(value: StoredChatPart) -> Self {
        match value {
            StoredChatPart::Text { text } => Self::Text { text },
            StoredChatPart::Thinking { text } => Self::Thinking { text },
            StoredChatPart::ToolCall {
                name,
                args,
                result,
                is_error,
            } => Self::ToolCall {
                name,
                args: serde_json::Value::from(args),
                result,
                is_error,
            },
        }
    }
}

impl From<&ChatSource> for StoredChatSource {
    fn from(value: &ChatSource) -> Self {
        Self {
            collection: value.collection.clone(),
            path: value.path.clone(),
            title: value.title.clone(),
        }
    }
}

impl From<StoredChatSource> for ChatSource {
    fn from(value: StoredChatSource) -> Self {
        Self {
            collection: value.collection,
            path: value.path,
            title: value.title,
        }
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
    fn deserialize_rejects_non_rkyv_bytes() {
        let json = serde_json::json!({
            "id": "conv-1",
            "title": "Chat",
            "created_at": 1,
            "updated_at": 2,
            "messages": []
        });

        let bytes = serde_json::to_vec(&json).unwrap();
        assert!(Conversation::deserialize(&bytes).is_err());
    }

    #[test]
    fn message_json_accepts_omitted_optional_fields() {
        let json = serde_json::json!({
            "id": "msg-1",
            "role": "user",
        });

        let message: ChatMessage = serde_json::from_value(json).unwrap();

        assert_eq!(message.actor, None);
        assert!(message.parts.is_empty());
        assert!(message.sources.is_none());
    }

    #[test]
    fn tool_call_args_roundtrip_through_stored_json() {
        let conv = Conversation {
            id: "conv-1".to_string(),
            title: "Chat".to_string(),
            created_at: 1,
            updated_at: 2,
            messages: vec![ChatMessage {
                id: "msg-1".to_string(),
                role: ChatRole::Assistant,
                actor: Some(ChatActor::Parent),
                parts: vec![ChatPart::ToolCall {
                    name: "search_hybrid".to_string(),
                    args: serde_json::json!({
                        "query": "rust",
                        "top_k": u64::MAX,
                        "filters": {
                            "tags": ["systems", "lang"],
                            "score": 0.75,
                            "enabled": true,
                            "note": null
                        }
                    }),
                    result: Some("[]".to_string()),
                    is_error: false,
                }],
                sources: None,
            }],
        };

        let bytes = conv.serialize().unwrap();
        let decoded = Conversation::deserialize(&bytes).unwrap();

        assert_eq!(decoded.messages[0].parts, conv.messages[0].parts);
    }

    #[test]
    fn message_json_wire_format_is_id_role_actor_parts_sources() {
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
                        text: "Planning".to_string(),
                    },
                    ChatPart::Text {
                        text: "Answer".to_string(),
                    },
                ],
                sources: None,
            }],
        };

        let json = serde_json::to_value(&conv).unwrap();
        let message = json["messages"][0].as_object().unwrap();

        let mut keys: Vec<&str> = message.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["actor", "id", "parts", "role"]);
    }
}
