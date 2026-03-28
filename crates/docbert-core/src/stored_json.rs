use rkyv::{Archive, Deserialize, Serialize};

/// rkyv-friendly representation of a JSON number that preserves whether the
/// original serde_json value was signed, unsigned, or floating-point.
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub enum StoredJsonNumber {
    I64(i64),
    U64(u64),
    F64(f64),
}

/// rkyv-friendly representation of `serde_json::Value`.
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
#[rkyv(serialize_bounds(
    __S: rkyv::ser::Writer + rkyv::ser::Allocator,
    __S::Error: rkyv::rancor::Source,
))]
#[rkyv(deserialize_bounds(__D::Error: rkyv::rancor::Source))]
#[rkyv(bytecheck(bounds(__C: rkyv::validation::ArchiveContext)))]
pub enum StoredJsonValue {
    Null,
    Bool(bool),
    Number(StoredJsonNumber),
    String(String),
    Array(#[rkyv(omit_bounds)] Vec<StoredJsonValue>),
    Object(#[rkyv(omit_bounds)] Vec<(String, StoredJsonValue)>),
}

impl From<serde_json::Value> for StoredJsonValue {
    fn from(value: serde_json::Value) -> Self {
        match value {
            serde_json::Value::Null => Self::Null,
            serde_json::Value::Bool(value) => Self::Bool(value),
            serde_json::Value::Number(value) => {
                if value.is_i64() {
                    Self::Number(StoredJsonNumber::I64(
                        value.as_i64().expect("signed JSON number"),
                    ))
                } else if value.is_u64() {
                    Self::Number(StoredJsonNumber::U64(
                        value.as_u64().expect("unsigned JSON number"),
                    ))
                } else {
                    Self::Number(StoredJsonNumber::F64(
                        value.as_f64().expect(
                            "serde_json numbers are representable as f64",
                        ),
                    ))
                }
            }
            serde_json::Value::String(value) => Self::String(value),
            serde_json::Value::Array(values) => {
                Self::Array(values.into_iter().map(Self::from).collect())
            }
            serde_json::Value::Object(values) => Self::Object(
                values
                    .into_iter()
                    .map(|(key, value)| (key, Self::from(value)))
                    .collect(),
            ),
        }
    }
}

impl From<StoredJsonValue> for serde_json::Value {
    fn from(value: StoredJsonValue) -> Self {
        match value {
            StoredJsonValue::Null => Self::Null,
            StoredJsonValue::Bool(value) => Self::Bool(value),
            StoredJsonValue::Number(value) => Self::Number(value.into()),
            StoredJsonValue::String(value) => Self::String(value),
            StoredJsonValue::Array(values) => {
                Self::Array(values.into_iter().map(Into::into).collect())
            }
            StoredJsonValue::Object(entries) => Self::Object(
                entries
                    .into_iter()
                    .map(|(key, value)| (key, value.into()))
                    .collect(),
            ),
        }
    }
}

impl From<StoredJsonNumber> for serde_json::Number {
    fn from(value: StoredJsonNumber) -> Self {
        match value {
            StoredJsonNumber::I64(value) => Self::from(value),
            StoredJsonNumber::U64(value) => Self::from(value),
            StoredJsonNumber::F64(value) => Self::from_f64(value)
                .expect("stored JSON floats are finite serde_json numbers"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_json_roundtrips_nested_value() {
        let value = serde_json::json!({
            "null": null,
            "bool": true,
            "string": "hello",
            "array": [1, { "nested": [false, "x"] }],
            "object": {
                "answer": 42,
                "fraction": 3.25,
                "unsigned": 18446744073709551615u64
            }
        });

        let stored = StoredJsonValue::from(value.clone());
        let restored = serde_json::Value::from(stored);

        assert_eq!(restored, value);
    }

    #[test]
    fn stored_json_preserves_number_kinds() {
        let signed = StoredJsonValue::from(serde_json::json!(-5));
        let unsigned = StoredJsonValue::from(serde_json::json!(u64::MAX));
        let float = StoredJsonValue::from(serde_json::json!(5.5));

        assert_eq!(signed, StoredJsonValue::Number(StoredJsonNumber::I64(-5)));
        assert_eq!(
            unsigned,
            StoredJsonValue::Number(StoredJsonNumber::U64(u64::MAX))
        );
        assert_eq!(float, StoredJsonValue::Number(StoredJsonNumber::F64(5.5)));

        assert_eq!(serde_json::Value::from(signed), serde_json::json!(-5));
        assert_eq!(
            serde_json::Value::from(unsigned),
            serde_json::json!(u64::MAX)
        );
        assert_eq!(serde_json::Value::from(float), serde_json::json!(5.5));
    }
}
