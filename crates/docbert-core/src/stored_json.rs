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
                // `serde_json::Number` is always exactly one of i64,
                // u64, or f64 by spec. Walk the cascade via
                // `or_else`; the final `unreachable!` is dead code
                // the compiler can't prove out but documents the
                // invariant instead of a silent default.
                let stored = value
                    .as_i64()
                    .map(StoredJsonNumber::I64)
                    .or_else(|| value.as_u64().map(StoredJsonNumber::U64))
                    .or_else(|| value.as_f64().map(StoredJsonNumber::F64))
                    .unwrap_or_else(|| {
                        unreachable!(
                            "serde_json::Number is always i64, u64, or f64",
                        )
                    });
                Self::Number(stored)
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
            // `from_f64` only rejects NaN/±Inf. Stored values come
            // from a previous `as_f64()` (which returns finite f64
            // for every real JSON number), so non-finite floats
            // don't reach this branch for valid roundtrips. Fall
            // back to 0 rather than a panic for the callers who
            // construct `StoredJsonNumber::F64` directly with a
            // non-finite float.
            StoredJsonNumber::F64(value) => serde_json::Number::from_f64(value)
                .unwrap_or_else(|| serde_json::Number::from(0)),
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
