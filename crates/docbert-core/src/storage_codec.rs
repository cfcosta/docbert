use rkyv::{
    Archive,
    Deserialize,
    Serialize,
    api::high::{HighSerializer, HighValidator, to_bytes},
    bytecheck::CheckBytes,
    rancor::Error as RkyvError,
    ser::allocator::ArenaHandle,
    util::AlignedVec,
};

use crate::Result;

/// Encode a value to rkyv bytes using the high-level checked serializer.
pub fn encode_bytes(
    value: &impl for<'a> Serialize<
        HighSerializer<AlignedVec, ArenaHandle<'a>, RkyvError>,
    >,
) -> Result<Vec<u8>> {
    Ok(to_bytes::<RkyvError>(value)?.into_vec())
}

/// Decode a value from rkyv bytes using the high-level checked deserializer.
pub fn decode_bytes<T>(bytes: &[u8]) -> Result<T>
where
    T: Archive,
    T::Archived: for<'a> CheckBytes<HighValidator<'a, RkyvError>>
        + Deserialize<T, rkyv::api::high::HighDeserializer<RkyvError>>,
{
    Ok(rkyv::from_bytes::<T, RkyvError>(bytes)?)
}

#[cfg(test)]
mod tests {
    use super::{decode_bytes, encode_bytes};
    use crate::stored_json::StoredJsonValue;

    #[test]
    fn storage_codec_roundtrips_string() {
        let value = "hello storage".to_string();

        let bytes = encode_bytes(&value).unwrap();
        let decoded: String = decode_bytes(&bytes).unwrap();

        assert_eq!(decoded, value);
    }

    #[test]
    fn storage_codec_roundtrips_stored_json_value() {
        let value = StoredJsonValue::from(serde_json::json!({
            "message": "hello",
            "items": [1, true, null, { "nested": 4.5 }]
        }));

        let bytes = encode_bytes(&value).unwrap();
        let decoded: StoredJsonValue = decode_bytes(&bytes).unwrap();

        assert_eq!(decoded, value);
    }
}
