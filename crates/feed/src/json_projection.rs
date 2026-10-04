//! JSON metadata projection validates the entire document while allocating only
//! selected root fields; ignored trajectory arrays must not become Value trees.

use serde::de::{Deserialize, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value};

struct Projection<'a> {
    keys: &'a [&'a str],
    array: Option<(&'a str, &'a [&'a str])>,
}

struct ObjectArray<'a>(&'a [&'a str]);

impl<'de> DeserializeSeed<'de> for ObjectArray<'_> {
    type Value = Vec<Value>;

    fn deserialize<D: serde::Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_seq(self)
    }
}

impl<'de> Visitor<'de> for ObjectArray<'_> {
    type Value = Vec<Value>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("an array of JSON objects")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut access: A) -> Result<Self::Value, A::Error> {
        let mut selected = Vec::new();
        while let Some(object) = access.next_element_seed(Projection {
            keys: self.0,
            array: None,
        })? {
            selected.push(Value::Object(object));
        }
        Ok(selected)
    }
}

struct CheckedDiscard;

impl<'de> Deserialize<'de> for CheckedDiscard {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(Self)
    }
}

impl<'de> Visitor<'de> for CheckedDiscard {
    type Value = Self;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a valid JSON value")
    }

    fn visit_bool<E: serde::de::Error>(self, _: bool) -> Result<Self, E> {
        Ok(self)
    }
    fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<Self, E> {
        Ok(self)
    }
    fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<Self, E> {
        Ok(self)
    }
    fn visit_f64<E: serde::de::Error>(self, _: f64) -> Result<Self, E> {
        Ok(self)
    }
    fn visit_str<E: serde::de::Error>(self, _: &str) -> Result<Self, E> {
        Ok(self)
    }
    fn visit_unit<E: serde::de::Error>(self) -> Result<Self, E> {
        Ok(self)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut access: A) -> Result<Self, A::Error> {
        while access.next_element::<Self>()?.is_some() {}
        Ok(self)
    }

    fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> Result<Self, A::Error> {
        while access.next_key::<Self>()?.is_some() {
            access.next_value::<Self>()?;
        }
        Ok(self)
    }
}

impl<'de> DeserializeSeed<'de> for Projection<'_> {
    type Value = Map<String, Value>;

    fn deserialize<D: serde::Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for Projection<'_> {
    type Value = Map<String, Value>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON object")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> Result<Self::Value, A::Error> {
        let mut selected = Map::new();
        while let Some(key) = access.next_key::<String>()? {
            if let Some((_, fields)) = self.array.filter(|(name, _)| *name == key) {
                selected.insert(
                    key,
                    Value::Array(access.next_value_seed(ObjectArray(fields))?),
                );
            } else if self.keys.contains(&key.as_str()) {
                selected.insert(key, access.next_value::<Value>()?);
            } else {
                // IgnoredAny bypasses number range and nesting checks; validating
                // discarded values must retain full Value rejection without a tree.
                access.next_value::<CheckedDiscard>()?;
            }
        }
        Ok(selected)
    }
}

/// Preserve serde_json's last-key-wins semantics and reject malformed skipped
/// values and trailing bytes; projection is not permission to skip validation.
pub fn object_fields(bytes: &[u8], keys: &[&str]) -> Result<Map<String, Value>, serde_json::Error> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let selected = Projection { keys, array: None }.deserialize(&mut deserializer)?;
    deserializer.end()?;
    Ok(selected)
}

/// Retain only selected fields in a root object array; complete JSON validation
/// remains mandatory so malformed discarded row data cannot pass a metadata gate.
pub fn object_array_fields(
    bytes: &[u8],
    array: &str,
    keys: &[&str],
) -> Result<Map<String, Value>, serde_json::Error> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let selected = Projection {
        keys: &[],
        array: Some((array, keys)),
    }
    .deserialize(&mut deserializer)?;
    deserializer.end()?;
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::{object_array_fields, object_fields};
    use serde_json::json;

    #[test]
    fn projects_each_game_wall_without_admitting_bad_discarded_rows() {
        let fields = object_array_fields(
            br#"{"games":[{"wall_id":"old","rows":[1,2],"wall_id":"w0"},{"wall_id":"w1","rows":[{"x":true}]}],"row_count":3}"#,
            "games", &["wall_id"],
        ).unwrap();
        assert_eq!(fields["games"], json!([{"wall_id":"w0"},{"wall_id":"w1"}]));
        assert!(
            object_array_fields(
                br#"{"games":[{"wall_id":"w0","rows":[1e400]}]}"#,
                "games",
                &["wall_id"]
            )
            .is_err()
        );
    }

    #[test]
    fn selects_metadata_without_changing_nested_values_or_duplicate_precedence() {
        let bytes = br#"{"rows":1,"games":[{"text":"escaped \" ] }","nested":[true,null,{"x":2}]}],"event_ref":{"digest":"d","locator":"l"},"rows":2}"#;
        let fields = object_fields(bytes, &["rows", "event_ref"]).unwrap();
        assert_eq!(fields.len(), 2);
        assert_eq!(fields["rows"], json!(2));
        assert_eq!(fields["event_ref"], json!({"digest":"d","locator":"l"}));
    }

    #[test]
    fn malformed_ignored_values_and_trailing_documents_fail() {
        for bytes in [
            br#"{"rows":2,"games":[true,]}"#.as_slice(),
            br#"{"rows":2} {}"#.as_slice(),
            br#"[1,2]"#.as_slice(),
            br#"{"games":"\uZZZZ","rows":2}"#.as_slice(),
            br#"{"rows":2,"games":[1e400]}"#.as_slice(),
            br#"{"rows":2,"games":["\uD800"]}"#.as_slice(),
        ] {
            assert!(object_fields(bytes, &["rows"]).is_err());
        }
        let nested = format!(
            "{{\"rows\":2,\"games\":{}0{}}}",
            "[".repeat(129),
            "]".repeat(129)
        );
        assert!(serde_json::from_slice::<serde_json::Value>(nested.as_bytes()).is_err());
        assert!(object_fields(nested.as_bytes(), &["rows"]).is_err());
    }
}
