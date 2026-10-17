//! Duplicate-key rejecting JSON for tokenizer/config identities.
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Number, Value};
use std::fmt;
struct Strict(Value);
impl<'de> Deserialize<'de> for Strict {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Strict;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result { f.write_str("JSON with unique object keys") }
            fn visit_bool<E: de::Error>(self, v: bool) -> Result<Strict, E> { Ok(Strict(Value::Bool(v))) }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Strict, E> { Ok(Strict(Value::Number(v.into()))) }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Strict, E> { Ok(Strict(Value::Number(v.into()))) }
            fn visit_f64<E: de::Error>(self, v: f64) -> Result<Strict, E> {
                Ok(Strict(Value::Number(Number::from_f64(v).ok_or_else(|| E::custom("nonfinite JSON"))?)))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<Strict, E> { Ok(Strict(Value::String(v.into()))) }
            fn visit_string<E: de::Error>(self, v: String) -> Result<Strict, E> { Ok(Strict(Value::String(v))) }
            fn visit_unit<E: de::Error>(self) -> Result<Strict, E> { Ok(Strict(Value::Null)) }
            fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Strict, A::Error> {
                let mut values = Vec::new();
                while let Some(v) = a.next_element::<Strict>()? { values.push(v.0); }
                Ok(Strict(Value::Array(values)))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<Strict, A::Error> {
                let mut values = Map::new();
                while let Some((k, v)) = a.next_entry::<String, Strict>()? {
                    if values.insert(k.clone(), v.0).is_some() { return Err(de::Error::custom(format!("duplicate JSON key: {k}"))); }
                }
                Ok(Strict(Value::Object(values)))
            }
        }
        d.deserialize_any(V)
    }
}
pub(super) fn parse(bytes: &[u8]) -> crate::Result<Value> {
    Ok(serde_json::from_slice::<Strict>(bytes)?.0)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn nested_duplicates_and_trailing_data_fail() {
        for s in [r#"{"x":{"a":1,"a":2}}"#, r#"{"a":1}{}"#] { assert!(parse(s.as_bytes()).is_err()); }
        assert_eq!(parse(br#"{"a":[true,null,1.25,"x"]}"#).unwrap()["a"][2],1.25);
    }
}
