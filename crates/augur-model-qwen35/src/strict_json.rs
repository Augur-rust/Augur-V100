use crate::Qwen35Error;
use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use std::collections::BTreeMap;
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum StrictValue {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<Self>),
    Object(BTreeMap<String, Self>),
}

impl StrictValue {
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, Qwen35Error> {
        let mut deserializer = serde_json::Deserializer::from_slice(bytes);
        let value = Self::deserialize(&mut deserializer)
            .map_err(|error| Qwen35Error::Json(error.to_string()))?;
        deserializer
            .end()
            .map_err(|error| Qwen35Error::Json(error.to_string()))?;
        Ok(value)
    }

    pub(crate) fn object(
        &self,
        field: impl Into<String>,
    ) -> Result<&BTreeMap<String, Self>, Qwen35Error> {
        match self {
            Self::Object(value) => Ok(value),
            _ => Err(Qwen35Error::invalid(field, "expected object")),
        }
    }

    pub(crate) fn array(&self, field: impl Into<String>) -> Result<&[Self], Qwen35Error> {
        match self {
            Self::Array(value) => Ok(value),
            _ => Err(Qwen35Error::invalid(field, "expected array")),
        }
    }

    pub(crate) fn string(&self, field: impl Into<String>) -> Result<&str, Qwen35Error> {
        match self {
            Self::String(value) => Ok(value),
            _ => Err(Qwen35Error::invalid(field, "expected string")),
        }
    }

    pub(crate) fn bool(&self, field: impl Into<String>) -> Result<bool, Qwen35Error> {
        match self {
            Self::Bool(value) => Ok(*value),
            _ => Err(Qwen35Error::invalid(field, "expected boolean")),
        }
    }

    pub(crate) fn u64(&self, field: impl Into<String>) -> Result<u64, Qwen35Error> {
        match self {
            Self::Number(value) => value
                .as_u64()
                .ok_or_else(|| Qwen35Error::invalid(field, "expected nonnegative integer")),
            _ => Err(Qwen35Error::invalid(field, "expected nonnegative integer")),
        }
    }

    pub(crate) fn usize(&self, field: impl Into<String>) -> Result<usize, Qwen35Error> {
        usize::try_from(self.u64(field)?).map_err(|_| Qwen35Error::SizeOverflow)
    }

    pub(crate) fn u32(&self, field: impl Into<String>) -> Result<u32, Qwen35Error> {
        let field = field.into();
        let value = self.u64(field.clone())?;
        u32::try_from(value).map_err(|_| Qwen35Error::invalid(field, "exceeds u32"))
    }

    pub(crate) fn f64(&self, field: impl Into<String>) -> Result<f64, Qwen35Error> {
        let field = field.into();
        let value = match self {
            Self::Number(value) => value
                .as_f64()
                .or_else(|| value.to_string().parse::<f64>().ok())
                .ok_or_else(|| Qwen35Error::invalid(field.clone(), "expected finite number"))?,
            _ => return Err(Qwen35Error::invalid(field, "expected finite number")),
        };
        if value.is_finite() {
            Ok(value)
        } else {
            Err(Qwen35Error::invalid(field, "expected finite number"))
        }
    }
}

impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct StrictVisitor;

        impl<'de> Visitor<'de> for StrictVisitor {
            type Value = StrictValue;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON value without duplicate object keys")
            }

            fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
                Ok(StrictValue::Bool(value))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
                Ok(StrictValue::Number(value.into()))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
                Ok(StrictValue::Number(value.into()))
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                serde_json::Number::from_f64(value)
                    .map(StrictValue::Number)
                    .ok_or_else(|| E::custom("non-finite JSON number"))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(StrictValue::String(value.to_owned()))
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
                Ok(StrictValue::String(value))
            }

            fn visit_none<E>(self) -> Result<Self::Value, E> {
                Ok(StrictValue::Null)
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(StrictValue::Null)
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut values = Vec::new();
                while let Some(value) = sequence.next_element()? {
                    values.push(value);
                }
                Ok(StrictValue::Array(values))
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut values = BTreeMap::new();
                while let Some((key, value)) = map.next_entry::<String, StrictValue>()? {
                    // With serde_json's `arbitrary_precision` feature, decimal
                    // numbers are delivered to custom visitors as a private
                    // one-entry map instead of `visit_f64`. Recover that
                    // representation without weakening ordinary object parsing.
                    if key == "$serde_json::private::Number" {
                        let StrictValue::String(raw) = value else {
                            return Err(serde::de::Error::custom(
                                "invalid arbitrary-precision JSON number",
                            ));
                        };
                        if !values.is_empty() {
                            return Err(serde::de::Error::custom(
                                "invalid arbitrary-precision JSON number object",
                            ));
                        }
                        let number = raw.parse::<serde_json::Number>().map_err(|_| {
                            serde::de::Error::custom("invalid arbitrary-precision JSON number")
                        })?;
                        return Ok(StrictValue::Number(number));
                    }
                    if values.insert(key.clone(), value).is_some() {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate JSON key: {key}"
                        )));
                    }
                }
                Ok(StrictValue::Object(values))
            }
        }

        deserializer.deserialize_any(StrictVisitor)
    }
}

pub(crate) fn required(
    object: &BTreeMap<String, StrictValue>,
    field: impl Into<String>,
) -> Result<&StrictValue, Qwen35Error> {
    let field = field.into();
    object.get(&field).ok_or(Qwen35Error::MissingField(field))
}
