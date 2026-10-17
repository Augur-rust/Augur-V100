use crate::FormatError;

pub trait ReferenceDecode {
    type Elem;

    fn decoded_len(&self) -> usize;
    fn decode_ref(&self, dst: &mut [Self::Elem]) -> Result<(), FormatError>;
}
