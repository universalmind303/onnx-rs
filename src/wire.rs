use crate::error::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WireType {
    Varint,
    Fixed64,
    LengthDelimited,
    Fixed32,
}

pub(crate) struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Cursor { buf, pos: 0 }
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    #[cfg(test)]
    pub fn position(&self) -> usize {
        self.pos
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pos >= self.buf.len()
    }

    #[inline]
    pub fn read_varint(&mut self) -> Result<u64, Error> {
        if self.pos >= self.buf.len() {
            return Err(Error::UnexpectedEof {
                offset: self.pos,
                context: "reading varint",
            });
        }
        let byte = self.buf[self.pos];
        if byte & 0x80 == 0 {
            self.pos += 1;
            return Ok(byte as u64);
        }
        self.read_varint_slow()
    }

    #[cold]
    fn read_varint_slow(&mut self) -> Result<u64, Error> {
        let mut result: u64 = 0;
        let mut shift: u32 = 0;

        loop {
            if self.is_empty() {
                return Err(Error::UnexpectedEof {
                    offset: self.pos,
                    context: "reading varint",
                });
            }

            let byte = self.buf[self.pos];
            self.pos += 1;

            if shift >= 63 && byte > 1 {
                return Err(Error::InvalidVarint { offset: self.pos - 1 });
            }

            result |= ((byte & 0x7F) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(result);
            }

            shift += 7;
            if shift >= 70 {
                return Err(Error::InvalidVarint { offset: self.pos });
            }
        }
    }

    #[inline]
    pub fn read_tag(&mut self) -> Result<(u32, WireType), Error> {
        let varint = self.read_varint()?;
        let wire_type_val = (varint & 0x07) as u8;
        let field_number = (varint >> 3) as u32;
        let wire_type = match wire_type_val {
            0 => WireType::Varint,
            1 => WireType::Fixed64,
            2 => WireType::LengthDelimited,
            5 => WireType::Fixed32,
            _ => {
                return Err(Error::InvalidWireType {
                    value: wire_type_val,
                    offset: self.pos,
                })
            }
        };
        Ok((field_number, wire_type))
    }

    #[inline]
    pub fn read_fixed32(&mut self) -> Result<u32, Error> {
        if self.remaining() < 4 {
            return Err(Error::UnexpectedEof {
                offset: self.pos,
                context: "reading fixed32",
            });
        }
        let bytes: [u8; 4] = self.buf[self.pos..self.pos + 4].try_into().unwrap();
        self.pos += 4;
        Ok(u32::from_le_bytes(bytes))
    }

    #[inline]
    pub fn read_fixed64(&mut self) -> Result<u64, Error> {
        if self.remaining() < 8 {
            return Err(Error::UnexpectedEof {
                offset: self.pos,
                context: "reading fixed64",
            });
        }
        let bytes: [u8; 8] = self.buf[self.pos..self.pos + 8].try_into().unwrap();
        self.pos += 8;
        Ok(u64::from_le_bytes(bytes))
    }

    #[inline]
    pub fn read_bytes(&mut self) -> Result<&'a [u8], Error> {
        let len = self.read_varint()? as usize;
        if self.remaining() < len {
            return Err(Error::TruncatedMessage {
                offset: self.pos,
                expected: len,
                available: self.remaining(),
            });
        }
        let data = &self.buf[self.pos..self.pos + len];
        self.pos += len;
        Ok(data)
    }

    #[inline]
    pub fn read_string(&mut self) -> Result<&'a str, Error> {
        let offset = self.pos;
        let bytes = self.read_bytes()?;
        core::str::from_utf8(bytes).map_err(|_| Error::InvalidUtf8 { offset })
    }

    pub fn read_sub_cursor(&mut self) -> Result<Cursor<'a>, Error> {
        let offset = self.pos;
        let len = self.read_varint()? as usize;
        if self.remaining() < len {
            return Err(Error::TruncatedMessage {
                offset,
                expected: len,
                available: self.remaining(),
            });
        }
        let sub = Cursor {
            buf: &self.buf[self.pos..self.pos + len],
            pos: 0,
        };
        self.pos += len;
        Ok(sub)
    }

    #[inline]
    pub fn read_f32_le(&mut self) -> Result<f32, Error> {
        let bits = self.read_fixed32()?;
        Ok(f32::from_bits(bits))
    }

    #[inline]
    pub fn read_f64_le(&mut self) -> Result<f64, Error> {
        let bits = self.read_fixed64()?;
        Ok(f64::from_bits(bits))
    }

    #[inline]
    pub fn remaining_slice(&self) -> &'a [u8] {
        &self.buf[self.pos..]
    }

    pub fn skip_field(&mut self, wire_type: WireType) -> Result<(), Error> {
        match wire_type {
            WireType::Varint => {
                self.read_varint()?;
            }
            WireType::Fixed64 => {
                if self.remaining() < 8 {
                    return Err(Error::UnexpectedEof {
                        offset: self.pos,
                        context: "skipping fixed64",
                    });
                }
                self.pos += 8;
            }
            WireType::Fixed32 => {
                if self.remaining() < 4 {
                    return Err(Error::UnexpectedEof {
                        offset: self.pos,
                        context: "skipping fixed32",
                    });
                }
                self.pos += 4;
            }
            WireType::LengthDelimited => {
                let len = self.read_varint()? as usize;
                if self.remaining() < len {
                    return Err(Error::TruncatedMessage {
                        offset: self.pos,
                        expected: len,
                        available: self.remaining(),
                    });
                }
                self.pos += len;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Cursor basics ===

    #[test]
    fn test_cursor_empty() {
        let cursor = Cursor::new(&[]);
        assert!(cursor.is_empty());
        assert_eq!(cursor.remaining(), 0);
        assert_eq!(cursor.position(), 0);
    }

    #[test]
    fn test_cursor_position_tracking() {
        let data = [0x08, 0x96, 0x01]; // tag(1, varint) + varint 150
        let mut cursor = Cursor::new(&data);
        assert_eq!(cursor.remaining(), 3);
        assert_eq!(cursor.position(), 0);
        let _ = cursor.read_tag().unwrap();
        assert_eq!(cursor.position(), 1);
        let _ = cursor.read_varint().unwrap();
        assert_eq!(cursor.position(), 3);
        assert!(cursor.is_empty());
    }

    // === Varint decoding ===

    #[test]
    fn test_read_varint_single_byte() {
        let mut cursor = Cursor::new(&[0x00]);
        assert_eq!(cursor.read_varint().unwrap(), 0);

        let mut cursor = Cursor::new(&[0x01]);
        assert_eq!(cursor.read_varint().unwrap(), 1);

        let mut cursor = Cursor::new(&[0x7F]);
        assert_eq!(cursor.read_varint().unwrap(), 127);
    }

    #[test]
    fn test_read_varint_two_bytes() {
        let mut cursor = Cursor::new(&[0x96, 0x01]);
        assert_eq!(cursor.read_varint().unwrap(), 150);
    }

    #[test]
    fn test_read_varint_max_u64() {
        let mut cursor =
            Cursor::new(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01]);
        assert_eq!(cursor.read_varint().unwrap(), u64::MAX);
    }

    #[test]
    fn test_read_varint_300() {
        let mut cursor = Cursor::new(&[0xAC, 0x02]);
        assert_eq!(cursor.read_varint().unwrap(), 300);
    }

    #[test]
    fn test_read_varint_eof() {
        let mut cursor = Cursor::new(&[]);
        assert!(cursor.read_varint().is_err());
    }

    #[test]
    fn test_read_varint_truncated() {
        let mut cursor = Cursor::new(&[0x80]);
        assert!(cursor.read_varint().is_err());
    }

    #[test]
    fn test_read_varint_too_long() {
        let mut cursor = Cursor::new(&[
            0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01,
        ]);
        assert!(cursor.read_varint().is_err());
    }

    // === Tag decoding ===

    #[test]
    fn test_read_tag_field1_varint() {
        let mut cursor = Cursor::new(&[0x08]);
        let (field, wt) = cursor.read_tag().unwrap();
        assert_eq!(field, 1);
        assert_eq!(wt, WireType::Varint);
    }

    #[test]
    fn test_read_tag_field2_length_delimited() {
        let mut cursor = Cursor::new(&[0x12]);
        let (field, wt) = cursor.read_tag().unwrap();
        assert_eq!(field, 2);
        assert_eq!(wt, WireType::LengthDelimited);
    }

    #[test]
    fn test_read_tag_field7_length_delimited() {
        let mut cursor = Cursor::new(&[0x3A]);
        let (field, wt) = cursor.read_tag().unwrap();
        assert_eq!(field, 7);
        assert_eq!(wt, WireType::LengthDelimited);
    }

    #[test]
    fn test_read_tag_field15_varint() {
        let mut cursor = Cursor::new(&[0x78]);
        let (field, wt) = cursor.read_tag().unwrap();
        assert_eq!(field, 15);
        assert_eq!(wt, WireType::Varint);
    }

    #[test]
    fn test_read_tag_field16_varint() {
        let mut cursor = Cursor::new(&[0x80, 0x01]);
        let (field, wt) = cursor.read_tag().unwrap();
        assert_eq!(field, 16);
        assert_eq!(wt, WireType::Varint);
    }

    #[test]
    fn test_read_tag_fixed64() {
        let mut cursor = Cursor::new(&[0x09]);
        let (field, wt) = cursor.read_tag().unwrap();
        assert_eq!(field, 1);
        assert_eq!(wt, WireType::Fixed64);
    }

    #[test]
    fn test_read_tag_fixed32() {
        let mut cursor = Cursor::new(&[0x0D]);
        let (field, wt) = cursor.read_tag().unwrap();
        assert_eq!(field, 1);
        assert_eq!(wt, WireType::Fixed32);
    }

    #[test]
    fn test_read_tag_invalid_wire_type() {
        let mut cursor = Cursor::new(&[0x0B]); // field 1, wire type 3
        assert!(cursor.read_tag().is_err());
    }

    // === Fixed-width reads ===

    #[test]
    fn test_read_fixed32() {
        let mut cursor = Cursor::new(&[0x01, 0x00, 0x00, 0x00]);
        assert_eq!(cursor.read_fixed32().unwrap(), 1);
    }

    #[test]
    fn test_read_fixed32_float_pi() {
        let bytes = std::f32::consts::PI.to_le_bytes();
        let mut cursor = Cursor::new(&bytes);
        let bits = cursor.read_fixed32().unwrap();
        assert_eq!(f32::from_le_bytes(bits.to_le_bytes()), std::f32::consts::PI);
    }

    #[test]
    fn test_read_fixed32_eof() {
        let mut cursor = Cursor::new(&[0x01, 0x02]);
        assert!(cursor.read_fixed32().is_err());
    }

    #[test]
    fn test_read_fixed64() {
        let mut cursor = Cursor::new(&[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
        assert_eq!(cursor.read_fixed64().unwrap(), 1);
    }

    #[test]
    fn test_read_fixed64_double_pi() {
        let bytes = std::f64::consts::PI.to_le_bytes();
        let mut cursor = Cursor::new(&bytes);
        let bits = cursor.read_fixed64().unwrap();
        assert_eq!(f64::from_le_bytes(bits.to_le_bytes()), std::f64::consts::PI);
    }

    #[test]
    fn test_read_fixed64_eof() {
        let mut cursor = Cursor::new(&[0x01, 0x02, 0x03]);
        assert!(cursor.read_fixed64().is_err());
    }

    // === Length-delimited reads ===

    #[test]
    fn test_read_bytes() {
        let mut cursor = Cursor::new(&[0x03, 0xAA, 0xBB, 0xCC]);
        let bytes = cursor.read_bytes().unwrap();
        assert_eq!(bytes, &[0xAA, 0xBB, 0xCC]);
        assert!(cursor.is_empty());
    }

    #[test]
    fn test_read_bytes_empty() {
        let mut cursor = Cursor::new(&[0x00]);
        let bytes = cursor.read_bytes().unwrap();
        assert_eq!(bytes, &[]);
    }

    #[test]
    fn test_read_bytes_truncated() {
        let mut cursor = Cursor::new(&[0x05, 0xAA, 0xBB]);
        assert!(cursor.read_bytes().is_err());
    }

    #[test]
    fn test_read_string_valid_utf8() {
        let mut cursor = Cursor::new(&[0x05, 0x68, 0x65, 0x6C, 0x6C, 0x6F]);
        assert_eq!(cursor.read_string().unwrap(), "hello");
    }

    #[test]
    fn test_read_string_empty() {
        let mut cursor = Cursor::new(&[0x00]);
        assert_eq!(cursor.read_string().unwrap(), "");
    }

    #[test]
    fn test_read_string_invalid_utf8() {
        let mut cursor = Cursor::new(&[0x02, 0xFF, 0xFE]);
        assert!(cursor.read_string().is_err());
    }

    // === Sub-cursor (nested messages) ===

    #[test]
    fn test_read_sub_cursor() {
        let mut cursor = Cursor::new(&[0x03, 0x08, 0x96, 0x01, 0xFF]);
        let mut sub = cursor.read_sub_cursor().unwrap();
        assert_eq!(sub.remaining(), 3);
        assert_eq!(cursor.remaining(), 1);
        let (field, wt) = sub.read_tag().unwrap();
        assert_eq!(field, 1);
        assert_eq!(wt, WireType::Varint);
        assert_eq!(sub.read_varint().unwrap(), 150);
        assert!(sub.is_empty());
    }

    // === Skip field ===

    #[test]
    fn test_skip_varint() {
        let mut cursor = Cursor::new(&[0x96, 0x01, 0xFF]);
        cursor.skip_field(WireType::Varint).unwrap();
        assert_eq!(cursor.remaining(), 1);
    }

    #[test]
    fn test_skip_fixed64() {
        let mut cursor = Cursor::new(&[0; 9]);
        cursor.skip_field(WireType::Fixed64).unwrap();
        assert_eq!(cursor.remaining(), 1);
    }

    #[test]
    fn test_skip_fixed32() {
        let mut cursor = Cursor::new(&[0; 5]);
        cursor.skip_field(WireType::Fixed32).unwrap();
        assert_eq!(cursor.remaining(), 1);
    }

    #[test]
    fn test_skip_length_delimited() {
        let mut cursor = Cursor::new(&[0x03, 0xAA, 0xBB, 0xCC, 0xFF]);
        cursor.skip_field(WireType::LengthDelimited).unwrap();
        assert_eq!(cursor.remaining(), 1);
    }
}
