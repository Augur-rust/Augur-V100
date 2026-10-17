//! Bounded loopback HTTP/1.1: one request per connection, no request chunking.
use augur_model_qwen35_runtime::augur_session::Result;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    net::TcpStream,
};
const HEAD_LIMIT: usize = 16384;
const BODY_LIMIT: usize = if cfg!(feature="qwen35-augur-vision") {augur_model_qwen35_runtime::augur_session::images::MAX_REQUEST_BYTES} else {16 * 1024 * 1024};

fn headers(raw: &[u8]) -> Result<(String, String, usize)> {
    let header = std::str::from_utf8(raw)?;
    if !header.is_ascii() || !header.ends_with("\r\n\r\n") {
        return Err("invalid HTTP header encoding/framing".into());
    }
    let mut lines = header[..header.len() - 4].split("\r\n");
    let fields: Vec<_> = lines
        .next()
        .ok_or("missing request line")?
        .split(' ')
        .collect();
    if fields.len() != 3
        || !["GET", "POST"].contains(&fields[0])
        || fields[2] != "HTTP/1.1"
        || !fields[1].starts_with('/')
        || fields[1].bytes().any(|c| c <= 32 || c == 127)
    {
        return Err("invalid bounded request line".into());
    }
    let mut map = BTreeMap::new();
    for line in lines {
        let (key, value) = line.split_once(':').ok_or("invalid HTTP header")?;
        if key.is_empty()
            || !key.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
            || value.bytes().any(|b| (b < 32 && b != b'\t') || b == 127)
            || map.insert(key.to_ascii_lowercase(), value.trim()).is_some()
        {
            return Err("invalid or duplicate HTTP header".into());
        }
    }
    if map.contains_key("transfer-encoding")
        || map.contains_key("content-encoding")
        || map.contains_key("expect")
        || !map.contains_key("host")
    {
        return Err("unsupported HTTP encoding/expect or missing host".into());
    }
    let n = match map.get("content-length") {
        Some(s) if !s.is_empty() && s.len() <= BODY_LIMIT.to_string().len() && s.bytes().all(|b| b.is_ascii_digit()) => {
            s.parse::<usize>()?
        }
        Some(_) => return Err("invalid Content-Length".into()),
        None if fields[0] == "GET" => 0,
        _ => return Err("Content-Length required".into()),
    };
    if n > BODY_LIMIT
        || (fields[0] == "GET" && n != 0)
        || (fields[0] == "POST"
            && (n == 0
                || !map.get("content-type").is_some_and(|s| {
                    s.split(';')
                        .next()
                        .unwrap()
                        .trim()
                        .eq_ignore_ascii_case("application/json")
                })))
    {
        return Err("HTTP body extent/type".into());
    }
    Ok((fields[0].into(), fields[1].into(), n))
}
pub(super) fn read_request(reader: &mut impl Read) -> Result<(String, String, Vec<u8>)> {
    let mut bytes = Vec::new();
    let mut chunk = [0; 8192];
    let end = loop {
        let n = reader.read(&mut chunk)?;
        if n == 0 {
            return Err("incomplete HTTP header".into());
        }
        bytes.extend_from_slice(&chunk[..n]);
        if let Some(i) = bytes.windows(4).position(|s| s == b"\r\n\r\n") {
            if i + 4 > HEAD_LIMIT {
                return Err("HTTP header bound".into());
            }
            break i + 4;
        }
        if bytes.len() > HEAD_LIMIT {
            return Err("HTTP header bound".into());
        }
    };
    let (method, path, n) = headers(&bytes[..end])?;
    while bytes.len() < end + n {
        let count = reader.read(&mut chunk)?;
        if count == 0 {
            return Err("incomplete HTTP body".into());
        }
        bytes.extend_from_slice(&chunk[..count]);
    }
    if bytes.len() != end + n {
        return Err("HTTP pipelining unsupported".into());
    }
    Ok((method, path, bytes[end..].to_vec()))
}
pub(super) fn json(stream: &mut TcpStream, status: u16, value: &Value) -> Result<()> {
    let raw = serde_json::to_vec(value)?;
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        _ => "Internal Server Error",
    };
    write!(stream,"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nX-AUGUR-Product-Ready: false\r\n\r\n",raw.len())?;
    stream.write_all(&raw)?;
    stream.flush()?;
    Ok(())
}
pub(super) fn begin(stream: &mut TcpStream) -> Result<()> {
    stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\nConnection: close\r\nX-AUGUR-Product-Ready: false\r\n\r\n")?;
    stream.flush()?;
    Ok(())
}
pub(super) fn chunk(stream: &mut TcpStream, value: &Value) -> Result<()> {
    let raw = serde_json::to_vec(value)?;
    static FUSED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *FUSED.get_or_init(|| matches!(std::env::var("AUGUR_HTTP_FINISH").as_deref(), Ok("fused_v1" | "receipt_handle_v2" | "receipt_ref_v3"))) {
        let mut chunk = format!("{:X}\r\n", raw.len() + 1).into_bytes();
        chunk.extend_from_slice(&raw);
        chunk.extend_from_slice(b"\n\r\n");
        stream.write_all(&chunk)?;
    } else {
        write!(stream, "{:X}\r\n", raw.len() + 1)?;
        stream.write_all(&raw)?;
        stream.write_all(b"\n\r\n")?;
    }
    stream.flush()?;
    Ok(())
}
pub(super) fn end(stream: &mut TcpStream) -> Result<()> {
    stream.write_all(b"0\r\n\r\n")?;
    stream.flush()?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_smuggling_duplicates_encoded_or_unbounded_requests() {
        let good = b"POST /augur/v1/turn HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\nContent-Type: application/json\r\n\r\n{}";
        assert_eq!(read_request(&mut good.as_slice()).unwrap().2, b"{}");
        for extra in [
            "content-length: 2\r\n",
            "Transfer-Encoding: chunked\r\n",
            "Content-Encoding: gzip\r\n",
            "Expect: 100-continue\r\n",
            " HOST: x\r\n",
        ] {
            let raw = format!("POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\nContent-Type: application/json\r\n{extra}\r\n{{}}");
            assert!(read_request(&mut raw.as_bytes()).is_err());
        }
        let mut pipeline = good.to_vec();
        pipeline.extend_from_slice(b"GET / HTTP/1.1\r\n\r\n");
        assert!(read_request(&mut pipeline.as_slice()).is_err());
        assert!(read_request(&mut &good[..good.len() - 1]).is_err());
        let at_limit=format!("POST / HTTP/1.1\r\nHost: x\r\nContent-Length: {BODY_LIMIT}\r\nContent-Type: application/json\r\n\r\n");
        assert!(headers(at_limit.as_bytes()).is_ok());
        let too_large=format!("POST / HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n",BODY_LIMIT+1);
        assert!(headers(too_large.as_bytes()).is_err());
    }
}
