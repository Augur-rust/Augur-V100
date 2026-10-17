//! Default-off full-vocabulary observation after the actual cold request input.
//! No teacher-forced tokens, changed schedule, or new GPU allocation. The host
//! readback/IO invalidates performance comparisons for this diagnostic request.
use super::*;
use std::ffi::OsStr;
use std::io::Write;

pub(super) fn directory(value: Option<&OsStr>) -> Result<Option<PathBuf>> {
    let Some(value) = value else { return Ok(None) };
    let path = PathBuf::from(value);
    if !path.is_absolute() || path.components().any(|p| matches!(p, std::path::Component::ParentDir)) {
        return Err(invalid("fixed-history logits requires an absolute existing diagnostic directory"));
    }
    for item in path.ancestors() {
        let metadata = std::fs::symlink_metadata(item).map_err(Qwen35Error::from)?;
        #[cfg(windows)]
        let reparse = {
            use std::os::windows::fs::MetadataExt;
            metadata.file_attributes() & 0x400 != 0
        };
        #[cfg(not(windows))]
        let reparse = false;
        if !metadata.is_dir() || metadata.file_type().is_symlink() || reparse {
            return Err(invalid("fixed-history logits directory must not traverse links/reparse points"));
        }
    }
    Ok(Some(path))
}

fn input_digest(ids: &[usize], cursor: usize, capacity: usize, vocab: usize) -> Result<String> {
    if ids.is_empty() || ids.len() != cursor || cursor > capacity || capacity > 196608
        || vocab < 2 || vocab > u32::MAX as usize || ids.iter().any(|&id| id >= vocab) {
        return Err(invalid("fixed-history logits input/cursor/vocabulary extent"));
    }
    let mut hash = Sha256::new();
    hash.update(b"QWEN35-SESSION-CONSUMED-U32LE-V1\0");
    for &token in ids { hash.update((token as u32).to_le_bytes()); }
    Ok(format!("{:x}", hash.finalize()))
}

pub(super) fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(path)
        .map_err(Qwen35Error::from)?;
    file.write_all(bytes).map_err(Qwen35Error::from)?;
    file.flush().map_err(Qwen35Error::from)?;
    Ok(())
}

pub(super) fn capture(d: &Driver, ids: &[usize]) -> Result<Option<Value>> {
    let Some(root) = directory(std::env::var_os("AUGUR_QWEN35_FIXED_HISTORY_LOGITS_DIR").as_deref())? else {
        return Ok(None);
    };
    let digest = input_digest(ids, d.cursor, d.capacity, V)?;
    let receipt = d.stream.submission_receipt()?;
    let logits = read_f32_buffer(&d.stream, &d.head.logits, receipt)?;
    if logits.len() != V { return Err(invalid("fixed-history full vocabulary extent")); }
    let summary = target_only_reference::summarize_logits(&logits, d.anchor, 0)?;
    let bytes = logits.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
    let name = format!("{digest}.logits.f32le.bin");
    // Same input may be captured only once per experiment directory. Never
    // overwrite or silently turn a reused output into fresh evidence.
    write_new(&root.join(&name), &bytes)?;
    Ok(Some(json!({
        "schema":"qwen35-fixed-history-full-logits-v1", "directory":root,"file":name,
        "input_token_digest":digest,"input_token_count":ids.len(),"cursor":d.cursor,
        "vocab_size":V,"bytes":bytes.len(),"sha256":sha256_hex(&bytes),
        "selected_token":d.anchor,"summary":summary,
        "scope":"cold_request_all_actual_input_before_first_publication",
        "host_serialized_diagnostic":true,"performance_evidence":false,
        "product_ready":false,"qualified":false,"semantic_qualification":false,"visible_tps":false
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn diagnostic_is_default_off_and_path_must_be_absolute() {
        assert!(directory(None).unwrap().is_none());
        assert!(directory(Some(OsStr::new("relative"))).is_err());
        assert!(directory(Some(OsStr::new(""))).is_err());
    }
    #[test]
    fn digest_binds_exact_history_order_cursor_and_capacity() {
        let a = input_digest(&[1,2,3],3,196608,4).unwrap();
        assert_ne!(a,input_digest(&[1,3,2],3,196608,4).unwrap());
        assert_ne!(a,input_digest(&[1,2],2,196608,4).unwrap());
        #[cfg(feature="qwen35-session-continuation")]
        assert_eq!(a,committed_session::token_hash(&[1,2,3]));
        for (ids,cursor,capacity,vocab) in [
            (&[][..],0,196608,4),(&[1,2][..],1,196608,4),
            (&[1,2][..],2,1,4),(&[1][..],1,196609,4),
            (&[4][..],1,196608,4),(&[0][..],1,196608,1),
        ] { assert!(input_digest(ids,cursor,capacity,vocab).is_err()); }
    }
    #[test]
    fn capture_file_creation_cannot_overwrite_existing_evidence() {
        let nonce=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let root=std::env::temp_dir().join(format!("augur-fixed-logits-{}-{nonce}",std::process::id()));
        std::fs::create_dir(&root).unwrap();
        assert!(directory(Some(root.as_os_str())).unwrap().is_some());
        let file=root.join("logits.bin");
        write_new(&file,&[0,1,2,3]).unwrap();
        assert!(write_new(&file,&[4,5]).is_err());
        assert_eq!(std::fs::read(&file).unwrap(),[0,1,2,3]);
        assert!(directory(Some(file.as_os_str())).is_err());
        std::fs::remove_file(file).unwrap();std::fs::remove_dir(root).unwrap();
    }
}
