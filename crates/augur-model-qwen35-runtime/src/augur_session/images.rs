//! Bounded RGB ingress shared by the native owner and its HTTP preflight.
use super::*;
use augur_model_qwen35::{Qwen35ImageGrid, Qwen35ImagePromptLayout, Qwen35PreparedF16Image,
    prepare_qwen35_aligned_rgb8_f16};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageInput {
    pub width: usize,
    pub height: usize,
    pub rgb8_hex: String,
    pub rgb8_sha256: String,
}
pub struct PreparedImages {
    pub images: Vec<Qwen35PreparedF16Image>,
    pub identity_sha256: String,
}
pub const MAX_IMAGE_PIXELS: usize = 1_048_576;
// No image-count ceiling. The owner encodes images serially; aggregate limits
// bound retained host pixels and language ingress embeddings, not the turns.
pub const MAX_TOTAL_IMAGE_PIXELS: usize = 32 * MAX_IMAGE_PIXELS;
pub const MAX_IMAGE_TOKENS: usize = MAX_TOTAL_IMAGE_PIXELS / 1024;
pub const MAX_REQUEST_BYTES: usize = 256 * 1024 * 1024;
pub fn prepare(tokens: &[u32], inputs: &[ImageInput]) -> Result<PreparedImages> {
    if inputs.is_empty() { return Err("at least one image required".into()); }
    let mut sum=0usize;
    for input in inputs {
        let pixels=input.width.checked_mul(input.height).ok_or("image extent overflow")?;
        sum=sum.checked_add(pixels).ok_or("image total overflow")?;
        if pixels==0 || pixels>MAX_IMAGE_PIXELS || sum>MAX_TOTAL_IMAGE_PIXELS {
            return Err("image history exceeds per-image or total pixel budget; no images were removed".into());
        }
    }
    let mut images = Vec::new();
    let mut hash = Sha256::new();
    hash.update(b"QUASAR-IMAGE-INPUT-RGB8-V1\0");
    let mut total_pixels = 0usize;
    for input in inputs {
        let pixels = input.width.checked_mul(input.height).ok_or("image extent overflow")?;
        total_pixels = total_pixels.checked_add(pixels).ok_or("image total overflow")?;
        if pixels == 0 || pixels > MAX_IMAGE_PIXELS || total_pixels > MAX_TOTAL_IMAGE_PIXELS || !valid_digest(&input.rgb8_sha256)
            || input.rgb8_hex.len() != pixels * 6 {
            return Err("image pixel/identity/encoded byte bound".into());
        }
        let grid = Qwen35ImageGrid::from_aligned_pixels(input.height, input.width)?;
        fn nibble(b: u8) -> Result<u8> {
            match b { b'0'..=b'9'=>Ok(b-b'0'), b'a'..=b'f'=>Ok(b-b'a'+10),
                _=>Err("RGB payload must use lowercase hex".into()) }
        }
        let rgb = input.rgb8_hex.as_bytes().chunks_exact(2)
            .map(|p| Ok((nibble(p[0])? << 4) | nibble(p[1])?)).collect::<Result<Vec<_>>>()?;
        if digest(&rgb) != input.rgb8_sha256 { return Err("image RGB digest mismatch".into()); }
        hash.update((input.height as u64).to_le_bytes());
        hash.update((input.width as u64).to_le_bytes());
        hash.update(Sha256::digest(&rgb));
        images.push(prepare_qwen35_aligned_rgb8_f16(&rgb, grid, 16 * 1024 * 1024)?);
    }
    let kinds: Vec<_> = tokens.iter().map(|&id| u8::from(id == 248056)).collect();
    let grids: Vec<_> = images.iter().map(|i| i.grid).collect();
    Qwen35ImagePromptLayout::new(tokens, &kinds, &grids)?;
    Ok(PreparedImages { images, identity_sha256: format!("{:x}", hash.finalize()) })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample() -> ImageInput { let rgb = vec![0u8; 32*32*3]; ImageInput {
        width:32,height:32,rgb8_hex:"00".repeat(rgb.len()),rgb8_sha256:digest(&rgb) } }
    #[test]
    fn identities_bind_pixels_geometry_and_order() {
        let ids = [248053,248056,248054];
        let a=sample(); let first=prepare(&ids,&[a.clone()]).unwrap();
        let mut b=a.clone(); b.rgb8_hex="ff".repeat(32*32*3);b.rgb8_sha256=digest(&vec![255;32*32*3]);
        assert_ne!(first.identity_sha256,prepare(&ids,&[b.clone()]).unwrap().identity_sha256);
        let pair=[248053,248056,248054,248053,248056,248054];
        assert_ne!(prepare(&pair,&[a.clone(),b.clone()]).unwrap().identity_sha256,
                   prepare(&pair,&[b,a]).unwrap().identity_sha256);
    }
    #[test]
    fn malformed_digest_extent_layout_and_limits_reject() {
        let ids=[248053,248056,248054];let a=sample();
        for change in 0..5 {let mut b=a.clone();match change {
            0=>b.width=usize::MAX,1=>b.rgb8_sha256="a".repeat(64),
            2=>b.rgb8_hex.replace_range(..1,"G"),3=>b.rgb8_hex.push('0'),_=>b.width=33 }
            assert!(prepare(&ids,&[b]).is_err());}
        assert!(prepare(&[248056],&[a.clone()]).is_err());
        assert!(prepare(&ids,&[]).is_err());
    }
    #[test]
    fn image_history_has_no_four_or_thirty_two_count_limit() {
        for count in [5,33,65] {
            let ids=[248053,248056,248054].repeat(count);
            let prepared=prepare(&ids,&vec![sample();count]).unwrap();
            assert_eq!(prepared.images.len(),count);
        }
        let over=ImageInput{width:1024,height:1024,rgb8_hex:String::new(),rgb8_sha256:"0".repeat(64)};
        let message=prepare(&[],&vec![over;33]).err().unwrap().to_string();
        assert!(message.contains("total pixel budget"));
    }
    #[test]
    fn four_high_budget_images_keep_individual_identity_and_layout() {
        let rgb=vec![0u8;736*736*3];
        let image=ImageInput {width:736,height:736,rgb8_hex:"00".repeat(rgb.len()),rgb8_sha256:digest(&rgb)};
        let mut ids=Vec::new();
        for _ in 0..4 {ids.push(248053);ids.extend(vec![248056;529]);ids.push(248054);}
        let prepared=prepare(&ids,&vec![image;4]).unwrap();
        assert_eq!(prepared.images.len(),4);
        let mut oversized=sample();oversized.width=2048;oversized.height=1024;
        assert!(prepare(&ids,&[oversized]).is_err());
    }
}
