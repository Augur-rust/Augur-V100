//! CPU-only, already-resized RGB8 image inputs for the local Qwen3.5/27B ABI.
//!
//! No image decoding/resizing, vision forward, CUDA, KV mutation or product
//! admission occurs here. Patch order and language positions are distinct:
//! vision patches are merge-group major; merged image tokens are raster major.

use crate::Qwen35Error;

const PATCH: usize = 16;
const MERGE: usize = 2;
const IMAGE: u32 = 248_056;
const VIDEO: u32 = 248_057;
const START: u32 = 248_053;
const END: u32 = 248_054;
const MAX_PIXELS: usize = 16_777_216;
const MAX_TOKENS: usize = 262_144;
pub const QWEN35_IMAGE_PATCH_ELEMENTS: usize = 3 * 2 * PATCH * PATCH;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35ImageGrid {
    height: usize,
    width: usize,
}

impl Qwen35ImageGrid {
    /// Already RGB8, resized to multiples of patch * merge. Tiny aligned images
    /// are allowed for component tests; this is not a resize policy.
    pub fn from_aligned_pixels(height: usize, width: usize) -> Result<Self, Qwen35Error> {
        let pixels = height.checked_mul(width).ok_or(Qwen35Error::SizeOverflow)?;
        if height == 0
            || width == 0
            || !height.is_multiple_of(32)
            || !width.is_multiple_of(32)
            || pixels > MAX_PIXELS
        {
            return Err(Qwen35Error::invalid(
                "image_extent",
                "expected nonzero 32-aligned RGB extent within 16777216 pixels",
            ));
        }
        Ok(Self { height, width })
    }

    pub fn grid_thw(self) -> [usize; 3] {
        [1, self.height / PATCH, self.width / PATCH]
    }
    pub fn merged_tokens(self) -> usize {
        (self.height / (PATCH * MERGE)) * (self.width / (PATCH * MERGE))
    }
    pub fn rgb8_bytes(self) -> usize {
        self.height * self.width * 3
    }
}

#[derive(Debug)]
pub struct Qwen35PreparedImage {
    pub grid: Qwen35ImageGrid,
    /// [patch_count, channels * temporal_patch * patch_height * patch_width].
    pub pixel_values: Vec<f32>,
    /// Unmerged (h,w) positions, in merge-group-major order.
    pub patch_positions_hw: Vec<[usize; 2]>,
}

/// Normalize HWC RGB8 with mean/std = 0.5, duplicate still-image temporal size 2,
/// then emit [group_h,group_w,local_h,local_w,c,t,y,x]. The explicit CPU budget
/// bounds the returned f32 payload before allocation.
pub fn prepare_qwen35_aligned_rgb8(
    rgb: &[u8],
    grid: Qwen35ImageGrid,
    max_pixel_value_bytes: usize,
) -> Result<Qwen35PreparedImage, Qwen35Error> {
    if rgb.len() != grid.rgb8_bytes() {
        return Err(Qwen35Error::invalid("rgb8", "input byte extent mismatch"));
    }
    let [_, gh, gw] = grid.grid_thw();
    let values = gh * gw * QWEN35_IMAGE_PATCH_ELEMENTS;
    if values * std::mem::size_of::<f32>() > max_pixel_value_bytes {
        return Err(Qwen35Error::invalid(
            "pixel_value_budget",
            "prepared image exceeds CPU payload budget",
        ));
    }
    let mut pixel_values = Vec::new();
    pixel_values
        .try_reserve_exact(values)
        .map_err(|_| Qwen35Error::invalid("pixel_values", "CPU allocation failed"))?;
    let mut patch_positions_hw = Vec::with_capacity(gh * gw);
    for group_h in 0..gh / MERGE {
        for group_w in 0..gw / MERGE {
            for local_h in 0..MERGE {
                for local_w in 0..MERGE {
                    let ph = group_h * MERGE + local_h;
                    let pw = group_w * MERGE + local_w;
                    patch_positions_hw.push([ph, pw]);
                    for c in 0..3 {
                        for _t in 0..2 {
                            for y in 0..PATCH {
                                for x in 0..PATCH {
                                    let offset =
                                        ((ph * PATCH + y) * grid.width + pw * PATCH + x) * 3 + c;
                                    let scaled = f32::from(rgb[offset]) * (1.0f32 / 255.0);
                                    pixel_values.push((scaled - 0.5) / 0.5);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(Qwen35PreparedImage {
        grid,
        pixel_values,
        patch_positions_hw,
    })
}

/// One unpadded image-only prompt's positions and scatter offsets. No video,
/// batched padding, persistent cache state or inference completion is implied.
#[derive(Debug)]
pub struct Qwen35ImagePromptLayout {
    positions_thw: Vec<[i64; 3]>,
    image_token_indices: Vec<usize>,
    rope_delta: i64,
}

impl Qwen35ImagePromptLayout {
    pub fn new(
        input_ids: &[u32],
        token_types: &[u8],
        grids: &[Qwen35ImageGrid],
    ) -> Result<Self, Qwen35Error> {
        if input_ids.is_empty()
            || input_ids.len() > MAX_TOKENS
            || input_ids.len() != token_types.len()
        {
            return Err(Qwen35Error::invalid(
                "image_prompt",
                "empty, overlong or mismatched token/type extent",
            ));
        }
        // Delimiters are text tokens; each image placeholder is modality type 1.
        for (i, (&id, &kind)) in input_ids.iter().zip(token_types).enumerate() {
            if id >= 248_320 || id == VIDEO || kind > 1 || (id == IMAGE) != (kind == 1) {
                return Err(Qwen35Error::invalid(
                    "image_prompt",
                    "unsupported token/modality pairing",
                ));
            }
            if id == START && token_types.get(i + 1) != Some(&1) {
                return Err(Qwen35Error::invalid(
                    "vision_start",
                    "must immediately precede an image run",
                ));
            }
            if id == END && (i == 0 || token_types[i - 1] != 1) {
                return Err(Qwen35Error::invalid(
                    "vision_end",
                    "must immediately follow an image run",
                ));
            }
        }
        let mut positions_thw = Vec::with_capacity(input_ids.len());
        let mut image_token_indices = Vec::new();
        let mut cursor = 0i64;
        let mut image_index = 0;
        let mut i = 0;
        while i < input_ids.len() {
            if token_types[i] == 0 {
                positions_thw.push([cursor; 3]);
                cursor += 1;
                i += 1;
                continue;
            }
            let begin = i;
            while i < input_ids.len() && token_types[i] == 1 {
                i += 1;
            }
            let grid = grids
                .get(image_index)
                .ok_or_else(|| Qwen35Error::invalid("image_grid", "missing grid for image run"))?;
            if begin == 0
                || input_ids[begin - 1] != START
                || input_ids.get(i) != Some(&END)
                || i - begin != grid.merged_tokens()
            {
                return Err(Qwen35Error::invalid(
                    "image_run",
                    "delimiter or placeholder count mismatch",
                ));
            }
            let [_, gh, gw] = grid.grid_thw();
            for h in 0..gh / MERGE {
                for w in 0..gw / MERGE {
                    positions_thw.push([cursor, cursor + h as i64, cursor + w as i64]);
                }
            }
            image_token_indices.extend(begin..i);
            cursor += (gh.max(gw) / MERGE) as i64;
            image_index += 1;
        }
        if image_index != grids.len() {
            return Err(Qwen35Error::invalid("image_grid", "unused image grids"));
        }
        Ok(Self {
            positions_thw,
            image_token_indices,
            rope_delta: cursor - input_ids.len() as i64,
        })
    }

    pub fn positions_thw(&self) -> &[[i64; 3]] {
        &self.positions_thw
    }
    pub fn image_token_indices(&self) -> &[usize] {
        &self.image_token_indices
    }
    pub fn rope_delta(&self) -> i64 {
        self.rope_delta
    }

    /// Pure position calculation, not a KV cursor advance or restore permit.
    pub fn continuation_positions(
        &self,
        absolute_start: usize,
        count: usize,
    ) -> Result<Vec<[i64; 3]>, Qwen35Error> {
        let end = absolute_start
            .checked_add(count)
            .ok_or(Qwen35Error::SizeOverflow)?;
        if absolute_start < self.positions_thw.len() || end > MAX_TOKENS {
            return Err(Qwen35Error::invalid(
                "continuation_position",
                "start precedes prompt or end exceeds model bound",
            ));
        }
        Ok((absolute_start..end)
            .map(|p| [p as i64 + self.rope_delta; 3])
            .collect())
    }

    /// Replace only placeholder rows. Validate all shapes/finiteness before the
    /// first write so invalid tensors cannot leave a partial destination.
    pub fn scatter_image_embeddings(
        &self,
        text: &mut [f32],
        image: &[f32],
        hidden: usize,
    ) -> Result<(), Qwen35Error> {
        let text_len = self
            .positions_thw
            .len()
            .checked_mul(hidden)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let image_len = self
            .image_token_indices
            .len()
            .checked_mul(hidden)
            .ok_or(Qwen35Error::SizeOverflow)?;
        if hidden == 0
            || text.len() != text_len
            || image.len() != image_len
            || !text.iter().chain(image).all(|x| x.is_finite())
        {
            return Err(Qwen35Error::invalid(
                "image_embeddings",
                "shape mismatch, zero hidden size or non-finite input",
            ));
        }
        for (row, &position) in self.image_token_indices.iter().enumerate() {
            text[position * hidden..(position + 1) * hidden]
                .copy_from_slice(&image[row * hidden..(row + 1) * hidden]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> Qwen35ImagePromptLayout {
        Qwen35ImagePromptLayout::new(
            &[9, START, IMAGE, IMAGE, IMAGE, IMAGE, END, 10],
            &[0, 0, 1, 1, 1, 1, 0, 0],
            &[Qwen35ImageGrid::from_aligned_pixels(64, 64).unwrap()],
        )
        .unwrap()
    }

    #[test]
    fn aligned_geometry_rejects_overflow_zero_unaligned_and_excess() {
        for (h, w) in [
            (0, 32),
            (32, 0),
            (16, 32),
            (33, 32),
            (usize::MAX, 32),
            (8192, 8192),
        ] {
            assert!(Qwen35ImageGrid::from_aligned_pixels(h, w).is_err());
        }
        assert_eq!(
            Qwen35ImageGrid::from_aligned_pixels(256, 384)
                .unwrap()
                .grid_thw(),
            [1, 16, 24]
        );
    }

    #[test]
    fn patches_are_merge_group_major_not_plain_raster() {
        let grid = Qwen35ImageGrid::from_aligned_pixels(32, 64).unwrap();
        let mut rgb = vec![0u8; grid.rgb8_bytes()];
        for y in 0..32 {
            for x in 0..64 {
                for c in 0..3 {
                    rgb[(y * 64 + x) * 3 + c] = ((y / 16) * 100 + (x / 16) * 20 + c) as u8;
                }
            }
        }
        let prepared = prepare_qwen35_aligned_rgb8(&rgb, grid, 8 * 1536 * 4).unwrap();
        assert_eq!(
            prepared.patch_positions_hw,
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [0, 2],
                [0, 3],
                [1, 2],
                [1, 3]
            ]
        );
        for (row, &[h, w]) in prepared.patch_positions_hw.iter().enumerate() {
            for c in 0..3 {
                let offset = row * 1536 + c * 512;
                let expected = (((h * 100 + w * 20 + c) as f32) * (1.0 / 255.0) - 0.5) / 0.5;
                assert!(prepared.pixel_values[offset..offset + 512]
                    .iter()
                    .all(|v| *v == expected));
            }
        }
    }

    #[test]
    fn budget_and_rgb_extent_fail_before_allocation() {
        let grid = Qwen35ImageGrid::from_aligned_pixels(32, 32).unwrap();
        let rgb = vec![0; grid.rgb8_bytes()];
        assert!(prepare_qwen35_aligned_rgb8(&rgb[..rgb.len() - 1], grid, usize::MAX).is_err());
        assert!(prepare_qwen35_aligned_rgb8(&rgb, grid, 4 * 1536 * 4 - 1).is_err());
        assert_eq!(
            prepare_qwen35_aligned_rgb8(&rgb, grid, 4 * 1536 * 4)
                .unwrap()
                .pixel_values
                .len(),
            4 * 1536
        );
    }

    #[test]
    fn square_image_positions_compress_language_coordinates() {
        let value = layout();
        assert_eq!(
            value.positions_thw(),
            &[
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [2, 2, 3],
                [2, 3, 2],
                [2, 3, 3],
                [4, 4, 4],
                [5, 5, 5]
            ]
        );
        assert_eq!(value.rope_delta(), -2);
        assert_eq!(
            value.continuation_positions(8, 2).unwrap(),
            [[6, 6, 6], [7, 7, 7]]
        );
    }

    #[test]
    fn rectangular_multi_image_positions_preserve_grid_order() {
        let small = Qwen35ImageGrid::from_aligned_pixels(32, 64).unwrap();
        let large = Qwen35ImageGrid::from_aligned_pixels(64, 96).unwrap();
        let ids = [
            START, IMAGE, IMAGE, END, 42, START, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, END,
        ];
        let kinds = [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0];
        let value = Qwen35ImagePromptLayout::new(&ids, &kinds, &[small, large]).unwrap();
        assert_eq!(value.positions_thw()[2], [1, 1, 2]);
        assert_eq!(value.positions_thw()[11], [6, 7, 8]);
        assert_eq!(value.positions_thw()[12], [9; 3]);
        assert_eq!(value.rope_delta(), -3);
    }

    #[test]
    fn text_only_remains_one_dimensional() {
        let x = Qwen35ImagePromptLayout::new(&[1, 2, 3], &[0, 0, 0], &[]).unwrap();
        assert_eq!(x.positions_thw(), &[[0; 3], [1; 3], [2; 3]]);
        assert_eq!(x.rope_delta(), 0);
    }

    #[test]
    fn malformed_prompt_and_grid_pairings_reject() {
        let grid = Qwen35ImageGrid::from_aligned_pixels(32, 32).unwrap();
        for (ids, kinds, grids) in [
            (vec![], vec![], vec![]),
            (vec![1], vec![], vec![]),
            (vec![IMAGE], vec![0], vec![]),
            (vec![1], vec![1], vec![grid]),
            (vec![VIDEO], vec![2], vec![grid]),
            (vec![248_320], vec![0], vec![]),
            (vec![START, END], vec![0, 0], vec![]),
            (vec![IMAGE, END], vec![1, 0], vec![grid]),
            (vec![START, IMAGE], vec![0, 1], vec![grid]),
            (vec![START, IMAGE, IMAGE, END], vec![0, 1, 1, 0], vec![grid]),
            (vec![START, IMAGE, END], vec![0, 1, 0], vec![]),
            (vec![1], vec![0], vec![grid]),
        ] {
            assert!(Qwen35ImagePromptLayout::new(&ids, &kinds, &grids).is_err());
        }
    }

    #[test]
    fn continuation_extent_is_checked_without_advancing_state() {
        let x = layout();
        assert!(x.continuation_positions(7, 1).is_err());
        assert!(x.continuation_positions(usize::MAX, 2).is_err());
        assert!(x.continuation_positions(MAX_TOKENS, 1).is_err());
        assert_eq!(
            x.continuation_positions(MAX_TOKENS - 1, 1).unwrap().len(),
            1
        );
    }

    #[test]
    fn scatter_only_changes_placeholder_rows() {
        let x = layout();
        let mut text: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let initial = text.clone();
        let image: Vec<f32> = (100..112).map(|i| i as f32).collect();
        x.scatter_image_embeddings(&mut text, &image, 3).unwrap();
        assert_eq!(&text[6..18], image.as_slice());
        assert_eq!(&text[..6], &initial[..6]);
        assert_eq!(&text[18..], &initial[18..]);
    }

    #[test]
    fn invalid_scatter_cannot_partially_mutate_destination() {
        let x = layout();
        for image in [vec![1.; 11], vec![f32::NAN; 12], vec![f32::INFINITY; 12]] {
            let mut text = vec![2.; 24];
            assert!(x.scatter_image_embeddings(&mut text, &image, 3).is_err());
            assert_eq!(text, vec![2.; 24]);
        }
        assert!(x.scatter_image_embeddings(&mut [], &[], 0).is_err());
        assert!(x
            .scatter_image_embeddings(&mut [], &[], usize::MAX)
            .is_err());
    }
}
