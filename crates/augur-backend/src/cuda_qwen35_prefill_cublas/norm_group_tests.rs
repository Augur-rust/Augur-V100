//! Fused direct-B8 norm group vs the separate add/norm/stage/project/copy sequence.
//! Every output word, record word and residual word must be bit-identical.
use super::*;
use crate::{cuda_device_allocation_snapshot, CudaAddF32ArgsV1};
use super::CudaQwen35AugurProjectionV1 as Projection;
use crate::cuda_qwen35_prefill_cublas::CudaQwen35PrefillRmsnormV1;
use augur_core::PhysicalLayoutId as L;
const H: usize = 5120;
fn f32s(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n).map(|i| {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        let v = ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * scale;
        if i % 409 == 7 { v * 40.0 } else { v } // outlier channels, like real activations
    }).collect()
}
fn upload(stream: &OwnedCudaStream, v: &[f32]) -> Rc<OwnedCudaBuffer> {
    let b = OwnedCudaBuffer::allocate(0, v.len() * 4).unwrap();
    b.copy_from_host_sync(stream, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>()).unwrap();
    Rc::new(b)
}
fn read(stream: &OwnedCudaStream, b: &OwnedCudaBuffer) -> Vec<u8> {
    let mut out = vec![0; b.len_bytes()]; b.copy_to_host_sync(stream, &mut out).unwrap(); out
}
fn fview(b: &OwnedCudaBuffer, n: usize) -> CudaF32View {
    CudaF32View::new(b.view(0, n * 4).unwrap(), L::DEVICE_CONTIGUOUS_F32_CUDA_V1, n).unwrap()
}
fn weight(stream: &OwnedCudaStream, rows: usize, seed: usize, scale: f32) -> Rc<OwnedCudaQwen35QpnWeightV1> {
    let packed: Vec<u8> = (0..rows * H / 2).map(|i| ((i * 37 + seed * 11 + i / 97) & 255) as u8).collect();
    let scales: Vec<u8> = (0..rows * H / 16).map(|i| ((i * 13 + seed + i / 7) % 120) as u8).collect();
    Rc::new(OwnedCudaQwen35QpnWeightV1::upload_checkpoint_nvfp4(stream, &packed, &scales, rows, H, scale).unwrap())
}
#[test]
#[ignore = "exclusive SM70; fused direct-B8 GDN input/post-attention glue bit identity"]
fn fused_norm_group_b8_is_bit_identical_to_separate_glue() {
    let before = cuda_device_allocation_snapshot();
    {
        let stream = Rc::new(OwnedCudaStream::create(0).unwrap());
        let plan = Rc::new(CudaQwen35PrefillCublasPlanV1::new_direct_b8(&stream, 17408).unwrap());
        let eps = 1e-6f32;
        // GDN input group: qkv, z, alpha64, beta64 (alpha/beta trimmed to 48).
        let shapes = [(10240usize, 10240usize), (6144, 6144), (64, 48), (64, 48)];
        let weights: Vec<_> = shapes.iter().enumerate().map(|(i, &(r, _))| weight(&stream, r, i, 0.0011 + i as f32 * 0.0003)).collect();
        for case in 0..3u64 {
            let x = upload(&stream, &f32s(8 * H, 11 + case, 3.0));
            let nw = upload(&stream, &f32s(H, 23 + case, 1.0));
            let normed = Rc::new(OwnedCudaBuffer::allocate(0, 8 * H * 4).unwrap());
            let norm = CudaQwen35PrefillRmsnormV1::bind(&stream, fview(&x, 8 * H), fview(&nw, H), fview(&normed, 8 * H), 8, H, eps).unwrap();
            // Separate path.
            let full: Vec<Rc<OwnedCudaBuffer>> = shapes.iter().map(|&(r, _)| Rc::new(OwnedCudaBuffer::allocate(0, 8 * r * 4).unwrap())).collect();
            let projections: Vec<_> = weights.iter().zip(&full).map(|(w, y)| {
                let mut p = Projection::bind(Rc::clone(&plan), Rc::clone(&stream), Rc::clone(w), Rc::clone(&normed), 0, Rc::clone(y), 0, 8).unwrap();
                p.use_direct_b8().unwrap(); p
            }).collect();
            norm.submit(&stream).unwrap();
            Projection::submit_same_input([&projections[0], &projections[1], &projections[2], &projections[3]]).unwrap();
            stream.synchronize().unwrap();
            let want_norm = read(&stream, &normed);
            let want: Vec<Vec<u8>> = shapes.iter().zip(&full).map(|(&(r, out), y)| {
                let all = read(&stream, y);
                (0..8).flat_map(|t| all[t * r * 4..(t * r + out) * 4].to_vec()).collect()
            }).collect();
            // Fused path into fresh planes: trimmed outputs + records.
            let outs: Vec<Rc<OwnedCudaBuffer>> = shapes.iter().map(|&(_, o)| Rc::new(OwnedCudaBuffer::allocate(0, 8 * o * 4).unwrap())).collect();
            let recs: Vec<Rc<OwnedCudaBuffer>> = shapes.iter().map(|&(_, o)| Rc::new(OwnedCudaBuffer::allocate(0, 8 * o * 4).unwrap())).collect();
            normed.copy_from_host_sync(&stream, &vec![0xa5; 8 * H * 4]).unwrap();
            let members: Vec<_> = (0..4).map(|i| {
                let (_, o) = shapes[i];
                (&projections[i], NormGroupMemberV1 { out_cols: o,
                    output: if o != shapes[i].0 { Some(outs[i].view(0, 8 * o * 4).unwrap()) } else { None },
                    record: Some(recs[i].view(0, 8 * o * 4).unwrap()) })
            }).collect();
            // Untrimmed members keep their bound output; redirect those reads to `full`.
            let group = CudaQwen35AugurNormGroupB8V1::bind(&norm, &members, None).unwrap();
            for y in &full { y.copy_from_host_sync(&stream, &vec![0x5a; y.len_bytes()]).unwrap(); }
            let hot = cuda_device_allocation_snapshot();
            group.submit().unwrap(); stream.synchronize().unwrap();
            assert_eq!(cuda_device_allocation_snapshot(), hot);
            assert_eq!(read(&stream, &normed), want_norm, "norm plane case {case}");
            for i in 0..4 {
                let (r, o) = shapes[i];
                let got = if o == r { read(&stream, &full[i]) } else { read(&stream, &outs[i]) };
                assert_eq!(got, want[i], "member {i} case {case}");
                assert_eq!(read(&stream, &recs[i]), want[i], "record {i} case {case}");
            }
            // Rejections: aliasing record/output, wrong trim, audit norm.
            let bad = [(&projections[2], NormGroupMemberV1 { out_cols: 48, output: Some(outs[2].view(0, 8 * 48 * 4).unwrap()), record: Some(outs[2].view(0, 8 * 48 * 4).unwrap()) }),
                (&projections[3], members[3].1)];
            assert!(CudaQwen35AugurNormGroupB8V1::bind(&norm, &bad, None).is_err());
            let bad = [(&projections[0], NormGroupMemberV1 { out_cols: 48, output: None, record: None }), (&projections[1], members[1].1)];
            assert!(CudaQwen35AugurNormGroupB8V1::bind(&norm, &bad, None).is_err());
            assert!(CudaQwen35AugurNormGroupB8V1::bind(&norm, &members[..1], None).is_err());
        }
        // Post-attention: residual add -> norm -> {gate, up} (17408 rows each).
        let gw = weight(&stream, 17408, 7, 0.0009); let uw = weight(&stream, 17408, 9, 0.0013);
        let a = upload(&stream, &f32s(8 * H, 31, 2.0)); let b = upload(&stream, &f32s(8 * H, 37, 2.0));
        let nw = upload(&stream, &f32s(H, 41, 1.0));
        let sum = Rc::new(OwnedCudaBuffer::allocate(0, 8 * H * 4).unwrap());
        let normed = Rc::new(OwnedCudaBuffer::allocate(0, 8 * H * 4).unwrap());
        let add = CudaAddF32ArgsV1::bind(&stream, fview(&a, 8 * H), fview(&b, 8 * H), fview(&sum, 8 * H)).unwrap();
        let norm = CudaQwen35PrefillRmsnormV1::bind(&stream, fview(&sum, 8 * H), fview(&nw, H), fview(&normed, 8 * H), 8, H, eps).unwrap();
        let g = Rc::new(OwnedCudaBuffer::allocate(0, 8 * 17408 * 4).unwrap()); let u = Rc::new(OwnedCudaBuffer::allocate(0, 8 * 17408 * 4).unwrap());
        let mk = |w: &Rc<OwnedCudaQwen35QpnWeightV1>, y: &Rc<OwnedCudaBuffer>| {
            let mut p = Projection::bind(Rc::clone(&plan), Rc::clone(&stream), Rc::clone(w), Rc::clone(&normed), 0, Rc::clone(y), 0, 8).unwrap();
            p.use_direct_b8().unwrap(); p
        };
        let (gp, up) = (mk(&gw, &g), mk(&uw, &u));
        add.submit_for_plan(&stream).unwrap(); norm.submit(&stream).unwrap();
        Projection::submit_same_input([&gp, &up]).unwrap(); stream.synchronize().unwrap();
        let want = [read(&stream, &sum), read(&stream, &normed), read(&stream, &g), read(&stream, &u)];
        for buf in [&sum, &normed, &g, &u] { buf.copy_from_host_sync(&stream, &vec![0x3c; buf.len_bytes()]).unwrap(); }
        let plain = NormGroupMemberV1 { out_cols: 17408, output: None, record: None };
        let group = CudaQwen35AugurNormGroupB8V1::bind(&norm, &[(&gp, plain), (&up, plain)], Some(&add)).unwrap();
        group.submit().unwrap(); stream.synchronize().unwrap();
        assert_eq!([read(&stream, &sum), read(&stream, &normed), read(&stream, &g), read(&stream, &u)], want);
    }
    let after = cuda_device_allocation_snapshot();
    assert_eq!(after.live_bytes, before.live_bytes); assert_eq!(after.live_allocations, before.live_allocations);
}
