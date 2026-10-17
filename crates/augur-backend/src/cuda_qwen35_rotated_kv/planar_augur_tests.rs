//! Real SM70 layout/consumer tests without external model weights.
//! Sparse initialized history is a structural test, not a long-context TPS claim.
use super::*;
use crate::{cuda_device_allocation_snapshot, OwnedCudaBuffer};
const CAP: usize = 262144;
const ROW: usize = 1088;
fn upload(stream: &OwnedCudaStream, n: usize, salt: usize) -> OwnedCudaBuffer {
    let raw: Vec<u8> = (0..n).flat_map(|i|
        (((i * 37 + salt) % 127) as f32 / 127.0 - 0.5).to_le_bytes()).collect();
    let mut b = OwnedCudaBuffer::allocate(0, raw.len()).unwrap();
    b.copy_from_host_sync(stream, &raw).unwrap(); b
}
fn read(stream: &OwnedCudaStream, b: &OwnedCudaBuffer, offset: usize, n: usize) -> Vec<u8> {
    let mut bytes = vec![0; n];
    b.view(offset, n).unwrap().copy_to_host_sync(stream, &mut bytes).unwrap(); bytes
}
fn unplanar(row: &[u8]) -> Vec<u8> {
    let mut plain = vec![0; ROW];
    for head in 0..4 { for block in 0..8 {
        let src = head * 272; let dst = (head * 8 + block) * 34;
        plain[dst..dst+2].copy_from_slice(&row[src+256+block*2..src+258+block*2]);
        plain[dst+2..dst+34].copy_from_slice(&row[src+block*32..src+(block+1)*32]);
    }} plain
}
fn close(a: &[u8], b: &[u8]) -> f32 {
    let mut max = 0f32;
    for (x,y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let x = f32::from_le_bytes(x.try_into().unwrap());
        let y = f32::from_le_bytes(y.try_into().unwrap());
        assert!(x.is_finite() && y.is_finite());
        max = max.max((x-y).abs());
        assert!((x-y).abs() <= 0.0001 + 0.002*x.abs(), "attention {x} vs {y}");
    } max
}
// Split-K changes F16 probability normalization and reduction groups. This is
// a bounded numerical-difference check, not canonical output equivalence.
fn split_difference(a: &[u8], b: &[u8]) -> (f32, f64) {
    assert_eq!(a.len(), b.len());
    let (mut max, mut error, mut norm) = (0f32, 0f64, 0f64);
    for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let x = f32::from_le_bytes(x.try_into().unwrap());
        let y = f32::from_le_bytes(y.try_into().unwrap());
        assert!(x.is_finite() && y.is_finite());
        let delta = x as f64 - y as f64;
        max = max.max((x-y).abs()); error += delta*delta; norm += (x as f64)*(x as f64);
    }
    let relative_l2 = (error / norm.max(f64::MIN_POSITIVE)).sqrt();
    eprintln!("SPLIT2_DIFFERENCE max_abs={max} relative_l2={relative_l2}");
    // Dense screen: <=0.00421 absolute and <=0.00743 relative L2 at 32K.
    assert!(max <= 0.01 && relative_l2 <= 0.02, "split difference {max}/{relative_l2}");
    (max, relative_l2)
}
#[test]
#[ignore = "exclusive SM70: Augur planar prefill/B1/B8/256Ki layout and replay"]
fn augur_planar_gqa8_full256k_layout_and_consumers() {
    let before = cuda_device_allocation_snapshot();
    {
        let s = OwnedCudaStream::create(0).unwrap();
        let q = upload(&s, 512*6144, 3); let gate = upload(&s, 512*6144, 11);
        let k = upload(&s, 512*1024, 17); let v = upload(&s, 512*1024, 23);
        let mut caches = Vec::new();
        for _ in 0..4 { let mut b=OwnedCudaBuffer::allocate(0,CAP*ROW).unwrap();
            b.zero_sync(&s).unwrap(); caches.push(b); }
        let outputs: Vec<_> = (0..4).map(|_| OwnedCudaBuffer::allocate(0,512*6144*4).unwrap()).collect();
        let wide = CudaQwen35RotatedKvInputPlanV1::new(&s).unwrap();
        let planar_wide = CudaQwen35RotatedKvInputPlanV1::new_planar(&s).unwrap();
        let small = CudaQwen35RotatedKvGqa8PlanV1::new(&s).unwrap();
        let planar_small = CudaQwen35RotatedKvGqa8PlanV1::new_planar(&s).unwrap();
        let scratch = OwnedCudaBuffer::allocate(0, input_geometry(CAP,0,512).unwrap()).unwrap();
        let views = |rows: usize, planar: bool, capacity: usize| {
            let view = |b: &OwnedCudaBuffer,n| b.view(0,n).unwrap();
            let base = if planar {2} else {0};
            Qwen35RotatedKvViews {
                query:view(&q,rows*6144*4), gate:view(&gate,rows*6144*4),
                key:view(&k,rows*1024*4), value:view(&v,rows*1024*4),
                key_cache:view(&caches[base],capacity*ROW), value_cache:view(&caches[base+1],capacity*ROW),
                attention:view(&outputs[base],rows*6144*4), gated:view(&outputs[base+1],rows*6144*4),
                workspace:view(&scratch,if rows >= 9 { input_geometry(CAP,0,rows).unwrap() } else { rows*32768+INPUT_META_BYTES }),
            }
        };
        // Same producers; Ninfer tiling and length-dependent partitions change reduction groups.
        for (pos,rows) in [(32543,224),(32544,224),(32545,224),(32287,480),(32288,480),(32289,480),(130560,480),(131072,224),(261920,224),(261664,480),(0,9),(247,16),(32760,32),(0,512),(511,512),(512,512),(513,512),(588,512),(767,512),(768,512),(1025,512),(1535,512),(1536,512),(1537,512),(7679,512),(7680,512),(7681,512),(32248,512),(32256,512),(32257,512),(48639,512),(48640,512),(48641,512),(65023,512),(65024,512),(65025,512),(97791,512),(97792,512),(97793,512),(130559,512),(130560,512),(130561,512),(32639,128),(32640,128),(32641,128),(32511,256),(32512,256),(32513,256),(65407,128),(65408,128),(65409,128),(65279,256),(65280,256),(65281,256),(65528,256),(196560,128),(196560,256),(261631,128),(261632,128),(261633,128),(261631,256),(261632,256),(261633,256),(262016,128),(261888,256),(196080,512),(196095,512),(196096,512),(196097,512),(245247,512),(245248,512),(245249,512),(261631,512),(261632,512)] {
            wide.submit(&s,views(rows,false,CAP),CAP,pos,rows).unwrap();
            planar_wide.submit(&s,views(rows,true,CAP),CAP,pos,rows).unwrap();
            let baseline = read(&s,&outputs[1],0,rows*6144*4);
            let candidate = read(&s,&outputs[3],0,rows*6144*4);
            let changed_prefill = (rows == 512 && (1024..=262144).contains(&(pos+rows)))
                || (matches!(rows,128|224|256|480) && (32768..=262144).contains(&(pos+rows)));
            let (max, relative_l2) = if changed_prefill { split_difference(&baseline, &candidate) }
                else { (close(&baseline, &candidate), 0.0) };
            planar_wide.submit(&s,views(rows,true,CAP),CAP,pos,rows).unwrap();
            assert_eq!(candidate,read(&s,&outputs[3],0,rows*6144*4),"wide replay {pos}/{rows}");
            for r in [pos,pos+rows/2,pos+rows-1] { for pair in 0..2 {
                assert_eq!(read(&s,&caches[pair],r*ROW,ROW),unplanar(&read(&s,&caches[pair+2],r*ROW,ROW)));
            }}
            eprintln!("PLANAR_WIDE position={pos} rows={rows} changed_prefill={changed_prefill} max_abs={max} relative_l2={relative_l2} replay_exact=true");
        }
        // Early-KV/shared-Q six-live-head cohort across pages and full capacity.
        for pos in [0,31,255,256,32760,65528,65536,196600,262136,262143] {
            for rows in 1..=8.min(CAP-pos) {
                planar_small.submit(&s,views(rows,true,CAP),CAP,pos,rows).unwrap();
                let first=read(&s,&outputs[3],0,rows*6144*4);
                assert!(first.chunks_exact(4).all(|x|f32::from_le_bytes(x.try_into().unwrap()).is_finite()));
                if pos+rows<=196608 {
                    small.submit(&s,views(rows,false,196608),196608,pos,rows).unwrap();
                    close(&read(&s,&outputs[1],0,rows*6144*4), &first);
                    for r in pos..pos+rows {for pair in 0..2 {
                        assert_eq!(read(&s,&caches[pair],r*ROW,ROW),unplanar(&read(&s,&caches[pair+2],r*ROW,ROW)));
                    }}
                }
                let end=(pos+rows).div_ceil(256)*256;
                for cache in &caches[2..] {
                    if end>pos+rows {assert!(read(&s,cache,(pos+rows)*ROW,(end-pos-rows)*ROW).iter().all(|&b|b==0));}
                }
                planar_small.submit(&s,views(rows,true,CAP),CAP,pos,rows).unwrap();
                assert_eq!(first,read(&s,&outputs[3],0,rows*6144*4),"replay {pos}/{rows}");
                eprintln!("PLANAR_GQA8 position={pos} rows={rows} finite=true replay_exact=true");
            }
        }
        // No enqueue on invalid admission; future output remains unchanged.
        let saved=read(&s,&outputs[3],0,6144*4);
        assert!(planar_small.submit(&s,views(8,true,CAP),CAP,CAP-7,8).is_err());
        assert!(planar_small.submit(&s,views(9,true,CAP),CAP,0,9).is_err());
        assert_eq!(saved,read(&s,&outputs[3],0,6144*4));
        s.synchronize().unwrap();
    }
    let after=cuda_device_allocation_snapshot();
    assert_eq!(before.live_bytes,after.live_bytes);
    assert_eq!(before.live_allocations,after.live_allocations);
}

// A host-computed oracle distinguishes retained prompt rows, generated rows,
// future rows and every causal suffix row. This is not a model quality test.
#[test]
#[ignore = "SM70 history-retention diagnostic; about 550 MiB device memory"]
fn augur_planar_gqa8_retains_generated_history() {
    const PROMPT: usize = 14478;
    let before = cuda_device_allocation_snapshot();
    let mut receipts = Vec::new();
    {
        let s = OwnedCudaStream::create(0).unwrap();
        let zero = |n: usize| {
            let mut b = OwnedCudaBuffer::allocate(0, n).unwrap();
            b.zero_sync(&s).unwrap(); b
        };
        let q = zero(8*6144*4); let gate = zero(8*6144*4);
        let k = zero(8*1024*4); let v = zero(8*1024*4);
        let kc = zero(CAP*ROW); let mut vc = zero(CAP*ROW);
        let attention = zero(8*6144*4); let gated = zero(8*6144*4);
        let scratch = zero(8*32768+INPUT_META_BYTES);
        let plan = CudaQwen35RotatedKvGqa8PlanV1::new_planar(&s).unwrap();
        // FWHT64 maps a constant original value into four independent DC
        // coordinates per head. Q8 scale 1/8 with signed code +/-64 gives
        // exactly +/-8 in rotated space, hence exactly +/-1 after FWHT64.
        let row = |value: i8, scale: u16| {
            let mut bytes = [0u8; ROW];
            for head in 0..4 { for block in [0usize,2,4,6] {
                bytes[head*272+block*32] = value as u8;
                let off = head*272+256+block*2;
                bytes[off..off+2].copy_from_slice(&scale.to_le_bytes());
            }} bytes
        };
        let prompt = row(64,0x3000); // +1 original space
        let generated = row(-64,0x3000); // -1 original space
        let future = row(64,0x4800); // +64, catches noncausal/global-capacity reads
        for position in [0,255,14478,24893,28792,32767,32768,65535,65536,80759,262136,262143] {
            for rows in [1,8.min(CAP-position)] {
                let mut initial = vec![0u8; CAP*ROW];
                for (i, bytes) in initial.chunks_exact_mut(ROW).enumerate() {
                    bytes.copy_from_slice(if i<position.min(PROMPT) {&prompt}
                        else if i<position {&generated} else {&future});
                }
                vc.copy_from_host_sync(&s,&initial).unwrap();
                let views = |vc: &OwnedCudaBuffer| Qwen35RotatedKvViews {
                    query:q.view(0,rows*6144*4).unwrap(), gate:gate.view(0,rows*6144*4).unwrap(),
                    key:k.view(0,rows*1024*4).unwrap(), value:v.view(0,rows*1024*4).unwrap(),
                    key_cache:kc.view(0,CAP*ROW).unwrap(), value_cache:vc.view(0,CAP*ROW).unwrap(),
                    attention:attention.view(0,rows*6144*4).unwrap(), gated:gated.view(0,rows*6144*4).unwrap(),
                    workspace:scratch.view(0,rows*32768+INPUT_META_BYTES).unwrap(),
                };
                plan.submit(&s,views(&vc),CAP,position,rows).unwrap();
                let actual=read(&s,&attention,0,rows*6144*4);
                let old=position.min(PROMPT);
                let recent=position-old;
                let mut max_error=0f64;
                for (r, bytes) in actual.chunks_exact(6144*4).enumerate() {
                    // Q=K=0: exactly uniform over [0, position+r], never future.
                    let expected=(old as f64-recent as f64)/(position+r+1) as f64;
                    for x in bytes.chunks_exact(4) {
                        let x=f32::from_le_bytes(x.try_into().unwrap()) as f64;
                        max_error=max_error.max((x-expected).abs());
                        assert!(x.is_finite() && (x-expected).abs() < 0.004,
                            "history oracle pos={position} rows={rows} row={r}: {x} vs {expected}");
                    }
                }
                // An append must not alter any accepted historical V byte.
                if position>0 {assert_eq!(read(&s,&vc,0,position*ROW),initial[..position*ROW],
                    "accepted V prefix overwritten at {position}/{rows}");}
                // Negative control: simulate losing only generated history.
                // The attention result must measurably change in the predicted direction.
                let mut lost_history_delta=None;
                if recent>0 {
                    initial[old*ROW..position*ROW].fill(0);
                    vc.copy_from_host_sync(&s,&initial).unwrap();
                    plan.submit(&s,views(&vc),CAP,position,rows).unwrap();
                    let lost=read(&s,&attention,0,rows*6144*4);
                    for (r,bytes) in lost.chunks_exact(6144*4).enumerate() {
                        let expected=old as f64/(position+r+1) as f64;
                        let expected_delta=recent as f64/(position+r+1) as f64;
                        for (j,x) in bytes.chunks_exact(4).enumerate() {
                            let x=f32::from_le_bytes(x.try_into().unwrap()) as f64;
                            let off=(r*6144+j)*4;
                            let retained=f32::from_le_bytes(actual[off..off+4].try_into().unwrap()) as f64;
                            assert!(x.is_finite() && (x-expected).abs()<0.004);
                            assert!(((x-retained)-expected_delta).abs()<0.008);
                            if r==0 && j==0 {lost_history_delta=Some(x-retained);}
                        }
                    }
                }
                receipts.push(serde_json::json!({"position":position,"rows":rows,"prompt_rows":old,
                    "generated_rows":recent,"max_abs_error":max_error,
                    "accepted_value_prefix_unchanged":true,"simulated_loss_delta":lost_history_delta}));
                eprintln!("KV_HISTORY position={position} rows={rows} max_abs={max_error} generated_rows={recent} loss_delta={lost_history_delta:?}");
            }
        }
        // Also exercise nonzero K/Q: clearing generated K alone must change
        // the result, even though the generated V bytes remain intact.
        q.copy_from_host_sync(&s,&vec![1.0f32;8*6144].iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()).unwrap();
        let key_row = |scale: u16| {
            let mut bytes=[0u8;ROW];
            for head in 0..4 {
                bytes[head*272]=64;
                bytes[head*272+256..head*272+258].copy_from_slice(&scale.to_le_bytes());
            } bytes
        };
        let retained_key=key_row(0x3000); // rotated K[0]=8, score=8
        let future_key=key_row(0x3400); // rotated K[0]=16, score=16
        for position in [24893,28792,32767,65535,80759,262136] {for rows in [1,8] {
            let old=position.min(PROMPT);let recent=position-old;
            let mut keys=vec![0u8;CAP*ROW];let mut values=vec![0u8;CAP*ROW];
            for i in 0..CAP {
                keys[i*ROW..(i+1)*ROW].copy_from_slice(if i<position {&retained_key} else {&future_key});
                values[i*ROW..(i+1)*ROW].copy_from_slice(if i<old {&prompt} else if i<position {&generated} else {&future});
            }
            kc.copy_from_host_sync(&s,&keys).unwrap();vc.copy_from_host_sync(&s,&values).unwrap();
            let views=||Qwen35RotatedKvViews {
                query:q.view(0,rows*6144*4).unwrap(),gate:gate.view(0,rows*6144*4).unwrap(),
                key:k.view(0,rows*1024*4).unwrap(),value:v.view(0,rows*1024*4).unwrap(),
                key_cache:kc.view(0,CAP*ROW).unwrap(),value_cache:vc.view(0,CAP*ROW).unwrap(),
                attention:attention.view(0,rows*6144*4).unwrap(),gated:gated.view(0,rows*6144*4).unwrap(),
                workspace:scratch.view(0,rows*32768+INPUT_META_BYTES).unwrap(),
            };
            plan.submit(&s,views(),CAP,position,rows).unwrap();
            let actual=read(&s,&attention,0,rows*6144*4);
            assert_eq!(read(&s,&kc,0,position*ROW),keys[..position*ROW],"accepted K overwritten");
            assert_eq!(read(&s,&vc,0,position*ROW),values[..position*ROW],"accepted V overwritten");
            keys[old*ROW..position*ROW].fill(0);
            kc.copy_from_host_sync(&s,&keys).unwrap();
            plan.submit(&s,views(),CAP,position,rows).unwrap();
            let lost=read(&s,&attention,0,rows*6144*4);
            let tiny=(-8.0f64).exp();let mut max_error=0f64;let mut loss_delta=0f64;
            for (r,(a,b)) in actual.chunks_exact(6144*4).zip(lost.chunks_exact(6144*4)).enumerate() {
                let expected=(old as f64-recent as f64)/(position as f64+(r+1) as f64*tiny);
                let expected_lost=(old as f64-recent as f64*tiny)/(old as f64+(recent+r+1) as f64*tiny);
                for (a,b) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
                    let a=f32::from_le_bytes(a.try_into().unwrap()) as f64;
                    let b=f32::from_le_bytes(b.try_into().unwrap()) as f64;
                    max_error=max_error.max((a-expected).abs()).max((b-expected_lost).abs());
                    assert!(a.is_finite() && b.is_finite() && max_error<0.004,
                        "key history pos={position} rows={rows} got={a}/{b} expected={expected}/{expected_lost}");
                    loss_delta=b-a;
                }
            }
            receipts.push(serde_json::json!({"position":position,"rows":rows,"prompt_rows":old,"generated_rows":recent,
                "component":"nonzero_query_and_key","max_abs_error":max_error,"accepted_kv_prefix_unchanged":true,
                "simulated_key_only_loss_delta":loss_delta}));
            eprintln!("KV_KEY_HISTORY position={position} rows={rows} max_abs={max_error} key_loss_delta={loss_delta}");
        }}
        s.synchronize().unwrap();
    }
    let after=cuda_device_allocation_snapshot();
    assert_eq!(before.live_bytes,after.live_bytes);
    assert_eq!(before.live_allocations,after.live_allocations);
    if let Some(path)=std::env::var_os("AUGUR_KV_HISTORY_AUDIT_OUT") {
        std::fs::write(path,serde_json::to_vec_pretty(&serde_json::json!({
            "status":"PASS","scope":"synthetic planar GQA8 history retention, not full model or GDN",
            "capacity":CAP,"cases":receipts,"device_allocation_closed":true})).unwrap()).unwrap();
    }
}
