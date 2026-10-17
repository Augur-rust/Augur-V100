//! Bounded, ordered copy -> per-plane SHA -> global SHA pipeline.
//! CUDA and optional snapshot IO stay on the caller; workers only own host bytes.
use sha2::{Digest, Sha256};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};

#[derive(Debug, PartialEq, Eq)]
pub struct Digests {
    pub global: [u8; 32],
    pub planes: Vec<[u8; 32]>,
}

#[derive(Clone)]
pub struct Plane {
    pub header: Vec<u8>,
    pub bytes: usize,
}

struct Chunk {
    data: Vec<u8>,
    plane: usize,
    offset: usize,
    used: usize,
}

fn receive(
    rx: &Receiver<Chunk>,
    plane: usize,
    offset: usize,
    used: usize,
) -> Result<Chunk, String> {
    let item = rx
        .recv()
        .map_err(|_| "hash input ended before declared extent")?;
    if item.plane != plane || item.offset != offset || item.used != used || item.data.len() < used {
        return Err("hash chunk ordering/extent drift".into());
    }
    Ok(item)
}

fn plane_worker(
    planes: &[Plane],
    chunk_bytes: usize,
    input: Receiver<Chunk>,
    output: SyncSender<Chunk>,
) -> Result<Vec<[u8; 32]>, String> {
    let mut hashes = Vec::new();
    for (index, plane) in planes.iter().enumerate() {
        let mut hash = Sha256::new();
        for offset in (0..plane.bytes).step_by(chunk_bytes) {
            let chunk = receive(&input, index, offset, chunk_bytes.min(plane.bytes - offset))?;
            hash.update(&chunk.data[..chunk.used]);
            output
                .send(chunk)
                .map_err(|_| "global hash worker stopped")?;
        }
        hashes.push(hash.finalize().into());
    }
    if input.recv().is_ok() {
        return Err("extra hash input past declared extent".into());
    }
    Ok(hashes)
}

fn global_worker(
    domain: &[u8],
    planes: &[Plane],
    chunk_bytes: usize,
    input: Receiver<Chunk>,
    recycled: SyncSender<Vec<u8>>,
) -> Result<[u8; 32], String> {
    let mut hash = Sha256::new();
    hash.update(domain);
    for (index, plane) in planes.iter().enumerate() {
        hash.update(&plane.header);
        for offset in (0..plane.bytes).step_by(chunk_bytes) {
            let chunk = receive(&input, index, offset, chunk_bytes.min(plane.bytes - offset))?;
            hash.update(&chunk.data[..chunk.used]);
            recycled
                .send(chunk.data)
                .map_err(|_| "hash producer stopped")?;
        }
    }
    if input.recv().is_ok() {
        return Err("extra global hash input past declared extent".into());
    }
    Ok(hash.finalize().into())
}

/// Reuses three byte buffers. Backpressure bounds all queued/active memory to
/// exactly 3 * chunk_bytes, excluding small metadata and the returned digests.
/// Every early return drops channels and joins both workers; none outlives the call.
pub fn run<F>(
    domain: &[u8],
    planes: &[Plane],
    chunk_bytes: usize,
    mut read: F,
) -> Result<Digests, String>
where
    F: FnMut(usize, usize, &mut [u8]) -> Result<(), String>,
{
    if chunk_bytes == 0
        || chunk_bytes > 16 * 1024 * 1024
        || planes.is_empty()
        || planes.len() > 1024
        || planes.iter().any(|p| p.bytes == 0)
    {
        return Err("bounded plane/chunk extent required".into());
    }
    let (free_tx, free_rx) = sync_channel(3);
    for _ in 0..3 {
        free_tx
            .send(vec![0; chunk_bytes])
            .map_err(|_| "buffer initialization")?;
    }
    let (local_tx, local_rx) = sync_channel(3);
    let (global_tx, global_rx) = sync_channel(3);
    std::thread::scope(move |scope| {
        // Handle thread-creation failure without leaving a scoped worker blocked.
        let local = std::thread::Builder::new()
            .name("seal-plane-sha".into())
            .spawn_scoped(scope, move || {
                plane_worker(planes, chunk_bytes, local_rx, global_tx)
            })
            .map_err(|e| e.to_string())?;
        let global = match std::thread::Builder::new()
            .name("seal-global-sha".into())
            .spawn_scoped(scope, move || {
                global_worker(domain, planes, chunk_bytes, global_rx, free_tx)
            }) {
            Ok(worker) => worker,
            Err(e) => {
                drop(local_tx);
                drop(free_rx);
                let _ = local.join();
                return Err(e.to_string());
            }
        };
        let produced: Result<(), String> = (|| {
            for (plane, extent) in planes.iter().enumerate() {
                for offset in (0..extent.bytes).step_by(chunk_bytes) {
                    let mut data = free_rx
                        .recv()
                        .map_err(|_| "hash buffer recycling stopped")?;
                    let used = chunk_bytes.min(extent.bytes - offset);
                    read(plane, offset, &mut data[..used])?;
                    local_tx
                        .send(Chunk {
                            data,
                            plane,
                            offset,
                            used,
                        })
                        .map_err(|_| "plane hash worker stopped")?;
                }
            }
            Ok(())
        })();
        drop(local_tx);
        // A failed read must disconnect recycling BEFORE joining: the global
        // worker may otherwise be blocked returning a buffer to an absent consumer.
        if produced.is_err() {
            drop(free_rx);
        }
        let local_result = local
            .join()
            .map_err(|_| "plane hash worker panicked".to_string());
        let global_result = global
            .join()
            .map_err(|_| "global hash worker panicked".to_string());
        produced?;
        Ok(Digests {
            global: global_result??,
            planes: local_result??,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Vec<Plane>, Vec<Vec<u8>>) {
        let data: Vec<Vec<u8>> = [1, 63, 64, 65, 127, 129, 4099]
            .into_iter()
            .map(|n| (0..n).map(|v| ((v * 23 + 17) % 251) as u8).collect())
            .collect();
        let planes = data
            .iter()
            .enumerate()
            .map(|(i, d)| Plane {
                header: [i.to_le_bytes(), d.len().to_le_bytes()].concat(),
                bytes: d.len(),
            })
            .collect();
        (planes, data)
    }
    #[test]
    fn all_bytes_and_headers_match_serial_sha_in_every_chunking() {
        let (planes, data) = fixture();
        let mut global = Sha256::new();
        global.update(b"test-domain\0");
        for (plane, bytes) in planes.iter().zip(&data) {
            global.update(&plane.header);
            global.update(bytes);
        }
        let expected = Digests {
            global: global.finalize().into(),
            planes: data.iter().map(|b| Sha256::digest(b).into()).collect(),
        };
        for size in [1, 63, 64, 65, 127, 1024, 4096, 8192] {
            let actual = run(b"test-domain\0", &planes, size, |p, o, out| {
                out.copy_from_slice(&data[p][o..o + out.len()]);
                Ok(())
            })
            .unwrap();
            assert_eq!(actual, expected, "chunk={size}");
        }
    }
    #[test]
    fn read_failures_at_each_pipeline_position_join_workers() {
        let (planes, data) = fixture();
        for fail_at in [0, 1, 2, 3, 4, 10, 20, 50] {
            let mut calls = 0;
            let result = run(b"d", &planes, 64, |p, o, out| {
                if calls == fail_at {
                    return Err("injected read failure".into());
                }
                calls += 1;
                out.copy_from_slice(&data[p][o..o + out.len()]);
                Ok(())
            });
            assert_eq!(result.unwrap_err(), "injected read failure");
        }
    }
    #[test]
    fn invalid_extent_rejects_before_reading() {
        let (mut planes, _) = fixture();
        for size in [0, 16 * 1024 * 1024 + 1] {
            assert!(run(b"d", &planes, size, |_, _, _| panic!("unexpected read")).is_err());
        }
        planes[0].bytes = 0;
        assert!(run(b"d", &planes, 64, |_, _, _| panic!("unexpected read")).is_err());
        assert!(run(b"d", &[], 64, |_, _, _| panic!("unexpected read")).is_err());
    }
    #[test]
    fn changed_byte_or_header_changes_digest() {
        let (planes, mut data) = fixture();
        let hash = |p: &[Plane], d: &[Vec<u8>]| {
            run(b"d", p, 64, |i, o, out| {
                out.copy_from_slice(&d[i][o..o + out.len()]);
                Ok(())
            })
            .unwrap()
        };
        let before = hash(&planes, &data);
        data[2][31] ^= 1;
        let after = hash(&planes, &data);
        assert_ne!(before.global, after.global);
        assert_ne!(before.planes[2], after.planes[2]);
        let mut changed = planes.clone();
        changed[1].header.push(1);
        let header = hash(&changed, &data);
        assert_ne!(after.global, header.global);
        assert_eq!(after.planes, header.planes);
    }
    #[test]
    fn caller_panic_drops_queues_before_joining_workers() {
        let (planes, _) = fixture();
        let caught = std::panic::catch_unwind(|| {
            let _ = run(b"d", &planes, 64, |_, _, _| {
                panic!("injected callback panic")
            });
        });
        assert!(caught.is_err());
    }
}
