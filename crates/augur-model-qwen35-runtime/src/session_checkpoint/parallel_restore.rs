//! Bounded CPU verification workers; all storage and destination I/O stays on
//! the calling thread. Each owned staging buffer is verified before GPU use.
use super::*;
use std::sync::{mpsc, Arc, Mutex};

struct Work {
    index: usize,
    component: Component,
    offset: u64,
    bytes: usize,
    digest: String,
    buffer: Vec<u8>,
}
struct Verified {
    index: usize,
    valid: bool,
    buffer: Vec<u8>,
}

pub fn restore_parallel_verified<R: CheckpointReader, T: RestoreTarget>(
    store: &mut R, target: &mut T, manifest_bytes: &[u8],
    expected_manifest_sha256: &str, expected: &TransferPlan, workers: usize,
) -> Result<TransferReceipt, TransferFailure> {
    require((2..=8).contains(&workers), "restore verifier workers must be 2..8")
        .map_err(|e| failure("restore preflight", e))?;
    let manifest=Manifest::decode(manifest_bytes,expected_manifest_sha256,expected)
        .map_err(|e|failure("restore preflight",e))?;
    let mut result=receipt(&manifest,expected_manifest_sha256.into())
        .map_err(|e|failure("restore preflight",e))?;
    let width=workers.min(manifest.chunks.len());
    result.staging_bytes=width*manifest.chunk_bytes;
    let mut buffers:Vec<_>=(0..width).map(|_|vec![0;manifest.chunk_bytes]).collect();
    let mut phase="restore verification workers";
    let operation=std::thread::scope(|scope| -> io::Result<()> {
        let (work_tx,work_rx)=mpsc::channel::<Work>();
        let work_rx=Arc::new(Mutex::new(work_rx));
        let (done_tx,done_rx)=mpsc::channel::<Verified>();
        let mut handles=Vec::with_capacity(width);
        let mut spawn_error=None;
        for index in 0..width {
            let incoming=work_rx.clone();let outgoing=done_tx.clone();
            match std::thread::Builder::new().name(format!("checkpoint-sha-{index}")).spawn_scoped(scope,move|| {
                loop {
                    let work=match incoming.lock(){Ok(rx)=>rx.recv(),Err(_)=>break};
                    let Ok(work)=work else {break;};
                    // Return a failed verification on panic so the coordinator
                    // never waits for a result abandoned by a worker.
                    let valid=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        chunk_digest(&work.component,work.offset,&work.buffer[..work.bytes])==work.digest
                    })).unwrap_or(false);
                    if outgoing.send(Verified{index:work.index,valid,buffer:work.buffer}).is_err(){break;}
                }
            }) {
                Ok(handle)=>handles.push(handle),Err(error)=>{spawn_error=Some(error);break;}
            }
        }
        drop(done_tx);
        let mut outcome=if let Some(error)=spawn_error {Err(error)} else {(|| -> io::Result<()> {
            phase="restore begin";target.begin(expected)?;
            for (batch_index,batch) in manifest.chunks.chunks(width).enumerate() {
                for (position,chunk) in batch.iter().enumerate() {
                    phase="chunk read";
                    let mut buffer=buffers.pop().ok_or_else(||io::Error::other("verification buffer budget"))?;
                    let mut reader=store.open_chunk(&chunk.sha256)?;
                    reader.read_exact(&mut buffer[..chunk.bytes])?;
                    let mut tail=[0];require(reader.read(&mut tail)?==0,"chunk trailing bytes")?;
                    let work=Work{index:batch_index*width+position,component:expected.components[chunk.component].clone(),
                        offset:chunk.offset,bytes:chunk.bytes,digest:chunk.sha256.clone(),buffer};
                    work_tx.send(work).map_err(|_|io::Error::other("verification workers stopped"))?;
                }
                phase="chunk verify";
                let mut verified=Vec::with_capacity(batch.len());
                for _ in batch {verified.push(done_rx.recv().map_err(|_|io::Error::other("verification result missing"))?);}
                verified.sort_unstable_by_key(|v|v.index);
                for (position,verified) in verified.into_iter().enumerate() {
                    phase="chunk verify";
                    let index=batch_index*width+position;
                    require(verified.index==index && verified.valid,"chunk content mismatch")?;
                    let chunk=&manifest.chunks[index];
                    phase="restore write";
                    target.write_exact(chunk.component,chunk.offset,&verified.buffer[..chunk.bytes])?;
                    buffers.push(verified.buffer);
                }
            }
            phase="store final verification";store.finish_read()?;
            phase="restore finish unpublished";target.finish_unpublished()
        })()};
        // Unbounded channels carry at most width owned buffers. Dropping both
        // coordinator endpoints lets every error path drain without deadlock.
        drop(work_tx);drop(done_rx);
        for handle in handles {
            if handle.join().is_err() && outcome.is_ok() {outcome=Err(io::Error::other("verification worker panic"));}
        }
        outcome
    });
    match operation {
        Ok(())=>Ok(result),
        Err(cause)=>{let mut error=failure(phase,cause);
            if let Err(e)=target.abort(){error.abort_errors.push(format!("destination: {e}"));}
            Err(error)}
    }
}
