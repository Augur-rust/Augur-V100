use super::*;
use std::sync::atomic::{AtomicU64,Ordering};
static NEXT:AtomicU64=AtomicU64::new(1);
fn root()->PathBuf {
    let base=std::env::var_os("AUGUR_CHECKPOINT_TEST_ROOT").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    let root=base.join(format!("augur-store-index-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::SeqCst)));
    fs::create_dir(&root).unwrap();root
}
fn budget()->StoreBudget {StoreBudget{max_checkpoints:4,max_reserved_bytes:4*(MAX_PAYLOAD_BYTES+OVERHEAD)}}
struct Tiny {plan:TransferPlan}
impl CaptureSource for Tiny {
    fn begin(&mut self)->io::Result<TransferPlan>{Ok(self.plan.clone())}
    fn read_exact(&mut self,_:usize,_:u64,out:&mut[u8])->io::Result<()>{out.fill(7);Ok(())}
    fn finish(&mut self)->io::Result<TransferPlan>{Ok(self.plan.clone())}
    fn abort(&mut self)->io::Result<()>{Ok(())}
}
fn publish(store:&mut HttpCheckpointStore,label:&[u8])->(String,String,TransferPlan) {
    let id=sha256(label);let plan=TransferPlan{binding_sha256:sha256(b"independent tiny plan"),components:vec![Component{name:"tiny.state".into(),bytes:16}]};
    let slot=store.reserve(&id,16).unwrap();let mut source=Tiny{plan:plan.clone()};
    let mut writer=LocalCheckpointWriter::create(&slot.native_path(),&plan).unwrap();
    let native=save(&mut source,&mut writer,&plan,8).unwrap();drop(writer);
    let digest=slot.publish_opaque(&id,label,&native.manifest_sha256,&plan).unwrap();(id,digest,plan)
}
#[test]
fn metadata_scan_reopens_only_sealed_entries_and_retains_partial_quota() {
    let root=root();let mut store=HttpCheckpointStore::open(&root,budget()).unwrap();
    let (id,digest,_)=publish(&mut store,b"sealed envelope");let partial=sha256(b"interrupted");
    drop(store.reserve(&partial,32).unwrap());let usage=store.usage().unwrap();drop(store);
    let store=HttpCheckpointStore::open(&root,budget()).unwrap();let scan=store.scan_opaque().unwrap();
    assert_eq!(scan.published.len(),1);assert_eq!(scan.published[0].checkpoint_id,id);
    assert_eq!(scan.published[0].checkpoint_sha256,digest);assert_eq!(scan.published[0].bytes,b"sealed envelope");
    assert_eq!(scan.published[0].payload_bytes,16);assert_eq!(scan.unavailable.len(),1);
    assert_eq!(scan.unavailable[0].checkpoint_id,partial);assert!(!scan.unavailable[0].reason.is_empty());
    assert_eq!(usage,store.usage().unwrap());assert!(!root.join("index.json").exists());
}
#[test]
fn metadata_scan_does_not_read_native_chunks_but_restore_detects_corruption() {
    let root=root();let mut store=HttpCheckpointStore::open(&root,budget()).unwrap();
    let (id,digest,plan)=publish(&mut store,b"lazy payload envelope");
    let path=root.join(&id).join("native");let reader=LocalCheckpointReader::open(&path,&{
        // Native manifest hash is not an authority for this store-only fixture.
        sha256(&fs::read(path.join("manifest.json")).unwrap())
    },&plan).unwrap();
    let bytes=reader.manifest_bytes().to_vec();drop(reader);
    let manifest=Manifest::decode(&bytes,&sha256(&bytes),&plan).unwrap();
    fs::write(path.join(chunk_name(&manifest.chunks[0].sha256).unwrap()),[0;8]).unwrap();
    let scan=store.scan_opaque().unwrap();assert_eq!(scan.published.len(),1);assert!(scan.unavailable.is_empty());
    let held=store.read_opaque(&id,&digest).unwrap();assert_eq!(held.payload_bytes,16);
    let mut reader=LocalCheckpointReader::open(&held.native_path(),&sha256(&bytes),&plan).unwrap();
    assert!(restore(&mut reader,&mut VerifyOnly,&bytes,&sha256(&bytes),&plan).is_err());
}
#[test]
fn metadata_scan_quarantines_changed_aliases_without_hiding_healthy_entries() {
    let root=root();let mut store=HttpCheckpointStore::open(&root,budget()).unwrap();
    let (good,_,_)=publish(&mut store,b"healthy");let (bad,_,_)=publish(&mut store,b"changed");
    fs::remove_file(root.join(&bad).join(OUTER_BLOB)).unwrap();
    fs::write(root.join(&bad).join(OUTER_BLOB),b"different").unwrap();
    let usage=store.usage().unwrap();let scan=store.scan_opaque().unwrap();
    assert_eq!(scan.published.len(),1);assert_eq!(scan.published[0].checkpoint_id,good);
    assert_eq!(scan.unavailable.len(),1);assert_eq!(scan.unavailable[0].checkpoint_id,bad);
    assert_eq!(store.usage().unwrap(),usage);
}
#[test]
fn metadata_scan_never_accepts_unknown_root_entries_or_bad_reservations() {
    for (leaf,contents) in [("index.json",b"{}".as_slice()),("owner.lock",b"bad".as_slice())] {
        let root=root();let store=HttpCheckpointStore::open(&root,budget()).unwrap();
        if leaf=="owner.lock" {drop(store);fs::write(root.join(leaf),contents).unwrap();assert!(HttpCheckpointStore::open(&root,budget()).is_err());}
        else{fs::write(root.join(leaf),contents).unwrap();assert!(store.scan_opaque().is_err());}
    }
}

#[test]
fn metadata_scan_bad_reservation_blocks_admission_even_with_a_healthy_seal() {
    let root=root();let mut store=HttpCheckpointStore::open(&root,budget()).unwrap();
    let (id,_,_)=publish(&mut store,b"bad quota reservation");
    fs::write(root.join(&id).join(RESERVATION),b"{}").unwrap();
    assert!(store.scan_opaque().is_err());
    assert!(store.has_capacity_for(16).is_err());
    drop(store);assert!(HttpCheckpointStore::open(&root,budget()).is_err());
}
