//! Augur reuses the existing exclusive, quota-bound Windows store. The bytes
//! here have no session authority; the Augur adapter validates its own schema.
use super::*;

pub struct OpaqueCheckpointRead {
    directory: Arc<DirectoryLease>,
    _store: Arc<StoreLease>,
    _native_directory: DirectoryLease,
    _files: [File; 2],
    pub bytes: Vec<u8>,
    pub payload_bytes: u64,
}
impl OpaqueCheckpointRead {
    pub fn native_path(&self) -> PathBuf { self.directory.root.join("native") }
    pub fn verify_namespace(&self) -> io::Result<()> { outer_namespace(&self.directory, true) }
}
/// Small, unpublished discovery result. The Augur owner must validate the live
/// bindings and descriptor-derived ID; native chunks are checked on restore.
pub struct OpaqueCheckpointMetadata {
    pub checkpoint_id: String,
    pub checkpoint_sha256: String,
    pub bytes: Vec<u8>,
    pub payload_bytes: u64,
}
pub struct OpaqueCheckpointIssue { pub checkpoint_id: String, pub reason: String }
pub struct OpaqueCheckpointScan {
    pub published: Vec<OpaqueCheckpointMetadata>,
    pub unavailable: Vec<OpaqueCheckpointIssue>,
}
impl HttpCheckpointStore {
    /// Externally hash-bound bytes, held under the same stable leases as GGUF.
    /// The caller must derive its own live identity and expected transfer plan.
    pub fn read_opaque(&self,id:&str,expected_sha256:&str)->io::Result<OpaqueCheckpointRead> {
        require(valid_digest(id) && valid_digest(expected_sha256),"checkpoint ID/digest")?;
        self.usage()?;
        self.read_opaque_metadata(id, Some(expected_sha256))
    }
    /// Rebuild a bounded index from sealed envelopes, without reading GPU state
    /// payloads. Interrupted/corrupt slots stay charged and are reported, never
    /// deleted or promoted into the index. Namespace/reservation failures abort.
    pub fn scan_opaque(&self) -> io::Result<OpaqueCheckpointScan> {
        self.usage()?;
        let mut ids=Vec::new();
        for entry in fs::read_dir(&self.lease.directory.root)? {
            let id=entry?.file_name().into_string().map_err(|_|invalid("checkpoint ID encoding"))?;
            if id!=LOCK { require(valid_digest(&id),"checkpoint discovery ID")?; ids.push(id); }
        }
        ids.sort();
        let mut scan=OpaqueCheckpointScan{published:Vec::new(),unavailable:Vec::new()};
        for id in ids {
            match self.read_opaque_metadata(&id,None) {
                Ok(read)=>scan.published.push(OpaqueCheckpointMetadata{checkpoint_id:id,
                    checkpoint_sha256:sha256(&read.bytes),bytes:read.bytes.clone(),payload_bytes:read.payload_bytes}),
                Err(error)=>scan.unavailable.push(OpaqueCheckpointIssue{checkpoint_id:id,reason:error.to_string()}),
            }
        }
        Ok(scan)
    }
    fn read_opaque_metadata(&self,id:&str,expected_sha256:Option<&str>)->io::Result<OpaqueCheckpointRead> {
        let directory=Arc::new(DirectoryLease::acquire(&self.lease.directory.root.join(id))?);
        let reservation=Reservation::read(&directory,id)?;
        let mut seal=directory.read_file(OUTER_SEAL,MAX_HTTP_METADATA_BYTES as u64)?;
        let bytes=bounded_bytes(&mut seal,MAX_HTTP_METADATA_BYTES)?;
        if let Some(expected)=expected_sha256 {require(sha256(&bytes)==expected,"checkpoint outer digest")?;}
        let mut blob=directory.read_file(OUTER_BLOB,MAX_HTTP_METADATA_BYTES as u64)?;
        require(bounded_bytes(&mut blob,MAX_HTTP_METADATA_BYTES)?==bytes,"checkpoint publication aliases")?;
        let native=DirectoryLease::acquire(&directory.root.join("native"))?;
        outer_namespace(&directory,true)?;
        Ok(OpaqueCheckpointRead{directory,_store:self.lease.clone(),_native_directory:native,
            _files:[seal,blob],bytes,payload_bytes:reservation.native_payload_bytes})
    }
}
struct VerifyOnly;
impl RestoreTarget for VerifyOnly {
    fn begin(&mut self,_:&TransferPlan)->io::Result<()>{Ok(())}
    fn write_exact(&mut self,_:usize,_:u64,_:&[u8])->io::Result<()>{Ok(())}
    fn finish_unpublished(&mut self)->io::Result<()>{Ok(())}
    fn abort(&mut self)->io::Result<()>{Ok(())}
}
impl HttpCheckpointSlot {
    /// Recheck every bounded native chunk before manifest-last outer publication.
    /// No overwrite, arbitrary request path, or partially sealed success exists.
    pub fn publish_opaque(self,id:&str,bytes:&[u8],native_sha:&str,plan:&TransferPlan)->io::Result<String> {
        require(id==self.reservation.checkpoint_id && !bytes.is_empty() && bytes.len()<=MAX_HTTP_METADATA_BYTES,
            "checkpoint envelope bound/reserved ID")?;
        require(plan.validate()?==self.reservation.native_payload_bytes,"checkpoint payload reservation")?;
        let mut reader=LocalCheckpointReader::open(&self.native_path(),native_sha,plan)?;
        let manifest_bytes=reader.manifest_bytes().to_vec();
        restore(&mut reader,&mut VerifyOnly,&manifest_bytes,native_sha,plan).map_err(io::Error::other)?;
        outer_namespace(&self.directory,false)?;
        self.directory.create_file(OUTER_BLOB,bytes)?;
        let mut blob=self.directory.read_file(OUTER_BLOB,MAX_HTTP_METADATA_BYTES as u64)?;
        require(bounded_bytes(&mut blob,MAX_HTTP_METADATA_BYTES)?==bytes,"checkpoint blob stable bytes")?;
        fs::hard_link(self.directory.root.join(OUTER_BLOB),self.directory.root.join(OUTER_SEAL))?;
        let mut seal=self.directory.read_file(OUTER_SEAL,MAX_HTTP_METADATA_BYTES as u64)?;
        require(bounded_bytes(&mut seal,MAX_HTTP_METADATA_BYTES)?==bytes,"checkpoint published bytes")?;
        outer_namespace(&self.directory,true)?;
        Ok(sha256(bytes))
    }
}

#[cfg(test)]
#[path="augur_store_tests.rs"]
mod tests;
