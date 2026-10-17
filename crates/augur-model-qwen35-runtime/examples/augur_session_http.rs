//! Serial loopback HTTP bridge to the separately owned Augur token producer.
//! No CUDA in this adapter. One exact child, no replacement before confirmed exit.
use augur_model_qwen35_runtime::augur_session::{manifest, *};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write, Seek, SeekFrom},
    net::{SocketAddr, TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    time::{Duration, Instant},
};
#[path = "resident_service/http_teardown.rs"]
mod http_teardown;
#[path = "resident_service/http_transport.rs"]
mod http_transport;
#[path = "augur_service/protocol.rs"]
mod protocol;
#[path = "augur_service/http_wire.rs"]
mod wire;
#[path = "augur_service/cancellation.rs"]
mod cancellation;
use http_transport::ProducerFrames;
const MAX_FRAME: usize = 4 * 1024 * 1024;
const RETIRE_GRACE: Duration = Duration::from_secs(30);
struct Options {
    bind: SocketAddr,
    producer: PathBuf,
    producer_sha: String,
    model: PathBuf,
    manifest: PathBuf,
    manifest_sha: String,
    metadata: Value,
    eos: Vec<u32>,
    evidence: PathBuf,
}
struct Owner {
    child: Child,
    stdin: Option<ChildStdin>,
    frames: Option<ProducerFrames>,
    evidence: PathBuf,
    manifest_sha: String,
    producer_sha: String,
    active: Option<protocol::Active>,
    retired: bool,
    recovery: bool,
    recovery_serial: u64,
    last_discard: Option<Value>,
    reserved_receipt: Option<(String, File)>,
}
fn bounded(path: &Path) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take(MAX_FRAME as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_FRAME {
        return Err("receipt extent".into());
    }
    Ok(bytes)
}
fn check_receipt(root: &Path, rid: &str, expected: &str, value: &Value) -> Result<()> {
    if !valid_id(rid) || !valid_digest(expected) {
        return Err("receipt identity".into());
    }
    let bytes = bounded(&root.join("turns").join(format!("{rid}.json")))?;
    if digest(&bytes) != expected || serde_json::from_slice::<Value>(&bytes)? != *value {
        return Err("persisted receipt mismatch".into());
    }
    Ok(())
}
fn check_receipt_reservation(frame: &Value, request_id: &str, session_id: &str,
    enabled: bool, seen: bool, tokens_seen: bool) -> Result<()> {
    if !enabled || seen || tokens_seen || frame["event"] != "receipt_reserved"
        || !valid_id(request_id) || !valid_id(session_id)
        || frame["request_id"] != request_id || frame["session_id"] != session_id
        || frame["product_ready"] != false {
        return Err("foreign, late or duplicate receipt reservation".into());
    }
    Ok(())
}
fn open_reserved_receipt(root: &Path, rid: &str) -> Result<File> {
    if !valid_id(rid) {return Err("invalid reserved receipt id".into());}
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(windows)] {
        use std::os::windows::fs::OpenOptionsExt;
        // Permit producer writes, deny delete/rename while this authenticated
        // receipt object is retained. Never replace pathname checks with a
        // stale handle to a different file.
        options.share_mode(0x1 | 0x2);
    }
    Ok(options.open(root.join("turns").join(format!("{rid}.json")))?)
}
fn check_cancelled_receipt_uncommitted(root: &Path, rid: &str,
    reservation: Option<&(String, File)>, enabled: bool) -> Result<()> {
    if !valid_id(rid) { return Err("cancelled receipt identity".into()); }
    if enabled {
        if let Some((bound, file)) = reservation {
            if bound == rid {
                // The authenticated discard has already drained/reset the producer.
                // Denied DELETE sharing pins this zero-byte reservation to its path.
                if file.metadata()?.len() == 0 { return Ok(()); }
                return Err("cancelled turn has a partial or committed receipt".into());
            }
        }
    }
    if root.join("turns").join(format!("{rid}.json")).try_exists()? {
        return Err("cancelled turn has an unbound or committed receipt".into());
    }
    Ok(())
}
fn read_reserved_receipt(file: &mut File, expected: &str) -> Result<Value> {
    if !valid_digest(expected) {return Err("receipt identity".into());}
    file.seek(SeekFrom::Start(0))?;
    let mut bytes = Vec::new();
    file.take(MAX_FRAME as u64 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > MAX_FRAME || digest(&bytes) != expected {
        return Err("persisted reserved receipt mismatch".into());
    }
    Ok(serde_json::from_slice(&bytes)?)
}
fn check_reserved_receipt(file: &mut File, expected: &str, value: &Value) -> Result<()> {
    if read_reserved_receipt(file, expected)? != *value {
        return Err("persisted reserved receipt value mismatch".into());
    }
    Ok(())
}
fn hydrate_receipt_reference(frame: &mut Value, file: &mut File) -> Result<()> {
    let committed=frame["committed"].as_object().ok_or("reference committed shape")?;
    if frame["receipt_transport"]!="bound_receipt_v1" || committed.len()!=1 {
        return Err("reference terminal layout mismatch".into());
    }
    let hash=committed.get("receipt_sha256").and_then(Value::as_str).ok_or("reference digest")?;
    let receipt=read_reserved_receipt(file,hash)?;
    frame["committed"]["receipt"]=receipt;
    Ok(())
}
fn compact_receipt_vectors(frame: &mut Value) -> Result<()> {
    let receipt=frame["committed"]["receipt"].as_object_mut().ok_or("receipt shape")?;
    let output=receipt.remove("output_token_ids").ok_or("receipt output vector")?;
    let session=receipt.get_mut("session").and_then(Value::as_object_mut).ok_or("receipt session")?;
    let consumed=session.remove("consumed_token_ids").ok_or("receipt history vector")?;
    frame["receipt_vectors"]=json!({"schema":"from_stream_v1",
        "output_tokens":output.as_array().ok_or("output vector type")?.len(),
        "consumed_tokens":consumed.as_array().ok_or("history vector type")?.len()});
    Ok(())
}

impl Owner {
    fn spawn(o: &Options, serial: u64, client: &TcpStream) -> Result<Self> {
        let recovery = match std::env::var("AUGUR_CANCEL_RECOVERY").as_deref() {
            Ok("discard") if cfg!(feature = "qwen35-augur-http") => true,
            Ok("retire") | Err(std::env::VarError::NotPresent) => false,
            _ => return Err("invalid AUGUR_CANCEL_RECOVERY".into()),
        };
        http_transport::client_connected(client)?;
        // Pin each spawned executable, including after a previous retirement.
        if digest(&fs::read(&o.producer)?) != o.producer_sha {
            return Err("producer binary identity drift".into());
        }
        let dir = o.evidence.join(format!("owner-{serial}"));
        fs::create_dir(&dir)?;
        let mut cmd = Command::new(&o.producer);
        cmd.arg("--model-root")
            .arg(&o.model)
            .arg("--manifest")
            .arg(&o.manifest)
            .arg("--manifest-sha256")
            .arg(&o.manifest_sha)
            .arg("--evidence-root")
            .arg(dir.join("native"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(dir.join("stderr.log"))?,
            );
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x08000000);
        }
        let child = cmd.spawn()?;
        let mut owner = Self {
            child,
            stdin: None,
            frames: None,
            evidence: dir,
            manifest_sha: o.manifest_sha.clone(),
            producer_sha: o.producer_sha.clone(),
            active: None,
            retired: false,
            recovery,
            recovery_serial: 0,
            last_discard: None,
            reserved_receipt: None,
        };
        let started = Instant::now();
        let result = (|| -> Result<()> {
            persist_receipt(
                &owner.evidence,
                "process",
                &json!({"pid":owner.child.id(),"owner_serial":serial,
                "producer_sha256":o.producer_sha,"manifest_sha256":o.manifest_sha,"product_ready":false}),
            )?;
            owner.stdin = Some(owner.child.stdin.take().ok_or("producer stdin")?);
            owner.frames = Some(ProducerFrames::spawn(
                owner.child.stdout.take().ok_or("producer stdout")?,
                MAX_FRAME,
            )?);
            let ready = owner.receive(client)?;
            protocol::ready(&ready, &o.manifest_sha)?;
            if ready["checkpoint_capabilities"]["bindings"].is_object()
                && ready["checkpoint_capabilities"]["bindings"]["producer_sha256"]!=o.producer_sha {
                return Err("checkpoint producer binding mismatch".into());
            }
            if owner.recovery && ready["request_bound_cancellation_discard"] != true {
                return Err("producer does not support bound cancellation discard".into());
            }
            persist_receipt(
                &owner.evidence,
                "ready",
                &json!({"native":ready,"observed_startup_wall_s":started.elapsed().as_secs_f64()}),
            )?;
            Ok(())
        })();
        if let Err(error) = result {
            owner.retire("startup_failure")?;
            return Err(error);
        }
        Ok(owner)
    }
    fn receive(&self, client: &TcpStream) -> Result<Value> {
        self.frames
            .as_ref()
            .ok_or("producer reader missing")?
            .receive(client)
    }
    fn send(&mut self, value: &Value) -> Result<()> {
        let mut bytes = serde_json::to_vec(value)?;
        bytes.push(b'\n');
        let input = self.stdin.as_mut().ok_or("retired producer stdin")?;
        input.write_all(&bytes)?;
        input.flush()?;
        Ok(())
    }
    fn turn(&mut self, p: &protocol::Prepared, eos: &[u32], client: &mut TcpStream) -> Result<()> {
        self.last_discard = None;
        let mut reserved_seen = false;
        self.send(&json!({"op":"turn","request":p.request}))?;
        let mut tokens = Vec::new();
        let result = (|| -> Result<()> {
        loop {
            let frame = if self.recovery {
                loop {
                    http_transport::client_connected(client)?;
                    if let Some(frame) = self.frames.as_ref().ok_or("producer frames missing")?.poll(Duration::from_millis(25))? { break frame; }
                }
            } else { self.receive(client)? };
            match frame["event"].as_str() {
                Some("receipt_reserved") => {
                    self.bind_receipt_reservation(&frame, p, &tokens, &mut reserved_seen)?;
                }
                Some("token") => {
                    protocol::token(&frame, p, &mut tokens, eos)?;
                    wire::chunk(client, &frame)?;
                }
                Some("terminal") => {
                    let adapter_received_ns = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)?.as_nanos().to_string();
                    let check_started = std::time::Instant::now();
                    let mut frame=frame;
                    let mut receipt_readback_s=0.0;
                    if receipt_reference_enabled() {
                        let (rid,file)=self.reserved_receipt.as_mut().ok_or("missing receipt reference binding")?;
                        if !reserved_seen || *rid!=p.request.request_id {return Err("receipt reference request mismatch".into());}
                        let readback_started=Instant::now();
                        hydrate_receipt_reference(&mut frame,file)?;
                        receipt_readback_s=readback_started.elapsed().as_secs_f64();
                    } else if frame["receipt_transport"]=="bound_receipt_v1" {
                        return Err("unexpected receipt reference".into());
                    }
                    let protocol_started=Instant::now();
                    let active = protocol::terminal(&frame, p, &tokens, eos, &self.manifest_sha)?;
                    let protocol_check_s = protocol_started.elapsed().as_secs_f64();
                    if !receipt_reference_enabled() {
                        let readback_started = Instant::now();
                        if receipt_handle_enabled() {
                            let (rid, file) = self.reserved_receipt.as_mut().ok_or("missing receipt reservation")?;
                            if !reserved_seen || *rid != p.request.request_id {return Err("receipt reservation binding".into());}
                            check_reserved_receipt(file, &active.receipt_sha256, &frame["committed"]["receipt"])?;
                        } else {
                            check_receipt(&self.evidence.join("native"), &p.request.request_id,
                                &active.receipt_sha256, &frame["committed"]["receipt"])?;
                        }
                        receipt_readback_s = readback_started.elapsed().as_secs_f64();
                    }
                    self.active = Some(active);
                    if receipt_reference_enabled() {compact_receipt_vectors(&mut frame)?;}
                    frame["adapter_terminal_check_s"] = json!(check_started.elapsed().as_secs_f64());
                    frame["adapter_protocol_check_s"] = json!(protocol_check_s);
                    frame["adapter_receipt_readback_s"] = json!(receipt_readback_s);
                    frame["adapter_terminal_received_unix_ns"] = json!(adapter_received_ns);
                    frame["adapter_before_terminal_unix_ns"] = json!(std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)?.as_nanos().to_string());
                    wire::chunk(client, &frame)?;
                    return Ok(());
                }
                Some("rejected")
                    if tokens.is_empty()
                        && frame["request_id"] == p.request.request_id
                        && frame["session_id"] == p.request.session_id
                        && frame["owner_retired"] == false
                        && frame["state_unchanged"] == true =>
                {
                    wire::chunk(client, &frame)?;
                    return Ok(());
                }
                Some("error") => {
                    return Err(format!("native turn error: {}", frame["reason"]).into())
                }
                _ => return Err("unexpected native turn frame".into()),
            }
        }
        })();
        if let Err(error) = &result {
            if self.recovery && http_transport::is_client_disconnect(error.as_ref()) {
                if let Err(discard_error) = self.cancel_disconnected(p, eos, &mut tokens, &mut reserved_seen) {
                    eprintln!("AUGUR_CANCEL_DISCARD_FALLBACK={discard_error}");
                    self.last_discard = None;
                }
            }
        }
        result
    }
    fn bind_receipt_reservation(&mut self, frame: &Value, p: &protocol::Prepared,
        tokens: &[u32], seen: &mut bool) -> Result<()> {
        check_receipt_reservation(frame, &p.request.request_id,
            &p.request.session_id, receipt_handle_enabled(), *seen, !tokens.is_empty())?;
        let file = open_reserved_receipt(&self.evidence.join("native"), &p.request.request_id)?;
        // At most one completed handle is kept; release it during the next admission.
        self.reserved_receipt = Some((p.request.request_id.clone(), file));
        *seen = true;
        Ok(())
    }
    fn cancel_disconnected(&mut self, p: &protocol::Prepared, eos: &[u32], tokens: &mut Vec<u32>,
        reserved_seen: &mut bool) -> Result<()> {
        let discard_id = format!("cancel-discard-{}", self.recovery_serial);
        self.recovery_serial = self.recovery_serial.checked_add(1).ok_or("discard serial overflow")?;
        let started = Instant::now(); let deadline = started + RETIRE_GRACE;
        self.send(&json!({"op":"cancel_turn","request_id":discard_id,
            "cancelled_request_id":p.request.request_id,"session_id":p.request.session_id}))?;
        loop {
            let remaining = deadline.checked_duration_since(Instant::now()).ok_or("cancel discard deadline")?;
            let Some(frame) = self.frames.as_ref().ok_or("producer frames missing")?.poll(remaining.min(Duration::from_millis(25)))? else { continue; };
            if frame["event"] == "receipt_reserved" {
                self.bind_receipt_reservation(&frame, p, tokens, reserved_seen)?;
                continue;
            }
            if frame["event"] == "token" { protocol::token(&frame, p, tokens, eos)?; continue; }
            cancellation::validate_discard(&frame, &p.request.request_id, &p.request.session_id, &discard_id, &self.manifest_sha)?;
            check_receipt(&self.evidence.join("native"), &discard_id,
                frame["discarded"]["receipt_sha256"].as_str().ok_or("discard digest")?, &frame["discarded"]["receipt"])?;
            check_cancelled_receipt_uncommitted(&self.evidence.join("native"),
                &p.request.request_id, self.reserved_receipt.as_ref(), receipt_handle_enabled())?;
            if self.child.try_wait()?.is_some() { return Err("cancelled producer already exited".into()); }
            persist_receipt(&self.evidence, &discard_id, &json!({"event":"cancelled_turn_discard_verified",
                "discarded":frame["discarded"],"producer_sha256":self.producer_sha,
                "pid":self.child.id(),"recovery_wall_s":started.elapsed().as_secs_f64(),"product_ready":false}))?;
            self.active = None;
            self.last_discard = Some(frame["discarded"].clone());
            return Ok(());
        }
    }
    fn close(&mut self, r: &Close, client: &mut TcpStream) -> Result<()> {
        self.send(&json!({"op":"close","request_id":r.request_id,"session_id":r.session_id,"revision":r.revision}))?;
        let frame = self.receive(client)?;
        let hash = frame["receipt_sha256"]
            .as_str()
            .filter(|s| valid_digest(s))
            .ok_or("close receipt digest")?;
        if frame["event"] != "closed"
            || frame["request_id"] != r.request_id
            || frame["session_id"] != r.session_id
            || frame["revision"] != r.revision
            || frame["product_ready"] != false
        {
            return Err("close frame mismatch".into());
        }
        let raw = bounded(
            &self
                .evidence
                .join("native/turns")
                .join(format!("{}.json", r.request_id)),
        )?;
        let receipt: Value = serde_json::from_slice(&raw)?;
        if digest(&raw) != hash
            || receipt["reset_completed"] != true
            || receipt["request_id"] != r.request_id
            || receipt["session_id"] != r.session_id
            || receipt["revision"] != r.revision
            || receipt["manifest_sha256"] != self.manifest_sha
            || receipt["last_turn_receipt_sha256"]
                != self
                    .active
                    .as_ref()
                    .ok_or("missing close state")?
                    .receipt_sha256
        {
            return Err("close persisted identity/reset mismatch".into());
        }
        self.active = None;
        wire::chunk(client, &frame)
    }
    fn retire(&mut self, reason: &str) -> Result<Value> {
        if self.retired {
            return Err("owner already retired".into());
        }
        self.stdin.take();
        if let Some(f) = &mut self.frames {
            f.close_receiver();
        }
        let retirement = match http_teardown::retire_after_disconnect(&mut self.child, RETIRE_GRACE)
        {
            Ok(r) => r,
            Err(error) => {
                eprintln!(
                    "AUGUR_OWNER_REAP_UNCONFIRMED pid={} error={error}",
                    self.child.id()
                );
                std::process::exit(70);
            }
        };
        if let Some(f) = &mut self.frames {
            f.join_after_child_exit();
        }
        self.retired = true;
        self.active = None;
        let cleanup = bounded(&self.evidence.join("native/owner-stopped.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());
        let clean = cleanup.as_ref().is_some_and(|v| {
            v["event"] == "owner_stopped"
                && v["allocation_closure"] == true
                && v["live_bytes_before"] == 0
                && v["live_bytes_after"] == 0
                && v["live_allocations_before"] == 0
                && v["live_allocations_after"] == 0
                && v["drain_error"].is_null()
        });
        let record = json!({"event":"owner_process_retired","reason":reason,"pid":self.child.id(),
            "confirmed_exit":true,"exit_code":retirement.exit.code(),"force_attempted":retirement.force_attempted,
            "elapsed_s":retirement.elapsed.as_secs_f64(),"native_cleanup_verified":clean,
            "graceful_cleanup_verified":clean&&!retirement.force_attempted,"native_cleanup":cleanup,
            "producer_sha256":self.producer_sha,"manifest_sha256":self.manifest_sha,"product_ready":false});
        persist_receipt(&self.evidence, "retirement", &record)?;
        Ok(record)
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        if self.retired {
            return;
        }
        self.stdin.take();
        if let Some(f) = &mut self.frames {
            f.close_receiver();
        }
        // Last-resort containment; never present it as graceful qualification.
        let reaped = (|| -> std::io::Result<()> {
            if self.child.try_wait()?.is_none() {
                if let Err(e) = self.child.kill() {
                    if self.child.try_wait()?.is_none() {
                        return Err(e);
                    }
                }
                self.child.wait()?;
            }
            Ok(())
        })();
        if let Err(e) = reaped {
            eprintln!(
                "AUGUR_OWNER_REAP_UNCONFIRMED pid={} error={e}",
                self.child.id()
            );
            std::process::exit(70);
        }
        if let Some(f) = &mut self.frames {
            f.join_after_child_exit();
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Close {
    request_id: String,
    session_id: String,
    revision: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Empty {}
enum Action {
    Turn(protocol::Prepared),
    Close(Close),
    #[cfg(all(windows,feature="qwen35-augur-http"))] SaveCheckpoint(checkpoint::SaveRequest),
    #[cfg(all(windows,feature="qwen35-augur-http"))] RestoreCheckpoint(checkpoint::RestoreRequest),
    Shutdown,
}
fn action(path: &str, body: &[u8], owner: Option<&Owner>) -> Result<Action> {
    match path {
        "/augur/v1/turn" => Ok(Action::Turn(protocol::prepare(
            serde_json::from_slice(body)?,
            owner.and_then(|o| o.active.as_ref()),
        )?)),
        "/augur/v1/close" => {
            let r: Close = serde_json::from_slice(body)?;
            let a = owner
                .and_then(|o| o.active.as_ref())
                .ok_or("no active Augur session")?;
            if !valid_id(&r.request_id)
                || a.snapshot.session_id != r.session_id
                || a.snapshot.revision != r.revision
            {
                return Err("close identity mismatch".into());
            }
            Ok(Action::Close(r))
        }
        #[cfg(all(windows,feature="qwen35-augur-http"))]
        "/augur/v1/checkpoint/save" => {
            let r:checkpoint::SaveRequest=serde_json::from_slice(body)?;
            let a=owner.and_then(|o|o.active.as_ref()).ok_or("save needs an active session")?;
            if !checkpoint::storage_requested() || !valid_id(&r.request_id) || !valid_digest(&r.checkpoint_id)
                || r.session_id!=a.snapshot.session_id || r.revision!=a.snapshot.revision || a.snapshot.image_input_sha256.is_some() {
                return Err("checkpoint save admission".into());}
            Ok(Action::SaveCheckpoint(r))
        }
        #[cfg(all(windows,feature="qwen35-augur-http"))]
        "/augur/v1/checkpoint/restore" => {
            let r:checkpoint::RestoreRequest=serde_json::from_slice(body)?;
            if !checkpoint::storage_requested() || owner.is_some_and(|o|o.active.is_some()) || !valid_id(&r.request_id)
                || !valid_id(&r.session_id) || !valid_digest(&r.checkpoint_id) || !valid_digest(&r.checkpoint_sha256) {
                return Err("checkpoint restore admission".into());}
            Ok(Action::RestoreCheckpoint(r))
        }
        "/augur/v1/shutdown" => {
            let _: Empty = serde_json::from_slice(body)?;
            Ok(Action::Shutdown)
        }
        _ => Err("unknown Augur route".into()),
    }
}
fn handle(
    mut client: TcpStream,
    o: &Options,
    owner: &mut Option<Owner>,
    serial: &mut u64,
) -> Result<bool> {
    client.set_nodelay(true)?;
    client.set_read_timeout(Some(Duration::from_secs(15)))?;
    client.set_write_timeout(Some(Duration::from_secs(5)))?;
    let (method, path, body) = match wire::read_request(&mut client) {
        Ok(r) => r,
        Err(e) => {
            wire::json(&mut client, 400, &json!({"error":e.to_string()}))?;
            return Ok(false);
        }
    };
    if method == "GET" && path == "/health" {
        if let Some(live) = owner.as_mut() {
            if live.child.try_wait()?.is_some() {
                owner.take().unwrap().retire("exited_before_health")?;
            }
        }
        wire::json(
            &mut client,
            200,
            &json!({"profile":requested_profile()?,"schema":"quasar-token-http-v1","owner_resident":owner.is_some(),
            "session_active":owner.as_ref().is_some_and(|o|o.active.is_some()),"capacity":context_capacity()?,"maximum_output_tokens":context_output_limit()?,
            "last_cancel_discard":owner.as_ref().and_then(|o|o.last_discard.as_ref()),
            "request_bound_cancellation_discard":owner.as_ref().is_some_and(|o|o.recovery),
            "manifest_sha256":o.manifest_sha,"model_manifest":o.metadata,"producer_sha256":o.producer_sha,
            "sampling":augur_model_qwen35_runtime::augur_sampling::effective_description()?,
            "lm_head":"original_bf16","kv":"Q8_0","dflash_enabled":requested_dflash_enabled()?,"vision_enabled":cfg!(feature="qwen35-augur-vision"),"restore_enabled":checkpoint::storage_requested(),
            "product_ready":false,"qualified":false,"actual_long_history_qualified":false}),
        )?;
        return Ok(false);
    }
    if method != "POST" {
        wire::json(&mut client, 404, &json!({"error":"unknown route"}))?;
        return Ok(false);
    }
    let action = match action(&path, &body, owner.as_ref()) {
        Ok(a) => a,
        Err(e) => {
            wire::json(
                &mut client,
                400,
                &json!({"error":e.to_string(),"state_unchanged":true}),
            )?;
            return Ok(false);
        }
    };
    let (rid, sid) = match &action {
        Action::Turn(p) => (
            Some(p.request.request_id.clone()),
            Some(p.request.session_id.clone()),
        ),
        Action::Close(r) => (Some(r.request_id.clone()), Some(r.session_id.clone())),
        #[cfg(all(windows,feature="qwen35-augur-http"))]
        Action::SaveCheckpoint(r) => (Some(r.request_id.clone()),Some(r.session_id.clone())),
        #[cfg(all(windows,feature="qwen35-augur-http"))]
        Action::RestoreCheckpoint(r) => (Some(r.request_id.clone()),Some(r.session_id.clone())),
        Action::Shutdown => (None, None),
    };
    wire::begin(&mut client)?;
    let stop = matches!(&action, Action::Shutdown);
    let is_turn = matches!(&action, Action::Turn(_));
    let result = (|| -> Result<()> {
        match action {
            Action::Turn(p) => {
                if owner.is_none() {
                    let next = *serial;
                    *serial = serial.checked_add(1).ok_or("owner serial overflow")?;
                    *owner = Some(Owner::spawn(o, next, &client)?);
                }
                owner.as_mut().unwrap().turn(&p, &o.eos, &mut client)?;
            }
            Action::Close(r) => owner
                .as_mut()
                .ok_or("owner missing")?
                .close(&r, &mut client)?,
            #[cfg(all(windows,feature="qwen35-augur-http"))]
            Action::SaveCheckpoint(r) => {
                owner.as_mut().ok_or("checkpoint owner missing")?.checkpoint(false,&serde_json::to_value(r)?,&mut client)?;
            }
            #[cfg(all(windows,feature="qwen35-augur-http"))]
            Action::RestoreCheckpoint(r) => {
                if owner.is_none() {
                    let next=*serial;*serial=serial.checked_add(1).ok_or("owner serial overflow")?;
                    *owner=Some(Owner::spawn(o,next,&client)?);
                }
                owner.as_mut().unwrap().checkpoint(true,&serde_json::to_value(r)?,&mut client)?;
            }
            Action::Shutdown => {
                if let Some(mut live) = owner.take() {
                    let stopped = (|| -> Result<()> {
                        live.send(&json!({"op":"shutdown"}))?;
                        let frame = live.receive(&client)?;
                        if frame["event"] != "owner_stopped"
                            || frame["reason"] != "shutdown"
                            || frame["allocation_closure"] != true
                        {
                            return Err("native shutdown failed".into());
                        }
                        Ok(())
                    })();
                    // Even a shutdown-client RST gets cooperative drain/reaping,
                    // not an implicit forced-kill through the fallback Drop.
                    let retirement = live.retire(if stopped.is_ok() {
                        "shutdown"
                    } else {
                        "shutdown_transport_failure"
                    })?;
                    stopped?;
                    if retirement["exit_code"] != 0
                        || retirement["graceful_cleanup_verified"] != true
                    {
                        return Err("shutdown cleanup unqualified".into());
                    }
                    wire::chunk(
                        &mut client,
                        &json!({"event":"shutdown","retirement":retirement,"product_ready":false}),
                    )?;
                } else {
                    wire::chunk(
                        &mut client,
                        &json!({"event":"shutdown","no_owner":true,"product_ready":false}),
                    )?;
                }
            }
        }
        wire::end(&mut client)
    })();
    if let Err(error) = result {
        let retained = is_turn && owner.as_ref().is_some_and(|o| o.active.is_none()
            && o.last_discard.as_ref().is_some_and(|v| v["receipt"]["cancelled_request_id"] == json!(rid)
                && v["receipt"]["session_id"] == json!(sid)));
        if !retained {
            if let Some(mut live) = owner.take() { live.retire("request_or_transport_failure")?; }
        }
        eprintln!("AUGUR_HTTP_REQUEST_ERROR={error}");
        if http_transport::client_connected(&client).is_ok() {
            let _ = wire::chunk(
                &mut client,
                &json!({"event":"error","request_id":rid,"session_id":sid,"owner_retired":!retained,
                "reason":error.to_string(),"product_ready":false}),
            );
            let _ = wire::end(&mut client);
        }
        return Ok(false);
    }
    Ok(stop)
}
fn options() -> Result<Options> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let mut map = BTreeMap::new();
    let keys = [
        "--bind",
        "--producer",
        "--producer-sha256",
        "--model-root",
        "--manifest",
        "--manifest-sha256",
        "--evidence-root",
    ];
    if args.len() != keys.len() * 2 {
        return Err("required: --bind LOOPBACK:PORT --producer EXE --producer-sha256 SHA --model-root DIR --manifest FILE --manifest-sha256 SHA --evidence-root NEW_DIR".into());
    }
    for pair in args.chunks_exact(2) {
        if !keys.contains(&pair[0].as_str())
            || map.insert(pair[0].as_str(), pair[1].as_str()).is_some()
        {
            return Err("unknown/duplicate option".into());
        }
    }
    let bind: SocketAddr = map["--bind"].parse()?;
    if !bind.ip().is_loopback() {
        return Err("loopback binding required".into());
    }
    let producer = PathBuf::from(map["--producer"]);
    let producer_sha = map["--producer-sha256"].to_owned();
    if !valid_digest(&producer_sha) || digest(&fs::read(&producer)?) != producer_sha {
        return Err("pinned producer digest mismatch".into());
    }
    let model = PathBuf::from(map["--model-root"]);
    let mp = PathBuf::from(map["--manifest"]);
    let mh = map["--manifest-sha256"].to_owned();
    let admitted = manifest::load(&model, &mp, &mh)?;
    let evidence = PathBuf::from(map["--evidence-root"]);
    fs::create_dir(&evidence)?;
    Ok(Options {
        bind,
        producer,
        producer_sha,
        model,
        manifest: mp,
        manifest_sha: mh,
        metadata: serde_json::to_value(admitted.manifest())?,
        eos: admitted.eos_token_ids().to_vec(),
        evidence,
    })
}
fn main() -> Result<()> {
    requested_profile()?; // Reject invalid mode before binding/spawning a producer.
    let o = options()?;
    let listener = TcpListener::bind(o.bind)?;
    println!(
        "{}",
        json!({"event":"augur_http_listening","address":listener.local_addr()?.to_string(),"profile":requested_profile()?,"product_ready":false})
    );
    let mut owner = None;
    let mut serial = 0;
    for connection in listener.incoming() {
        match connection {
            Ok(client) => match handle(client, &o, &mut owner, &mut serial) {
                Ok(true) => break,
                Ok(false) => (),
                Err(e) => eprintln!("AUGUR_HTTP_ERROR={e}"),
            },
            Err(e) => eprintln!("AUGUR_HTTP_ACCEPT_ERROR={e}"),
        }
    }
    if let Some(mut live) = owner {
        live.retire("adapter_exit")?;
    }
    Ok(())
}

#[cfg(all(windows,feature="qwen35-augur-http"))]
impl Owner {
    fn checkpoint(&mut self,restore:bool,request:&Value,client:&mut TcpStream)->Result<()> {
        self.send(&json!({"op":if restore{"restore_checkpoint"}else{"save_checkpoint"},"request":request}))?;
        let frame=self.receive(client)?;
        let expected=if restore{"checkpoint_restored"}else{"checkpoint_saved"};
        if frame["event"]!=expected || frame["request_id"]!=request["request_id"] || frame["session_id"]!=request["session_id"] || frame["product_ready"]!=false {
            return Err(format!("checkpoint producer frame: {}",frame["reason"]).into());}
        let committed=&frame["committed"];let receipt=&committed["receipt"];
        let hash=committed["receipt_sha256"].as_str().ok_or("checkpoint receipt digest")?;
        check_receipt(&self.evidence.join("native"),request["request_id"].as_str().ok_or("checkpoint request ID")?,hash,receipt)?;
        let snapshot:SessionSnapshot=serde_json::from_value(receipt["session"].clone())?;
        let tokens:Vec<u8>=snapshot.consumed_token_ids.iter().flat_map(|t|t.to_le_bytes()).collect();
        if snapshot.profile!=requested_profile()? || snapshot.manifest_sha256!=self.manifest_sha
            || snapshot.session_id!=request["session_id"] || snapshot.image_input_sha256.is_some()
            || snapshot.consumed_token_ids.is_empty() || snapshot.consumed_token_ids.len()>=context_capacity()?
            || snapshot.consumed_token_ids.iter().any(|&t|t>=VOCAB) || snapshot.pending_token>=VOCAB
            || snapshot.consumed_input_sha256!=digest(&tokens) || receipt["request_id"]!=request["request_id"]
            || receipt["checkpoint"]["checkpoint_id"]!=request["checkpoint_id"] {
            return Err("checkpoint persisted session boundary".into());}
        if restore {
            if snapshot.revision!=1 || receipt["recomputed_prefix_tokens"]!=0
                || receipt["checkpoint"]["checkpoint_sha256"]!=request["checkpoint_sha256"] {
                return Err("checkpoint restore recomputation/identity".into());}
            self.active=Some(protocol::Active{snapshot,receipt_sha256:hash.into()});
        } else if self.active.as_ref().is_none_or(|a|a.snapshot!=snapshot) || receipt["state_unchanged"]!=true {
            return Err("checkpoint save changed active state".into());
        }
        if self.child.try_wait()?.is_some(){return Err("checkpoint producer exited before publication".into());}
        wire::chunk(client,&frame)
    }
}

#[cfg(test)]
mod receipt_handle_tests {
    use super::*;
    fn fixture() -> PathBuf {
        let root = std::env::temp_dir().join(format!("augur-http-receipt-handle-{}-{}",
            std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        fs::create_dir(&root).unwrap(); fs::create_dir(root.join("turns")).unwrap(); root
    }
    #[test]
    fn cancelled_receipt_accepts_only_current_empty_reservation_after_discard() {
        let root=fixture();
        assert!(check_cancelled_receipt_uncommitted(&root,"r1",None,false).is_ok());
        let mut writer=reserve_receipt(&root.join("turns"),"r1").unwrap();
        let bound=(String::from("r1"),open_reserved_receipt(&root,"r1").unwrap());
        assert!(check_cancelled_receipt_uncommitted(&root,"r1",Some(&bound),true).is_ok());
        assert!(check_cancelled_receipt_uncommitted(&root,"r1",Some(&bound),false).is_err());
        assert!(check_cancelled_receipt_uncommitted(&root,"r1",None,true).is_err());
        let foreign=(String::from("foreign"),open_reserved_receipt(&root,"r1").unwrap());
        assert!(check_cancelled_receipt_uncommitted(&root,"r1",Some(&foreign),true).is_err());
        writer.write_all(b"{").unwrap(); writer.sync_all().unwrap();
        assert!(check_cancelled_receipt_uncommitted(&root,"r1",Some(&bound),true).is_err());
        drop(foreign); drop(bound); drop(writer);
        fs::remove_file(root.join("turns/r1.json")).unwrap();
        fs::remove_dir(root.join("turns")).unwrap(); fs::remove_dir(root).unwrap();
    }
    #[test]
    fn reference_terminal_reads_durable_bytes_and_compacts_only_redundant_vectors() {
        let root=fixture();let value=json!({"request_id":"r1","output_token_ids":[4,5],
            "session":{"consumed_token_ids":[1,2,4,5],"pending_token":6},"other":{"kept":true}});
        let mut writer=reserve_receipt(&root.join("turns"),"r1").unwrap();
        let mut reader=open_reserved_receipt(&root,"r1").unwrap();
        let hash=persist_reserved_receipt(&mut writer,&value).unwrap();
        let reference=json!({"receipt_transport":"bound_receipt_v1","committed":{"receipt_sha256":hash}});
        let mut frame=reference.clone();hydrate_receipt_reference(&mut frame,&mut reader).unwrap();
        assert_eq!(frame["committed"]["receipt"],value);
        compact_receipt_vectors(&mut frame).unwrap();
        assert_eq!(frame["receipt_vectors"],json!({"schema":"from_stream_v1","output_tokens":2,"consumed_tokens":4}));
        assert!(frame["committed"]["receipt"].get("output_token_ids").is_none());
        assert!(frame["committed"]["receipt"]["session"].get("consumed_token_ids").is_none());
        assert_eq!(frame["committed"]["receipt"]["other"],value["other"]);
        assert_eq!(frame["committed"]["receipt_sha256"],hash);
        for bad in [json!({"receipt_transport":"inline_v1","committed":{"receipt_sha256":hash}}),
            json!({"receipt_transport":"bound_receipt_v1","committed":{"receipt_sha256":hash,"receipt":value}}),
            json!({"receipt_transport":"bound_receipt_v1","committed":{"receipt_sha256":"0".repeat(64)}})] {
            assert!(hydrate_receipt_reference(&mut bad.clone(),&mut reader).is_err());
        }
        drop(reader);drop(writer);fs::remove_file(root.join("turns/r1.json")).unwrap();
        fs::remove_dir(root.join("turns")).unwrap();fs::remove_dir(root).unwrap();
    }
    #[test]
    fn reservation_must_precede_tokens_and_bind_both_ids() {
        let frame = json!({"event":"receipt_reserved","request_id":"r1","session_id":"s1","product_ready":false});
        assert!(check_receipt_reservation(&frame,"r1","s1",true,false,false).is_ok());
        for (enabled,seen,tokens) in [(false,false,false),(true,true,false),(true,false,true)] {
            assert!(check_receipt_reservation(&frame,"r1","s1",enabled,seen,tokens).is_err());
        }
        for (field,value) in [("request_id",json!("foreign")),("session_id",json!("foreign")),
            ("event",json!("token")),("product_ready",json!(true)),("product_ready",Value::Null)] {
            let mut bad=frame.clone(); bad[field]=value;
            assert!(check_receipt_reservation(&bad,"r1","s1",true,false,false).is_err());
        }
    }
    #[test]
    fn bound_readback_requires_committed_bytes_and_exact_value() {
        let root=fixture(); let value=json!({"request_id":"r1","unicode":"中文","revision":1});
        let mut writer=reserve_receipt(&root.join("turns"),"r1").unwrap();
        let mut reader=open_reserved_receipt(&root,"r1").unwrap();
        assert!(open_reserved_receipt(&root,"../escape").is_err());
        assert!(check_reserved_receipt(&mut reader,&"0".repeat(64),&value).is_err());
        let hash=persist_reserved_receipt(&mut writer,&value).unwrap();
        assert!(check_reserved_receipt(&mut reader,&hash,&value).is_ok());
        // A second read must rewind the bound object, not reopen the pathname.
        assert!(check_reserved_receipt(&mut reader,&hash,&value).is_ok());
        assert!(check_reserved_receipt(&mut reader,"bad",&value).is_err());
        assert!(check_reserved_receipt(&mut reader,&"0".repeat(64),&value).is_err());
        assert!(check_reserved_receipt(&mut reader,&hash,&json!({"revision":2})).is_err());
        writer.set_len(1).unwrap(); writer.sync_all().unwrap();
        assert!(check_reserved_receipt(&mut reader,&hash,&value).is_err());
        writer.set_len(MAX_FRAME as u64+1).unwrap(); writer.sync_all().unwrap();
        assert!(check_reserved_receipt(&mut reader,&hash,&value).is_err());
        drop(reader); drop(writer);
        fs::remove_file(root.join("turns/r1.json")).unwrap();
        fs::remove_dir(root.join("turns")).unwrap(); fs::remove_dir(root).unwrap();
    }
    #[cfg(windows)]
    #[test]
    fn bound_handle_prevents_receipt_path_replacement_until_released() {
        let root=fixture(); let file=root.join("turns/r1.json");
        let mut writer=reserve_receipt(&root.join("turns"),"r1").unwrap();
        let mut reader=open_reserved_receipt(&root,"r1").unwrap();
        let value=json!({"request_id":"r1"});
        let hash=persist_reserved_receipt(&mut writer,&value).unwrap();
        drop(writer);
        assert!(fs::remove_file(&file).is_err());
        assert!(fs::rename(&file,root.join("turns/replaced.json")).is_err());
        assert!(check_reserved_receipt(&mut reader,&hash,&value).is_ok());
        drop(reader); fs::remove_file(&file).unwrap();
        fs::remove_dir(root.join("turns")).unwrap(); fs::remove_dir(root).unwrap();
    }
}
