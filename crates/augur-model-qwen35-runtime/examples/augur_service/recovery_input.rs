//! Request-scoped control over the same private producer stdin. EOF, malformed
//! control and pipelining remain owner-fatal; no cancellation token is reset.
use super::Command;
use augur_model_qwen35_runtime::{augur_session::valid_id, session_cancellation::SessionCancellationToken};
use serde::Deserialize;
use std::{io::{self, BufRead, BufReader, Read}, sync::{Arc, Mutex, mpsc::{self, Receiver}}, thread};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Cancel {
    pub op: String,
    pub request_id: String,
    pub cancelled_request_id: String,
    pub session_id: String,
}
pub(super) struct Work { pub raw: Vec<u8>, pub token: SessionCancellationToken }
struct Binding { request_id: String, session_id: String, token: SessionCancellationToken, cancel: Option<Cancel> }
pub(super) struct Input {
    receiver: Receiver<io::Result<Work>>,
    binding: Arc<Mutex<Option<Binding>>>,
    lifetime: SessionCancellationToken,
}
fn failure(message: &str) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, message) }
fn terminate(lifetime: &SessionCancellationToken, binding: &Mutex<Option<Binding>>) {
    lifetime.request_cancel();
    if let Ok(slot) = binding.lock() { if let Some(active) = slot.as_ref() { active.token.request_cancel(); } }
}
impl Input {
    pub(super) fn start(reader: impl Read + Send + 'static, limit: u64, lifetime: SessionCancellationToken) -> io::Result<Self> {
        if limit == 0 || limit == u64::MAX { return Err(failure("bounded input required")); }
        let (sender, receiver) = mpsc::sync_channel(1);
        let binding = Arc::new(Mutex::new(None::<Binding>));
        let worker_slot = binding.clone(); let worker_life = lifetime.clone();
        thread::Builder::new().name("augur-turn-control".into()).spawn(move || {
            let mut reader = BufReader::new(reader);
            loop {
                let result = (|| -> io::Result<Option<Work>> {
                    let mut raw = Vec::new();
                    reader.by_ref().take(limit + 1).read_until(b'\n', &mut raw)?;
                    if raw.is_empty() { return Err(failure("producer control EOF")); }
                    if raw.len() as u64 > limit || !raw.ends_with(b"\n") { return Err(failure("oversized or truncated control")); }
                    let value: serde_json::Value = serde_json::from_slice(&raw).map_err(|_| failure("invalid control JSON"))?;
                    let mut slot = worker_slot.lock().map_err(|_| failure("control lock poisoned"))?;
                    if value["op"] == "cancel_turn" {
                        let cancel: Cancel = serde_json::from_value(value).map_err(|_| failure("invalid cancellation fields"))?;
                        let active = slot.as_mut().ok_or_else(|| failure("cancel has no active request"))?;
                        if cancel.op != "cancel_turn" || !valid_id(&cancel.request_id)
                            || cancel.request_id == cancel.cancelled_request_id
                            || active.request_id != cancel.cancelled_request_id || active.session_id != cancel.session_id
                            || active.cancel.is_some() { return Err(failure("foreign or repeated cancellation")); }
                        active.cancel = Some(cancel);
                        active.token.request_cancel();
                        return Ok(None);
                    }
                    if slot.is_some() { return Err(failure("pipelined command while a turn is active")); }
                    let command: Command = serde_json::from_slice(&raw).map_err(|_| failure("invalid producer command"))?;
                    let token = SessionCancellationToken::new();
                    if let Command::Turn { request } = command {
                        if !valid_id(&request.request_id) || !valid_id(&request.session_id) { return Err(failure("invalid turn binding")); }
                        *slot = Some(Binding { request_id: request.request_id, session_id: request.session_id,
                            token: token.clone(), cancel: None });
                    }
                    Ok(Some(Work { raw, token }))
                })();
                match result {
                    Ok(None) => (),
                    Ok(Some(work)) => if sender.try_send(Ok(work)).is_err() {
                        terminate(&worker_life, &worker_slot); break;
                    },
                    Err(error) => {
                        terminate(&worker_life, &worker_slot);
                        let _ = sender.try_send(Err(error)); break;
                    }
                }
            }
        })?;
        Ok(Self { receiver, binding, lifetime })
    }
    pub(super) fn next(&self) -> io::Result<Option<Work>> {
        let work = self.receiver.recv();
        if self.lifetime.is_requested() { return Err(failure("owner control terminated")); }
        match work { Ok(value) => value.map(Some), Err(_) => Ok(None) }
    }
    pub(super) fn cancellation(&self, rid: &str, sid: &str) -> io::Result<Option<Cancel>> {
        if self.lifetime.is_requested() { return Err(failure("EOF/fault cannot authorize recovery")); }
        let slot = self.binding.lock().map_err(|_| failure("control lock poisoned"))?;
        let active = slot.as_ref().ok_or_else(|| failure("missing turn binding"))?;
        if active.request_id != rid || active.session_id != sid { return Err(failure("turn control identity mismatch")); }
        Ok(active.cancel.clone())
    }
    pub(super) fn finish(&self, rid: &str, sid: &str, discarded: bool) -> io::Result<()> {
        if self.lifetime.is_requested() { return Err(failure("owner control terminated at completion")); }
        let mut slot = self.binding.lock().map_err(|_| failure("control lock poisoned"))?;
        let active = slot.as_ref().ok_or_else(|| failure("missing completion binding"))?;
        if active.request_id != rid || active.session_id != sid || active.cancel.is_some() != discarded {
            return Err(failure("late cancellation or completion identity mismatch"));
        }
        *slot = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{io::Write, net::{TcpListener, TcpStream}, time::{Duration, Instant}};
    fn setup() -> (Input, TcpStream, SessionCancellationToken) {
        let server = TcpListener::bind("127.0.0.1:0").unwrap();
        let peer = TcpStream::connect(server.local_addr().unwrap()).unwrap();
        let reader = server.accept().unwrap().0; let life = SessionCancellationToken::new();
        (Input::start(reader, 2048, life.clone()).unwrap(), peer, life)
    }
    fn turn(peer: &mut TcpStream, rid: &str) {
        writeln!(peer, "{{\"op\":\"turn\",\"request\":{{\"request_id\":\"{rid}\",\"session_id\":\"s\",\"max_new_tokens\":4,\"input\":{{\"kind\":\"open\",\"prompt_token_ids\":[1,2]}}}}}}").unwrap();
    }
    fn wait(token: &SessionCancellationToken) {
        let until = Instant::now() + Duration::from_secs(2);
        while !token.is_requested() && Instant::now() < until { thread::sleep(Duration::from_millis(1)); }
        assert!(token.is_requested());
    }
    #[test]
    fn exact_control_cancels_only_current_turn_then_new_token_is_fresh() {
        let (input, mut peer, life) = setup(); turn(&mut peer, "r1"); let first = input.next().unwrap().unwrap();
        peer.write_all(b"{\"op\":\"cancel_turn\",\"request_id\":\"d1\",\"cancelled_request_id\":\"r1\",\"session_id\":\"s\"}\n").unwrap();
        wait(&first.token); assert!(!life.is_requested());
        assert_eq!(input.cancellation("r1", "s").unwrap().unwrap().request_id, "d1");
        assert!(input.finish("r1", "s", false).is_err()); input.finish("r1", "s", true).unwrap();
        turn(&mut peer, "r2"); let next = input.next().unwrap().unwrap();
        assert!(!next.token.is_requested()); assert!(first.token.is_requested());
        input.finish("r2", "s", false).unwrap();
    }
    #[test]
    fn eof_cancels_lifetime_and_active_turn_but_never_authorizes_discard() {
        let (input, mut peer, life) = setup(); turn(&mut peer, "r1"); let work = input.next().unwrap().unwrap();
        drop(peer); wait(&life); wait(&work.token); assert!(input.cancellation("r1", "s").is_err());
    }
    #[test]
    fn foreign_unknown_and_pipelined_controls_are_owner_fatal() {
        for bytes in [b"{\"op\":\"cancel_turn\",\"request_id\":\"d\",\"cancelled_request_id\":\"wrong\",\"session_id\":\"s\"}\n".as_slice(),
            b"{\"op\":\"shutdown\"}\n", b"{\"op\":\"cancel_turn\",\"request_id\":\"d\",\"cancelled_request_id\":\"r1\",\"session_id\":\"s\",\"extra\":true}\n"] {
            let (input, mut peer, life) = setup(); turn(&mut peer, "r1"); let work = input.next().unwrap().unwrap();
            peer.write_all(bytes).unwrap(); wait(&life); wait(&work.token);
            assert!(input.cancellation("r1", "s").is_err());
        }
    }
    #[test]
    fn repeated_cancel_is_not_idempotent_authority() {
        let (input, mut peer, life) = setup(); turn(&mut peer, "r1"); let work = input.next().unwrap().unwrap();
        let raw = b"{\"op\":\"cancel_turn\",\"request_id\":\"d\",\"cancelled_request_id\":\"r1\",\"session_id\":\"s\"}\n";
        peer.write_all(raw).unwrap(); wait(&work.token); peer.write_all(raw).unwrap(); wait(&life);
        assert!(input.finish("r1", "s", true).is_err());
    }
    #[test]
    fn completion_foreign_ids_do_not_clear_live_binding() {
        let (input, mut peer, _) = setup(); turn(&mut peer, "r1"); input.next().unwrap().unwrap();
        assert!(input.finish("wrong", "s", false).is_err()); assert!(input.cancellation("r1", "wrong").is_err());
        input.finish("r1", "s", false).unwrap();
    }
    #[test]
    fn late_cancel_after_terminal_is_fatal_not_a_second_commit() {
        let (input, mut peer, life) = setup(); turn(&mut peer, "r1"); input.next().unwrap().unwrap();
        input.finish("r1", "s", false).unwrap();
        peer.write_all(b"{\"op\":\"cancel_turn\",\"request_id\":\"d\",\"cancelled_request_id\":\"r1\",\"session_id\":\"s\"}\n").unwrap();
        wait(&life); assert!(input.next().is_err());
    }
}
