//! Model-neutral control, own Nerkyor wire binding. Never resets a cancellation flag.
//! One completed turn binding is retained until the next command so a late
//! disconnect can close an already committed session without pretending rollback.
use super::{parse_command, Command};
use augur_model_qwen35_runtime::{
    augur_session::valid_id, session_cancellation::SessionCancellationToken,
};
use std::{
    io::{self, BufRead, BufReader, Read},
    sync::{
        mpsc::{self, Receiver},
        Arc, Mutex,
    },
    thread,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Cancel {
    pub request_id: String,
    pub cancelled_request_id: String,
    pub session_id: String,
}
pub(super) struct Work {
    pub raw: Vec<u8>,
    pub token: SessionCancellationToken,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Running,
    Finished,
    Resolving,
}
struct Binding {
    rid: String,
    sid: String,
    token: SessionCancellationToken,
    cancel: Option<Cancel>,
    phase: Phase,
}
pub(super) struct Input {
    receiver: Receiver<io::Result<Work>>,
    binding: Arc<Mutex<Option<Binding>>>,
    lifetime: SessionCancellationToken,
}
fn failure(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn terminate(lifetime: &SessionCancellationToken, binding: &Mutex<Option<Binding>>) {
    lifetime.request_cancel();
    if let Ok(slot) = binding.lock() {
        if let Some(active) = slot.as_ref() {
            active.token.request_cancel();
        }
    }
}
impl Input {
    pub(super) fn start(
        reader: impl Read + Send + 'static,
        limit: u64,
        lifetime: SessionCancellationToken,
    ) -> io::Result<Self> {
        if limit == 0 || limit == u64::MAX {
            return Err(failure("bounded control input required"));
        }
        let (sender, receiver) = mpsc::sync_channel(1);
        let binding = Arc::new(Mutex::new(None::<Binding>));
        let worker_slot = binding.clone();
        let worker_life = lifetime.clone();
        thread::Builder::new()
            .name("nerkyor-request-control".into())
            .spawn(move || {
                let mut reader = BufReader::new(reader);
                loop {
                    let result = (|| -> io::Result<Option<Work>> {
                        let mut raw = Vec::new();
                        reader
                            .by_ref()
                            .take(limit + 1)
                            .read_until(b'\n', &mut raw)?;
                        if raw.is_empty() {
                            return Err(failure("producer control EOF"));
                        }
                        if raw.len() as u64 > limit || !raw.ends_with(b"\n") {
                            return Err(failure("oversized or truncated control"));
                        }
                        // This also rejects duplicate keys and unknown fields, before binding authority.
                        let command = parse_command(&raw)
                            .map_err(|_| failure("invalid strict producer command"))?;
                        let mut slot = worker_slot
                            .lock()
                            .map_err(|_| failure("control lock poisoned"))?;
                        if let Command::CancelTurn {
                            request_id,
                            cancelled_request_id,
                            session_id,
                        } = command
                        {
                            let active = slot
                                .as_mut()
                                .ok_or_else(|| failure("cancel has no bound request"))?;
                            if !valid_id(&request_id)
                                || request_id == cancelled_request_id
                                || active.rid != cancelled_request_id
                                || active.sid != session_id
                                || active.cancel.is_some()
                                || active.phase == Phase::Resolving
                            {
                                return Err(failure("foreign or repeated request cancellation"));
                            }
                            active.cancel = Some(Cancel {
                                request_id,
                                cancelled_request_id,
                                session_id,
                            });
                            active.token.request_cancel();
                            if active.phase == Phase::Finished {
                                active.phase = Phase::Resolving;
                                return Ok(Some(Work {
                                    raw,
                                    token: active.token.clone(),
                                }));
                            }
                            return Ok(None);
                        }
                        if slot.as_ref().is_some_and(|s| s.phase != Phase::Finished) {
                            return Err(failure(
                                "pipelined command while turn or cancellation is active",
                            ));
                        }
                        *slot = None;
                        let token = SessionCancellationToken::new();
                        if let Command::Turn { request } = command {
                            if !valid_id(&request.request_id) || !valid_id(&request.session_id) {
                                return Err(failure("invalid turn binding"));
                            }
                            *slot = Some(Binding {
                                rid: request.request_id,
                                sid: request.session_id,
                                token: token.clone(),
                                cancel: None,
                                phase: Phase::Running,
                            });
                        }
                        Ok(Some(Work { raw, token }))
                    })();
                    match result {
                        Ok(None) => (),
                        Ok(Some(work)) => {
                            if sender.try_send(Ok(work)).is_err() {
                                terminate(&worker_life, &worker_slot);
                                break;
                            }
                        }
                        Err(error) => {
                            terminate(&worker_life, &worker_slot);
                            let _ = sender.try_send(Err(error));
                            break;
                        }
                    }
                }
            })?;
        Ok(Self {
            receiver,
            binding,
            lifetime,
        })
    }
    pub(super) fn next(&self) -> io::Result<Option<Work>> {
        let work = self.receiver.recv();
        if self.lifetime.is_requested() {
            return Err(failure("owner control terminated"));
        }
        match work {
            Ok(result) => result.map(Some),
            Err(_) => Ok(None),
        }
    }
    /// Publication does not revoke the binding. A matching late cancel is an
    /// explicit close of a durable result, never an unpublished-turn discard.
    pub(super) fn completed(&self, rid: &str, sid: &str) -> io::Result<Option<Cancel>> {
        let mut slot = self
            .binding
            .lock()
            .map_err(|_| failure("control lock poisoned"))?;
        if self.lifetime.is_requested() {
            return Err(failure("owner terminated at completion"));
        }
        let active = slot
            .as_mut()
            .ok_or_else(|| failure("missing turn binding"))?;
        if active.rid != rid || active.sid != sid || active.phase != Phase::Running {
            return Err(failure("completion identity/state mismatch"));
        }
        active.phase = if active.cancel.is_some() {
            Phase::Resolving
        } else {
            Phase::Finished
        };
        Ok(active.cancel.clone())
    }
    pub(super) fn cancellation(&self, rid: &str, sid: &str) -> io::Result<Cancel> {
        let slot = self
            .binding
            .lock()
            .map_err(|_| failure("control lock poisoned"))?;
        if self.lifetime.is_requested() {
            return Err(failure("owner terminated during cancellation"));
        }
        let active = slot
            .as_ref()
            .ok_or_else(|| failure("missing cancellation binding"))?;
        if active.rid != rid || active.sid != sid || active.phase != Phase::Resolving {
            return Err(failure("cancellation identity/state mismatch"));
        }
        active
            .cancel
            .clone()
            .ok_or_else(|| failure("no explicit cancellation control"))
    }
    pub(super) fn resolved(&self, cancel: &Cancel) -> io::Result<()> {
        let mut slot = self
            .binding
            .lock()
            .map_err(|_| failure("control lock poisoned"))?;
        if self.lifetime.is_requested() {
            return Err(failure("owner terminated before recovery publication"));
        }
        let active = slot
            .as_ref()
            .ok_or_else(|| failure("missing resolved binding"))?;
        if active.phase != Phase::Resolving || active.cancel.as_ref() != Some(cancel) {
            return Err(failure("resolution identity/state mismatch"));
        }
        *slot = None;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::Write,
        net::{TcpListener, TcpStream},
        time::{Duration, Instant},
    };
    fn setup() -> (Input, TcpStream, SessionCancellationToken) {
        let server = TcpListener::bind("127.0.0.1:0").unwrap();
        let peer = TcpStream::connect(server.local_addr().unwrap()).unwrap();
        let reader = server.accept().unwrap().0;
        let life = SessionCancellationToken::new();
        (
            Input::start(reader, 2048, life.clone()).unwrap(),
            peer,
            life,
        )
    }
    fn turn(peer: &mut TcpStream, rid: &str) {
        writeln!(peer,"{{\"op\":\"turn\",\"request\":{{\"request_id\":\"{rid}\",\"session_id\":\"s\",\"max_new_tokens\":4,\"input\":{{\"kind\":\"open\",\"prompt_token_ids\":[1,2]}}}}}}").unwrap();
    }
    fn cancel(peer: &mut TcpStream) {
        peer.write_all(b"{\"op\":\"cancel_turn\",\"request_id\":\"d1\",\"cancelled_request_id\":\"r1\",\"session_id\":\"s\"}\n").unwrap();
    }
    fn wait(token: &SessionCancellationToken) {
        let until = Instant::now() + Duration::from_secs(2);
        while !token.is_requested() && Instant::now() < until {
            thread::sleep(Duration::from_millis(1));
        }
        assert!(token.is_requested());
    }
    #[test]
    fn live_cancel_resolves_once_and_next_turn_gets_fresh_token() {
        let (input, mut peer, life) = setup();
        turn(&mut peer, "r1");
        let old = input.next().unwrap().unwrap();
        cancel(&mut peer);
        wait(&old.token);
        let c = input.completed("r1", "s").unwrap().unwrap();
        assert_eq!(input.cancellation("r1", "s").unwrap(), c);
        input.resolved(&c).unwrap();
        assert!(!life.is_requested());
        assert!(input.resolved(&c).is_err());
        turn(&mut peer, "r2");
        let next = input.next().unwrap().unwrap();
        assert!(!next.token.is_requested());
        assert!(old.token.is_requested());
        assert!(input.completed("r2", "s").unwrap().is_none());
    }
    #[test]
    fn matching_late_cancel_is_queued_for_committed_close_not_retroactive_discard() {
        let (input, mut peer, life) = setup();
        turn(&mut peer, "r1");
        input.next().unwrap().unwrap();
        assert!(input.completed("r1", "s").unwrap().is_none());
        cancel(&mut peer);
        let work = input.next().unwrap().unwrap();
        assert!(matches!(
            parse_command(&work.raw).unwrap(),
            Command::CancelTurn { .. }
        ));
        let c = input.cancellation("r1", "s").unwrap();
        assert!(input.completed("r1", "s").is_err());
        input.resolved(&c).unwrap();
        assert!(!life.is_requested());
    }
    #[test]
    fn wrong_duplicate_malformed_and_pipelined_controls_fail_owner_closed() {
        for raw in [b"{\"op\":\"cancel_turn\",\"request_id\":\"d\",\"cancelled_request_id\":\"wrong\",\"session_id\":\"s\"}\n".as_slice(),
            b"{\"op\":\"cancel_turn\",\"request_id\":\"d\",\"cancelled_request_id\":\"r1\",\"session_id\":\"s\",\"extra\":true}\n",
            b"{\"op\":\"cancel_turn\",\"request_id\":\"d\",\"request_id\":\"other\",\"cancelled_request_id\":\"r1\",\"session_id\":\"s\"}\n",
            b"{\"op\":\"shutdown\"}\n",b"[]\n",b"{broken}\n"]{
            let(input,mut peer,life)=setup();turn(&mut peer,"r1");let work=input.next().unwrap().unwrap();peer.write_all(raw).unwrap();wait(&life);wait(&work.token);
            assert!(input.completed("r1","s").is_err());
        }
    }
    #[test]
    fn repeated_cancel_never_grants_second_resolution() {
        let (input, mut peer, life) = setup();
        turn(&mut peer, "r1");
        let work = input.next().unwrap().unwrap();
        cancel(&mut peer);
        wait(&work.token);
        cancel(&mut peer);
        wait(&life);
        assert!(input.completed("r1", "s").is_err());
    }
    #[test]
    fn eof_cancels_owner_and_active_work_without_recovery_authority() {
        let (input, mut peer, life) = setup();
        turn(&mut peer, "r1");
        let work = input.next().unwrap().unwrap();
        drop(peer);
        wait(&life);
        wait(&work.token);
        assert!(input.completed("r1", "s").is_err());
        assert!(input.next().is_err());
    }
    #[test]
    fn foreign_completion_does_not_clear_the_real_binding() {
        let (input, mut peer, _) = setup();
        turn(&mut peer, "r1");
        input.next().unwrap().unwrap();
        assert!(input.completed("wrong", "s").is_err());
        assert!(input.completed("r1", "wrong").is_err());
        assert!(input.completed("r1", "s").unwrap().is_none());
    }
    #[test]
    fn old_cancel_cannot_cross_into_next_request() {
        let (input, mut peer, life) = setup();
        turn(&mut peer, "r1");
        input.next().unwrap().unwrap();
        input.completed("r1", "s").unwrap();
        turn(&mut peer, "r2");
        let next = input.next().unwrap().unwrap();
        cancel(&mut peer);
        wait(&life);
        wait(&next.token);
        assert!(input.completed("r2", "s").is_err());
    }
    #[test]
    fn oversize_and_truncated_controls_are_fatal() {
        for raw in [vec![b'x'; 2050], b"{\"op\":\"shutdown\"}".to_vec()] {
            let life = SessionCancellationToken::new();
            let input = Input::start(io::Cursor::new(raw), 2048, life.clone()).unwrap();
            assert!(input.next().is_err());
            assert!(life.is_requested());
        }
    }
}
