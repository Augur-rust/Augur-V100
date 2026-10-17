//! Bounded producer reader and request-scoped disconnect observation. No CUDA.
use serde_json::Value;
use std::{
    error::Error,
    io::{self, BufRead, BufReader, Read},
    net::TcpStream,
    sync::mpsc::{self, Receiver, RecvTimeoutError},
    thread::{self, JoinHandle},
    time::Duration,
};

const FRAME_QUEUE_CAPACITY: usize = 4;
const CLIENT_POLL_INTERVAL: Duration = Duration::from_millis(25);

#[derive(Debug)]
pub(super) struct ClientDisconnected(pub String);
impl std::fmt::Display for ClientDisconnected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.0.fmt(f) }
}
impl Error for ClientDisconnected {}
pub(super) fn is_client_disconnect(error: &(dyn Error + 'static)) -> bool {
    error.is::<ClientDisconnected>() || error.downcast_ref::<io::Error>().is_some_and(|e|
        matches!(e.kind(), io::ErrorKind::BrokenPipe | io::ErrorKind::ConnectionReset
            | io::ErrorKind::ConnectionAborted | io::ErrorKind::NotConnected))
}

pub(super) fn client_connected(stream: &TcpStream) -> Result<(), Box<dyn Error>> {
    // The adapter supports one complete request per connection. A FIN (including
    // write-half-close) is cancellation under this bounded protocol; pipelining
    // is unsupported. This check never consumes bytes or writes heartbeat data.
    stream.set_nonblocking(true)?;
    let result = stream.peek(&mut [0u8; 1]);
    // Restore blocking writes even on peer error. Only this request thread uses
    // this socket; the producer reader owns a different, pipe-only descriptor.
    stream.set_nonblocking(false)?;
    match result {
        Err(error) if error.kind() == io::ErrorKind::WouldBlock => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::Interrupted => Ok(()),
        Ok(0) => Err(Box::new(ClientDisconnected("client disconnected before producer boundary (FIN)".into()))),
        Ok(_) => Err("client sent unsupported pipelined bytes".into()),
        Err(error) if is_client_disconnect(&error) => Err(Box::new(ClientDisconnected(format!("client disconnected before producer boundary: {error}")))),
        Err(error) => Err(error.into()),
    }
}

fn bounded_frame(reader: &mut impl BufRead, maximum: usize) -> Result<Value, String> {
    let mut bytes = Vec::new();
    // Limit BEFORE growing the Vec; read_until alone would allocate an unbounded
    // unterminated frame before its old post-read length check ran.
    reader
        .take(maximum as u64 + 1)
        .read_until(b'\n', &mut bytes)
        .map_err(|e| format!("producer read: {e}"))?;
    if bytes.is_empty() {
        return Err("producer exited before a complete frame".into());
    }
    if bytes.len() > maximum {
        return Err("producer frame exceeds bound".into());
    }
    if bytes.last() != Some(&b'\n') {
        return Err("producer exited in an unterminated frame".into());
    }
    let frame: Value = serde_json::from_slice(&bytes).map_err(|e| format!("producer JSON: {e}"))?;
    if !frame.is_object() {
        return Err("producer frame is not an object".into());
    }
    Ok(frame)
}

#[derive(Debug)]
pub(super) struct ProducerFrames {
    receiver: Option<Receiver<Result<Value, String>>>,
    worker: Option<JoinHandle<()>>,
}

impl ProducerFrames {
    pub(super) fn spawn(source: impl Read + Send + 'static, maximum: usize) -> io::Result<Self> {
        let (sender, receiver) = mpsc::sync_channel(FRAME_QUEUE_CAPACITY);
        let worker = thread::Builder::new()
            .name("qwen35-producer-frames".into())
            .spawn(move || {
                let mut reader = BufReader::new(source);
                loop {
                    let frame = bounded_frame(&mut reader, maximum);
                    let terminal_error = frame.is_err();
                    if sender.send(frame).is_err() || terminal_error {
                        break;
                    }
                }
            })?;
        Ok(Self {
            receiver: Some(receiver),
            worker: Some(worker),
        })
    }

    pub(super) fn receive(&self, client: &TcpStream) -> Result<Value, Box<dyn Error>> {
        let receiver = self
            .receiver
            .as_ref()
            .ok_or("producer receiver already closed")?;
        loop {
            client_connected(client)?;
            match receiver.recv_timeout(CLIENT_POLL_INTERVAL) {
                Ok(frame) => {
                    client_connected(client)?;
                    return frame.map_err(Into::into);
                }
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => return Err("producer reader stopped".into()),
            }
        }
    }

    /// Augur recovery consumes each producer frame exactly once even if the
    /// original client is gone. Caller supplies a total deadline and validates
    /// frame identities; a poll timeout is not a producer terminal event.
    pub(super) fn poll(&self, timeout: Duration) -> Result<Option<Value>, Box<dyn Error>> {
        let receiver = self.receiver.as_ref().ok_or("producer receiver already closed")?;
        match receiver.recv_timeout(timeout) {
            Ok(frame) => frame.map(Some).map_err(Into::into),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => Err("producer reader stopped".into()),
        }
    }

    pub(super) fn close_receiver(&mut self) {
        // Unblock a worker waiting to publish into a full channel before waiting
        // for child exit. It must never pin process teardown via queue backpressure.
        self.receiver.take();
    }

    pub(super) fn join_after_child_exit(&mut self) {
        self.close_receiver();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{Cursor, Write},
        net::{Shutdown, TcpListener},
        time::Instant,
    };

    fn pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let a = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let b = listener.accept().unwrap().0;
        (a, b)
    }
    #[test]
    fn recovery_disconnect_classification_excludes_pipelining_and_producer_errors() {
        assert!(is_client_disconnect(&ClientDisconnected("FIN".into())));
        assert!(is_client_disconnect(&io::Error::new(io::ErrorKind::BrokenPipe, "write")));
        assert!(!is_client_disconnect(&io::Error::new(io::ErrorKind::InvalidData, "bad frame")));
        let (mut client, server) = pair(); client.write_all(b"extra").unwrap();
        let error = client_connected(&server).unwrap_err();
        assert!(!is_client_disconnect(error.as_ref()));
    }
    #[test]
    fn detached_poll_preserves_order_and_has_bounded_empty_wait() {
        let (mut producer, reader) = pair(); let mut frames = ProducerFrames::spawn(reader, 1024).unwrap();
        assert!(frames.poll(Duration::from_millis(1)).unwrap().is_none());
        producer.write_all(b"{\"sequence\":1}\n{\"sequence\":2}\n").unwrap();
        assert_eq!(frames.poll(Duration::from_secs(1)).unwrap().unwrap()["sequence"],1);
        assert_eq!(frames.poll(Duration::from_secs(1)).unwrap().unwrap()["sequence"],2);
        drop(producer); frames.join_after_child_exit();
    }

    #[test]
    fn bounded_frames_require_object_newline_and_exact_limit() {
        assert!(bounded_frame(&mut Cursor::new(b"{}\n"), 3).is_ok());
        for raw in [b"{}".as_slice(), b"[]\n", b"{broken\n", b"{} \n", b""] {
            assert!(bounded_frame(&mut Cursor::new(raw), 3).is_err());
        }
        let mut large = Cursor::new(vec![b'x'; 1024 * 1024]);
        assert!(bounded_frame(&mut large, 16).is_err());
        assert_eq!(large.position(), 17);
    }

    #[test]
    fn live_socket_and_healthy_delayed_producer_are_not_cancelled() {
        let (mut producer, reader) = pair();
        let (client, server) = pair();
        let mut frames = ProducerFrames::spawn(reader, 1024).unwrap();
        let writer = thread::spawn(move || {
            thread::sleep(Duration::from_millis(90));
            producer
                .write_all(b"{\"event\":\"text\",\"token\":7}\n")
                .unwrap();
        });
        let frame = frames.receive(&server).unwrap();
        assert_eq!(frame["token"], 7);
        writer.join().unwrap();
        frames.join_after_child_exit();
        drop(client);
    }

    #[test]
    fn silent_producer_does_not_hide_client_fin() {
        let (producer, reader) = pair();
        reader
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let (client, server) = pair();
        let mut frames = ProducerFrames::spawn(reader, 1024).unwrap();
        let closer = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            client.shutdown(Shutdown::Both).unwrap();
        });
        let start = Instant::now();
        let error = frames.receive(&server).unwrap_err().to_string();
        assert!(error.contains("client disconnected"), "{error}");
        assert!(start.elapsed() < Duration::from_secs(2));
        closer.join().unwrap();
        drop(producer);
        frames.join_after_child_exit();
    }

    #[test]
    fn protocol_explicitly_rejects_half_close_and_pipelining() {
        let (client, server) = pair();
        client.shutdown(Shutdown::Write).unwrap();
        for _ in 0..20 {
            if client_connected(&server).is_err() {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        assert!(client_connected(&server).is_err());
        let (mut client, server) = pair();
        client.write_all(b"POST extra").unwrap();
        assert!(client_connected(&server).is_err());
    }

    #[test]
    fn bounded_queue_preserves_order_and_releases_backpressure() {
        let (client, server) = pair();
        let raw = (0..20)
            .map(|i| format!("{{\"sequence\":{i}}}\n"))
            .collect::<String>();
        let mut frames = ProducerFrames::spawn(Cursor::new(raw.into_bytes()), 128).unwrap();
        for i in 0..10 {
            assert_eq!(frames.receive(&server).unwrap()["sequence"], i);
        }
        // Stop consuming with more than four frames left: dropping the receiver
        // must release the blocked sender without draining any queued frames.
        frames.close_receiver();
        frames.join_after_child_exit();
        drop(client);
    }

    #[test]
    fn incomplete_producer_frame_never_becomes_a_success() {
        let (client, server) = pair();
        let mut frames =
            ProducerFrames::spawn(Cursor::new(b"{\"event\":\"terminal\"}"), 1024).unwrap();
        assert!(frames
            .receive(&server)
            .unwrap_err()
            .to_string()
            .contains("unterminated"));
        frames.join_after_child_exit();
        drop(client);
    }
}
