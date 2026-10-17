//! Bounded stdin ingress for the explicitly cooperative HTTP-owned producer.
//!
//! This CPU-only reader never owns model state. EOF/read failure marks the
//! owner's cancellation token while its execution thread may be in prefill.
//! One queued command is permitted; pipelining beyond that fails closed.
use augur_model_qwen35_runtime::session_cancellation::SessionCancellationToken;
use std::{
    io::{self, BufRead, BufReader, Read},
    sync::mpsc::{self, Receiver, SyncSender, TrySendError},
    thread,
};

type Message = io::Result<Vec<u8>>;

pub(super) struct CancellableInput {
    receiver: Receiver<Message>,
    cancellation: SessionCancellationToken,
}

fn pump<R: Read>(
    reader: R,
    limit: u64,
    token: SessionCancellationToken,
    sender: SyncSender<Message>,
) {
    let mut reader = BufReader::new(reader);
    loop {
        let mut raw = Vec::new();
        let result = reader.by_ref().take(limit + 1).read_until(b'\n', &mut raw);
        let message = match result {
            Ok(0) => {
                token.request_cancel();
                break;
            }
            Ok(_) if raw.len() as u64 > limit => {
                token.request_cancel();
                Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "oversized session command",
                ))
            }
            Ok(_) if !raw.ends_with(b"\n") => {
                token.request_cancel();
                Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "unterminated session command",
                ))
            }
            Ok(_) => Ok(raw),
            Err(error) => {
                token.request_cancel();
                Err(error)
            }
        };
        let failed = message.is_err();
        match sender.try_send(message) {
            Ok(()) if !failed => (),
            Ok(()) => break,
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                token.request_cancel();
                break;
            }
        }
    }
}

impl CancellableInput {
    pub(super) fn start<R: Read + Send + 'static>(
        reader: R,
        limit: u64,
        token: SessionCancellationToken,
    ) -> io::Result<Self> {
        if limit == 0 || limit == u64::MAX {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "bounded command limit required",
            ));
        }
        let (sender, receiver) = mpsc::sync_channel(1);
        let worker_token = token.clone();
        // The one reader may block on stdin until the broker closes it. Do not
        // join it while handling shutdown: the broker needs the final receipt
        // before closing stdin. It owns no GPU resources and ends with this
        // dedicated native process. A dropped receiver never blocks its sender.
        thread::Builder::new()
            .name("qwen35-owner-input".into())
            .spawn(move || pump(reader, limit, worker_token, sender))?;
        Ok(Self {
            receiver,
            cancellation: token,
        })
    }

    pub(super) fn next(&self) -> io::Result<Option<Vec<u8>>> {
        match self.receiver.recv() {
            Ok(Ok(_)) if self.cancellation.is_requested() => Err(io::Error::new(
                io::ErrorKind::ConnectionAborted,
                "owner input cancelled before command admission",
            )),
            Ok(result) => result.map(Some),
            Err(_) => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::Write,
        net::{TcpListener, TcpStream},
        time::Duration,
    };

    #[test]
    fn eof_notifies_during_silent_work_without_needing_an_output_frame() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut writer = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (reader, _) = listener.accept().unwrap();
        let token = SessionCancellationToken::new();
        let input = CancellableInput::start(reader, 64, token.clone()).unwrap();
        writer.write_all(b"{\"op\":\"open\"}\n").unwrap();
        assert_eq!(input.next().unwrap().unwrap(), b"{\"op\":\"open\"}\n");
        assert!(!token.is_requested());
        // Equivalent to broker stdin closure. No native output is produced.
        writer.shutdown(std::net::Shutdown::Write).unwrap();
        assert!(input.next().unwrap().is_none());
        assert!(token.check("silent batch").is_err());
    }

    #[test]
    fn oversized_truncated_and_read_error_cancel_without_admitting_a_command() {
        for raw in [b"123456\n".as_slice(), b"x".as_slice()] {
            let token = SessionCancellationToken::new();
            let input =
                CancellableInput::start(io::Cursor::new(raw.to_vec()), 4, token.clone()).unwrap();
            assert!(input.next().is_err());
            assert!(token.is_requested());
        }
        struct Failed;
        impl Read for Failed {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::other("synthetic read failure"))
            }
        }
        let token = SessionCancellationToken::new();
        let input = CancellableInput::start(Failed, 64, token.clone()).unwrap();
        assert!(input.next().is_err());
        assert!(token.is_requested());
    }

    #[test]
    fn extra_queued_command_fails_closed_instead_of_blocking_eof_detection() {
        let token = SessionCancellationToken::new();
        let (tx, rx) = mpsc::sync_channel(1);
        pump(io::Cursor::new(b"first\nsecond\n"), 64, token.clone(), tx);
        assert!(token.is_requested());
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(1)).unwrap().unwrap(),
            b"first\n"
        );
        assert!(rx.recv().is_err());
    }

    #[test]
    fn dropping_consumer_cannot_trap_the_sender() {
        let token = SessionCancellationToken::new();
        let (tx, rx) = mpsc::sync_channel(1);
        drop(rx);
        pump(io::Cursor::new(b"command\n"), 64, token.clone(), tx);
        assert!(token.is_requested());
    }

    #[test]
    fn queued_command_does_not_authorize_work_after_eof() {
        let token = SessionCancellationToken::new();
        let (tx, rx) = mpsc::sync_channel(1);
        pump(io::Cursor::new(b"open\n"), 64, token.clone(), tx);
        let input = CancellableInput {
            receiver: rx,
            cancellation: token,
        };
        assert_eq!(
            input.next().unwrap_err().kind(),
            io::ErrorKind::ConnectionAborted
        );
    }
}
