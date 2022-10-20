use std::thread::{self, JoinHandle};
use std::sync::mpsc::{channel, Sender, SendError, RecvError};
use std::error::Error;
use std::fmt;

type Job = Box<dyn FnOnce() + Send + 'static>;

pub enum Message {
    Execute(Job),
    Stop,
}

pub struct Janitor {
    thread: Option<JoinHandle<Result<(), RecvError>>>,
    channel: Sender<Message>,
}

#[derive(Debug)]
pub enum JanitorError {
    Send(SendError<Message>),
}

impl fmt::Display for JanitorError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "could not contact janitor: ")?;
        match self {
            JanitorError::Send(err) =>
                err.fmt(f)
        }
    }
}

impl Error for JanitorError {}

impl From<SendError<Message>> for JanitorError {
    fn from(err: SendError<Message>) -> Self {
        JanitorError::Send(err)
    }
}

impl Janitor {
    pub fn new() -> Janitor {
        let (sender, receiver) = channel();
        Janitor {
            thread: Some(thread::spawn(move || {
                loop {
                    match receiver.recv()? {
                        Message::Execute(job) => job(),
                        Message::Stop => return Ok(()),
                    }
                }
            })),
            channel: sender,
        }
    }

    pub fn execute<F: FnOnce() + Send + 'static>(&self, f: F) -> Result<(), JanitorError> {
        self.channel.send(Message::Execute(Box::new(f)))?;
        Ok(())
    }

    pub fn dispose<T : Send + 'static>(&self, value: T) -> Result<(), JanitorError> {
        self.execute(move || drop(value))
    }
}

impl Drop for Janitor {
    fn drop(&mut self) {
        self.channel.send(Message::Stop).unwrap();
        std::mem::replace(&mut self.thread, None).unwrap().join().unwrap().unwrap();
    }
}
