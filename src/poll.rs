use std::iter::zip;
use std::time::Duration;

pub trait Port<Event> {
    fn port_fd(&self) -> libc::c_int;
    fn get_event(&self) -> Option<Event>;
}

pub fn wait_for_event<'a, Event, Iter>(
    ports: Iter,
    timeout: Option<(Duration, Event)>,
) -> Vec<Event>
where
    Event: 'a,
    Iter: Clone + IntoIterator<Item = &'a dyn Port<Event>>,
{
    let mut pollfds: Vec<_> = ports
        .clone()
        .into_iter()
        .map(|port| libc::pollfd {
            fd: port.port_fd(),
            events: libc::POLLIN,
            revents: 0,
        })
        .collect();
    let (timeout_ms, timeout_event) = match timeout {
        None => (-1, None),
        Some((timeout_duration, timeout_event)) => {
            (timeout_duration.as_millis() as i32 + 1, Some(timeout_event))
        }
    };
    let status = unsafe { libc::poll(pollfds.as_mut_ptr(), pollfds.len() as u64, timeout_ms) };
    assert!(status != -1);
    if status == 0 {
        vec![timeout_event.expect("timed out without timeout")]
    } else {
        zip(pollfds.into_iter(), ports.into_iter())
            .filter_map(|(item, port)| {
                if item.revents != 0 {
                    port.get_event()
                } else {
                    None
                }
            })
            .collect()
    }
}
