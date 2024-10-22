use std::sync::{Arc, Mutex, MutexGuard, Weak};

#[allow(unused_macros)]
macro_rules! cached {
    (|$($xs:ident),*| $body:expr) => {
        CachedNode::new(($($xs.clone()),*), |($($xs),*)| $body)
    };
    (|| $body:expr) => {
        CachedNode::new((), |()| $body)
    }
}

macro_rules! cached_result {
    (|$($xs:ident),*| $body:expr) => {
        CachedResultNode::new(($($xs.clone()),*), |($($xs),*)| $body)
    };
    (|| $body:expr) => {
        CachedResultNode::new((), |()| $body)
    }
}

pub trait Source: Clone + 'static {
    type Item;

    fn get(&self) -> Self::Item;
    fn add_sink(&self, sink: impl FnMut() + Clone + 'static);
}

struct ConstNodeCell<T> {
    sinks: Vec<Box<dyn FnMut()>>,
    value: T,
}

#[derive(Clone)]
pub struct ConstNode<T>(Arc<Mutex<ConstNodeCell<T>>>);

impl<T: Clone> ConstNode<T> {
    pub fn new(value: T) -> Self {
        Self(Arc::new(Mutex::new(ConstNodeCell {
            sinks: Vec::new(),
            value,
        })))
    }

    pub fn put(&self, value: T) {
        let mut cell = self.0.lock().unwrap();
        cell.value = value;
        for sink in &mut cell.sinks {
            sink()
        }
    }

    fn lock(&self) -> MutexGuard<ConstNodeCell<T>> {
        self.0.lock().unwrap()
    }
}

impl<T: Clone + 'static> Source for ConstNode<T> {
    type Item = T;

    fn get(&self) -> T {
        self.lock().value.clone()
    }

    fn add_sink(&self, sink: impl FnMut() + 'static) {
        self.lock().sinks.push(Box::new(sink))
    }
}

struct CachedNodeCell<T, E> {
    compute: Box<dyn Fn() -> Result<T, E>>,
    sinks: Vec<Box<dyn FnMut()>>,
    value: Option<T>,
}

pub struct CachedNode<T, E = !>(Arc<Mutex<CachedNodeCell<T, E>>>);

struct WeakCachedNode<T, E>(Weak<Mutex<CachedNodeCell<T, E>>>);

impl<T: Clone + 'static, E: 'static> CachedNode<T, E> {
    pub fn new_result<S: Source + 'static>(
        source: S,
        f: impl Fn(S::Item) -> Result<T, E> + 'static,
    ) -> Self {
        Self(Arc::new_cyclic(|weak_ref| {
            let weak_node = WeakCachedNode(weak_ref.clone());
            source.add_sink(move || {
                if let Some(sink_node) = weak_node.upgrade() {
                    let mut cell = sink_node.lock();
                    if cell.value.is_some() {
                        cell.value = None;
                        for sink in &mut cell.sinks {
                            sink();
                        }
                    }
                }
            });
            Mutex::new(CachedNodeCell {
                compute: Box::new(move || f(source.get())),
                sinks: Vec::new(),
                value: None,
            })
        }))
    }

    fn get_result(&self) -> Result<T, E> {
        let mut cell = self.lock();
        if let Some(ref value) = cell.value {
            Ok(value.clone())
        } else {
            let value = (cell.compute)()?;
            cell.value = Some(value.clone());
            Ok(value)
        }
    }

    fn add_sink_result(&self, sink: impl FnMut() + 'static) {
        self.lock().sinks.push(Box::new(sink));
    }

    fn lock(&self) -> MutexGuard<CachedNodeCell<T, E>> {
        self.0.lock().unwrap()
    }
}

impl<T, E> WeakCachedNode<T, E> {
    fn upgrade(&self) -> Option<CachedNode<T, E>> {
        Some(CachedNode(self.0.upgrade()?))
    }
}

impl<T: Clone + 'static> CachedNode<T> {
    #[allow(unused)]
    pub fn new<S: Source + 'static>(source: S, f: impl Fn(S::Item) -> T + 'static) -> Self {
        Self::new_result(source, move |x| Ok(f(x)))
    }
}

impl<T: 'static, E> Clone for CachedNode<T, E> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: 'static, E> Clone for WeakCachedNode<T, E> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: Clone + 'static> Source for CachedNode<T> {
    type Item = T;

    fn get(&self) -> T {
        let Ok(value) = self.get_result();
        value
    }

    fn add_sink(&self, sink: impl FnMut() + 'static) {
        self.add_sink_result(sink)
    }
}

pub struct CachedResultNode<T, E>(CachedNode<T, E>);

impl<T: Clone + 'static, E: 'static> CachedResultNode<T, E> {
    pub fn new<S: Source + 'static>(
        source: S,
        f: impl Fn(S::Item) -> Result<T, E> + 'static,
    ) -> Self {
        Self(CachedNode::new_result(source, f))
    }

    pub fn map<U: Clone + 'static>(&self, f: impl Fn(T) -> U + 'static) -> CachedResultNode<U, E> {
        CachedResultNode::new(self.clone(), move |x| x.map(|y| f(y)))
    }
}

impl<T: Clone + 'static, E> Clone for CachedResultNode<T, E> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: Clone + 'static, E: 'static> Source for CachedResultNode<T, E> {
    type Item = Result<T, E>;

    fn get(&self) -> Result<T, E> {
        self.0.get_result()
    }

    fn add_sink(&self, sink: impl FnMut() + Clone + 'static) {
        self.0.add_sink_result(sink)
    }
}

impl Source for () {
    type Item = ();

    fn get(&self) {}

    fn add_sink(&self, _sink: impl FnMut() + 'static) {}
}

impl<S1, S2> Source for (S1, S2)
where
    S1: Source,
    S2: Source,
{
    type Item = (S1::Item, S2::Item);

    fn get(&self) -> (S1::Item, S2::Item) {
        (self.0.get(), self.1.get())
    }

    fn add_sink(&self, sink: impl FnMut() + Clone + 'static) {
        self.0.add_sink(sink.clone());
        self.1.add_sink(sink);
    }
}

impl<S1, S2, S3> Source for (S1, S2, S3)
where
    S1: Source,
    S2: Source,
    S3: Source,
{
    type Item = (S1::Item, S2::Item, S3::Item);

    fn get(&self) -> (S1::Item, S2::Item, S3::Item) {
        (self.0.get(), self.1.get(), self.2.get())
    }

    fn add_sink(&self, sink: impl FnMut() + Clone + 'static) {
        self.0.add_sink(sink.clone());
        self.1.add_sink(sink.clone());
        self.2.add_sink(sink);
    }
}

mod test {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn const_get() {
        let value = 42;
        let source = ConstNode::new(value);
        assert_eq!(source.get(), value);
    }

    #[test]
    fn const_and_node() {
        let source = ConstNode::new(7);
        let node = CachedNode::new(source.clone(), |x| x + 1);
        assert_eq!(node.get(), 8);
        source.put(11);
        assert_eq!(node.get(), 12);
    }

    #[test]
    fn multiple_sources() {
        let source1 = ConstNode::new(2);
        let source2 = ConstNode::new(3);
        let node = CachedNode::new((source1.clone(), source2.clone()), |(x, y)| x * y);
        assert_eq!(node.get(), 6);
        source1.put(5);
        assert_eq!(node.get(), 15);
        source2.put(7);
        assert_eq!(node.get(), 35);
    }

    #[test]
    fn propagate() {
        let source = ConstNode::new(28);
        let node = CachedNode::new(source.clone(), |x| x + 1);
        let sink = CachedNode::new(node.clone(), |x| x + 1);
        assert_eq!(sink.get(), 30);
        source.put(32);
        assert_eq!(sink.get(), 34);
    }
}
