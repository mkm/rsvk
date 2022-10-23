pub trait Cached {
    fn cached(self) -> Self;
}

impl<T, E> Cached for Box<dyn FnMut() -> Result<T, E>>
    where
        T: Clone + 'static,
        E: 'static,
{
    fn cached(mut self) -> Self {
        let mut result: Option<T> = None;
        Box::new(move || {
            match &result {
                Some(value) => {
                    Ok(value.clone())
                },
                _ => {
                    let value = self()?;
                    result = Some(value.clone());
                    Ok(value)
                }
            }
        })
    }
}

impl<A0, T, E> Cached for Box<dyn FnMut(A0) -> Result<T, E>>
    where
        A0: Clone + PartialEq + 'static,
        T: Clone + 'static,
        E: 'static,
{
    fn cached(mut self) -> Self {
        let mut result: Option<(A0, T)> = None;
        Box::new(move |a0| {
            match &result {
                Some((arg0, value)) if a0 == *arg0 => {
                    Ok(value.clone())
                },
                _ => {
                    let value = self(a0.clone())?;
                    result = Some((a0, value.clone()));
                    Ok(value)
                }
            }
        })
    }
}

impl<A0, A1, T, E> Cached for Box<dyn FnMut(A0, A1) -> Result<T, E>>
    where
        A0: Clone + PartialEq + 'static,
        A1: Clone + PartialEq + 'static,
        T: Clone + 'static,
        E: 'static,
{
    fn cached(mut self) -> Self {
        let mut result: Option<(A0, A1, T)> = None;
        Box::new(move |a0, a1| {
            match &result {
                Some((arg0, arg1, value)) if a0 == *arg0 && a1 == *arg1 => {
                    Ok(value.clone())
                },
                _ => {
                    let value = self(a0.clone(), a1.clone())?;
                    result = Some((a0, a1, value.clone()));
                    Ok(value)
                }
            }
        })
    }
}
