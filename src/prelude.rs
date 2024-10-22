use std::error::Error;

pub type E = Box<dyn Error + 'static>;
pub type R<T> = Result<T, E>;

macro_rules! dont {
    {$($s:stmt);*} => {
            if false {
                $($s)*
            }
        };
    {$($s:stmt);*;} => {
            if false {
                $($s)*
            }
        }
}
