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
