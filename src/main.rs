use std::error::Error;
use std::borrow::Borrow;

fn print_error_message<'a>(err: &(dyn Error + 'a)) {
    print!("{err}");
    if let Some(suberr) = err.source() {
        print!("; ");
        print_error_message(suberr)
    }
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));

    match rsvk::run() {
        Ok(()) =>
            {},
        Err(e) => {
            println!("{e:#?}");
            print!("ERROR: ");
            print_error_message(e.borrow());
            println!();
        }
    }
}
