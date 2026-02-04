#[derive(Debug, Clone, clap::Parser)]
pub enum Options {
    Collection,
    Context,
    Search,
    Get,
}

fn main() {
    println!("Hello world");
}
