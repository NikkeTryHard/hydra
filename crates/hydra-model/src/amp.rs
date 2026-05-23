/// Executes `f` without autocast. Rust LibTorch support has been removed.
pub fn maybe_autocast<T, F>(_enabled: bool, f: F) -> T
where
    F: FnOnce() -> T,
{
    f()
}
