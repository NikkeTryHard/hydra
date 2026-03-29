pub fn maybe_autocast<T, F>(enabled: bool, f: F) -> T
where
    F: FnOnce() -> T,
{
    if enabled {
        tch::autocast(true, f)
    } else {
        f()
    }
}
