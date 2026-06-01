pub(super) fn normalize_long_flag(arg: &str) -> String {
    arg.replace('_', "-")
}

pub(super) fn parse_positive_usize_text(flag: &str, raw: &str) -> Result<usize, String> {
    if raw.is_empty() || raw.starts_with('+') || raw.starts_with('-') {
        return Err(format!(
            "invalid {flag} value '{raw}': expected positive integer"
        ));
    }
    let value = raw
        .parse::<usize>()
        .map_err(|err| format!("invalid {flag} value '{raw}': {err}"))?;
    if value == 0 {
        return Err(format!("{flag} must be greater than 0"));
    }
    Ok(value)
}

pub(super) fn parse_usize_flag_allowing_zero(
    flag: &str,
    value: Option<String>,
    allow_zero: bool,
) -> Result<usize, String> {
    let raw = value.ok_or_else(|| format!("missing value for {flag}"))?;
    if raw != raw.trim()
        || raw.contains(char::is_whitespace)
        || raw.starts_with('+')
        || raw.starts_with('-')
    {
        return Err(format!("invalid {flag} value '{raw}': expected integer"));
    }
    let parsed = raw
        .parse::<usize>()
        .map_err(|err| format!("invalid {flag} value '{raw}': {err}"))?;
    if !allow_zero && parsed == 0 {
        return Err(format!("{flag} must be greater than 0"));
    }
    Ok(parsed)
}

pub(super) fn parse_u64_flag_allowing_zero(
    flag: &str,
    value: Option<String>,
    allow_zero: bool,
) -> Result<u64, String> {
    let raw = value.ok_or_else(|| format!("missing value for {flag}"))?;
    if raw != raw.trim()
        || raw.contains(char::is_whitespace)
        || raw.starts_with('+')
        || raw.starts_with('-')
    {
        return Err(format!("invalid {flag} value '{raw}': expected integer"));
    }
    let parsed = raw
        .parse::<u64>()
        .map_err(|err| format!("invalid {flag} value '{raw}': {err}"))?;
    if !allow_zero && parsed == 0 {
        return Err(format!("{flag} must be greater than 0"));
    }
    Ok(parsed)
}

pub(super) fn parse_f64_flag(flag: &str, value: Option<String>) -> Result<f64, String> {
    let raw = value.ok_or_else(|| format!("missing value for {flag}"))?;
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.contains(char::is_whitespace) {
        return Err(format!(
            "invalid {flag} value '{raw}': expected finite number"
        ));
    }
    let parsed = trimmed
        .parse::<f64>()
        .map_err(|err| format!("invalid {flag} value '{raw}': {err}"))?;
    if !parsed.is_finite() {
        return Err(format!("{flag} must be finite"));
    }
    Ok(parsed)
}

pub(super) fn parse_bool_flag(flag: &str, value: Option<String>) -> Result<bool, String> {
    let raw = value.ok_or_else(|| format!("missing value for {flag}"))?;
    match raw.as_str() {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err(format!("invalid {flag} value '{raw}'; expected 0 or 1")),
    }
}

pub(super) fn parse_usize_range_list(flag: &str, raw: &str) -> Result<Vec<usize>, String> {
    let mut values = Vec::new();
    for segment in raw.trim().split(',') {
        let atom = segment.trim();
        if atom.is_empty() {
            return Err(format!("invalid {flag} value '{raw}': empty range segment"));
        }
        if atom.contains(char::is_whitespace) {
            return Err(format!(
                "invalid {flag} value '{raw}': whitespace inside range atom"
            ));
        }
        parse_usize_range_atom(flag, atom, &mut values)?;
    }
    Ok(values)
}

fn parse_usize_range_atom(flag: &str, atom: &str, out: &mut Vec<usize>) -> Result<(), String> {
    if let Some((start, rest)) = atom.split_once('-') {
        let (end, step, multiply) = if let Some((end, step)) = rest.split_once('+') {
            (end, parse_positive_usize_text(flag, step)?, false)
        } else if let Some((end, factor)) = rest.split_once('*') {
            (end, parse_positive_usize_text(flag, factor)?, true)
        } else {
            (rest, 1, false)
        };
        let start = parse_positive_usize_text(flag, start)?;
        let end = parse_positive_usize_text(flag, end)?;
        if start > end {
            return Err(format!("invalid {flag} range '{atom}': start exceeds end"));
        }
        if multiply && step == 1 {
            return Err(format!(
                "invalid {flag} range '{atom}': multiplicative step must be greater than 1"
            ));
        }
        let mut current = start;
        while current <= end {
            out.push(current);
            current = if multiply {
                current.checked_mul(step)
            } else {
                current.checked_add(step)
            }
            .ok_or_else(|| format!("invalid {flag} range '{atom}': overflow"))?;
        }
    } else {
        out.push(parse_positive_usize_text(flag, atom)?);
    }
    Ok(())
}

pub(super) fn parse_usize_flag(flag: &str, value: Option<String>) -> Result<usize, String> {
    let raw = value.ok_or_else(|| format!("missing value for {flag}"))?;
    raw.parse::<usize>()
        .map_err(|err| format!("invalid {flag} value '{raw}': {err}"))
}

pub(super) fn parse_f64_list(flag: &str, raw: &str) -> Result<Vec<f64>, String> {
    let mut values = Vec::new();
    for segment in raw.trim().split(',') {
        let atom = segment.trim();
        if atom.is_empty() {
            return Err(format!("invalid {flag} value '{raw}': empty float segment"));
        }
        if atom.contains(char::is_whitespace) {
            return Err(format!(
                "invalid {flag} value '{raw}': whitespace inside float atom"
            ));
        }
        let value = atom
            .parse::<f64>()
            .map_err(|err| format!("invalid {flag} value '{atom}': {err}"))?;
        if !value.is_finite() {
            return Err(format!("{flag} entries must be finite"));
        }
        if value <= 0.0 {
            return Err(format!("{flag} entries must be greater than 0"));
        }
        values.push(value);
    }
    Ok(values)
}
