use std::io::Read;
use std::path::Path;

/// Read the single CSV file inside a Binance .zip and return its bytes.
pub fn read_zip_csv(path: &Path) -> Option<Vec<u8>> {
    let f = std::fs::File::open(path).ok()?;
    let mut z = zip::ZipArchive::new(f).ok()?;
    let mut e = z.by_index(0).ok()?;
    let mut buf = Vec::with_capacity(e.size() as usize);
    e.read_to_end(&mut buf).ok()?;
    Some(buf)
}

/// Iterate CSV rows as &[&[u8]] fields, skipping a header row if the first
/// field of row 1 is non-numeric.
pub fn for_each_row<F: FnMut(&[&[u8]])>(data: &[u8], mut f: F) {
    let mut first = true;
    for line in data.split(|&b| b == b'\n') {
        if line.is_empty() { continue; }
        let line = if line.last() == Some(&b'\r') { &line[..line.len()-1] } else { line };
        let mut fields: Vec<&[u8]> = Vec::with_capacity(12);
        let mut start = 0usize;
        for (i, &b) in line.iter().enumerate() {
            if b == b',' { fields.push(&line[start..i]); start = i + 1; }
        }
        fields.push(&line[start..]);
        if first {
            first = false;
            let f0 = fields[0];
            let numeric = !f0.is_empty() && f0.iter().all(|&c| c.is_ascii_digit() || c == b'.' || c == b'-' || c == b'e' || c == b'E' || c == b'+');
            if !numeric { continue; }
        }
        f(&fields);
    }
}

pub fn pf(b: &[u8]) -> f64 { std::str::from_utf8(b).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(f64::NAN) }
pub fn pi(b: &[u8]) -> i64 { std::str::from_utf8(b).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0) }
