// aggTrades zip(s) -> per-bar microstructure features -> CSV on stdout.
// usage: tape <SYMBOL> <BAR_MIN> <dir_with_zips> <YYYY-MM-DD>...
mod common;
use common::*;
use rayon::prelude::*;
use std::io::Write;
use std::path::PathBuf;

struct Bar {
    t0: i64, ofi: f64, buy_n: f64, sell_n: f64, cnt: f64,
    first_p: f64, last_p: f64, hi: f64, lo: f64, spv: f64, sv: f64,
    // kyle sums
    sx: f64, sy: f64, sxx: f64, sxy: f64, kn: f64,
    // big trade
    big_s: f64, big_a: f64,
    // sign ac1
    ss: f64, ss1: f64, sp: f64, prev_sign: f64, have_prev: bool,
    // roll spread + rv
    rc: f64, rn: f64, prev_dp: f64, have_dp: bool, prev_lp: f64, have_lp: bool,
    rv2: f64, rvn: f64,
}
impl Bar {
    fn new(t0: i64) -> Self { Bar{ t0, ofi:0.,buy_n:0.,sell_n:0.,cnt:0.,first_p:f64::NAN,last_p:f64::NAN,
        hi:f64::MIN,lo:f64::MAX,spv:0.,sv:0.,sx:0.,sy:0.,sxx:0.,sxy:0.,kn:0.,big_s:0.,big_a:0.,
        ss:0.,ss1:0.,sp:0.,prev_sign:0.,have_prev:false,rc:0.,rn:0.,prev_dp:0.,have_dp:false,
        prev_lp:0.,have_lp:false,rv2:0.,rvn:0. } }
}

fn day_bars(bytes: &[u8], bar_ms: i64, p90: f64) -> Vec<Bar> {
    // rows: agg_id, price, qty, first_id, last_id, ts, is_buyer_maker
    let mut rows: Vec<(i64,f64,f64,bool)> = Vec::with_capacity(4_000_000);
    for_each_row(bytes, |f| {
        if f.len() < 7 { return; }
        let price = pf(f[1]); let qty = pf(f[2]);
        let mut ts = pi(f[5]) as i64;
        if ts > 2_000_000_000_000_000 { ts /= 1000; }
        let bm = f[6].first().map(|&c| c==b't'||c==b'T'||c==b'1').unwrap_or(false);
        rows.push((ts, price, qty, bm));
    });
    if rows.is_empty() { return vec![]; }
    rows.sort_by_key(|r| r.0);
    let t0 = (rows[0].0 / bar_ms) * bar_ms;
    let nb = ((rows[rows.len()-1].0 - t0) / bar_ms + 1).max(1) as usize;
    let mut bars: Vec<Bar> = (0..nb).map(|i| Bar::new(t0 + i as i64 * bar_ms)).collect();
    let mut prev_p_global = rows[0].1;
    for &(ts, price, qty, bm) in &rows {
        let b = ((ts - t0) / bar_ms) as usize;
        if b >= nb { continue; }
        let s = if bm { -1.0 } else { 1.0 };
        let nz = qty * price;
        let bar = &mut bars[b];
        bar.ofi += s*nz;
        if s>0.0 { bar.buy_n += nz; } else { bar.sell_n += nz; }
        bar.cnt += 1.0;
        if bar.first_p.is_nan() { bar.first_p = price; }
        bar.last_p = price;
        bar.spv += price*nz; bar.sv += nz;
        if price > bar.hi { bar.hi = price; }
        if price < bar.lo { bar.lo = price; }
        let dpk = price - prev_p_global;
        let xv = s * nz.sqrt();
        bar.sx += xv; bar.sy += dpk; bar.sxx += xv*xv; bar.sxy += xv*dpk; bar.kn += 1.0;
        prev_p_global = price;
        if nz >= p90 { bar.big_s += s*nz; bar.big_a += nz; }
        if bar.have_prev { bar.ss1 += s*bar.prev_sign; bar.sp += 1.0; }
        bar.ss += s; bar.prev_sign = s; bar.have_prev = true;
        if bar.have_lp {
            let dp = (price / bar.prev_lp).ln();
            bar.rv2 += dp*dp; bar.rvn += 1.0;
            if bar.have_dp { bar.rc += dp*bar.prev_dp; bar.rn += 1.0; }
            bar.prev_dp = dp; bar.have_dp = true;
        }
        bar.prev_lp = price; bar.have_lp = true;
    }
    bars
}

fn quantile90(bytes: &[u8]) -> f64 {
    let mut v: Vec<f64> = Vec::with_capacity(4_000_000);
    for_each_row(bytes, |f| { if f.len()>=3 { v.push(pf(f[1])*pf(f[2])); } });
    if v.is_empty() { return f64::INFINITY; }
    v.sort_by(|a,b| a.partial_cmp(b).unwrap());
    v[(v.len() as f64 * 0.9) as usize]
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let sym = &a[1]; let bar_min: i64 = a[2].parse().unwrap(); let dir = &a[3];
    let days: Vec<String> = a[4..].to_vec();
    let bar_ms = bar_min * 60_000;
    let mut all: Vec<(i64,String)> = days.par_iter().flat_map(|d| {
        let p = PathBuf::from(format!("{dir}/{sym}-aggTrades-{d}.zip"));
        let bytes = match read_zip_csv(&p) { Some(b)=>b, None=>return vec![] };
        let p90 = quantile90(&bytes);
        let bars = day_bars(&bytes, bar_ms, p90);
        bars.into_iter().filter(|b| b.cnt>0.0).map(|b| {
            let tot = b.buy_n + b.sell_n;
            let kd = b.kn*b.sxx - b.sx*b.sx;
            let kyle = if kd!=0.0 && b.kn>5.0 { (b.kn*b.sxy - b.sx*b.sy)/kd } else { f64::NAN };
            let ac1 = if b.sp>4.0 { let m=b.ss/b.cnt; let cov=b.ss1/b.sp - m*m; let var=1.0-m*m;
                if var>1e-9 { cov/var } else { f64::NAN } } else { f64::NAN };
            let roll = if b.rn>2.0 { let c=b.rc/b.rn; if c<0.0 {2.0*(-c).sqrt()*1e4} else {0.0} } else { f64::NAN };
            let rv = if b.rvn>2.0 { (b.rv2/b.rvn).sqrt() } else { f64::NAN };
            let bigimb = if b.big_a>0.0 { b.big_s/b.big_a } else { f64::NAN };
            let vwap = if b.sv>0.0 { b.spv/b.sv } else { f64::NAN };
            let buyf = if tot>0.0 { b.buy_n/tot } else { f64::NAN };
            let aggr = if tot>0.0 { (b.buy_n-b.sell_n)/tot } else { f64::NAN };
            (b.t0, format!("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                b.t0, b.first_p, b.hi, b.lo, b.last_p, tot, b.cnt, b.ofi, buyf, aggr, vwap,
                kyle, roll, ac1, bigimb, rv))
        }).collect()
    }).collect();
    all.sort_by_key(|x| x.0);
    let out = std::io::stdout(); let mut w = std::io::BufWriter::new(out.lock());
    writeln!(w, "t0,open,high,low,close,notional,n_trades,ofi_usd,buy_frac,aggr_imb,vwap,kyle_lambda,roll_spread_bps,trade_sign_ac1,big_trade_imb,realised_vol").unwrap();
    for (_,line) in all { writeln!(w, "{}", line).unwrap(); }
}
