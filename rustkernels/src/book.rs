// bookDepth zip(s) -> per-bar DOM features -> CSV on stdout.
// usage: book <SYMBOL> <BAR_MIN> <dir> <YYYY-MM-DD>...
// raw rows: timestamp(str "YYYY-MM-DD HH:MM:SS"), percentage, depth, notional
mod common;
use common::*;
use rayon::prelude::*;
use std::io::Write;
use std::path::PathBuf;
use std::collections::HashMap;

fn parse_ts(s: &[u8]) -> i64 {
    // "2024-03-15 00:00:09" -> epoch ms (UTC)
    let g = |a:usize,b:usize| std::str::from_utf8(&s[a..b]).unwrap().parse::<i64>().unwrap_or(0);
    if s.len() < 19 { return 0; }
    let (y,mo,d,h,mi,se) = (g(0,4),g(5,7),g(8,10),g(11,13),g(14,16),g(17,19));
    // days from civil (Howard Hinnant)
    let y2 = if mo<=2 { y-1 } else { y };
    let era = if y2>=0 { y2 } else { y2-399 } / 400;
    let yoe = y2 - era*400;
    let mp = (mo + 9) % 12;
    let doy = (153*mp + 2)/5 + d - 1;
    let doe = yoe*365 + yoe/4 - yoe/100 + doy;
    let days = era*146097 + doe - 719468;
    (days*86400 + h*3600 + mi*60 + se) * 1000
}

struct Snap { imb1: f64, imb5: f64, slope: f64, total: f64, mp: f64 }

fn day(bytes: &[u8], bar_ms: i64) -> Vec<(i64, f64,f64,f64,f64,f64,f64,f64)> {
    // group rows by snapshot ts -> per-band notional & depth
    let mut cur_ts: i64 = -1;
    let mut n: HashMap<i32,f64> = HashMap::new();
    let mut dp: HashMap<i32,f64> = HashMap::new();
    let mut snaps: Vec<(i64,Snap)> = Vec::new();
    let flush = |ts:i64, n:&HashMap<i32,f64>, dp:&HashMap<i32,f64>, out:&mut Vec<(i64,Snap)>| {
        let g=|m:&HashMap<i32,f64>,k:i32| *m.get(&k).unwrap_or(&f64::NAN);
        let (b1,a1,b5,a5)=(g(n,-1),g(n,1),g(n,-5),g(n,5));
        let imb1=(b1-a1)/(b1+a1); let imb5=(b5-a5)/(b5+a5);
        let slope=((b1/b5)+(a1/a5))/2.0;
        let total=b5+a5;
        let (db1,da1)=(g(dp,-1),g(dp,1));
        let mp=(db1-da1)/(db1+da1);
        out.push((ts, Snap{imb1,imb5,slope,total,mp}));
    };
    for_each_row(bytes, |f| {
        if f.len()<4 { return; }
        let ts = parse_ts(f[0]);
        if ts != cur_ts {
            if cur_ts>=0 { flush(cur_ts,&n,&dp,&mut snaps); }
            n.clear(); dp.clear(); cur_ts = ts;
        }
        let pct = pf(f[1]).round() as i32;
        dp.insert(pct, pf(f[2]));
        n.insert(pct, pf(f[3]));
    });
    if cur_ts>=0 { flush(cur_ts,&n,&dp,&mut snaps); }
    // aggregate snaps to bars
    let mut bars: HashMap<i64,(f64,f64,f64,f64,f64,f64,f64,f64)> = HashMap::new();
    // (sum_imb1, sum_imb1_sq, sum_imb5, vec_slope..., ) - keep it simple: means + count
    let mut acc: HashMap<i64,(f64,f64,f64,f64,f64,f64,Vec<f64>)> = HashMap::new();
    for (ts,s) in snaps {
        let bt = (ts/bar_ms)*bar_ms;
        let e = acc.entry(bt).or_insert((0.,0.,0.,0.,0.,0.,Vec::new()));
        if s.imb1.is_finite(){ e.0+=s.imb1; e.1+=s.imb1*s.imb1; }
        if s.imb5.is_finite(){ e.2+=s.imb5; }
        if s.total.is_finite(){ e.3+=s.total; }
        if s.mp.is_finite(){ e.4+=s.mp; }
        e.5+=1.0;
        if s.slope.is_finite(){ e.6.push(s.slope); }
    }
    let mut v: Vec<_> = acc.into_iter().map(|(bt,e)| {
        let c=e.5.max(1.0);
        let imb1=e.0/c; let imb1v=(e.1/c - imb1*imb1).max(0.0).sqrt();
        let imb5=e.2/c; let total=e.3/c; let mp=e.4/c;
        let mut sl=e.6.clone(); sl.sort_by(|a,b|a.partial_cmp(b).unwrap());
        let slope = if sl.is_empty(){f64::NAN}else{sl[sl.len()/2]};
        (bt, imb1, imb5, imb1v, slope, total, mp, e.5)
    }).collect();
    v.sort_by_key(|x|x.0);
    let _ = &mut bars;
    v
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let sym=&a[1]; let bar_min:i64=a[2].parse().unwrap(); let dir=&a[3];
    let days:Vec<String>=a[4..].to_vec();
    let bar_ms=bar_min*60_000;
    let mut all: Vec<(i64,f64,f64,f64,f64,f64,f64,f64)> = days.par_iter().flat_map(|d| {
        let p=PathBuf::from(format!("{dir}/{sym}-bookDepth-{d}.zip"));
        match read_zip_csv(&p) { Some(b)=>day(&b,bar_ms), None=>vec![] }
    }).collect();
    all.sort_by_key(|x|x.0);
    all.dedup_by_key(|x|x.0);
    let out=std::io::stdout(); let mut w=std::io::BufWriter::new(out.lock());
    writeln!(w,"t0,depth_imb_1pct,depth_imb_5pct,imb_vol,book_slope,total_depth,microprice_tilt,n_snap").unwrap();
    let mut prev_total=f64::NAN;
    for (t0,i1,i5,iv,sl,tot,mp,ns) in all {
        let wd = if prev_total.is_finite() && prev_total>0.0 { tot/prev_total-1.0 } else { f64::NAN };
        prev_total=tot;
        writeln!(w,"{},{},{},{},{},{},{},{}",t0,i1,i5,iv,sl,tot,mp,ns).unwrap();
        let _=wd;
    }
}
