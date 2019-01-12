#[macro_use]
extern crate bencher;
use bencher::Bencher;
fn bench_black_scholes_american(b: &mut Bencher) {
    let r=0.03;
    let sig=0.3;
    let s0=50.0 as f64;
    let maturity=1.0;
    let strike=50.0;
    let alpha_div_sigma=|_t:f64, _underlying:f64, _dt:f64, _width:usize| r/sig;
    let sigma_pr=|_t:f64, _underlying:f64, _dt:f64, _width:usize| sig;
    let sigma_inv=|_t:f64, x:f64, _dt:f64, _width:usize| (x*sig).exp();
    let py_off=|_t:f64, underlying:f64, _dt:f64, _width:usize| if strike<underlying {underlying-strike} else {0.0};
    let disc=|_t:f64, _underlying:f64, dt:f64, _width:usize| (-r*dt).exp();
    b.iter(|| binomial_tree::compute_price_american(
        &alpha_div_sigma,
        &sigma_pr,
        &sigma_inv,
        &py_off,
        &disc,
        s0.ln()/sig,
        maturity,
        5000
    ));
}

benchmark_group!(benches, bench_black_scholes_american);
benchmark_main!(benches);
#[cfg(never)]
fn main() { }