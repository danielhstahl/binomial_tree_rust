#[macro_use]
extern crate criterion;
extern crate black_scholes;

use criterion::{Criterion, ParameterizedBenchmark};
fn bench_black_scholes_american(c: &mut Criterion) {
    let r=0.03;
    let sig=0.3;
    let s0=50.0 as f64;
    let maturity=1.0;
    let strike=50.0;
    let alpha_div_sigma=move |_t:f64, _underlying:f64, _dt:f64, _width:usize| r/sig;
    let sigma_pr=move |_t:f64, _underlying:f64, _dt:f64, _width:usize| sig;
    let sigma_inv=move |_t:f64, x:f64, _dt:f64, _width:usize| (x*sig).exp();
    let py_off=move |_t:f64, underlying:f64, _dt:f64, _width:usize| if strike<underlying {underlying-strike} else {0.0};
    let disc=move |_t:f64, _underlying:f64, dt:f64, _width:usize| (-r*dt).exp();
    c.bench("compare binomial and black scholes",
        ParameterizedBenchmark::new(
            "black scholes",
            move |b, _i|{
                b.iter(|| {
                    black_scholes::call(s0, strike, r, sig, maturity)
                })
            },
            vec![100, 500, 1000, 5000]
        ).with_function("binomial", move |b, i|{
            b.iter(||{
                binomial_tree::compute_price_american(
                    &alpha_div_sigma,
                    &sigma_pr,
                    &sigma_inv,
                    &py_off,
                    &disc,
                    s0.ln()/sig,
                    maturity,
                    *i
                )
            });
        })
    );
}

criterion_group!(benches, bench_black_scholes_american);
criterion_main!(benches);