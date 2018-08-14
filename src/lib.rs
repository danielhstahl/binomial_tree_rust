#[cfg(test)]
extern crate black_scholes;
#[macro_use]
#[cfg(test)]
extern crate approx;

fn between_zero_and_one(v:f64)->f64{
    if v>1.0{1.0} else if v<0.0 {0.0} else {v}
}

fn compute_p(
    alph:f64,
    sig:f64,
    sqrt_dt:f64
)->f64{
    between_zero_and_one(
        (alph-sig*0.5)*0.5*sqrt_dt+0.5
    )
}

fn compute_expectation(
    p:f64,
    upper:f64,
    lower:f64,
    discount:f64
)->f64{
    (p*upper+(1.0-p)*lower)*discount
}

fn compute_skeleton(
    x0:f64,
    height:i32,
    sqrt_dt:f64
)->f64{
    x0+sqrt_dt*(height as f64)
}

fn max(v1:f64, v2:f64)->f64{
    if v1>v2 { v1 } else{ v2 }
}

fn get_height(width:usize, index:usize)->i32{
    (width as i32)-(index as i32)*2
}

//while this feels like too many inputs to a function...not sure what else I can do.  but try to cut it down
pub fn compute_price_raw(
    alpha_over_sigma:&Fn(f64, f64, f64, usize)->f64,
    sigma_prime:&Fn(f64, f64, f64, usize)->f64,
    sigma_inverse:&Fn(f64, f64, f64, usize)->f64,
    payoff:&Fn(f64, f64, f64, usize)->f64,
    discount:&Fn(f64, f64, f64, usize)->f64,
    y0:f64, //see if this can't be calculated inside.  in theory can be numerically solved for form sigma_inverse and initial underlying
    maturity:f64,
    n_time_periods:usize, //n=1 for a simple binomial tree with 2 terminal nodes
    is_american:bool
)->f64{
    let dt=maturity/(n_time_periods as f64);
    let sqrt_dt=dt.sqrt();

    let mut track_option_price:Vec<f64>=(0..(n_time_periods+1)).map(|height_index|{
        let height=get_height(n_time_periods, height_index);
        let w=compute_skeleton(y0, height, sqrt_dt);
        let underlying=sigma_inverse(maturity, w, dt, n_time_periods);
        
        payoff(maturity, underlying, dt, n_time_periods)
    }).collect();

    (0..n_time_periods).rev().for_each(|width|{
        let t=(width as f64)*dt;
        (0..(track_option_price.len()-1)).for_each(|height_index|{
            
            let upper=track_option_price[height_index];
            let lower=track_option_price[height_index+1];

            let height=get_height(width, height_index);
            let w=compute_skeleton(y0, height, sqrt_dt);
            let underlying=sigma_inverse(t, w, dt, width);

            let expectation=compute_expectation(
                compute_p(
                    alpha_over_sigma(t, underlying, dt, width),
                    sigma_prime(t, underlying, dt, width), 
                    sqrt_dt
                ),
                upper,
                lower,
                discount(t, underlying, dt, width)
            );
            if is_american{
                track_option_price[height_index]=max(expectation, payoff(t, underlying, dt, width));
            }
            else {
                track_option_price[height_index]=expectation;
            }
        });
        track_option_price.pop();
    });
    *track_option_price.first().unwrap()
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn binomial_approx_black_scholes() {
        let r=0.03;
        let sig=0.3;
        let s0=50.0 as f64;
        let maturity=1.0;
        let strike=50.0;
        let alpha_div_sigma=|_t:f64, _underlying:f64, _dt:f64, _width:usize| r/sig;
        let sigma_pr=|_t:f64, _underlying:f64, _dt:f64, _width:usize| sig;
        let sigma_inv=|_t:f64, x:f64, _dt:f64, _width:usize| (x*sig).exp();
        let py_off=|_t:f64, underlying:f64, _dt:f64, _width:usize| if strike>underlying {strike-underlying} else {0.0};
        let disc=|_t:f64, _underlying:f64, dt:f64, _width:usize| (-r*dt).exp();
        let now = std::time::Instant::now();
        let price=compute_price_raw(
            &alpha_div_sigma,
            &sigma_pr,
            &sigma_inv,
            &py_off,
            &disc,
            s0.ln()/sig,
            maturity,
            5000,
            false
        );
        let new_now=std::time::Instant::now();
        println!("Binomial tree time: {:?}", new_now.duration_since(now));
        assert_abs_diff_eq!(price, black_scholes::put(s0, strike, r, sig, maturity), epsilon=0.001);
    }
    #[test]
    fn binomial_approx_1_period() {
        let r=0.00;
        let sig=0.3;
        let s0=50.0 as f64;
        let maturity=1.0;
        let strike=50.0;
        let alpha_div_sigma=|_t:f64, _underlying:f64, _dt:f64, _width:usize| r/sig;
        let sigma_pr=|_t:f64, _underlying:f64, _dt:f64, _width:usize| sig;
        let sigma_inv=|_t:f64, x:f64, _dt:f64, _width:usize| (x*sig).exp();
        let py_off=|_t:f64, underlying:f64, _dt:f64, _width:usize| if strike>underlying {strike-underlying} else {0.0};
        let disc=|_t:f64, _underlying:f64, dt:f64, _width:usize| (-r*dt).exp();
        let price=compute_price_raw(
            &alpha_div_sigma,
            &sigma_pr,
            &sigma_inv,
            &py_off,
            &disc,
            s0.ln()/sig,
            maturity,
            1,
            false
        );
        assert_abs_diff_eq!(price, 7.451476, epsilon=0.0001);
    }
    #[test]
    fn binomial_approx_cir() {
        let r=0.03 as f64;
        let sig=0.3;
        let initial_r=2.0*r.sqrt()/sig;
        let maturity=1.0;
        let a=1.0;
        let b=0.05;

        let alpha_div_sigma=|_t:f64, underlying:f64, _dt:f64, _width:usize| (a*(b-underlying))/(sig*underlying.sqrt());
        let sigma_pr=|_t:f64, underlying:f64, _dt:f64, _width:usize| (0.5*sig)/underlying.sqrt();
        let sigma_inv=|_t:f64, x:f64, _dt:f64, _width:usize| sig.powi(2)*x.powi(2)*0.25;
        let py_off=|_t:f64, _underlying:f64, _dt:f64, _width:usize| 1.0;
        let disc=|_t:f64, underlying:f64, dt:f64, _width:usize| (-underlying*dt).exp();
        let price=compute_price_raw(
            &alpha_div_sigma,
            &sigma_pr,
            &sigma_inv,
            &py_off,
            &disc,
            initial_r,
            maturity,
            5000,
            false
        );

        let bondcir=|r:f64, a:f64, b:f64, sigma:f64, tau:f64|{
            let h=(a.powi(2)+2.0*sigma.powi(2)).sqrt();
            let expt=(tau*h).exp()-1.0;
            let den=2.0*h+(a+h)*expt;
            let at_t=((2.0*a*b)/(sigma.powi(2)))*(2.0*h*((a+h)*tau*0.5).exp()/den).ln();
            let bt_t=2.0*expt/den;
            (at_t-r*bt_t).exp()
        };
        assert_abs_diff_eq!(price, bondcir(r, a, b, sig, maturity), epsilon=0.0001);
    }
}
