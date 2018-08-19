//! Binomial tree approach to pricing options.  This approach is 
//! extremely general.  It allows pricing of any style option when 
//! the underlying is a 1-dimensional diffusion.  Currently only
//! European and American optionality is provided but it isn't 
//! difficult to add Bermudan optionality.  The approach works
//! by transforming a diffusion into a pure Brownian component
//! and building a tree off the pure Browning component.  Given
//! a diffusion dX=alpha(X, t)dt+sigma(X, t)dW, the user must
//! specify the function alpha(X, t)/sigma(X, t), the function
//! sigma'(X, t) (the derivative of sigma with respect to X), and
//! the inverse function of the indefinite integral of 1/sigma(y, t) 
//! with respect to y.

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


/// Returns increment of t between time steps
/// # Examples
/// 
/// ```
/// let maturity = 5.0;
/// let n_time_periods = 400;
/// let dt = binomial_tree::get_dt(maturity, n_time_periods);
/// ```
pub fn get_dt(
    maturity:f64,
    n_time_periods:usize
)->f64{
    maturity/(n_time_periods as f64)
}
/// Returns t at some time step
/// # Examples
/// 
/// ```
/// let dt = 0.01;
/// let width = 200;
/// let t = binomial_tree::get_t(dt, width);
/// ```
pub fn get_t(dt:f64, width:usize)->f64{
    (width as f64)*dt
}

/// Returns iterator over every t except the maturity
/// # Examples
/// 
/// ```
/// let maturity = 5.0;
/// let n_time_periods = 400;
/// let t_iter = binomial_tree::get_all_t(maturity, n_time_periods);
/// ```
pub fn get_all_t(
    maturity:f64,
    n_time_periods:usize
)->impl Iterator<Item = f64>+DoubleEndedIterator+ExactSizeIterator{
    let dt=get_dt(maturity, n_time_periods);
    (0..n_time_periods).map(move |index|get_t(dt, index))
}

/// Returns price using tree method
/// # Examples
/// 
/// ```
/// //Black Scholes tree
/// let r:f64 = 0.05;
/// let sig:f64 = 0.3;
/// let asset:f64 = 50.0;
/// let strike:f64 = 50.0;
/// let alpha_div_sigma = |_t:f64, _underlying:f64, _dt:f64, _width:usize| r/sig;
/// let sigma_pr = |_t:f64, _underlying:f64, _dt:f64, _width:usize| sig;
/// let sigma_inv = |_t:f64, x:f64, _dt:f64, _width:usize| (x*sig).exp();
/// let py_off = |_t:f64, underlying:f64, _dt:f64, _width:usize| {
///     if strike>underlying {strike-underlying} else {0.0}
/// };
/// let disc=|_t:f64, _underlying:f64, dt:f64, _width:usize| (-r*dt).exp();
/// let y0 = asset.ln()/sig; //inverse of signa_inv
/// let maturity = 0.9;
/// let n_time_periods = 50;
/// let is_american = false;
/// let price = binomial_tree::compute_price_raw(
///     &alpha_div_sigma,
///     &sigma_pr,
///     &sigma_inv,
///     &py_off,
///     &disc,
///     y0,
///     maturity,
///     n_time_periods,
///     is_american
/// );
/// ```
pub fn compute_price_raw(
    alpha_over_sigma:&Fn(f64, f64, f64, usize)->f64,
    sigma_prime:&Fn(f64, f64, f64, usize)->f64,
    sigma_inverse:&Fn(f64, f64, f64, usize)->f64,
    payoff:&Fn(f64, f64, f64, usize)->f64,
    discount:&Fn(f64, f64, f64, usize)->f64,
    y0:f64, 
    maturity:f64,
    n_time_periods:usize, //n=1 for a simple binomial tree with 2 terminal nodes
    is_american:bool
)->f64{
    let dt=get_dt(maturity, n_time_periods);
    let sqrt_dt=dt.sqrt();

    let mut track_option_price:Vec<f64>=(0..(n_time_periods+1)).map(|height_index|{
        let height=get_height(n_time_periods, height_index);
        let w=compute_skeleton(y0, height, sqrt_dt);
        let underlying=sigma_inverse(maturity, w, dt, n_time_periods);
        
        payoff(maturity, underlying, dt, n_time_periods)
    }).collect();
    get_all_t(maturity, n_time_periods).enumerate().rev().for_each(|(width, t)|{
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
/// Returns American option price using tree method
/// # Examples
/// 
/// ```
/// //Black Scholes tree
/// let r:f64 = 0.05;
/// let sig:f64 = 0.3;
/// let asset:f64 = 50.0;
/// let strike:f64 = 50.0;
/// let alpha_div_sigma = |_t:f64, _underlying:f64, _dt:f64, _width:usize| r/sig;
/// let sigma_pr = |_t:f64, _underlying:f64, _dt:f64, _width:usize| sig;
/// let sigma_inv = |_t:f64, x:f64, _dt:f64, _width:usize| (x*sig).exp();
/// let py_off = |_t:f64, underlying:f64, _dt:f64, _width:usize| {
///     if strike>underlying {strike-underlying} else {0.0}
/// };
/// let disc=|_t:f64, _underlying:f64, dt:f64, _width:usize| (-r*dt).exp();
/// let y0 = asset.ln()/sig; //inverse of signa_inv
/// let maturity = 0.9;
/// let n_time_periods = 50;
/// let price = binomial_tree::compute_price_american(
///     &alpha_div_sigma,
///     &sigma_pr,
///     &sigma_inv,
///     &py_off,
///     &disc,
///     y0,
///     maturity,
///     n_time_periods
/// );
/// ```
pub fn compute_price_american(
    alpha_over_sigma:&Fn(f64, f64, f64, usize)->f64,
    sigma_prime:&Fn(f64, f64, f64, usize)->f64,
    sigma_inverse:&Fn(f64, f64, f64, usize)->f64,
    payoff:&Fn(f64, f64, f64, usize)->f64,
    discount:&Fn(f64, f64, f64, usize)->f64,
    y0:f64, 
    maturity:f64,
    n_time_periods:usize //n=1 for a simple binomial tree with 2 terminal nodes
)->f64{
    compute_price_raw(
        alpha_over_sigma,
        sigma_prime,
        sigma_inverse,
        payoff,
        discount,
        y0,
        maturity,
        n_time_periods,
        true
    )
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
    #[test]
    fn american_option_test(){
        let rate=0.09;
        let sigma=0.2;
        let maturity=1.5;
        let stock:f64=50.0;
        let strike=55.0;
        let alpha_div_sigma=|_t:f64, _underlying:f64, _dt:f64, _width:usize| rate/sigma;
        let sigma_pr=|_t:f64, _underlying:f64, _dt:f64, _width:usize| sigma;
        let sigma_inv=|_t:f64, x:f64, _dt:f64, _width:usize| (x*sigma).exp();
        let py_off=|_t:f64, underlying:f64, _dt:f64, _width:usize| if strike>underlying {strike-underlying} else {0.0};
        let disc=|_t:f64, _underlying:f64, dt:f64, _width:usize| (-rate*dt).exp();
        let price=compute_price_american(
            &alpha_div_sigma,
            &sigma_pr,
            &sigma_inv,
            &py_off,
            &disc,
            stock.ln()/sigma,
            maturity,
            5000
        );
        //price comes from http://www.math.columbia.edu/~smirnov/options13.html with days=547
        assert_abs_diff_eq!(price, 5.627853492616043, epsilon=0.001);
    }
    #[test]
    fn binomial_approx_black_scholes_american() {
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
        let price=compute_price_american(
            &alpha_div_sigma,
            &sigma_pr,
            &sigma_inv,
            &py_off,
            &disc,
            s0.ln()/sig,
            maturity,
            5000
        );
        assert_abs_diff_eq!(price, black_scholes::call(s0, strike, r, sig, maturity), epsilon=0.001); //american call and european call should be the same with no dividends
    }
}
