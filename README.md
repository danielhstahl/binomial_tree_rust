| [Linux][lin-link] | [Codecov][cov-link]   |
| :---------------: | :-------------------: |
| ![lin-badge]      | ![cov-badge]          |

[lin-badge]: https://github.com/phillyfan1138/binomial_tree_rust/workflows/Rust/badge.svg
[lin-link]:  https://github.com/phillyfan1138/binomial_tree_rust/actions
[cov-badge]: https://codecov.io/gh/phillyfan1138/binomial_tree_rust/branch/master/graph/badge.svg
[cov-link]:  https://codecov.io/gh/phillyfan1138/binomial_tree_rust

## Binomial Tree Option Calculator

This is a very generic binomial tree calculator.   The calculator can be used to price American and European style options for any payoff and any single dimensional SDE of the form dS=alpha(S, t)dt+sigma(S, t)dW_t.

Requires 4 functions:
* The ratio of drift over volatility: (alpha(S, t)/sigma(S, t))
* The derivative of sigma with respect to the underlying: sigma'(S, t)
* The discount factor
* The payoff function

To demonstrate the flexibility, the tests compute the Black Scholes model price and a bond price under a CIR process.

## Speed

This library takes roughly .4 seconds compared to .7 seconds for my C++ library for a 5000 step European call option.  Benchmarks at https://phillyfan1138.github.io/binomial_tree_rust/report/index.html.