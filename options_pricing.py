"""Utility functions for pricing European call and put options.

This module currently implements the Black-Scholes-Merton model to price
European-style options on non-dividend-paying stocks.  The implementation uses
only the standard library to remain lightweight and easily portable.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, sqrt


def _norm_cdf(x: float) -> float:
    """Return the cumulative distribution function for a standard normal variable."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


@dataclass(frozen=True)
class BlackScholesInputs:
    """Container for the inputs required by the Black-Scholes-Merton model."""

    spot_price: float
    strike_price: float
    time_to_expiry: float
    risk_free_rate: float
    volatility: float

    def validate(self) -> None:
        """Validate that the parameters are in the expected ranges.

        Raises:
            ValueError: If any input violates the Black-Scholes assumptions.
        """

        if self.spot_price <= 0:
            raise ValueError("Spot price must be strictly positive")
        if self.strike_price <= 0:
            raise ValueError("Strike price must be strictly positive")
        if self.time_to_expiry <= 0:
            raise ValueError("Time to expiry must be strictly positive")
        if self.volatility <= 0:
            raise ValueError("Volatility must be strictly positive")


def _d1_d2(params: BlackScholesInputs) -> tuple[float, float]:
    """Compute the intermediate d1 and d2 terms used in Black-Scholes pricing."""
    numerator = log(params.spot_price / params.strike_price) + (
        params.risk_free_rate + 0.5 * params.volatility**2
    ) * params.time_to_expiry
    denominator = params.volatility * sqrt(params.time_to_expiry)
    d1 = numerator / denominator
    d2 = d1 - denominator
    return d1, d2


def black_scholes_call_price(params: BlackScholesInputs) -> float:
    """Return the Black-Scholes price of a European call option."""
    params.validate()
    d1, d2 = _d1_d2(params)
    discounted_strike = params.strike_price * exp(-params.risk_free_rate * params.time_to_expiry)
    return params.spot_price * _norm_cdf(d1) - discounted_strike * _norm_cdf(d2)


def black_scholes_put_price(params: BlackScholesInputs) -> float:
    """Return the Black-Scholes price of a European put option."""
    params.validate()
    d1, d2 = _d1_d2(params)
    discounted_strike = params.strike_price * exp(-params.risk_free_rate * params.time_to_expiry)
    return discounted_strike * _norm_cdf(-d2) - params.spot_price * _norm_cdf(-d1)


__all__ = [
    "BlackScholesInputs",
    "black_scholes_call_price",
    "black_scholes_put_price",
]
