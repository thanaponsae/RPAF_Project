"""
parameters.py — Central parameter store
========================================
All scripts import from here. Change once, all results update.
Source: Table 1, Saengsuwan & Kaewpia (2025) Utilities Policy
"""
CAPACITY_KWP      = 501.1
IRRADIATION       = 5.1
PERFORMANCE_RATIO = 0.80
AVAILABILITY      = 0.99
DEGRADATION       = 0.008
LIFETIME          = 25
FIT_RATE          = 4.12
DISCOUNT_RATE     = 0.0631
CARBON_PRICE      = 200.0
EMISSION_FACTOR   = 0.5692
OM_RATE           = 0.05
STAFF_COST        = 180_000.0
INFLATION         = 0.025
EXCHANGE_RATE     = 33.5
TAX_RATE          = 0.0        # Govt university: Revenue Code §3(1)
CAPEX             = 24_930_000.0

RPAF_WEIGHTS = {
    "Policy Stability":        0.30,
    "Financial Incentives":    0.25,
    "Regulatory Environment":  0.20,
    "Market Maturity":         0.15,
    "Implementation Barriers": 0.10,
}

COUNTRY_SCORES = {
    "Thailand":    {"Policy Stability":85,"Financial Incentives":78,
                    "Regulatory Environment":72,"Market Maturity":80,
                    "Implementation Barriers":75},
    "Singapore":   {"Policy Stability":90,"Financial Incentives":65,
                    "Regulatory Environment":85,"Market Maturity":85,
                    "Implementation Barriers":90},
    "Malaysia":    {"Policy Stability":75,"Financial Incentives":70,
                    "Regulatory Environment":68,"Market Maturity":75,
                    "Implementation Barriers":80},
    "Vietnam":     {"Policy Stability":70,"Financial Incentives":85,
                    "Regulatory Environment":60,"Market Maturity":70,
                    "Implementation Barriers":65},
    "Philippines": {"Policy Stability":65,"Financial Incentives":75,
                    "Regulatory Environment":65,"Market Maturity":65,
                    "Implementation Barriers":60},
    "Indonesia":   {"Policy Stability":60,"Financial Incentives":60,
                    "Regulatory Environment":55,"Market Maturity":60,
                    "Implementation Barriers":55},
}
COUNTRY_ORDER = ["Thailand","Singapore","Malaysia","Vietnam","Philippines","Indonesia"]
DIMS          = list(RPAF_WEIGHTS.keys())
