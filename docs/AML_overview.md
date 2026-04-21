# AML Overview

This project treats AML as a risk-based detection problem on temporal transaction graphs.

Core money laundering stages:
- Placement: illicit funds enter the financial system.
- Layering: multiple transactions obscure the audit trail.
- Integration: illicit funds re-enter the economy as apparently legitimate value.

Common typologies relevant to this project:
- Structuring / smurfing: many smaller transactions over a short period.
- Layering: repeated transfers across multiple entities or accounts.
- Fan-in / fan-out: many-to-one or one-to-many flow concentration.
- Cycle / chain patterns: value moves through intermediate nodes and may return or settle later.

Threshold policy:
- There is no universal numeric threshold valid for all products or jurisdictions.
- Thresholds must be configurable and calibrated per segment, product, and risk profile.
- Default to risk-based thresholds, not hard-coded global constants.
