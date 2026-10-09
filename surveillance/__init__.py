"""Multi-site sepsis / infection surveillance layer for the Sepsis Digital Twin.

Pipeline: per-patient risk scores -> virtual hospital sites -> privacy-preserving
daily aggregates -> temporal + geographic outbreak detection (EWMA / CUSUM).

NOTE: site assignment, admission dates and the outbreak/viral signals are SIMULATED.
The patient-level risk scores come from real model output on the ICU dataset.
"""
