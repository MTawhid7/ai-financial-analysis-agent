"""Sector benchmark data — live Damodaran + static JSON fallback."""

from .lookup import get_sector_benchmarks, normalise_sector

__all__ = ["get_sector_benchmarks", "normalise_sector"]
