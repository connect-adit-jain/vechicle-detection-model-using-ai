"""
Re-export VehicleDistanceEstimator from traffic module
"""
from traffic.distance_estimator import (
    VehicleDistanceEstimator,
    VEHICLE_DIMENSION_PRIORS,
    ZONE_IMMEDIATE,
    ZONE_NEAR,
    ZONE_MID
)

__all__ = [
    "VehicleDistanceEstimator",
    "VEHICLE_DIMENSION_PRIORS",
    "ZONE_IMMEDIATE",
    "ZONE_NEAR",
    "ZONE_MID"
]

