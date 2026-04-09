from __future__ import annotations

from pydantic import BaseModel, Field


class HousePriceInput(BaseModel):
    medinc: float = Field(..., ge=0.0, le=100.0)
    houseage: float = Field(..., ge=0.0, le=200.0)
    averooms: float = Field(..., ge=0.0, le=100.0)
    avebedrms: float = Field(..., ge=0.0, le=50.0)
    population: float = Field(..., ge=0.0, le=200000.0)
    aveoccup: float = Field(..., ge=0.0, le=100.0)
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "medinc": 8.3252,
                "houseage": 41.0,
                "averooms": 6.984127,
                "avebedrms": 1.0238,
                "population": 322.0,
                "aveoccup": 2.555556,
                "latitude": 37.88,
                "longitude": -122.23,
            }
        }
    }


class HousePriceOutput(BaseModel):
    predicted_price: float = Field(..., ge=0.0)

    model_config = {"json_schema_extra": {"example": {"predicted_price": 2.5}}}
