from pydantic import BaseModel, Field


class HousePriceInput(BaseModel):
    medinc: float = Field(..., example=8.3252)
    houseage: float = Field(..., example=41.0)
    averooms: float = Field(..., example=6.984127)
    avebedrms: float = Field(..., example=1.0238)
    population: float = Field(..., example=322.0)
    aveoccup: float = Field(..., example=2.555556)
    latitude: float = Field(..., example=37.88)
    longitude: float = Field(..., example=-122.23)
