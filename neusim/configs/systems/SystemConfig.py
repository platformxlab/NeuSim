from pydantic import BaseModel


class SystemConfig(BaseModel):
    '''
    System configuration for a single virtual NPU slice.
    '''
    PUE: float = 1.1
    carbon_intensity_kgCO2_per_kWh: float = 0.5
