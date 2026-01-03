### Helper functions for modeling power, energy, and carbon emissions of NPUs


def get_operational_energy_consumption_kWh(
    PUE: float,
    lifetime: float,
    power: float,
) -> float:
    '''
    @PUE: Power Usage Effectiveness of the data center
    @lifetime: Lifetime of the data center in years
    @power: Power consumption of the NPU in Watts

    @return: Total operational energy consumption in kWh
    '''

    # Convert power to kW
    power_kW = power / 1000
    # convert lifetime to hours
    lifetime_hours = lifetime * 365 * 24
    # Calculate total energy consumed in kWh
    total_energy_kWh = power_kW * lifetime_hours * PUE

    return total_energy_kWh


def get_operational_carbon_emission_kgCO2e(
    PUE: float,
    carbon_intensity: float,
    lifetime: float,
    power: float,
) -> float:
    '''
    @PUE: Power Usage Effectiveness of the data center
    @carbon_intensity: Carbon intensity of the electricity used in the data center (kgCO2e/kWh)
    @lifetime: Lifetime of the data center in years
    @power: Power consumption of the NPU in Watts

    @return: Total operational carbon emissions in kgCO2e
    '''

    total_energy_kWh = get_operational_energy_consumption_kWh(PUE, lifetime, power)
    total_carbon_emission_kgCO2e = total_energy_kWh * carbon_intensity

    return total_carbon_emission_kgCO2e


def get_total_carbon_emission_kgCO2e(
    PUE: float,
    carbon_intensity: float,
    lifetime: float,
    power: float,
    embodied_carbon: float,
) -> float:
    '''
    @PUE: Power Usage Effectiveness of the data center
    @carbon_intensity: Carbon intensity of the electricity used in the data center (kgCO2e/kWh)
    @lifetime: Lifetime of the data center in years
    @power: Power consumption of the NPU in Watts
    @embodied_carbon: Embodied carbon of the NPU in kgCO2e

    @return: Total carbon emissions in kgCO2e
    '''

    total_carbon_emission_kgCO2e = get_operational_carbon_emission_kgCO2e(
        PUE, carbon_intensity, lifetime, power
    ) + embodied_carbon

    return total_carbon_emission_kgCO2e


def get_total_work_done(
    goodput: float,
    lifetime: float,
) -> float:
    '''
    @goodput: Goodput of the NPU in unit/second (e.g., requests/second, samples/second)
    @lifetime: Lifetime of the data center in years

    @return: Total work done in the same "unit" as in @goodput
    '''

    # convert lifetime to seconds
    lifetime_s = lifetime * 365 * 24 * 3600
    # Calculate total work done in TOPS
    total_work_done = goodput * lifetime_s

    return total_work_done


def get_power_efficiency(
    goodput: float,
    PUE: float,
    power: float,
) -> float:
    '''
    @goodput: Goodput of the NPU in unit/second (e.g., requests/second, samples/second)
    @PUE: Power Usage Effectiveness of the data center
    @power: Power consumption of the NPU in Watts

    @return: Power efficiency in unit/second/W
    '''

    return goodput / (PUE * power)


def get_total_carbon_efficiency(
    goodput: float,
    PUE: float,
    lifetime: float,
    carbon_intensity: float,
    power: float,
    embodied_carbon: float,
) -> float:
    '''
    @goodput: Goodput of the NPU in unit/second (e.g., requests/second, samples/second)
    @PUE: Power Usage Effectiveness of the data center
    @lifetime: Lifetime of the data center in years
    @carbon_intensity: Carbon intensity of the electricity used in the data center (kgCO2e/kWh)
    @power: Power consumption of the NPU in Watts
    @embodied_carbon: Embodied carbon of the NPU in kgCO2e

    @return: Total carbon efficiency in kgCO2e/unit
    '''

    total_carbon_emission_kgCO2e = get_total_carbon_emission_kgCO2e(
        PUE, carbon_intensity, lifetime, power, embodied_carbon
    )
    total_work = get_total_work_done(goodput, lifetime)
    return total_carbon_emission_kgCO2e / total_work
