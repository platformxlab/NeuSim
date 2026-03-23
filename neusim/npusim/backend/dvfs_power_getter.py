from absl import logging
from functools import lru_cache
import itertools
from typing import NamedTuple

import neusim.npusim.frontend.Operator as Operator
from neusim.npusim.frontend.Operator import DVFSPolicy, DVFSConfig, ComponentDVFSConfig

"""
Map DVFS config (voltage_V, frequency_GHz) to dynamic/static power
using discrete SA/VU/SRAM/HBM/ICI tables.
"""

# Type aliases for table structures
class VfPoint(NamedTuple):
    '''Voltage/frequency point for SA/VU/SRAM components.'''
    voltage_V: float = 0.7
    frequency_GHz: float = 1.7
    static_power_W: float = 0.0
    dynamic_power_W: float = 0.0

class VBWPoint(NamedTuple):
    '''Voltage/bandwidth point for HBM/ICI components.'''
    voltage_V: float = 0.7
    bandwidth_GBs: float = 0.0
    static_power_W: float = 0.0
    dynamic_power_W: float = 0.0

class PowerEfficiencyPoint(NamedTuple):
    '''DVFS voltage regulator power conversion efficiency point for each scaling time, activity factor, and voltage.'''
    scaling_time_ns: int = 0
    activity_factor: float = 1.0
    voltage_V: float = 0.7
    power_efficiency_percent: float = 100.0

# Row = tuple[float, float, float]
Row = NamedTuple('Row', [('x', float), ('s', float), ('d', float)])  # x can be frequency_GHz or bandwidth_GBs
Groups = dict[float, list[Row]]  # keyed by voltage_V, value is list of Rows

# =========================
# Lookup tables for each component at different V/f points.
# The tables are only for TPUv5p's HW spec assuming 7nm FinFET node for now.
# SA and VU points are for a single SA/VU.
# =========================

_SA_POINTS = [
    # voltage_V, frequency_GHz, static_power_W, dynamic_power_W
    (0.45, 0,           0.5883612773214286, 0.0),
    (0.45, 0.05,        0.5883612773214286, 0.244),
    (0.45, 0.1,         0.5883612773214286, 0.489),
    (0.45, 0.150015002, 0.5883612773214286, 0.731),
    (0.45, 0.2,         0.5883612773214286, 0.975),
    (0.45, 0.25,        0.5883612773214286, 1.22),
    (0.45, 0.300120048, 0.5883612773214286, 1.46),
    (0.45, 0.350140056, 0.5883612773214286, 1.71),
    (0.45, 0.4,         0.5883612773214286, 1.95),
    (0.45, 0.450045005, 0.5883612773214286, 2.2),
    (0.45, 0.5,         0.5883612773214286, 2.44),
    (0.45, 0.550055006, 0.5883612773214286, 2.68),
    (0.45, 0.600240096, 0.5883612773214286, 2.93),
    (0.5,  0.650195059, 0.7066400908035716, 3.97),
    (0.5,  0.700280112, 0.7066400908035716, 4.28),
    (0.5,  0.750750751, 0.7066400908035716, 4.58),
    (0.5,  0.8,         0.7066400908035716, 4.89),
    (0.5,  0.850340136, 0.7066400908035716, 5.2),
    (0.55, 0.900900901, 0.8370500646428571, 6.76),
    (0.55, 0.950570342, 0.8370500646428571, 7.13),
    (0.55, 1.0,         0.8370500646428571, 7.49),
    (0.55, 1.050420168, 0.8370500646428571, 7.86),
    (0.55, 1.101321586, 0.8370500646428571, 8.24),
    (0.6,  1.149425287, 0.9947551492857144, 10.368),
    (0.6,  1.201923077, 0.9947551492857144, 10.8544),
    (0.6,  1.25,        0.9947551492857144, 11.264),
    (0.6,  1.302083333, 0.9947551492857144, 11.7504),
    (0.6,  1.351351351, 0.9947551492857144, 12.2112),
    (0.65, 1.400560224, 1.1645913942857145, 15.0016),
    (0.65, 1.449275362, 1.1645913942857145, 15.5136),
    (0.65, 1.501501502, 1.1645913942857145, 16.0768),
    (0.65, 1.552795031, 1.1645913942857145, 16.6144),
    (0.65, 1.602564103, 1.1645913942857145, 17.152),
    (0.7,  1.650165017, 1.35868996        , 20.6848),
    (0.7,  1.7,         1.35868996        , 21.3248),
]
SA_POINTS: list[VfPoint] = [VfPoint(*point) for point in _SA_POINTS]

_VU_POINTS = [
    # voltage_V, frequency_GHz, static_power_W, dynamic_power_W
    (0.45, 0,           0.2054646559675127,  0.0),
    (0.45, 0.05,        0.2054646559675127,  0.0571),
    (0.45, 0.1,         0.2054646559675127,  0.114),
    (0.45, 0.150015002, 0.2054646559675127,  0.171),
    (0.45, 0.2,         0.2054646559675127,  0.228),
    (0.45, 0.25,        0.2054646559675127,  0.285),
    (0.45, 0.300120048, 0.2054646559675127,  0.343),
    (0.45, 0.350140056, 0.2054646559675127,  0.4),
    (0.45, 0.4,         0.2054646559675127,  0.457),
    (0.45, 0.450045005, 0.2054646559675127,  0.514),
    (0.45, 0.5,         0.2054646559675127,  0.571),
    (0.45, 0.550055006, 0.2054646559675127,  0.629),
    (0.45, 0.600240096, 0.2054646559675127,  0.686),
    (0.5,  0.650195059, 0.2459788134822335,  0.936),
    (0.5,  0.700280112, 0.2459788134822335,  1.01),
    (0.5,  0.750750751, 0.2459788134822335,  1.08),
    (0.5,  0.8,         0.2459788134822335,  1.15),
    (0.5,  0.850340136, 0.2459788134822335,  1.22),
    (0.55, 0.900900901, 0.29324533058274116, 1.6),
    (0.55, 0.950570342, 0.29324533058274116, 1.68),
    (0.55, 1.0,         0.29324533058274116, 1.77),
    (0.55, 1.050420168, 0.29324533058274116, 1.86),
    (0.55, 1.101321586, 0.29324533058274116, 1.95),
    (0.55, 1.149425287, 0.29324533058274116, 2.04),
    (0.6,  1.201923077, 0.3458172730720812,  2.57),
    (0.6,  1.25,        0.3458172730720812,  2.68),
    (0.6,  1.302083333, 0.3458172730720812,  2.79),
    (0.6,  1.351351351, 0.3458172730720812,  2.89),
    (0.6,  1.400560224, 0.3458172730720812,  3.0),
    (0.65, 1.449275362, 0.40707082074314727, 3.69),
    (0.65, 1.501501502, 0.40707082074314727, 3.83),
    (0.65, 1.552795031, 0.40707082074314727, 3.96),
    (0.65, 1.602564103, 0.40707082074314727, 4.08),
    (0.7,  1.650165017, 0.475076728,         4.94),
    (0.7,  1.7,         0.475076728,         5.10),
]
VU_POINTS: list[VfPoint] = [VfPoint(*point) for point in _VU_POINTS]

_SRAM_POINTS = [
    # voltage_V, frequency_GHz, static_power_W, dynamic_power_W
    (0.45, 0,           6.43650960949367,   0.0),
    (0.45, 0.05,        6.43650960949367,   0.791),
    (0.45, 0.1,         6.43650960949367,   1.58),
    (0.45, 0.150015002, 6.43650960949367,   2.37),
    (0.45, 0.2,         6.43650960949367,   3.16),
    (0.45, 0.25,        6.43650960949367,   3.95),
    (0.45, 0.300120048, 6.43650960949367,   4.74),
    (0.45, 0.350140056, 6.43650960949367,   5.53),
    (0.45, 0.4,         6.43650960949367,   6.32),
    (0.45, 0.450045005, 6.43650960949367,   7.11),
    (0.45, 0.5,         6.43650960949367,   7.90),
    (0.5,  0.550055006, 8.837429860654009,  10.7),
    (0.5,  0.600240096, 8.837429860654009,  11.7),
    (0.5,  0.650195059, 8.837429860654009,  12.7),
    (0.5,  0.700280112, 8.837429860654009,  13.7),
    (0.5,  0.750750751, 8.837429860654009,  14.6),
    (0.55, 0.8,         11.749184207805905, 18.9),
    (0.55, 0.850340136, 11.749184207805905, 20.1),
    (0.55, 0.900900901, 11.749184207805905, 21.3),
    (0.55, 0.950570342, 11.749184207805905, 22.4),
    (0.55, 1.0,         11.749184207805905, 23.6),
    (0.55, 1.050420168, 11.749184207805905, 24.8),
    (0.6,  1.101321586, 15.248397765348098, 30.9),
    (0.6,  1.149425287, 15.248397765348098, 32.3),
    (0.6,  1.201923077, 15.248397765348098, 33.8),
    (0.6,  1.25,        15.248397765348098, 35.1),
    (0.65, 1.302083333, 19.386153942879744, 42.9),
    (0.65, 1.351351351, 19.386153942879744, 44.6),
    (0.65, 1.400560224, 19.386153942879744, 46.17327223),
    (0.65, 1.449275362, 19.386153942879744, 47.77934338),
    (0.65, 1.501501502, 19.386153942879744, 49.50132232),
    (0.7,  1.552795031, 24.21353615,        59.4),
    (0.7,  1.602564103, 24.21353615,        61.3),
    (0.7,  1.650165017, 24.21353615,        63.1),
    (0.7,  1.7,  24.21353615,        65.0),
]
SRAM_POINTS: list[VfPoint] = [VfPoint(*point) for point in _SRAM_POINTS]

_HBM_POINTS = [
    ### This table has the following assumptions:
    ###  - Only DVFS the memory controller, not the PHY, I/O bus, and DRAM arrays.
    ###  - The power split between MC and PHY is 40%/60% at max BW/freq point (an empirical estimate).
    # voltage_V, bandwidth_GBs, static_power_W, dynamic_power_W
    (0.45,  0.00,            21.861016,  0.000000),  # 0.000000 (scaled ref frequency)
    (0.45,  81.029412,       21.861016,  0.796834),  # 0.050000000
    (0.45,  162.058824,      21.861016,  1.593668),  # 0.100000000
    (0.45,  243.088235,      21.861016,  2.390502),  # 0.150000000
    (0.45,  324.117647,      21.861016,  3.187337),  # 0.200000000
    (0.45,  405.147059,      21.861016,  3.984171),  # 0.250000000
    (0.45,  486.176471,      21.861016,  4.781005),  # 0.300000000
    (0.45,  567.205882,      21.861016,  5.577839),  # 0.350000000
    (0.45,  648.235294,      21.861016,  6.374673),  # 0.400000000
    (0.45,  729.264706,      21.861016,  7.171507),  # 0.450000000
    (0.45,  810.294118,      21.861016,  7.968341),  # 0.500000000
    (0.50,  891.323529,      22.478464,  9.124189),  # 0.550000000
    (0.50,  972.352941,      22.478464,  9.953661),  # 0.600000000
    (0.50,  1053.382353,     22.478464,  10.783133), # 0.650000000
    (0.50,  1134.411765,     22.478464,  11.612604), # 0.700000000
    (0.50,  1215.441176,     22.478464,  12.442076), # 0.750000000
    (0.55,  1296.470588,     23.191154,  13.848718), # 0.800000000
    (0.55,  1377.500000,     23.191154,  14.714263), # 0.850000000
    (0.55,  1458.529412,     23.191154,  15.579808), # 0.900000000
    (0.55,  1539.558824,     23.191154,  16.445353), # 0.950000000
    (0.55,  1620.588235,     23.191154,  17.310897), # 1.000000000
    (0.55,  1701.617647,     23.191154,  18.176442), # 1.050000000
    (0.60,  1782.647059,     23.956602,  19.911178), # 1.100000000
    (0.60,  1863.676471,     23.956602,  20.816232), # 1.150000000
    (0.60,  1900.000000,     23.956602,  21.221945), # 1.172413793
    (0.60,  1925.000000,     23.956602,  21.501181), # 1.187840290
    (0.60,  1950.000000,     23.956602,  21.780417), # 1.203266788
    (0.60,  1975.000000,     23.956602,  22.059654), # 1.218693285
    (0.60,  2000.000000,     23.956602,  22.338890), # 1.234119782
    (0.60,  2025.000000,     23.956602,  22.618126), # 1.249546279
    (0.65,  2050.000000,     24.778335,  23.983827), # 1.264972777
    (0.65,  2075.000000,     24.778335,  24.276313), # 1.280399274
    (0.65,  2100.000000,     24.778335,  24.568798), # 1.295825771
    (0.65,  2125.000000,     24.778335,  24.861284), # 1.311252269
    (0.65,  2150.000000,     24.778335,  25.153770), # 1.326678766
    (0.65,  2175.000000,     24.778335,  25.446255), # 1.342105263
    (0.65,  2200.000000,     24.778335,  25.738741), # 1.357531760
    (0.65,  2225.000000,     24.778335,  26.031227), # 1.372958258
    (0.65,  2240.000000,     24.778335,  26.206718), # 1.382214156
    (0.65,  2250.000000,     24.778335,  26.323712), # 1.388384755
    (0.65,  2265.000000,     24.778335,  26.499204), # 1.397640653
    (0.65,  2290.000000,     24.778335,  26.791690), # 1.413067151
    (0.65,  2315.000000,     24.778335,  27.084175), # 1.428493648
    (0.65,  2340.000000,     24.778335,  27.376661), # 1.443920145
    (0.65,  2365.000000,     24.778335,  27.669147), # 1.459346642
    (0.65,  2372.000000,     24.778335,  27.751043), # 1.463666062
    (0.65,  2397.000000,     24.778335,  28.043528), # 1.479092559
    (0.65,  2422.000000,     24.778335,  28.336014), # 1.494519056
    (0.70,  2447.000000,     25.660103,  30.029117), # 1.509945554
    (0.70,  2472.000000,     25.660103,  30.335913), # 1.525372051
    (0.70,  2497.000000,     25.660103,  30.642708), # 1.540798548
    (0.70,  2502.000000,     25.660103,  30.704067), # 1.543883848
    (0.70,  2527.000000,     25.660103,  31.010862), # 1.559310345
    (0.70,  2552.000000,     25.660103,  31.317657), # 1.574736842
    (0.70,  2577.000000,     25.660103,  31.624453), # 1.590163339
    (0.70,  2602.000000,     25.660103,  31.931248), # 1.605589837
    (0.70,  2627.000000,     25.660103,  32.238043), # 1.621016334
    (0.70,  2629.000000,     25.660103,  32.262587), # 1.622250454
    (0.70,  2654.000000,     25.660103,  32.569382), # 1.637676951
    (0.70,  2679.000000,     25.660103,  32.876177), # 1.653103448
    (0.70,  2704.000000,     25.660103,  33.182972), # 1.668529946
    (0.70,  2729.000000,     25.660103,  33.489768), # 1.683956443
    (0.70,  2754.000000,     25.660103,  33.796563), # 1.699382940
    (0.70,  2755.000000,     25.660103,  33.808835), # 1.700000000
]
HBM_POINTS: list[VBWPoint] = [VBWPoint(*point) for point in _HBM_POINTS]

_ICI_POINTS = [
    # voltage_V, bandwidth_GBs, static_power_W, dynamic_power_W
    (0.45, 0.00,    5.208886, 0.000000),   # 0.000000 (scaled ref frequency)
    (0.45, 17.63,   5.208886, 0.249735),   # 0.049987
    (0.45, 35.27,   5.208886, 0.499612),   # 0.100003
    (0.45, 52.90,   5.208886, 0.749347),   # 0.149991
    (0.45, 70.54,   5.208886, 0.999224),   # 0.200007
    (0.45, 88.17,   5.208886, 1.248960),   # 0.249994
    (0.45, 105.81,  5.208886, 1.498836),   # 0.300010
    (0.45, 123.44,  5.208886, 1.748572),   # 0.349997
    (0.45, 141.08,  5.208886, 1.998449),   # 0.400013
    (0.45, 158.71,  5.208886, 2.248184),   # 0.450001
    (0.45, 176.34,  5.208886, 2.497919),   # 0.499988
    (0.50, 193.98,  5.356007, 2.859937),   # 0.550004
    (0.50, 211.61,  5.356007, 3.119864),   # 0.599992
    (0.50, 229.25,  5.356007, 3.379939),   # 0.650008
    (0.50, 246.88,  5.356007, 3.639866),   # 0.699995
    (0.50, 264.52,  5.356007, 3.899941),   # 0.750011
    (0.55, 282.15,  5.525821, 4.340150),   # 0.799998
    (0.55, 299.78,  5.525821, 4.611342),   # 0.849986
    (0.55, 306.38,  5.525821, 4.712866),   # 0.868699
    (0.55, 319.15,  5.525821, 4.909299),   # 0.904907
    (0.55, 331.91,  5.525821, 5.105579),   # 0.941086
    (0.55, 344.68,  5.525821, 5.302013),   # 0.977294
    (0.55, 357.45,  5.525821, 5.498446),   # 1.013501
    (0.55, 370.21,  5.525821, 5.694726),   # 1.049681
    (0.60, 382.98,  5.708207, 6.159173),   # 1.085888
    (0.60, 395.74,  5.708207, 6.364382),   # 1.122067
    (0.60, 408.51,  5.708207, 6.569752),   # 1.158275
    (0.60, 421.28,  5.708207, 6.775123),   # 1.194483
    (0.60, 434.04,  5.708207, 6.980332),   # 1.230662
    (0.65, 446.81,  5.904003, 7.525575),   # 1.266870
    (0.65, 459.57,  5.904003, 7.740490),   # 1.303049
    (0.65, 461.57,  5.904003, 7.774176),   # 1.308720
    (0.65, 463.57,  5.904003, 7.807861),   # 1.314390
    (0.65, 465.57,  5.904003, 7.841547),   # 1.320061
    (0.65, 467.57,  5.904003, 7.875233),   # 1.325732
    (0.65, 469.57,  5.904003, 7.908919),   # 1.331403
    (0.65, 471.57,  5.904003, 7.942605),   # 1.337073
    (0.65, 473.57,  5.904003, 7.976290),   # 1.342744
    (0.65, 475.57,  5.904003, 8.009976),   # 1.348415
    (0.65, 477.57,  5.904003, 8.043662),   # 1.354085
    (0.65, 479.57,  5.904003, 8.077348),   # 1.359756
    (0.65, 481.57,  5.904003, 8.111034),   # 1.365427
    (0.65, 483.57,  5.904003, 8.144719),   # 1.371098
    (0.65, 485.57,  5.904003, 8.178405),   # 1.376768
    (0.65, 487.57,  5.904003, 8.212091),   # 1.382439
    (0.65, 489.57,  5.904003, 8.245777),   # 1.388110
    (0.65, 491.57,  5.904003, 8.279463),   # 1.393781
    (0.65, 493.57,  5.904003, 8.313148),   # 1.399451
    (0.65, 495.57,  5.904003, 8.346834),   # 1.405122
    (0.65, 497.57,  5.904003, 8.380520),   # 1.410793
    (0.65, 499.57,  5.904003, 8.414206),   # 1.416463
    (0.65, 501.57,  5.904003, 8.447892),   # 1.422134
    (0.65, 503.57,  5.904003, 8.481577),   # 1.427805
    (0.65, 505.57,  5.904003, 8.515263),   # 1.433476
    (0.65, 507.57,  5.904003, 8.548949),   # 1.439146
    (0.65, 509.57,  5.904003, 8.582635),   # 1.444817
    (0.65, 511.57,  5.904003, 8.616320),   # 1.450488
    (0.65, 513.57,  5.904003, 8.650006),   # 1.456159
    (0.65, 515.57,  5.904003, 8.683692),   # 1.461829
    (0.65, 517.57,  5.904003, 8.717378),   # 1.467500
    (0.65, 519.57,  5.904003, 8.751064),   # 1.473171
    (0.65, 521.57,  5.904003, 8.784749),   # 1.478842
    (0.65, 523.57,  5.904003, 8.818435),   # 1.484512
    (0.65, 525.57,  5.904003, 8.852121),   # 1.490183
    (0.65, 527.57,  5.904003, 8.885807),   # 1.495854
    (0.70, 529.57,  6.114105, 9.354544),   # 1.501524
    (0.70, 531.57,  6.114105, 9.389873),   # 1.507195
    (0.70, 533.57,  6.114105, 9.425201),   # 1.512866
    (0.70, 535.57,  6.114105, 9.460530),   # 1.518537
    (0.70, 537.57,  6.114105, 9.495859),   # 1.524207
    (0.70, 539.57,  6.114105, 9.531188),   # 1.529878
    (0.70, 541.57,  6.114105, 9.566517),   # 1.535549
    (0.70, 543.57,  6.114105, 9.601846),   # 1.541220
    (0.70, 545.57,  6.114105, 9.637174),   # 1.546890
    (0.70, 547.57,  6.114105, 9.672503),   # 1.552561
    (0.70, 549.57,  6.114105, 9.707832),   # 1.558232
    (0.70, 551.57,  6.114105, 9.743161),   # 1.563902
    (0.70, 553.57,  6.114105, 9.778490),   # 1.569573
    (0.70, 555.57,  6.114105, 9.813819),   # 1.575244
    (0.70, 557.57,  6.114105, 9.849147),   # 1.580915
    (0.70, 559.57,  6.114105, 9.884476),   # 1.586585
    (0.70, 561.57,  6.114105, 9.919805),   # 1.592256
    (0.70, 563.57,  6.114105, 9.955134),   # 1.597927
    (0.70, 565.57,  6.114105, 9.990463),   # 1.603598
    (0.70, 567.57,  6.114105, 10.025792),   # 1.609268
    (0.70, 569.57,  6.114105, 10.061120),   # 1.614939
    (0.70, 571.57,  6.114105, 10.096449),   # 1.620610
    (0.70, 573.57,  6.114105, 10.131778),   # 1.626281
    (0.70, 575.57,  6.114105, 10.167107),   # 1.631951
    (0.70, 577.57,  6.114105, 10.202436),   # 1.637622
    (0.70, 579.57,  6.114105, 10.237764),   # 1.643293
    (0.70, 581.57,  6.114105, 10.273093),   # 1.648963
    (0.70, 583.57,  6.114105, 10.308422),   # 1.654634
    (0.70, 585.57,  6.114105, 10.343751),   # 1.660305
    (0.70, 587.57,  6.114105, 10.379080),   # 1.665976
    (0.70, 589.57,  6.114105, 10.414409),   # 1.671646
    (0.70, 591.57,  6.114105, 10.449737),   # 1.677317
    (0.70, 593.57,  6.114105, 10.485066),   # 1.682988
    (0.70, 595.57,  6.114105, 10.520395),   # 1.688659
    (0.70, 597.57,  6.114105, 10.555724),   # 1.694329
    (0.70, 599.57,  6.114105, 10.591053),   # 1.700000
]
ICI_POINTS: list[VBWPoint] = [VBWPoint(*point) for point in _ICI_POINTS]

_DVFS_VOLTAGE_REGULATOR_OVERHEAD_TABLE = [
    ### (scaling time in ns, activity factor, voltage in V, power efficiency (percentage))

    # scaling_time_ns = 2
    (2, 0.0, 0.45, 63.95031056), (2, 0.0, 0.5, 66.08660107), (2, 0.0, 0.55, 68.22289157),
    (2, 0.0, 0.6, 69.22986684), (2, 0.0, 0.65, 70.23684211), (2, 0.0, 0.7, 72.12129462),
    (2, 0.1, 0.45, 64.83850932), (2, 0.1, 0.5, 67.20841129), (2, 0.1, 0.55, 69.57831325),
    (2, 0.1, 0.6, 70.57863031), (2, 0.1, 0.65, 71.57894737), (2, 0.1, 0.7, 73.01935874),
    (2, 0.2, 0.45, 66.17080745), (2, 0.2, 0.5, 68.5522712),  (2, 0.2, 0.55, 70.93373494),
    (2, 0.2, 0.6, 72.151078),    (2, 0.2, 0.65, 73.36842105), (2, 0.2, 0.7, 74.82214156),
    (2, 0.3, 0.45, 67.50310559), (2, 0.3, 0.5, 69.89613111), (2, 0.3, 0.55, 72.28915663),
    (2, 0.3, 0.6, 73.05247305), (2, 0.3, 0.65, 73.81578947), (2, 0.3, 0.7, 75.27283726),
    (2, 0.4, 0.45, 68.39130435), (2, 0.4, 0.5, 70.79203771), (2, 0.4, 0.55, 73.19277108),
    (2, 0.4, 0.6, 73.9516487),  (2, 0.4, 0.65, 74.71052632), (2, 0.4, 0.7, 76.06072293),
    (2, 0.5, 0.45, 69.27950311), (2, 0.5, 0.5, 71.68794432), (2, 0.5, 0.55, 74.09638554),
    (2, 0.5, 0.6, 74.85082435), (2, 0.5, 0.65, 75.60526316), (2, 0.5, 0.7, 76.84860859),
    (2, 0.6, 0.45, 70.16770186), (2, 0.6, 0.5, 72.19981479), (2, 0.6, 0.55, 74.23192771),
    (2, 0.6, 0.6, 74.98570069), (2, 0.6, 0.65, 75.73947368), (2, 0.6, 0.7, 77.0746219),
    (2, 0.7, 0.45, 71.05590062), (2, 0.7, 0.5, 72.71168525), (2, 0.7, 0.55, 74.36746988),
    (2, 0.7, 0.6, 75.12057705), (2, 0.7, 0.65, 75.87368421), (2, 0.7, 0.7, 77.30063521),
    (2, 0.8, 0.45, 71.20393375), (2, 0.8, 0.5, 72.8911235),  (2, 0.8, 0.55, 74.57831325),
    (2, 0.8, 0.6, 75.33038469), (2, 0.8, 0.65, 76.08245614), (2, 0.8, 0.7, 77.45042347),
    (2, 0.9, 0.45, 71.35196687), (2, 0.9, 0.5, 73.07056175), (2, 0.9, 0.55, 74.78915663),
    (2, 0.9, 0.6, 75.54019235), (2, 0.9, 0.65, 76.29122807), (2, 0.9, 0.7, 77.60021174),
    (2, 1.0, 0.45, 71.5),       (2, 1.0, 0.5, 73.25),       (2, 1.0, 0.55, 75.0),
    (2, 1.0, 0.6, 75.75),       (2, 1.0, 0.65, 76.5),       (2, 1.0, 0.7, 77.75),

    # scaling_time_ns = 20
    (20, 0.0, 0.45, 72.0), (20, 0.0, 0.5, 73.75), (20, 0.0, 0.55, 75.5),
    (20, 0.0, 0.6, 77.0), (20, 0.0, 0.65, 78.5), (20, 0.0, 0.7, 80.0),
    (20, 0.1, 0.45, 73.0), (20, 0.1, 0.5, 75.0), (20, 0.1, 0.55, 77.0),
    (20, 0.1, 0.6, 78.5), (20, 0.1, 0.65, 80.0), (20, 0.1, 0.7, 81.0),
    (20, 0.2, 0.45, 74.5), (20, 0.2, 0.5, 76.5), (20, 0.2, 0.55, 78.5),
    (20, 0.2, 0.6, 80.25), (20, 0.2, 0.65, 82.0), (20, 0.2, 0.7, 83.0),
    (20, 0.3, 0.45, 76.0), (20, 0.3, 0.5, 78.0), (20, 0.3, 0.55, 80.0),
    (20, 0.3, 0.6, 81.25), (20, 0.3, 0.65, 82.5), (20, 0.3, 0.7, 83.5),
    (20, 0.4, 0.45, 77.0), (20, 0.4, 0.5, 79.0), (20, 0.4, 0.55, 81.0),
    (20, 0.4, 0.6, 82.25), (20, 0.4, 0.65, 83.5), (20, 0.4, 0.7, 84.375),
    (20, 0.5, 0.45, 78.0), (20, 0.5, 0.5, 80.0), (20, 0.5, 0.55, 82.0),
    (20, 0.5, 0.6, 83.25), (20, 0.5, 0.65, 84.5), (20, 0.5, 0.7, 85.25),
    (20, 0.6, 0.45, 79.0), (20, 0.6, 0.5, 80.575), (20, 0.6, 0.55, 82.15),
    (20, 0.6, 0.6, 83.4), (20, 0.6, 0.65, 84.65), (20, 0.6, 0.7, 85.5),
    (20, 0.7, 0.45, 80.0), (20, 0.7, 0.5, 81.15), (20, 0.7, 0.55, 82.3),
    (20, 0.7, 0.6, 83.55), (20, 0.7, 0.65, 84.8), (20, 0.7, 0.7, 85.75),
    (20, 0.8, 0.45, 80.16666667), (20, 0.8, 0.5, 81.35), (20, 0.8, 0.55, 82.53333333),
    (20, 0.8, 0.6, 83.78333333), (20, 0.8, 0.65, 85.03333333), (20, 0.8, 0.7, 85.91666667),
    (20, 0.9, 0.45, 80.33333333), (20, 0.9, 0.5, 81.55), (20, 0.9, 0.55, 82.76666667),
    (20, 0.9, 0.6, 84.01666667), (20, 0.9, 0.65, 85.26666667), (20, 0.9, 0.7, 86.08333334),
    (20, 1.0, 0.45, 80.5), (20, 1.0, 0.5, 81.75), (20, 1.0, 0.55, 83.0),
    (20, 1.0, 0.6, 84.25), (20, 1.0, 0.65, 85.5), (20, 1.0, 0.7, 86.25),

    # scaling_time_ns = 200
    (200, 0.0, 0.45, 76.11428571), (200, 0.0, 0.5, 77.62641997), (200, 0.0, 0.55, 79.13855422),
    (200, 0.0, 0.6, 79.96693793), (200, 0.0, 0.65, 80.79532164), (200, 0.0, 0.7, 82.31863783),
    (200, 0.1, 0.45, 77.17142857), (200, 0.1, 0.5, 78.94113597), (200, 0.1, 0.55, 80.71084337),
    (200, 0.1, 0.6, 81.52501233), (200, 0.1, 0.65, 82.33918129), (200, 0.1, 0.7, 83.34775157),
    (200, 0.2, 0.45, 78.75714286), (200, 0.2, 0.5, 80.5201377),  (200, 0.2, 0.55, 82.28313253),
    (200, 0.2, 0.6, 83.34039668), (200, 0.2, 0.65, 84.39766082), (200, 0.2, 0.7, 85.40572696),
    (200, 0.3, 0.45, 80.34285714), (200, 0.3, 0.5, 82.09913942), (200, 0.3, 0.55, 83.85542169),
    (200, 0.3, 0.6, 84.3838512),  (200, 0.3, 0.65, 84.9122807),  (200, 0.3, 0.7, 85.92022081),
    (200, 0.4, 0.45, 81.4),        (200, 0.4, 0.5, 83.15180723), (200, 0.4, 0.55, 84.90361446),
    (200, 0.4, 0.6, 85.42256747), (200, 0.4, 0.65, 85.94152047), (200, 0.4, 0.7, 86.82061656),
    (200, 0.5, 0.45, 82.45714286), (200, 0.5, 0.5, 84.20447505), (200, 0.5, 0.55, 85.95180723),
    (200, 0.5, 0.6, 86.46128373), (200, 0.5, 0.65, 86.97076023), (200, 0.5, 0.7, 87.7210123),
    (200, 0.6, 0.45, 83.51428571), (200, 0.6, 0.5, 84.81166093), (200, 0.6, 0.55, 86.10903614),
    (200, 0.6, 0.6, 86.61709117), (200, 0.6, 0.65, 87.1251462),  (200, 0.6, 0.7, 87.97823402),
    (200, 0.7, 0.45, 84.57142857), (200, 0.7, 0.5, 85.41884682), (200, 0.7, 0.55, 86.26626506),
    (200, 0.7, 0.6, 86.77289861), (200, 0.7, 0.65, 87.27953216), (200, 0.7, 0.7, 88.23545574),
    (200, 0.8, 0.45, 84.74761905), (200, 0.8, 0.5, 85.62923121), (200, 0.8, 0.55, 86.51084337),
    (200, 0.8, 0.6, 87.01526574), (200, 0.8, 0.65, 87.51968811), (200, 0.8, 0.7, 88.40697049),
    (200, 0.9, 0.45, 84.92380952), (200, 0.9, 0.5, 85.83961561), (200, 0.9, 0.55, 86.75542169),
    (200, 0.9, 0.6, 87.25763287), (200, 0.9, 0.65, 87.75984405), (200, 0.9, 0.7, 88.57848525),
    (200, 1.0, 0.45, 85.1),        (200, 1.0, 0.5, 86.05),       (200, 1.0, 0.55, 87.0),
    (200, 1.0, 0.6, 87.5),        (200, 1.0, 0.65, 88.0),        (200, 1.0, 0.7, 88.75),
]
DVFS_VOLTAGE_REGULATOR_OVERHEAD_TABLE: list[PowerEfficiencyPoint] = [
    PowerEfficiencyPoint(*point) for point in _DVFS_VOLTAGE_REGULATOR_OVERHEAD_TABLE
]

_FIXED_VOLTAGE_REGULATOR_OVERHEAD_TABLE = [
    ### (scaling time in ns (unused), activity factor, voltage in V (always 0.7), power efficiency (percentage))
    (0, 0.0, 0.7, 67.0),
    (0, 0.1, 0.7, 85.0),
    (0, 0.2, 0.7, 86.0),
    (0, 0.3, 0.7, 86.5),
    (0, 0.4, 0.7, 87.0),
    (0, 0.5, 0.7, 87.5),
    (0, 0.6, 0.7, 88.0),
    (0, 0.7, 0.7, 88.5),
    (0, 0.8, 0.7, 89.0),
    (0, 0.9, 0.7, 89.5),
    (0, 1.0, 0.7, 90.0),
]
FIXED_VOLTAGE_REGULATOR_OVERHEAD_TABLE: list[PowerEfficiencyPoint] = [
    PowerEfficiencyPoint(*point) for point in _FIXED_VOLTAGE_REGULATOR_OVERHEAD_TABLE
]


# =========================
# Helpers
# =========================

def _group_by_voltage(points: list[VfPoint]) -> Groups:
    """Group (v, x, s, d) points by v (voltage), and sort each group by x (can be frequency or bandwidth)."""
    groups: Groups = {}
    for v, x, s, d in points:
        groups.setdefault(v, []).append(Row(x, s, d))
    for v in groups:
        groups[v].sort(key=lambda t: t.x)
    return groups


def _choose_voltage_by_request_or_range(
    groups: Groups,
    target_x: float,
    requested_v: float | None = None,
) -> tuple[float | None, list[Row] | None]:
    """
    Choose voltage rows using:
      - If requested_v is not None: pick voltage closest to requested_v.
      - Else: pick segment whose [min_x, max_x] best matches target_x.
    Returns (v, rows).
    """
    if requested_v is not None and len(groups) > 0:
        best_v, best_rows = min(groups.items(), key=lambda item: abs(item[0] - requested_v))
        return best_v, best_rows

    # No requested voltage: infer from x-range
    best_v = None
    best_rows = None
    best_dist: float | None = None
    for v, rows in groups.items():
        x_min = rows[0].x
        x_max = rows[-1].x
        if x_min <= target_x <= x_max:
            dist = 0.0
        elif target_x < x_min:
            dist = x_min - target_x
        else:
            dist = target_x - x_max
        if (
            best_dist is None
            or dist < best_dist
            or (dist == best_dist and (best_v is None or v < best_v))
        ):
            best_dist = dist
            best_v = v
            best_rows = rows
    return best_v, best_rows


def _nearest_point(rows: list[Row], target_x: float) -> Row:
    """Return (x_ref, s_ref, d_ref) where x_ref is closest to target_x."""
    best = min(rows, key=lambda row: abs(row.x - target_x))
    return best


def _scale_dynamic(base_dyn_W: float, base_x: float, new_x: float) -> float:
    """Scale dynamic power linearly with x at fixed voltage."""
    assert base_dyn_W >= 0.0
    assert base_x >= 0.0
    assert new_x >= 0.0
    if base_x == 0.0:
        return base_dyn_W
    else:
        return base_dyn_W * (new_x / base_x)


def _baseline_freq_ghz(points: list[VfPoint]) -> float:
    """Max frequency_GHz from (v, f, s, d) table."""
    return max(p.frequency_GHz for p in points)


@lru_cache(maxsize=None)
def _baseline_bw_hbm() -> float:
    """Max bandwidth_GBs from HBM table."""
    return max(p.bandwidth_GBs for p in HBM_POINTS)


@lru_cache(maxsize=None)
def _baseline_bw_ici() -> float:
    """Max bandwidth_GBs from ICI table."""
    return max(p.bandwidth_GBs for p in ICI_POINTS)


@lru_cache(maxsize=None)
def _max_perf_point(component: str) -> VfPoint | VBWPoint:
    """Return the (v, x, s, d) row with max x (frequency or bandwidth) for component."""
    comp = str(component).strip().lower()
    if comp == "sa":
        return max(SA_POINTS, key=lambda p: p.frequency_GHz)
    elif comp == "vu":
        return max(VU_POINTS, key=lambda p: p.frequency_GHz)
    elif comp == "sram":
        return max(SRAM_POINTS, key=lambda p: p.frequency_GHz)
    elif comp == "hbm":
        return max(HBM_POINTS, key=lambda p: p.bandwidth_GBs)
    elif comp == "ici":
        return max(ICI_POINTS, key=lambda p: p.bandwidth_GBs)
    else:
        raise ValueError(f"Unsupported component: {component!r}")


@lru_cache(maxsize=None)
def _min_power_point(component: str) -> VfPoint | VBWPoint:
    """Return the (v, x, s, d) row with min power (min voltage and frequency) for component."""
    comp = str(component).strip().lower()
    if comp == "sa":
        return min(SA_POINTS, key=lambda p: (p.voltage_V, p.frequency_GHz))
    elif comp == "vu":
        return min(VU_POINTS, key=lambda p: (p.voltage_V, p.frequency_GHz))
    elif comp == "sram":
        return min(SRAM_POINTS, key=lambda p: (p.voltage_V, p.frequency_GHz))
    elif comp == "hbm":
        return min(HBM_POINTS, key=lambda p: (p.voltage_V, p.bandwidth_GBs))
    elif comp == "ici":
        return min(ICI_POINTS, key=lambda p: (p.voltage_V, p.bandwidth_GBs))
    else:
        raise ValueError(f"Unsupported component: {component!r}")


# =========================
# Main API
# =========================

@lru_cache(maxsize=None)
def get_power_from_dvfs(component: str, dvfs: ComponentDVFSConfig) -> tuple[float, float]:
    """
    Compute (dynamic_power_W, static_power_W) for a component from DVFSConfig.
    """
    comp = str(component).strip().lower()
    v_req = dvfs.voltage_V
    f_req_GHz = dvfs.frequency_GHz

    # default No DVFS policy: use max performance point (peak voltage and freq/bandwidth)
    if dvfs.policy == DVFSPolicy.NONE or (
        (v_req is None or v_req <= 0.0)
        and (f_req_GHz is None or f_req_GHz <= 0.0)
    ):
        max_point = _max_perf_point(comp)
        return max_point.dynamic_power_W, max_point.static_power_W

    # SA / VU / SRAM: use frequency in GHz
    if comp in ("sa", "vu", "sram"):
        points = SA_POINTS if comp == "sa" else VU_POINTS if comp == "vu" else SRAM_POINTS
        base_freq_ghz = _baseline_freq_ghz(points)

        if f_req_GHz is None or f_req_GHz <= 0.0:
            f_req_GHz = base_freq_ghz

        # first find the nearest voltage corner
        groups = _group_by_voltage(points)
        _, rows = _choose_voltage_by_request_or_range(groups, f_req_GHz, v_req)
        assert rows is not None, f"No voltage rows available for {comp} selection."
        x_ref, s_ref, d_ref = _nearest_point(rows, f_req_GHz)

        # then extrapolate the dynamic power from the existing corners using the given target frequency (Dynamic Power ~ freq)
        dyn = _scale_dynamic(d_ref, x_ref, f_req_GHz)
        static = s_ref
        return dyn, static

    # HBM: map DVFS freq proxy to bandwidth
    if comp in ("hbm", "ici"):
        base_bw = _baseline_bw_hbm() if comp == "hbm" else _baseline_bw_ici()
        base_freq_ghz = 1.7
        points = HBM_POINTS if comp == "hbm" else ICI_POINTS

        if f_req_GHz is not None and f_req_GHz > 0.0:
            bw_target = base_bw * (f_req_GHz / base_freq_ghz)
        else:
            bw_target = base_bw

        # group by voltage: {v: [(bw, s, d), ...]}
        groups: Groups = {}
        for v, bw, st_p, dyn_p in points:
            groups.setdefault(v, []).append(Row(bw, st_p, dyn_p))
        for v in groups:
            groups[v].sort(key=lambda t: t.x)

        _, rows = _choose_voltage_by_request_or_range(groups, bw_target, v_req)
        assert rows is not None, f"No voltage rows available for {comp} selection."
        bw_ref, s_ref, d_ref = _nearest_point(rows, bw_target)

        dyn = _scale_dynamic(d_ref, bw_ref, bw_target)
        static = s_ref
        return dyn, static

    raise ValueError(f"Unsupported component: {component!r}")


@lru_cache(maxsize=None)
def get_all_dvfs_configs_for_component(
    component: str,
    policy: DVFSPolicy = DVFSPolicy.IDEAL,
) -> list[ComponentDVFSConfig]:
    """
    Return all possible DVFS configurations for a given component.
    """
    comp = str(component).strip().lower()
    configs: list[ComponentDVFSConfig] = []

    if comp in ("sa", "vu", "sram"):
        points = SA_POINTS if comp == "sa" else VU_POINTS if comp == "vu" else SRAM_POINTS
        for v, f, s, d in points:
            configs.append(
                ComponentDVFSConfig(
                    policy=policy,
                    voltage_V=v,
                    frequency_GHz=f,
                )
            )
        return configs

    if comp in ("hbm", "ici"):
        points = HBM_POINTS if comp == "hbm" else ICI_POINTS
        for v, bw, s, d in points:
            # Map bandwidth back to frequency proxy
            base_bw = _baseline_bw_hbm() if comp == "hbm" else _baseline_bw_ici()
            base_freq_ghz = 1.7
            f = base_freq_ghz * (bw / base_bw)
            configs.append(
                ComponentDVFSConfig(
                    policy=policy,
                    voltage_V=v,
                    frequency_GHz=f,
                )
            )
        return configs

    raise ValueError(f"Unsupported component: {component!r}")


def get_all_dvfs_configs_for_op(
    op: Operator.Operator,
    policy: DVFSPolicy = DVFSPolicy.IDEAL,
    perf_degrade_threshold: float = 0,
    total_exe_time_ns: float | None = None,
) -> list[dict[str, ComponentDVFSConfig]]:
    """
    Return all possible DVFS configurations for each component used by the given operator.
    Sweeps through all frequencies for each component and pick the lowest voltage accordingly.
    If the component is unused, just set it to the lowest power state.
    - policy: DVFSPolicy to use for filling out ComponentDVFSConfig.
    - perf_degrade_threshold: float, maximum allowed performance degradation for the entire workload.
    """
    logging.set_verbosity(logging.INFO)

    configs: list[dict[str, ComponentDVFSConfig]] = []
    comp_names = ("sa", "vu", "sram", "hbm", "ici")

    def _get_exe_time_ns(comp_name: str) -> float:
        if comp_name == "sa":
            return op.stats.sa_time_ns
        elif comp_name == "vu":
            return op.stats.vu_time_ns
        elif comp_name == "sram":
            return op.stats.vmem_time_ns
        elif comp_name == "hbm":
            return op.stats.memory_time_ns
        elif comp_name == "ici":
            return op.stats.ici_time_ns
        else:
            raise ValueError(f"Unsupported component name: {comp_name!r}")

    # 1. Compute max allowed execution time
    #    max_allowed_exe_time_ns is derived from the perf_degrade_threshold and the program's original execution time.
    #    If total_exe_time_ns is not provided, use op.stats.execution_time_ns as the baseline.
    if not total_exe_time_ns:
        total_exe_time_ns = op.stats.execution_time_ns
    max_slack_time_ns = total_exe_time_ns * perf_degrade_threshold
    max_allowed_exe_time_ns = op.stats.execution_time_ns + max_slack_time_ns / op.stats.count

    logging.info(f"op_name: {op.name}, original_exe_time_ns: {op.stats.execution_time_ns}, max_allowed_exe_time_ns: {max_allowed_exe_time_ns}, max_op_perf_degrade: {(max_allowed_exe_time_ns / op.stats.execution_time_ns - 1) * 100:.4f}%")

    # 2. For each component, generate a list of ComponentDVFSConfigs that
    #    satisfy op.stats.execution_time_ns <= component exe time <= max_allowed_exe_time_ns.
    #    If the component is unused, just include the min power point config.
    comp_configs_list: list[list[ComponentDVFSConfig]] = []  # list of ComponentDVFSConfigs for each component
    for comp_name in comp_names:
        exe_time_ns = _get_exe_time_ns(comp_name)
        # we can always slow down the component without perf degradation if it is not the bottleneck
        slowdown_factor_max = exe_time_ns / op.stats.execution_time_ns
        # must have component exe time <= max_allowed_exe_time_ns
        slowdown_factor_min = exe_time_ns / max_allowed_exe_time_ns
        max_freq_GHz_required = min(
            (
                _baseline_freq_ghz(SA_POINTS) * slowdown_factor_max
                if comp_name in ("sa", "vu", "sram")
                else 1.7 * slowdown_factor_max  # for HBM and ICI
            ) + 0.05,  # step size is 0.05 GHz
            1.7,
        )
        min_freq_GHz_allowed = max(
            (
                _baseline_freq_ghz(SA_POINTS) * slowdown_factor_min
                if comp_name in ("sa", "vu", "sram")
                else 1.7 * slowdown_factor_min  # for HBM and ICI
            ) - 0.05,  # step size is 0.05 GHz
            0,
        )

        # print(f"Component: {comp_name}, Max freq required: {max_freq_GHz_required}, Min freq allowed: {min_freq_GHz_allowed}")

        if exe_time_ns > 0:
            all_cfgs = get_all_dvfs_configs_for_component(comp_name, policy)
            all_cfgs = [
                cfg for cfg in all_cfgs
                if cfg.frequency_GHz and max_freq_GHz_required >= cfg.frequency_GHz >= min_freq_GHz_allowed
            ]
            # print(f"Component: {comp_name}, Possible DVFS configs count: {len(all_cfgs)}")
            comp_configs_list.append(all_cfgs)
        else:
            min_power_point = _min_power_point(comp_name)
            comp_configs_list.append([
                ComponentDVFSConfig(
                    policy=policy,
                    voltage_V=min_power_point.voltage_V,
                    frequency_GHz=0.05,  # assume lowest freq is 0.05 GHz for now
                )
            ])

    for combo in itertools.product(*comp_configs_list):  # combo: (sa, vu, sram, ici, hbm) ComponentDVFSConfig
        config_dict = {comp_name: config for comp_name, config in zip(comp_names, combo)}
        configs.append(config_dict)

    logging.info(f"Total DVFS configurations generated for op {op.name}: {len(configs)}")

    return configs
