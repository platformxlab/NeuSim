from pydantic import BaseModel
from pydantic import TypeAdapter


# @dataclass(kw_only=True)
class ChipConfig(BaseModel):
    '''
    Configuration of an NPU chip.
    Default values are based on TPUv5p.
    '''

    name: str = "5p"

    num_sa: int = 8
    num_vu: int = 6
    num_vu_ports: int = 6
    hbm_bw_GBps: float = 2765
    hbm_latency_ns: int = 500
    vmem_size_MB: int = 128
    freq_GHz: float = 1.75

    sa_dim: int = 128

    hbm_size_GB: int = 95
    # vmem_bw_GBps: float = 9999999999999  # calculated based on frequency and num_vu_ports

    @property
    def vmem_bw_GBps(self) -> float:
        return self.num_vu_ports * self.freq_GHz * 8 * 128 * 4  # 8*128 elements, each 4 Bytes, per cycle per port

    ici_bw_GBps: float = 200
    dcn_bw_GBps: float = 25
    pcie_bw_GBps: float = 32
    # taken from DeepSeek ISCA'25 paper: https://arxiv.org/pdf/2505.09343
    # using nvlink latency number for ICI and infiniband for DCN
    ici_latency_ns: int = int(3.33e3)
    dcn_latency_ns: int = int(3.7e3)
    pcie_latency_ns: int = 400

    TDP_W: float = 350
    min_power_W: float = 1
    avg_power_W: float = 1
    max_power_W: float = 331
    HBM_GBps_per_W: float = 123.5
    ICI_GBps_per_W: float = 56.583
    ICI_topology: str = "TORUS_3D"

    embodied_carbon_kgCO2: float = 585

    use_vu_for_small_matmul: bool = True
    '''
    Lower the MatMul ops that are too small to VU to avoid SA padding overhead,
    if using VU results in faster compute time.
    '''

    ### detailed power model
    static_power_W_per_sa: float = 1.35868996
    '''Static power of a single SA in Watts'''
    static_power_W_per_vu: float = 0.475076728
    '''Static power of a single VU in Watts'''
    static_power_vmem_W: float = 24.21353615
    '''Static power of vmem in Watts'''
    static_power_ici_W: float = 6.114104803
    '''Static power of ICI in Watts'''
    static_power_hbm_mc_W: float = 10.264041296
    '''Static power of HBM controller (digital logic) in Watts'''
    static_power_hbm_phy_W: float = 15.396061944
    '''Static power of HBM PHY (analog part) in Watts'''
    static_power_other_W: float = 44.82811018
    '''Static power of other components on chip in Watts'''

    @property
    def static_power_hbm_W(self) -> float:
        '''Static power of HBM controller+PHY in Watts'''
        return self.static_power_hbm_mc_W + self.static_power_hbm_phy_W

    dynamic_power_W_per_SA: float = 28.19413333
    '''Dynamic power of a single SA in Watts'''
    dynamic_power_W_per_VU: float = 2.65216
    '''Dynamic power of a single VU (vector ALU) in Watts'''
    dynamic_power_vmem_W: float = 50.18368
    '''Dynamic power of vmem in Watts'''
    dynamic_power_ici_W_per_GBps: float = 0.01767315271
    '''Dynamic power of ICI in Watts/GBps'''
    dynamic_power_hbm_W_per_GBps: float = 0.01261538462
    '''Dynamic power of HBM controller+PHY in Watts/GBps'''
    dynamic_power_other_W: float = 0
    '''Dynamic power of other components in Watts'''

    pg_config: str = "NoPG"
    '''Power gating configuration. See neusim.npusim.frontend.power_analysis_lib for details.'''

    enable_dvfs: bool = False
    '''Enable dynamic voltage and frequency scaling (experimental).'''

    @property
    def peak_SA_tflops_per_sec(self) -> float:
        '''
        Peak TFLOPs per second of all SAs.
        '''
        return self.num_sa * (self.sa_dim ** 2) * 2 * self.freq_GHz * 1e9 / 1e12

    @property
    def peak_VU_tflops_per_sec(self) -> float:
        '''
        Peak TFLOPs per second of all VUs.
        Assume each VU is 8*128 SIMD ALU.
        '''
        return self.num_vu * (8 * 128) * self.freq_GHz * 1e9 / 1e12

    @property
    def peak_tflops_per_sec(self) -> float:
        '''
        Peak TFLOPs per second of the chip.
        '''
        return self.peak_SA_tflops_per_sec + self.peak_VU_tflops_per_sec

    @property
    def static_power_sa_W(self) -> float:
        return self.static_power_W_per_sa * self.num_sa

    @property
    def static_power_vu_W(self) -> float:
        return self.static_power_W_per_vu * self.num_vu

    @property
    def static_power_vmem_W_per_MB(self) -> float:
        return self.static_power_vmem_W / self.vmem_size_MB

    @property
    def static_power_W(self) -> float:
        return (
            self.static_power_sa_W +
            self.static_power_vu_W +
            self.static_power_vmem_W +
            self.static_power_ici_W +
            self.static_power_hbm_W +
            self.static_power_other_W
        )

    @property
    def idle_power_W(self) -> float:
        '''Currently, assume idle power is the same as static power.'''
        return self.static_power_W

    @property
    def dynamic_power_sa_W(self) -> float:
        return self.dynamic_power_W_per_SA * self.num_sa

    @property
    def dynamic_power_vu_W(self) -> float:
        return self.dynamic_power_W_per_VU * self.num_vu

    @property
    def dynamic_power_hbm_W(self) -> float:
        return self.hbm_bw_GBps * self.dynamic_power_hbm_W_per_GBps

    @property
    def dynamic_power_ici_W(self) -> float:
        return self.ici_bw_GBps * self.dynamic_power_ici_W_per_GBps

    @property
    def dynamic_power_peak_W(self) -> float:
        return (
            self.dynamic_power_sa_W +
            self.dynamic_power_vu_W +
            self.dynamic_power_vmem_W +
            self.dynamic_power_ici_W +
            self.dynamic_power_hbm_W +
            self.dynamic_power_other_W
        )

    @property
    def total_power_peak_W(self) -> float:
        '''
        Total peak power of the chip.
        '''
        return self.static_power_W + self.dynamic_power_peak_W
