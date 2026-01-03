# HLO Parser helper structures

import re
from typing import Any, Dict, List, Optional, Tuple

# Axis type
class HLOAxis:
    def __init__(self, name: str, index: int, size: int, data_type: str | None = None):
        self.name: str = name
        self.index: int = index
        self.size: int = size
        self.data_type: str = data_type if data_type and len(data_type) > 0 else "DT_FLOAT"
        self.tile_size: int = size  # not tiled by default

    def __str__(self) -> str:
        return f"<{self.name}, index={self.index}, size={self.size}, tile_size={self.tile_size}, dtype={self.data_type}>"

    def __repr__(self) -> str:
        return str(self)

class HLOType:
    def __init__(self, type: str | None = None, raw_string: str | None = None) -> None:
        '''
        Supported type list: tuple, tensor
        '''
        self.type: str = type or "unknown"
        self.raw_string: str = raw_string or ""

    def __str__(self) -> str:
        return self.type + ": " + self.raw_string

    def is_scalar(self) -> bool:
        raise NotImplementedError

# tuple type
class HLOTuple(HLOType):
    def __init__(self, type_list: List[HLOType] | None = None, type_str: str | None = None) -> None:
        super().__init__("tuple", type_str)
        if type_list == None:
            self.type_list: List[HLOType] = []
        else:
            self.type_list: List[HLOType] = type_list

    def is_scalar(self) -> bool:
        return False

# tensor shape and layout, including tiling, for a variable
class HLOTensorType(HLOType):
    def __init__(self, scalar_type: str | None = None,
                    shape: List[int] | None = None,
                    minor_to_major_order: List[int] | None = None,
                    tile_shape: List[Tuple[int, ...]] | None = None,
                    type_str: str | None = None) -> None:
        '''
        For now, tile_shape only supports 2-D tiling for simplicity
        '''
        super().__init__("tensor", type_str)
        self.scalar_type: str = scalar_type or ""
        self.shape: List[int] = shape or []
        self.minor_to_major_order: List[int] = minor_to_major_order or []
        self.tile_shape: List[Tuple[int, ...]] = tile_shape or []

    def is_scalar(self) -> bool:
        return len(self.shape) == 1 and self.shape[0] == 1

# a variable; note this is different from the sense of a conventional LLVM Value
class HLOValue:
    def __init__(self, type: HLOType | None = None,
                    name: str | None = None,
                    is_constant: bool = False,
                    value: int | float | None = None) -> None:
        self.type: HLOType = type or HLOType()
        self.name: str = name or ""
        self.is_constant: bool = is_constant
        self.value: int | float | None = value

    def __str__(self) -> str:
        return str(self.type) + ":\t" + self.name + ",\t" + \
            (str(self.value) if self.value else "")

    @property
    def is_scalar(self) -> bool:
        return self.type.is_scalar()

# HLO instruction, including input variables, output variable, opcode, metadata, etc.
class HLOInstruction:
    def __init__(self, result: HLOValue,
                    operands: List[HLOValue] | None = None,
                    opcode: str | None = None,
                    metadata: Dict[str, str] | None = None,
                    raw_string: str | None = None,
                    is_root: bool = False) -> None:
        self.result: HLOValue = result
        if operands == None:
            self.operands: List[HLOValue] = []
        else:
            self.operands: List[HLOValue] = operands
        self.opcode: str = opcode or "unknown"
        if metadata == None:
            self.metadata: Dict[str, str] = {}
        else:
            self.metadata: Dict[str, str] = metadata
        self.raw_string: str = raw_string or ""
        self.is_root: bool = is_root

        self.parse_dim_labels()
        self.parse_conv_config()

    def parse_dim_labels(self):
        '''
        parse the dimension labels from the metadata
        '''
        if "dim_labels" not in self.metadata:
            return

        # only works for conv/matmul now
        # not sure if other ops may also need this
        if self.opcode != "convolution":
            return

        dim_labels = self.metadata["dim_labels"]
        self.input_axes: List[List[HLOAxis]] = [[], []]
        self.output_axes: List[HLOAxis] = []

        dim_labels = re.split(r"_|;", dim_labels)
        assert len(dim_labels) == 2, f"dim_labels={dim_labels} not supported"
        dim_labels[1:] = dim_labels[1].split("->")
        assert len(dim_labels) == 3, f"dim_labels={dim_labels} not supported"

        lhs, rhs = dim_labels[:2]
        out = dim_labels[2]
        assert len(lhs) == len(rhs) == len(out), f"dim_labels={dim_labels} not supported, length mismatch"
        assert len(lhs) >= 2, f"dim_labels={dim_labels} not supported, rank must be >= 2"

        # lhs
        for i, c in enumerate(lhs):
            if c == 'b':
                self.input_axes[0].append(HLOAxis("batch", i, -1))
            elif c == 'f':
                self.input_axes[0].append(HLOAxis("input_channel", i, -1))
            elif c.isdigit():
                self.input_axes[0].append(HLOAxis(f"spatial{c}", i, -1))
            else:
                assert False, f"dim_labels={dim_labels} not supported, unexpected label {c}"

        # rhs
        for i, c in enumerate(rhs):
            if c == 'i':
                self.input_axes[1].append(HLOAxis("input_channel", i, -1))
            elif c == 'o':
                self.input_axes[1].append(HLOAxis("output_channel", i, -1))
            elif c.isdigit():
                self.input_axes[1].append(HLOAxis(f"spatial{c}", i, -1))
            else:
                assert False, f"dim_labels={dim_labels} not supported, unexpected label {c}"

        # out
        for i, c in enumerate(out):
            if c == 'b':
                self.output_axes.append(HLOAxis("batch", i, -1))
            elif c == 'f':
                self.output_axes.append(HLOAxis("output_channel", i, -1))
            elif c.isdigit():
                self.output_axes.append(HLOAxis(f"spatial{c}", i, -1))
            else:
                assert False, f"dim_labels={dim_labels} not supported, unexpected label {c}"

        # def get_axis_by_name(axes: List[HLOAxis], name: str) -> Optional[HLOAxis]:
        #     for a in axes:
        #         if a.name == name:
        #             return a
        #     return None

        # # check validity
        # assert get_axis_by_name(self.input_axes[0], "batch") == get_axis_by_name(self.output_axes, "batch"), f"dim_labels={dim_labels} not supported, batch mismatch"
        # assert get_axis_by_name(self.input_axes[0], "input_channel") == get_axis_by_name(self.input_axes[1], "input_channel"), f"dim_labels={dim_labels} not supported, input channel mismatch"
        # assert get_axis_by_name(self.input_axes[1], "output_channel") == get_axis_by_name(self.output_axes, "output_channel"), f"dim_labels={dim_labels} not supported, output channel mismatch"

    def parse_conv_config(self):
        '''
        parse the convolution config like stride size, etc.
        '''
        if self.opcode != "convolution":
            return
        if "window" not in self.metadata:
            return

        # Example window_str:
        #   """window={size=3x3 stride=2x2 pad=0_1x0_1}"""
        window_str = self.metadata["window"].removeprefix("{").removesuffix("}")
        window_attrs = window_str.split(" ")
        self.convolution_window: Dict[str, Any] = {}
        for attr in window_attrs:
            key, value = attr.split("=")
            self.convolution_window[key] = value

        if "size" in self.convolution_window:
            self.convolution_window["size"] = [
                int(s) for s in self.convolution_window["size"].split("x")
            ]
        if "stride" in self.convolution_window:
            self.convolution_window["stride"] = [
                int(s) for s in self.convolution_window["stride"].split("x")
            ]

    def isConvolution(self) -> bool:
        '''
        determines if this instruction performs a Convolution/MatMul
        '''
        return self.opcode == "convolution"

    @property
    def name(self) -> str:
        return self.result.name

    def __str__(self) -> str:
        return self.raw_string

    def __repr__(self) -> str:
        return self.raw_string

# HLO function, including func signature and a list of HLOInstruction
class HLOFunction:
    def __init__(self, name: str | None = None,
                    parameters: List[HLOValue] | None = None,
                    return_type: HLOType | None = None,
                    instructions: List[HLOInstruction] | None = None,
                    is_entry: bool = False,
                    raw_string: str | None = None) -> None:
        self.name: str = name or "unknown_hlo_function"
        if parameters == None:
            self.parameters: List[HLOValue] = []
        else:
            self.parameters: List[HLOValue] = parameters
        self.return_type: HLOType = return_type or HLOType()
        if instructions == None:
            self.instructions: List[HLOInstruction] = []
        else:
            self.instructions: List[HLOInstruction] = instructions
        self.is_entry: bool = is_entry

        self.sig_raw_string: str = raw_string or ""

    @property
    def ROOT_instruction(self) -> HLOInstruction:
        # Root instruction is the instruction that contains the "ROOT" prefix.
        # Root instruction is usually the last instruction of the function,
        # but for dlrm tf291 format, this might not be true though...
        for I in reversed(self.instructions):
            if I.is_root:
                return I
        assert False, "ROOT instruction not found"

    @property
    def ROOT_value(self) -> HLOValue:
        # root (return) value is the output value of the last instruction
        return self.instructions[-1].result

    def getInstructionByName(self, iname: str) -> Optional[HLOInstruction]:
        for I in self.instructions:
            if I.result.name == iname:
                return I

    def containsOpcode(self, opcode: str) -> Tuple[bool, Optional[HLOInstruction]]:
        '''
        returns True and the instruction if some instructions of this function has the given opcode;
        returns False otherwise
        '''
        for I in self.instructions:
            if I.opcode == opcode:
                return True, I

        return False, None

    def isConvolutionFusion(self) -> bool:
        '''
        determines if this function is a convolution fusion
        '''
        if "fused_computation" in self.name and self.containsOpcode("convolution")[0]:
            return True
        else:
            return False

    def __str__(self) -> str:
        return self.sig_raw_string + " {\n" + \
            "\t" + "\n\t".join([str(ins) for ins in self.instructions]) + "\n\t}"

# the highest-level HLO module, contains a list of functions, including the ENTRY function
class HLOModule:
    def __init__(self, name: str | None = None,
                    functions: List[HLOFunction] | None = None,
                    properties: str | None = None) -> None:
        self.name: str = name or "unknown_hlo_module"
        if functions == None:
            self.functions: List[HLOFunction] = []
        else:
            self.functions: List[HLOFunction] = functions
        self.properties: str = properties or ""

    @property
    def ENTRY(self) -> HLOFunction:
        '''
        returns the ENTRY function of this module
        '''
        # entry function is always the last function in the module
        return self.functions[-1]

    def addHLOFunction(self, func: HLOFunction) -> None:
        '''
        add an HLOFunction instance to the function list
        '''
        self.functions.append(func)

    def getHLOFunctions(self) -> List[HLOFunction]:
        '''
        get the function list
        '''
        return self.functions

    def getFunctionByName(self, fname: str) -> Optional[HLOFunction]:
        for F in self.functions:
            if F.name == fname:
                return F

    def getInstructionByName(self, iname: str) -> Optional[HLOInstruction]:
        for F in self.functions:
            I = F.getInstructionByName(iname)
            if I != None:
                return I

    def __str__(self) -> str:
        return self.name + ", " + self.properties + ":\n" + \
            "\t" + "\n\t".join([str(func) for func in self.functions])

class HLOModel:
    '''
    A DNN model in HLO. May contain one or multiple HLOModules.
    The modules may contain train, eval, or predict step functions.
    '''
    def __init__(self, name: str | None = None, modules: List[HLOModule] | None = None):
        '''
        name: an arbitrary name;
        modules: list of HLOModules
        '''
        self.name: str = name or "unknown_hlo_model"
        if modules == None:
            self.modules: List[HLOModule] = []
        else:
            self.modules: List[HLOModule] = modules

    def getFunctionByName(self, fname: str) -> Optional[Dict[HLOModule, HLOFunction]]:
        '''
        Get an HLOFunction instance by searching its name @fname in all modules.
        May return None if no functions found, or a Dict[HLOModule, HLOFunction] if at least one match is found.
        The returned dict contains the matched functions as well as which module each function is found.
        '''
        matches: Dict[HLOModule, HLOFunction] = {}
        for M in self.modules:
            matched_func = M.getFunctionByName(fname)
            if matched_func != None:
                matches[M] = matched_func

        if len(matches) == 0:
            return None
        else:
            return matches

    def searchModuleByName(self, mname: str) -> List[HLOModule]:
        '''
        return a list of HLOModules whose name matches the given @mname;
        return an empty list if none found
        '''
        matches: List[HLOModule] = []
        for M in self.modules:
            if mname in M.name:
                matches.append(M)

        return matches

    def __str__(self) -> str:
        return "Model '" + self.name + "' with HLO modules:\n" + ",\n".join([M.name for M in self.modules])


###### derived classes ######

class HLOFusedOpInstruction(HLOInstruction):
    '''
    A fusion or while instruction that calls an HLOFunction
    '''
    def __init__(self, result: HLOValue,
                    operands: List[HLOValue] | None = None,
                    opcode: str | None = None,
                    metadata: Dict[str, str] | None = None,
                    raw_string: str | None = None,
                    is_root: bool = False,
                    target_name: str | None = None,
                    target_func: HLOFunction | None = None):
        '''
        fusion_type is either "while" or "fusion"
        '''
        super().__init__(result, operands, opcode, metadata, raw_string, is_root)

        self.target_name: str = target_name or "unknown"
        self.target: HLOFunction | None = target_func

    def fusion_type(self) -> str:
        assert self.opcode in ["while", "fusion"], f"{self.opcode} is not a fused op"
        return self.opcode

    def isConvolutionFusion(self) -> bool:
        assert self.target != None, "target function not initialized, fusion type unknown"
        return self.target.isConvolutionFusion()

    def isConvolution(self) -> bool:
        return self.isConvolutionFusion()

def isFusedOp(I: HLOInstruction) -> bool:
    '''
    prefer directly using isinstance(I, HLOFusedOpInstruction) for better type annotation
    '''
    return isinstance(I, HLOFusedOpInstruction)
