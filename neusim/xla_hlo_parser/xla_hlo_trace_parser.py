#!/usr/bin/python3

# a very naive script to parse an hlo IR module in txt format

from os import listdir
from os.path import join, isfile
import itertools
import re

from neusim.xla_hlo_parser.xla_hlo_structures import HLOFunction, HLOInstruction, HLOFusedOpInstruction, HLOModule, HLOModel, HLOType, HLOTensorType, HLOTuple, HLOValue

# remove comments ("/*...*/") to ease succeeding parsing effort
def comment_remover(text):
    '''
    See https://stackoverflow.com/a/241506
    '''
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return ""
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )

    return re.sub(pattern, replacer, text)

def parseHLOTensorType(type_str: str) -> HLOTensorType:
    # scalar element type
    scalar_type: str = type_str[:type_str.find('[')]

    # parse tensor shape
    shape_raw_str = type_str[type_str.find('[')+1:type_str.find(']')].split(',')
    if len(shape_raw_str) == 1 and shape_raw_str[0] == '':
        # scalar type
        shape: list[int] = [1]
        minor_to_major_order: list[int] = [0]
    else:
        # multi-dimensional
        shape_str = [re.sub("[^0-9]", "", s) for s in shape_raw_str]
        shape: list[int] = [int(s) for s in shape_str]
        if type_str.find('{') == -1:
            # by default, use row-major layout
            minor_to_major_order: list[int] = list(reversed(range(len(shape))))
        else:
            minor_to_major_order_str = type_str[type_str.find('{')+1:type_str.find(':T')].split(',')
            minor_to_major_order: list[int] = [int(s) for s in minor_to_major_order_str]
            assert len(minor_to_major_order) == len(shape), "dimension mismatch: shape and layout"

    if type_str.find(':T(') == -1:
        # no tiling
        tile_shape: list[tuple[int, ...]] | None = None
    else:
        re_find_tile = re.search("[}S]", type_str)
        assert re_find_tile != None, f"tile shape not found in: {type_str}"
        tile_raw_str = type_str[type_str.find(':T(')+1:re_find_tile.start()]
        tile_shape_str = tile_raw_str[2:-1].split(")(")
        tile_shape_str = [s.split(',') for s in tile_shape_str]
        tile_shape = [tuple([int(i) for i in d]) for d in tile_shape_str]

    # reserved for handling "S(i)" syntax here; I don't know what this mean for now
    if type_str.find(")S(") == -1:
        pass
    else:
        tile_shape_reserved = type_str[type_str.find(")S(")+1:-1]
        # TODO: what does "S(6)" mean here?

    type: HLOTensorType = HLOTensorType(scalar_type = scalar_type,
                                        shape = shape,
                                        minor_to_major_order = minor_to_major_order,
                                        tile_shape = tile_shape,
                                        type_str = type_str)
    return type

def parseHLOTuple(type_str: str) -> HLOTuple:
    '''
    nested tuples are not tested
    '''
    type_list_str = type_str[1:-1].split(", ")
    type_list: list[HLOType] = [
        t for t in
        [parseHLOType(s) for s in type_list_str]
        if t != None
    ]

    type: HLOTuple = HLOTuple(type_list = type_list, type_str = type_str)
    return type

# parse HLOType from a string
def parseHLOType(type_str: str) -> HLOType | None:
    '''
    Supported type list: HLOTuple, HLOTensorType
    nested tuples are not tested
    '''
    type_str = type_str.strip()
    if type_str == "":  # handle empty tuple "()" for dlrm tf291 format...
        return None
    if type_str[0] == '(' and type_str[-1] == ')':
        return parseHLOTuple(type_str)
    else:
        return parseHLOTensorType(type_str)

# construct an HLOInstruction object from a string
def parseHLOInstruction(ins_str: str) -> HLOInstruction:
    ins_str = ins_str.strip()

    ### determine if this is ROOT instruction
    is_root: bool = "ROOT" in ins_str

    # split lhs and rhs of the statement
    # lhs is the name of the produced result
    if is_root:
        result_name, rhs = ins_str[len("ROOT "):].split(" = ")
    else:
        result_name, rhs = ins_str.split(" = ")
    if result_name[0] == '%':
        result_name = result_name[1:]   # remove leading '%'

    ### result type
    re_result_end = re.search(r"[\]\}\)]\s", rhs)
    assert re_result_end != None, f"result type not found in: {rhs}"
    result_type_str = rhs[:re_result_end.start()+1]
    result_type: HLOType | None = parseHLOType(result_type_str)

    result: HLOValue = HLOValue(type = result_type, name = result_name)

    # opcodes and operands and metadata
    operation_str = rhs[re_result_end.end():]

    ### opcode
    opcode: str = operation_str[:operation_str.find("(")]

    ### operands
    # similar to function parameters, operands are either
    #   a tuple, or multiple non-tuple types
    # TODO: this is a wrong assumption... Luckily, we don't need operands right now
    operands_list: list[HLOValue] = []
    re_op_find = re.search(r"(\), )|(\)$)", operation_str)
    assert re_op_find != None, f"operands not found in: {operation_str}"
    operands_str = operation_str[operation_str.find("(")+1:re_op_find.start()]
    if ") %" in operands_str:
        # # one tuple
        # # TODO: handle tf 2.9.1 HLO output which does not have %-prefix for variable names; not needed for now
        # operand_type_str, operand_name = operands_str.split(") %")
        # operand_type_str = operand_type_str + ")"
        # op_type: HLOType = parseHLOType(operand_type_str)
        # tuple_op: HLOValue = HLOValue(type = op_type, name = operand_name)
        # operands_list.append(tuple_op)
        pass
    elif re.search(r"\([0-9]+", operands_str):
        # TODO: one constant operand; not needed for now
        pass
    else:
        # TODO: multiple (potentially only one) non-tuple types; not needed for now
        pass

    ### metadata
    if operation_str.rfind("), ") == -1:
        metadata_str: str | None = None
    else:
        metadata_str = operation_str[operation_str.rfind("), ")+3:]

    # find call target or loop body
    call_target: str | None = None
    dim_labels: str | None = None
    op_type: str | None = None
    window: str | None = None

    if metadata_str != None:
        if opcode == "fusion":
            re_result_end = re.search("calls=", metadata_str)
            assert re_result_end != None, f"'calls' attribute not found in: {ins_str}"
            call_target = metadata_str[re_result_end.end():]
            if call_target[0] == '%':
                call_target = call_target[1:]
            re_end = re.search(",|$", call_target)
            assert re_end != None, f"call target format not recognized: {call_target}"
            call_target = call_target[:re_end.start()]
        elif opcode == "while":
            re_result_end = re.search("body=", metadata_str)
            assert re_result_end != None, f"'body' attribute not found in: {ins_str}"
            call_target = metadata_str[re_result_end.end():]
            if call_target[0] == '%':
                call_target = call_target[1:]
            re_end = re.search(",|$", call_target)
            assert re_end != None, f"call target format not recognized: {call_target}"
            call_target = call_target[:re_end.start()]
        elif "convolution" in opcode:
            # parse dim_labels
            re_result_end = re.search("dim_labels=", metadata_str)
            assert re_result_end != None, f"'dim_labels' attribute not found in: {ins_str}"
            dim_labels = metadata_str[re_result_end.end():]
            assert "_" in dim_labels and "->" in dim_labels, f"dim_labels format not recognized: {dim_labels}"
            re_end = re.search(",|$", dim_labels)
            assert re_end != None, f"dim_labels format not recognized: {dim_labels}"
            dim_labels = dim_labels[:re_end.start()]

            # parse op_type "Einsum" or "Conv2D"
            re_result_end = re.search("op_type=", metadata_str)
            # If op_type is present, use this in the metadata
            if re_result_end != None:
                op_type = metadata_str[re_result_end.end():]
                op_type = op_type.split(" ")[0][1:-1]  # remove quotes
            else:  # If op_type is not present, mark it as unknown and guess later
                op_type = "unknown"

            # parse window attribute (including kernel size, stride, pad, etc.)
            re_result_end = re.search("window=", metadata_str)
            if re_result_end:
                window = metadata_str[re_result_end.end():]
                re_end = re.search(",|$", window)
                assert re_end != None, f"window format not recognized: {window}"
                window = window[:re_end.start()]

    if metadata_str == None:
        metadata: dict[str, str] = {}
    else:
        metadata: dict[str, str] = {
            "raw_string": metadata_str
        }

    if dim_labels != None:
        metadata["dim_labels"] = dim_labels
    if op_type != None:
        metadata["op_type"] = op_type
    if window != None:
        metadata["window"] = window

    if call_target != None:
        # call ops
        ins = HLOFusedOpInstruction(
            result = result,
            operands = operands_list,
            opcode = opcode,
            metadata = metadata,
            raw_string = ins_str,
            is_root = is_root,
            target_name=call_target
        )
    else:
        # regular ops
        ins = HLOInstruction(
            result = result,
            operands = operands_list,
            opcode = opcode,
            metadata = metadata,
            raw_string = ins_str,
            is_root = is_root
        )
    return ins

# construct an HLOFunction object from a list of strings that represent the function
def parseHLOFunction(F_lines: list[str]) -> HLOFunction:
    # first line is function signature, last line is "}"
    func_sig: str = F_lines[0].strip()[:-2]

    # if this is entry function
    is_entry: bool = "ENTRY" in func_sig

    # function name
    if is_entry:
        func_name: str = func_sig.split(' ')[1]
    else:
        func_name: str = func_sig.split(' ')[0]

    if func_name[0] == '%':
        func_name = func_name[1:]

    # function return type
    if " -> " in func_sig:
        ret_type_str = func_sig.split(" -> ")[1]
        func_ret_type: HLOType | None = parseHLOType(ret_type_str)
        assert func_ret_type != None, f"return type not recognized: {ret_type_str}"
    else:
        ret_type_str = None     # dlrm tf291 format does not have this in function signature...
        func_ret_type = None

    # function parameters
    # Either only one tuple parameter, or multiple non-tuple parameters
    # first and last characters are "(" and ")"
    if " -> " in func_sig:
        param_str = func_sig[func_sig.find(" (") + 1:func_sig.find(" -> ")][1:-1]
        func_params: list[HLOValue] | None = []

        if param_str == "":
            # no parameters
            pass
        elif ": (" in param_str:
            # one tuple parameter
            param_str = param_str.split(": ")
            arg: HLOValue = HLOValue(type = parseHLOType(param_str[1]),
                                    name = param_str[0])
            func_params.append(arg)
        else:
            # multiple non-tuple parameters
            param_str = param_str.split(", ")
            param_str_list = [s.split(": ") for s in param_str]
            for p in param_str_list:
                arg: HLOValue = HLOValue(type = parseHLOType(p[1]),
                                        name = p[0])
                func_params.append(arg)
    else:
        func_params = None  # dlrm tf291 does not have param list in func sig...

    # get list of instructions
    # line 0 is function signature
    # last line is "}"
    F_lines = F_lines[1:-1]
    ins_list: list[HLOInstruction] = [parseHLOInstruction(line) for line in F_lines]

    func: HLOFunction = HLOFunction(func_name, func_params, func_ret_type, ins_list, is_entry, func_sig)
    return func

def parseHLOModuleFromFile(mFilePath: str) -> HLOModule:
    # module file
    with open(mFilePath, "r") as hlo_file:
        hlo_lines = hlo_file.readlines()

    hlo_lines = [comment_remover(line) for line in hlo_lines]

    # line 1: module header
    # HloModule [module name], property1=value1, property2=value2, ...
    module_name = hlo_lines[0].strip().split(" ")[1][:-1]
    module_properties = hlo_lines[0].strip()[len(f"HloModule {module_name}, "):]
    hlo_module: HLOModule = HLOModule(name = module_name, properties = module_properties)

    # TODO: parse memory allocation information
    buf_assign_filepath = f"{module_name[:-4]}-buffer-assignment.txt"

    hlo_lines = hlo_lines[1:]

    # line 2+: hlo functions (ENTRY function is last one)
    # each function is separated by an empty line ('\n')
    # This magic statement splits the hlo_lines into lists,
    #   where each list contains all lines for an HLO function
    hlo_functions = [list(y) for x, y in itertools.groupby(hlo_lines, lambda z: z == '\n') if not x]

    # Pass 1: iterate through each function
    for F_lines in hlo_functions:
        hlo_module.addHLOFunction(parseHLOFunction(F_lines))

    # Pass 2: find target HLOFunction for each call op
    for F in hlo_module.functions:
        for I in F.instructions:
            if isinstance(I, HLOFusedOpInstruction):
                assert I.target == None, f"call target already exists for: {str(I)}"
                I.target = hlo_module.getFunctionByName(I.target_name)
                assert I.target != None, f"call target not found for: {str(I)}"

    return hlo_module

def parseHLOModel(model_name: str, model_hlo_dir: str) -> HLOModel:
    '''
    Parse a model that may contain multiple HLO modules from a directory.
    The directory should only contain files that represent HLO module source codes.
    '''

    # get list of HLO module file names
    mfiles = [join(model_hlo_dir, f) for f in listdir(model_hlo_dir) if isfile(join(model_hlo_dir, f))]
    mfiles = [f for f in mfiles if "after_optimizations" in f and "buffer-assignment" not in f and "buffer_assignment" not in f]
    mfiles = [f for f in mfiles if ".txt" in f]

    # construct hlo model
    hlo_model = HLOModel(model_name)

    # parse hlo modules
    for f in mfiles:
        hlo_module = parseHLOModuleFromFile(f)
        hlo_model.modules.append(hlo_module)

    return hlo_model
