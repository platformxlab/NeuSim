import unittest
from neusim.xla_hlo_parser.xla_hlo_structures import (
    HLOAxis, HLOType, HLOTuple, HLOTensorType, HLOValue,
    HLOInstruction, HLOFunction, HLOModule, HLOModel,
    HLOFusedOpInstruction, isFusedOp
)

class TestXLAHLOStructures(unittest.TestCase):
    def test_hlo_axis(self):
        axis = HLOAxis(name="batch", index=0, size=32)
        self.assertEqual(axis.name, "batch")
        self.assertEqual(axis.index, 0)
        self.assertEqual(axis.size, 32)
        self.assertEqual(axis.tile_size, 32)
        self.assertEqual(axis.data_type, "DT_FLOAT")

    def test_hlo_type(self):
        t = HLOType(type="tensor", raw_string="f32[32,128]")
        self.assertEqual(t.type, "tensor")
        self.assertEqual(t.raw_string, "f32[32,128]")
        
        with self.assertRaises(NotImplementedError):
            t.is_scalar()

    def test_hlo_tuple(self):
        t1 = HLOType(type="tensor")
        t2 = HLOType(type="tensor")
        tuple_type = HLOTuple(type_list=[t1, t2], type_str="(f32[], f32[])")
        self.assertEqual(tuple_type.type, "tuple")
        self.assertFalse(tuple_type.is_scalar())
        self.assertEqual(len(tuple_type.type_list), 2)

    def test_hlo_tensor_type(self):
        # Scalar
        scalar = HLOTensorType(shape=[1])
        self.assertTrue(scalar.is_scalar())
        
        # Not scalar
        tensor = HLOTensorType(shape=[32, 128])
        self.assertFalse(tensor.is_scalar())
        self.assertEqual(tensor.shape, [32, 128])

    def test_hlo_value(self):
        t = HLOTensorType(shape=[1])
        val = HLOValue(type=t, name="val1", value=3.14)
        self.assertEqual(val.name, "val1")
        self.assertEqual(val.value, 3.14)
        self.assertTrue(val.is_scalar)

    def test_hlo_instruction_dim_labels(self):
        # Conv/Einsum case
        res = HLOValue(name="res")
        # Standard Conv2D pattern: batch, feature_in, spatial... -> batch, feature_out, spatial...
        metadata = {"dim_labels": "b01f_01io->b01f"}
        instr = HLOInstruction(result=res, opcode="convolution", metadata=metadata)
        
        # Check parsed axes
        self.assertTrue(hasattr(instr, "input_axes"))
        self.assertTrue(hasattr(instr, "output_axes"))
        
        # lhs: b, 0, 1, f -> batch, spatial0, spatial1, input_channel
        lhs_axes = instr.input_axes[0]
        self.assertEqual(lhs_axes[0].name, "batch")
        self.assertEqual(lhs_axes[1].name, "spatial0")
        self.assertEqual(lhs_axes[2].name, "spatial1")
        self.assertEqual(lhs_axes[3].name, "input_channel")

        # rhs: 0, 1, i, o -> spatial0, spatial1, input_channel, output_channel
        rhs_axes = instr.input_axes[1]
        self.assertEqual(rhs_axes[0].name, "spatial0")
        self.assertEqual(rhs_axes[2].name, "input_channel")
        self.assertEqual(rhs_axes[3].name, "output_channel")

        # out: b, 0, 1, f -> batch, spatial0, spatial1, output_channel
        out_axes = instr.output_axes
        self.assertEqual(out_axes[0].name, "batch")
        self.assertEqual(out_axes[3].name, "output_channel")

    def test_hlo_instruction_conv_config(self):
        res = HLOValue(name="res")
        metadata = {"window": "{size=3x3 stride=2x2 pad=1_1x1_1}"}
        instr = HLOInstruction(result=res, opcode="convolution", metadata=metadata)
        
        self.assertTrue(hasattr(instr, "convolution_window"))
        self.assertEqual(instr.convolution_window["size"], [3, 3])
        self.assertEqual(instr.convolution_window["stride"], [2, 2])
        self.assertEqual(instr.convolution_window["pad"], "1_1x1_1")
        self.assertTrue(instr.isConvolution())

    def test_hlo_instruction_conv_config_complex(self):
        # Complex window with dilations
        res = HLOValue(name="res")
        metadata = {"window": "{size=3x3 stride=2x2 pad=0_1x0_1 lhs_dilation=2x2 rhs_dilation=1x1}"}
        instr = HLOInstruction(result=res, opcode="convolution", metadata=metadata)
        
        self.assertTrue(hasattr(instr, "convolution_window"))
        self.assertEqual(instr.convolution_window["size"], [3, 3])
        self.assertEqual(instr.convolution_window["stride"], [2, 2])
        self.assertEqual(instr.convolution_window["pad"], "0_1x0_1")
        # Ensure it doesn't crash on extra fields like dilation
        self.assertTrue(instr.isConvolution())

    def test_hlo_function(self):
        # Instructions
        val1 = HLOValue(name="v1")
        instr1 = HLOInstruction(result=val1, opcode="add")
        
        val2 = HLOValue(name="v2")
        instr2 = HLOInstruction(result=val2, opcode="multiply", is_root=True)
        
        func = HLOFunction(name="main_func", instructions=[instr1, instr2])
        
        self.assertEqual(func.name, "main_func")
        self.assertEqual(func.ROOT_instruction, instr2)
        self.assertEqual(func.ROOT_value, val2)
        
        # Query by name
        self.assertEqual(func.getInstructionByName("v1"), instr1)
        self.assertIsNone(func.getInstructionByName("v_not_exist"))
        
        # Contains opcode
        has_add, i_add = func.containsOpcode("add")
        self.assertTrue(has_add)
        self.assertEqual(i_add, instr1)
        
        has_conv, _ = func.containsOpcode("convolution")
        self.assertFalse(has_conv)

    def test_hlo_module(self):
        func1 = HLOFunction(name="func1")
        func_entry = HLOFunction(name="main", is_entry=True)
        
        module = HLOModule(name="module1", functions=[func1, func_entry])
        
        self.assertEqual(module.ENTRY, func_entry)
        self.assertEqual(len(module.getHLOFunctions()), 2)
        
        self.assertEqual(module.getFunctionByName("func1"), func1)
        self.assertIsNone(module.getFunctionByName("func_missing"))

    def test_hlo_model(self):
        module1 = HLOModule(name="module1")
        model = HLOModel(name="model", modules=[module1])
        
        # Search module
        matches = model.searchModuleByName("module1")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0], module1)
        
        self.assertEqual(len(model.searchModuleByName("missing")), 0)

    def test_hlo_fused_op_instruction(self):
        target_func = HLOFunction(name="fused_computation.1")
        # To make it a convolution fusion, it needs "fused_computation" in name AND contain "convolution" op
        # Let's add a conv op to target func
        mock_conv_res = HLOValue(name="conv_res")
        conv_op = HLOInstruction(result=mock_conv_res, opcode="convolution")
        target_func.instructions.append(conv_op)
        
        res = HLOValue(name="res")
        fused_op = HLOFusedOpInstruction(
            result=res, opcode="fusion", target_name="fused_computation.1", target_func=target_func
        )
        
        self.assertEqual(fused_op.fusion_type(), "fusion")
        self.assertTrue(fused_op.isConvolutionFusion())
        self.assertTrue(fused_op.isConvolution())

    def test_hlo_tuple_init_none(self):
        t = HLOTuple(type_list=None)
        self.assertEqual(len(t.type_list), 0)

    def test_hlo_instruction_parse_dim_labels_non_conv(self):
        res = HLOValue(name="res")
        # Should return early and not raise error or set input_axes
        instr = HLOInstruction(result=res, opcode="add", metadata={"dim_labels": "..."})
        self.assertFalse(hasattr(instr, "input_axes"))

    def test_hlo_function_init_default(self):
        func = HLOFunction(name="f")
        self.assertEqual(len(func.parameters), 0)
        self.assertEqual(len(func.instructions), 0)

    def test_hlo_function_root_not_found(self):
        func = HLOFunction(name="f", instructions=[
            HLOInstruction(result=HLOValue(name="v"), opcode="add")
        ])
        with self.assertRaises(AssertionError):
            _ = func.ROOT_instruction

    def test_hlo_module_add_get(self):
        module = HLOModule(name="mod", properties="props")
        func = HLOFunction(name="f")
        module.addHLOFunction(func)
        
        self.assertEqual(module.getHLOFunctions(), [func])
        
        # getInstructionByName
        res = HLOValue(name="res")
        instr = HLOInstruction(result=res, opcode="add")
        func.instructions.append(instr)
        
        self.assertEqual(module.getInstructionByName("res"), instr)
        self.assertIsNone(module.getInstructionByName("missing"))

    def test_hlo_model_get_func(self):
        module = HLOModule(name="mod")
        func = HLOFunction(name="f")
        module.addHLOFunction(func)
        model = HLOModel(name="model", modules=[module])
        
        matches = model.getFunctionByName("f")
        self.assertIn(module, matches)
        self.assertEqual(matches[module], func)
        
        self.assertIsNone(model.getFunctionByName("missing"))

    def test_is_fused_op(self):
        res = HLOValue(name="res")
        
        # Not fused
        instr = HLOInstruction(result=res, opcode="add")
        self.assertFalse(isFusedOp(instr))
        
        # Fused
        fused = HLOFusedOpInstruction(result=res, opcode="fusion")
        self.assertTrue(isFusedOp(fused))
