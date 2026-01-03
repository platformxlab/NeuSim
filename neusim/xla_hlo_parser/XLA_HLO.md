# xla_hlo_parser

This directory contains documentation and resources related to parsing XLA HLO IR for use in the NeuSim simulator.

# XLA HLO IR

aXelerated Linear Algebra High-Level Operators Intermediate Representation

## List of Resources
- [XLA Operation Semantics in Compiler](https://www.tensorflow.org/xla/operation_semantics)
- [XLA-Report](https://github.com/TensorflowXLABeginner/XLA-Report)
- [Chromium Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/edit)

## XLA HLO IR Module Source Codes
https://drive.google.com/file/d/1yXoGoOje2_WbrdB60ynLV4yJDIy_taLI/view?usp=sharing

## TODO
- List of XLA instructions that should be supported by NPU
- How to generate XLA traces from current benchmarks in Tensorflow
    - How to parse TensorBoard trace files (`*.trace` or `*.tracetable`)
- How to support XLA instructions in the simulator
    - As a reference, how does TPU compiler/hardware handle XLA operations?

## Useful Notes
- `with tf.device("/TPU:0"):` runs the code on the specified TPU core 0
- `@tf.function(jit_compile=True)` enables XLA JIT compilation
- `func.experimental_get_compiler_ir(a, b)(stage="optimized_hlo")` returns a string that represents the XLA code for `func` taking parameters of type `a` and `b`
- For Keras models, use:

        XLA_FLAGS="--xla_dump_to=./xla/xla_hlo" TF_XLA_FLAGS="--tf_xla_auto_jit=2" python [...]
- `tensorflow/core/tpu/tpu_ops_c_api.h` contains C APIs for Tensorflow TPU
- `hlo_instruction.h`: [HLO instruction definitions in compiler source code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/hlo_instruction.h)
- `hlo_opcode.h`: [List of all HLO opcodes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/hlo_opcode.h)

## XLA HLO IR Language Semantics

### General Syntax
- Comments in the HLO IR are enclosed by C-style "`/* some comments */`".
- Variables declarations prefixed by "`%`". For exmample, "`%param_0.1`".
- Each HLO module corresponding to a DNN model.
- Each HLO module has an `ENTRY` function as the "main" function.

### HLO Functions (a.k.a. HLO Computations)
- Function Definition Example:

        %bitcast_fusion.1 (bf16input.1: f32[4,4,8,4]) -> f32[4,4,8,4]

    - `bitcast_fusion.1`: function name
    - `(bf16input.1: f32[4,4,8,4])`: function parameter(s). Multiple
        parameters can be written as `(param.1: type1, param.2: type2)`.
    - `f32[4,4,8,4]`: return type
- Function call: `calls=%function_name`
- Each function has a `ROOT` variable as the return instruction:
    - Example: `ROOT %bitcast.4 = f32[4,4,8,4]{2,3,1,0:T(4,128)} ...`

### Variable Semanics
- [Shapes and Layout](https://www.tensorflow.org/xla/shapes):
    basics about shapes and dimension index ordering conventions
- [Tiled Layout](https://www.tensorflow.org/xla/tiled_layout):
    explains what does it mean by "`F32[3,5]{1,0:T(2,2)}`"
    - `F32`: each element is float32 data type
    - `[3,5]`: original tensor shape
    - `1,0`: minor_to_major layout ordering. In this example, it implies
        a 3x5 matrix stored in row-major in the linear memory
    - `T(2,2)`: each tile is 2x2 shape
    - Implies padding
    - Visualization see Figure 1 in the link
- Repeated Tiling: "`bf16[256,128,128,8]{0,3,2,1:T(8,128)(2,1)}`"
    - `bf16`: each element is bf16 data type
    - `[256,128,128,8]`: original tensor shape
    - `0,3,2,1`: memory layout in row-major is 128x128x8x256
    - `T(8,128)(2,1)`: first divide into 8x128-sized tiles,
        then divide each tile further into 2x1 tiles. The 2x1 tile is
        used to group two bf16 values into a 32-bit input for each PE
        in the systolic array for TPU.

### Compute
- `convolution`
- `add`
- `compare`
- `select`

### Data Movement
- `copy`
- `reshape`
- `constant`
- `get-tuple-element`

### Type Conversion
- `bitcast`, `bitcast-convert`: [BitcastConvertType](https://www.tensorflow.org/xla/operation_semantics#bitcastconverttype)

### IR-Specific Instructions
- `parameter(i)`: get the `i`-th parameter of the HLO function
- `constant(x)`: returns the constant value `x`
- `get-tuple-element(t), index=i`: get the `i`-th element from the tuple `t`


### Special Ops
- `custom-call`
