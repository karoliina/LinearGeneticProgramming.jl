module LGPInstruction

using Lumberjack
using LGPOperator, LGPState
import Base.show, Base.copy
export Instruction, UnaryInstruction, BinaryInstruction, random_instruction, random_branch_instruction
export exec!, clone, is_branch, show_data

"An instruction that has an output register, one or two input registers, and an operator."
abstract Instruction

"An instruction that has an output register, one input register, and a unary operator."
type UnaryInstruction <: Instruction
    op::UnaryOperator
    out::Int64
    in1::Int64
end

"An instruction that has an output register, two input registers, and a binary operator."
type BinaryInstruction <: Instruction
    op::BinaryOperator
    out::Int64
    in1::Int64
    in2::Int64
end


"Show a string representation of a unary instruction."
function show(io::IO, instr::UnaryInstruction)
    print(io, "r$(instr.out) = $(instr.op.func)(r$(instr.in1))")
end


"Show a string representation of a binary instruction."
function show(io::IO, instr::BinaryInstruction)
    if is_branch(instr)
        print(io, "if r$(instr.in1) $(to_string(instr.op)) r$(instr.in2)")
    else
        print(io, "r$(instr.out) = r$(instr.in1) $(instr.op.func) r$(instr.in2)")
    end
end


"Show a string representation of the data in the input register of a unary instruction."
function show_data(instr::UnaryInstruction, registers::Array{Float64})
    data = @sprintf "%.4f" registers[instr.in1]
    println("r$(instr.out) = $(instr.op.func)($(data))")
end


"Show a string representation of the data in the input registers of a binary instruction."
function show_data(instr::BinaryInstruction, registers::Array{Float64})
    data1 = @sprintf "%.4f" registers[instr.in1]
    data2 = @sprintf "%.4f" registers[instr.in2]
    if is_branch(instr)
        println("if $(data1) $(to_string(instr.op)) $(data2)")
    else
        println("r$(instr.out) = $(data1) $(instr.op.func) $(data2)")
    end
end


"""
Initializes and returns a random instruction.
"""
function random_instruction(state::State)
    return create_random_instruction(state, state.calc_idx)
end


"""
Initializes and returns a random instruction, with the possible outputs restricted to the
given set of registers indices.
"""
function random_instruction(state::State, output_idx::Array{Int64})
    return create_random_instruction(state, output_idx)
end


"""
Initializes and returns a random branch instruction.
"""
function random_branch_instruction(state::State)
    return create_random_instruction(state, state.calc_idx, true)
end


"Private method for creating a random instruction."
function create_random_instruction(state::State, output_idx::Array{Int64}, branch=false)
    # combine input and calculation register indices to allow choosing non-constant registers easily
    input_and_calc = cat(1, state.input_idx, state.calc_idx)

    # ensure we have at least one possible output register
    if length(output_idx) == 0
        output_idx = [state.calc_idx[1]]
    end

    if branch
        branch_operators = filter(is_branch_operator, state.operators)
        op = rand(branch_operators)
    else
        op = rand(state.operators)
    end

    has_constant = rand()
    out = rand(output_idx)

    if isa(op, BinaryOperator)
        in1 = nothing; in2 = nothing;
        if has_constant < state.params["CONSTANTS_RATE"]
            const_register = rand(state.const_idx)
            not_const_register = rand(input_and_calc)
            if rand([1, 2]) == 1
                in1 = const_register
                in2 = not_const_register
            else
                in1 = not_const_register
                in2 = const_register
            end
        else
            in1 = rand(input_and_calc)
            in2 = rand(input_and_calc)
        end
        return BinaryInstruction(op, out, in1, in2)
    else
        in1 = has_constant < state.params["CONSTANTS_RATE"] ?
            in1 = rand(state.const_idx) : rand(input_and_calc)

        return UnaryInstruction(op, out, in1)
    end
end


"Executes a unary instruction."
function exec!(instr::UnaryInstruction, registers::Array{Float64})
    registers[instr.out] = func(instr.op, registers[instr.in1])
end


"Executes a binary instruction."
function exec!(instr::BinaryInstruction, registers::Array{Float64})
    registers[instr.out] = func(instr.op, registers[instr.in1], registers[instr.in2])
end


"Creates and returns a shallow copy of a unary instruction."
function clone(instr::UnaryInstruction)
    return UnaryInstruction(instr.op, instr.out, instr.in1)
end


"Creates and returns a shallow copy of a binary instruction."
function clone(instr::BinaryInstruction)
    return BinaryInstruction(instr.op, instr.out, instr.in1, instr.in2)
end


"""
Returns true if the instruction is a branch instruction, false otherwise.
"""
function is_branch(instr::Instruction)
    return isa(instr, BinaryInstruction) && (instr.op.func == if_less || instr.op.func == if_greater)
end


end
