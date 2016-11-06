module LGPOperator

import Base.show
export Operator, UnaryOperator, BinaryOperator, func, supported_operator, if_less, if_greater, to_string
export is_branch_operator

"The if < function."
function if_less(a::Float64, b::Float64)
    return a < b ? 1.0 : 0.0
end


"The if > function."
function if_greater(a::Float64, b::Float64)
    return a > b ? 1.0 : 0.0
end

const UNARY_FUNCTIONS = [exp, log, sin, cos, sqrt, ~]
const BINARY_FUNCTIONS = [+, -, *, /, ^, if_less, if_greater, &, |, $, div, %]
const BOOLEAN_FUNCTIONS = [&, |, ~]
const BRANCH_FUNCTIONS = [if_less, if_greater]
const C_UNDEF = 1.0 # value used by protected instructions

"An operator that represents the associated function."
abstract Operator


"A unary operator."
immutable UnaryOperator <: Operator
    func::Function

    function UnaryOperator(func::Function)
        if func in UNARY_FUNCTIONS
            new(func)
        else
            error("Function $(func) is not a unary function or is not supported")
        end
    end
end


"A binary operator"
immutable BinaryOperator <: Operator
    func::Function

    function BinaryOperator(func::Function)
        if func in BINARY_FUNCTIONS
            new(func)
        else
            error("Function $(func) is not a binary function or is not supported")
        end
    end
end


"Return a string representation of the given operator."
function to_string(op::Operator)
    if op.func == if_less
        return "<"
    elseif op.func == if_greater
        return ">"
    end
    return op.func
end


function show(io::IO, operator::UnaryOperator)
    print(io, "Unary operator $(to_string(operator))")
end


function show(io::IO, operator::BinaryOperator)
    print(io, "Binary operator $(to_string(operator))")
end


"Execute the function represented by a unary operator on one register's contents."
function func(op::UnaryOperator, arg::Float64)
    try
        # convert infinite register value to constant, if necessary
        data = arg
        if abs(data) == Inf
            data = C_UNDEF
        end

        # protected instructions
        if (op.func == exp)
            return abs(data) <= 32 ? exp(abs(data)) : data + C_UNDEF
        elseif (op.func == log)
            return data != 0 ? log(abs(data)) : data + C_UNDEF
        elseif (op.func == sqrt)
            return sqrt(abs(data))
        end

        # boolean-only functions: convert input values to boolean, and outputs back to Float64 0 or 1
        if op.func in BOOLEAN_FUNCTIONS
            data = data > 0 ? true : false
            return convert(Float64, op.func(data))
        end

        # non-protected instruction
        return op.func(data)
    catch e
        println("$(op.func)($(data))")
        println("Error in executing unary operator: $(e)")
        return C_UNDEF
    end
end


"Execute the function represented by a binary operator on two registers's contents."
function func(op::BinaryOperator, arg1::Float64, arg2::Float64)
    data1 = arg1
    data2 = arg2
    try
        # convert infinite register values to constants, if necessary
        if abs(data1) == Inf
            data1 = C_UNDEF
        end
        if abs(data2) == Inf
            data2 = C_UNDEF
        end

        # protected instructions
        if ((op.func == /) || (op.func == %)) && data2 == 0
            return data1 + C_UNDEF
        elseif (op.func == ^)
            return abs(data1) <= 10 ? abs(data1)^data2 : data1 + data2 + C_UNDEF
        end

        # boolean-only functions: convert input values to boolean, and outputs back to Float64 0 or 1
        if op.func in BOOLEAN_FUNCTIONS
            data1 = data1 > 0 ? true : false
            data2 = data2 > 0 ? true : false
            return convert(Float64, op.func(data1, round(data2)))
        end

        # execute non-protected instruction
        op.func(data1, data2)
    catch e
        println("$(data1) $(op.func) $(data2)")
        println("Error in executing binary operator: $(e)")
        return C_UNDEF
    end
end


"""
Checks if the given function is a supported unary or binary operator.
"""
function supported_operator(f::Function)
    if f in UNARY_FUNCTIONS
        return 1
    elseif f in BINARY_FUNCTIONS
        return 2
    else
        return 0
    end
end


"""
Returns true if the operator is a branch operator, false otherwise.
"""
function is_branch_operator(op::Operator)
    return op.func in BRANCH_FUNCTIONS
end

end
