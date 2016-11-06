module LGPIndividual

using Lumberjack
using LGPOperator, LGPInstruction, LGPState
import Base.show
import LGPInstruction.clone # need to import in order to add new method
export Individual, show_eff, show_features, random_individual, effective_random_individual, clone, run!
export find_effective_program!, find_effective_calc_registers, UNDEF_FITNESS

"""
The default, large fitness value for individuals. Will also be used when an overflow in the fitness
computation occurs.
"""
const UNDEF_FITNESS = 1e9


"""
An individual that has an output register, a list of instructions (its program), and its fitness value as
well as some additional properties.
"""
type Individual
    output::Int64 # index of output register in state.registers
    program::Array{Instruction}
    effective_program::Array{Instruction}
    fitness::Float64
    validation_fitness::Float64
    outputs::Array{Float64} # output values on each of the training cases before applying the fitness function
    validation_outputs::Array{Float64} # output values on each of the validation cases before applying the
                                       # fitness function
    effective_features::Array{Int64} # indices of effective input registers in state.registers

    function Individual(output::Int64, program::Array{Instruction}, effective_program::Array{Instruction},
                        fitness::Float64, validation_fitness::Float64, outputs::Array{Float64},
                        validation_outputs::Array{Float64}, effective_features::Array{Int64})
        new(output, program, effective_program, fitness, validation_fitness, outputs, validation_outputs,
            effective_features)
    end
end


"Constructs an Individual without the output value arrays or effective features specified."
function Individual(output::Int64, program::Array{Instruction}, effective_program::Array{Instruction},
                    fitness::Float64, validation_fitness::Float64)
    return Individual(output, program, effective_program, fitness, validation_fitness, Float64[], Float64[],
                      Int64[])
end

"Constructs an Individual without the effective features specified."
function Individual(output::Int64, program::Array{Instruction}, effective_program::Array{Instruction},
                    fitness::Float64, validation_fitness::Float64, outputs::Array{Float64},
                    validation_outputs::Array{Float64})
    return Individual(output, program, effective_program, fitness, validation_fitness, outputs, validation_outputs,
                      Int64[])
end


"Shows a string representation of an individual's training fitness and the length of its program."
function show(io::IO, ind::Individual)
    fit = @sprintf "Individual: Fitness = %.2f, %d instructions" ind.fitness length(ind.program)
    print(io, fit)
end


"Shows a string representation of an individual's fitness and its program."
function show_program(io::IO, ind::Individual)
    fit = @sprintf "Fitness = %.4f, program:\n" ind.fitness
    println(io, fit)
    for instr in ind.program
        println(io, "\t$(instr)")
    end
end


"Shows a string representation of an individual's fitness and its effective program."
function show_eff(io::IO, ind::Individual)
    fit = @sprintf "Fitness = %.4f, effective program:\n" ind.fitness
    println(io, fit)
    for instr in ind.effective_program
        println(io, "\t$(instr)")
    end
end

"Shows a string representation of an individual's fitness and its effective features."
function show_features(io::IO, ind::Individual)
    sorted = sort!(ind.effective_features)
    eff_features = join(sorted, ", ")
    str = @sprintf "%d effective features: %s" length(sorted) eff_features
    print(io, str)
end


"Constructs and returns a random individual."
function random_individual(state::State)
    program = Instruction[]

    # randomly select initial length
    length = rand(state.params["INIT_MIN_LENGTH"]:state.params["INIT_MAX_LENGTH"])

    # randomly construct instructions
    for i=1:length
        branches_used = any(is_branch_operator, state.operators)
        if branches_used && rand() < state.params["BRANCH_INITIALIZATION_RATE"]
            push!(program, random_branch_instruction(state))
        else
            push!(program, random_instruction(state))
        end
    end

    # set output register
    output = state.calc_idx[1]

    return Individual(output, program, Instruction[], UNDEF_FITNESS, UNDEF_FITNESS)
end


"Constructs and returns a random individual with a fully effective program."
function effective_random_individual(state::State)
    # check if branch instructions are used
    branches_used = any(is_branch_operator, state.operators)

    # set output register
    output = state.calc_idx[1]

    program = Instruction[]
    reff = Int64[output]

    # randomly select initial length
    len = rand(state.params["INIT_MIN_LENGTH"]:state.params["INIT_MAX_LENGTH"])

    # construct first instruction
    push!(program, random_instruction(state, reff))

    # construct effective instructions randomly
    for i=2:len
        if !is_branch(program[1])
            # a branch instruction's output register does not exist in reff, don't attempt to delete it
            # if the previously generated instruction was a branch
            deleteat!(reff, findfirst(reff, program[1].out))
        end

        if !in(program[1].in1, reff) && is_calc(program[1].in1, state)
            push!(reff, program[1].in1)
        end
        if isa(program[1], BinaryInstruction) && !in(program[1].in2, reff) && is_calc(program[1].in2, state)
            push!(reff, program[1].in2)
        end
        # if an effective register doesn't exist, use the output register as the output for the new
        # instruction
        if isempty(reff)
            push!(reff, output)
        end
        if branches_used && rand() < state.params["BRANCH_INITIALIZATION_RATE"]
            # branch instructions preceding an effective instruction are always effective, and their
            # output registers don't need to be effective for the instruction to be effective
            unshift!(program, random_branch_instruction(state))
        else
            unshift!(program, random_instruction(state, reff))
        end
    end

    return Individual(output, program, Instruction[], UNDEF_FITNESS, UNDEF_FITNESS)
end


"Creates and returns a shallow copy of the individual."
function clone(ind::Individual, state::State)
    program = Instruction[clone(instr) for instr in ind.program]
    cloned_ind = Individual(ind.output, program, Instruction[], ind.fitness, ind.validation_fitness,
                            ind.outputs, ind.validation_outputs)
    find_effective_program!(cloned_ind, state)
    return cloned_ind
end


"""
Runs the individual's effective program on the data currently contained in the registers. Assumes
that the effective program is up to date (that is, find_effective_program has been run since the
program was last changed).
"""
function run!(ind::Individual, registers::Array{Float64})
    execute_next = true
    for instr in ind.effective_program
        # don't execute this instruction if the preceding instruction was a branch instruction
        # that returned false
        if execute_next
            exec!(instr, registers)
            # determine whether to execute the following instruction
            execute_next = !is_branch(instr) ||
              (is_branch(instr) && registers[instr.out] == 1.0) ? true : false
        else
            # if this instruction was skipped because the preceding instruction was a branch
            # instruction that evaluated to false, and this instruction is also a branch instruction,
            # the following instruction should also be skipped
            execute_next = !is_branch(instr)
        end
    end
    return registers[ind.output]
end


"""
Computes the individual's effective program using the algorithm for structural intron detection
and saves it as ind.effective_program.
"""
function find_effective_program!(ind::Individual, state::State, stop_at::Integer=1)
    # effective registers at current program position
    reff = Int64[ind.output]
    # effective instructions
    ieff = Instruction[]

    for i in length(ind.program):-1:stop_at
        instr = ind.program[i]

        # any branch instruction that precedes an effective instruction has already been marked.
        # if the branch does not precede an effective instruction, it is not effective.
        if is_branch(instr)
            if instr in ieff
                # the input registers of an effective branch instruction haven't been marked as effective yet,
                # mark them now
                if !in(instr.in1, reff) && !is_const(instr.in1, state)
                    push!(reff, instr.in1)
                end
                if !in(instr.in2, reff) && !is_const(instr.in2, state)
                    push!(reff, instr.in2)
                end
            end
            continue
        end

        if instr.out in reff
            # mark instruction as effective
            unshift!(ieff, instr)
            # if the operation directly follows a branch or sequence of branches, mark these
            # instructions as effective. do not mark their input registers yet!
            j = i-1
            branches_marked = false
            while j >= 1 && is_branch(ind.program[j])
                unshift!(ieff, ind.program[j])
                branches_marked = true
                j -= 1
            end
            # otherwise remove output register from reff
            if !branches_marked
                deleteat!(reff, findfirst(reff, instr.out))
            end
            # mark (non-constant) registers as effective
            if !in(instr.in1, reff) && !is_const(instr.in1, state)
                push!(reff, instr.in1)
            end
            if isa(instr, BinaryInstruction) && !in(instr.in2, reff) && !is_const(instr.in2, state)
                push!(reff, instr.in2)
            end
        end
    end

    # if a partial run, no need to find effective features
    if stop_at > 1
        return (ieff, reff)
    end

    # set the individual's effective program and features
    ind.effective_program = ieff
    ind.effective_features = filter(r -> is_input(r, state), reff)
    return (ieff, reff)
end


"""
Finds the effective calculation registers at position stop_at in the individual's program.
Does not modify the attributes of the individual.
"""
function find_effective_calc_registers(ind::Individual, state::State, stop_at::Int64=1)
    # effective registers at current program position
    reff = Int64[ind.output]

    # find effective program
    for i in length(ind.program):-1:stop_at
        instr = ind.program[i]
        if instr.out in reff
            # remove output register from reff
            deleteat!(reff, findfirst(reff, instr.out))
            # mark registers as effective if they are calculation registers
            if !in(instr.in1, reff) && is_calc(instr.in1, state)
                push!(reff, instr.in1)
            end
            if isa(instr, BinaryInstruction) && !in(instr.in2, reff) && is_calc(instr.in2, state)
                push!(reff, instr.in2)
            end
        end
    end

    return reff
end

end
