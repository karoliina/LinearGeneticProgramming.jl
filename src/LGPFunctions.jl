module LGPFunctions

using Lumberjack
using StatsBase # for sample
using DataFrames
using LGPOperator, LGPInstruction, LGPIndividual, LGPState

export get_output_values, fitness!, logistic, SE, RMSE, MCE, MSE_and_MCE, mean_SE, MCE_interval
export cross!, mut!, effmut2!
export select_parents!, tournament_selection!, lexicase_selection!, fitness_proportional_selection!
export tournament, select_survivors
export neutreffmut!, micromut!, effmicromut!, insert_random_instruction!
export best_individual, compute_statistics!, sensitivity_and_specificity, insert_random_individuals!

# === FITNESS FUNCTIONS ===

"""
Computes and returns the output values for the given individual on each of the given set of test cases. For
classification problems, the outputs are not mapped to the discrete classes in this function.
"""
function get_output_values(ind::Individual, dataset::Array{Float64,2}, state::State)
    N = size(dataset, 2)
    outputs = Array{Float64}(N)

    find_effective_program!(ind, state)

    if length(ind.effective_program) == 0
        outputs[:] = UNDEF_FITNESS
    else
        for i = 1:N
            # initialize registers
            @inbounds state.registers[state.input_idx] = dataset[state.params["INPUT_COLUMNS"], i]
            @inbounds state.registers[state.calc_idx] = ones(length(state.calc_idx))
            # run program and collect results
            run!(ind, state.registers)
            @inbounds outputs[i] = state.registers[ind.output]
        end
    end

    return outputs
end


"""
Computes the fitness by comparing the output values stored in the individual's output array to the dataset's
output values.
"""
function fitness_from_outputs!(ind::Individual, dataset::Array{Float64}, state::State, is_validation::Bool=false)
    func = state.params["FITNESS_FUNC"]
    N = size(dataset, 2)

    if is_validation
        fit = func(ind.validation_outputs, dataset[state.params["OUTPUT_COLUMN"], 1:N], N, state)
        ind.validation_fitness = fit
    else
        fit = func(ind.outputs, dataset[state.params["OUTPUT_COLUMN"], 1:N], N, state)
        ind.fitness = fit
    end

    return fit
end


"""
Computes and returns the fitness value of the given individual on the given dataset, using
the given fitness function. Also saves the fitness value in the individual's `fitness` or
`validation_fitness` attribute, and the output values on each of the dataset's test cases as the `outputs` or
`validation_outputs` attribute.

The fitness function specified in the state must take the following parameters:
    predicted::Array{Float64}, desired::Array{Float64}, N::Int64, ind::Individual, state::State
"""
function fitness!(ind::Individual, dataset::Array{Float64,2}, state::State, is_validation::Bool=false)
    if is_validation
        ind.validation_outputs = get_output_values(ind, dataset, state)
    else
        ind.outputs = get_output_values(ind, dataset, state)
    end
    return fitness_from_outputs!(ind, dataset, state, is_validation)
end


"The logistic function."
logistic(x::Float64) = 1/(1 + exp(-x))


"""
The fitness function SE (sum of errors).
"""
function SE(predicted::Array{Float64}, desired::Array{Float64}, N)
    result = 0
    for i=1:N
        @inbounds val = abs(desired[i] - predicted[i])
        if isinf(result + val)
            return UNDEF_FITNESS
        end
        result += val
    end
    return result
end

SE(predicted::Array{Float64}, desired::Array{Float64}, N::Int64, state::State) = SE(predicted, desired, N)
SE(predicted::Float64, desired::Float64) = abs(desired - predicted)


"""
The fitness function mean_SE (mean of the sum of errors).
"""
function mean_SE(predicted::Array{Float64}, desired::Array{Float64}, N)
    return SE(predicted, desired, N)/N
end

mean_SE(predicted::Array{Float64}, desired::Array{Float64}, N::Int64, state::State) = mean_SE(predicted, desired, N)
mean_SE(predicted::Float64, desired::Float64) = SE(predicted, desired)


"""
The fitness function RMSE. Given the predicted and desired output values and the sample size, returns the root
mean square error.
"""
RMSE(predicted::Array{Float64}, desired::Array{Float64}, N::Int64, state::State) = sqrt(MSE(predicted, desired, N))
RMSE(predicted::Float64, desired::Float64) = sqrt(MSE(predicted, desired))


"""
The fitness function MSE (mean square error).
"""
function MSE(predicted::Array{Float64}, desired::Array{Float64}, N)
    SSE = 0
    for i = 1:N
        # @printf "Predicted = %5.2f, desired = %5.2f\n" predicted[i] desired[i]
        @inbounds val = (desired[i] - predicted[i])^2
        if isinf(SSE + val)
            return UNDEF_FITNESS
        end
        SSE += val
    end

    return SSE / N
end

MSE(predicted::Array{Float64}, desired::Array{Float64}, N::Int64, state::State) = MSE(predicted, desired, N)
MSE(predicted::Float64, desired::Float64) = (desired - predicted)^2


"""
The fitness function MCE (mean classification error) for classification problems. The predicted array must
include discrete values (usually 0's and 1's).
"""
function MCE(predicted::Array{Float64}, desired::Array{Float64}, N::Int64)
    CE = 0
    for i = 1:N
        # @printf "Output = %5.2f, predicted = %5.2f, desired = %5.2f\n" predicted[i] logistic(predicted[i]) desired[i]
        @inbounds CE += abs(logistic(predicted[i]) - desired[i]) < 0.5 ? 0 : 1
    end

    return CE / N
end


MCE(predicted::Array{Float64}, desired::Array{Float64}, N::Int64, state::State) = MCE(predicted, desired, N)
MCE(predicted::Float64, desired::Float64, N::Int64, state::State) = MCE(predicted, desired)

function MCE(predicted::Float64, desired::Float64)
    return abs(logistic(predicted) - desired) < 0.5 ? 0 : 1
end


"""
The fitness function MCE_interval (mean classification error with interval classification) for classification problems. The predicted array must
include discrete values (usually 0's and 1's).
"""
function MCE_interval(predicted::Array{Float64}, desired::Array{Float64}, N::Int64)
    CE = 0
    for i = 1:N
        # @printf "Output = %5.2f, predicted = %5.2f, desired = %5.2f\n" predicted[i] logistic(predicted[i]) desired[i]
        @inbounds CE += abs(predicted[i] - desired[i]) < 0.5 ? 0 : 1
    end

    return CE / N
end


MCE_interval(predicted::Array{Float64}, desired::Array{Float64}, N::Int64, state::State) = MCE_interval(predicted, desired, N)
MCE_interval(predicted::Float64, desired::Float64, N::Int64, state::State) = MCE_interval(predicted, desired)
function MCE_interval(predicted::Float64, desired::Float64)
    return abs(predicted - desired) < 0.5 ? 0 : 1
end


"""
The fitness function MSE + w*MCE for classification problems, where w is a parameter.
"""
function MSE_and_MCE(predicted::Array{Float64}, desired::Array{Float64}, N::Int64, state::State)
    mean_square = MSE(predicted, desired, N,)
    mean_classification = MCE(predicted, desired, N)
    return mean_square + state.params["CLASSIFICATION_ERROR_WEIGHT"]*mean_classification
end


function MSE_and_MCE(predicted::Float64, desired::Float64, N::Int64, state::State)
    return MSE(predicted, desired) + state.params["CLASSIFICATION_ERROR_WEIGHT"]*MCE(predicted, desired)
end

# === PARENT SELECTION FUNCTIONS ===

"""
Select parents from the population, returning the parents (or an empty array, if they are kept in the
population) and copies of them (= children).
Will only modify the population array when using certain selection methods.
"""
function select_parents!(population::Array{Individual,1}, state::State, dataset)
    selection_func = state.params["PARENT_SELECTION_FUNC"]
    if selection_func == lexicase_selection!
        parents, children = selection_func(population, state, dataset)
    else
        parents, children = selection_func(population, state)
    end
    return parents, children
end


"""
Performs tournament selection on the population according to the parameters in the state, and returns the
new sets of parents and children (where the children are new copies of the selected parents).
If replacement is used in the tournaments, the returned set of parents is an empty array, since they stay in
the population.
"""
function tournament_selection!(population::Array{Individual}, state::State)
    nparents = state.params["NUM_PARENTS"]
    replacement = state.params["REPLACEMENT_PARENTS"]
    parents = replacement ? Individual[] : Array{Individual}(nparents)
    children = Array{Individual}(nparents)

    for i=1:nparents
        winner, idx = tournament(population, state.params["PARENTS_TOURNAMENT_SIZE"], replacement)
        children[i] = clone(winner, state)

        # the winner gets to stay in the population for the next tournament only when
        # replacement is being used
        if !replacement
            splice!(population, idx)
            parents[i] = winner
        end
    end

    return parents, children
end


"""
Performs lexicase selection on the population according to the parameters in the state, and returns an empty
array of parents as well as an array of copies of the selected parents (= children). The parents are not
removed from the population.
"""
function lexicase_selection!(population::Array{Individual}, state::State, dataset::Array{Float64,2})
    nparents = state.params["NUM_PARENTS"]
    parents = Individual[] # empty array for compatibility with other selection methods
    children = Array{Individual}(nparents)
    popsize = length(population)
    N = size(dataset, 2)
    func = state.params["FITNESS_FUNC"]

    # find fitness values of each individual in the population on each case
    # rows = individuals, columns = cases
    fitness_values = Array{Float64,2}(popsize, N)
    for i=1:popsize
        for j=1:N
            fitness_values[i,j] = func(population[i].outputs[j],
                                       dataset[state.params["OUTPUT_COLUMN"],j])
        end
    end

    for p=1:nparents
        # set candidates to be the entire population
        candidate_indices = 1:popsize
        # set cases to be a list of all of the test cases in random order
        case_indices = sample(1:N, N, replace=false)

        while true
            # set candidates to be the subset of the current candidates that have exactly the best performance of
            # any individual currently in candidates for the first case in cases
            vals = view(fitness_values, candidate_indices, case_indices)
            best_value = findmin(vals[:,1])[1]
            candidate_indices = candidate_indices[find(x -> x == best_value, vals[:,1])]

            # if candidates contains just a single individual, select it as a parent
            if length(candidate_indices) == 1
                children[p] = clone(population[candidate_indices[1]], state)
                break
            # if cases contains just a single test case, then select a random individual from candidates as a
            # parent
            elseif length(case_indices) == 1
                parent_idx = sample(candidate_indices)
                children[p] = clone(population[parent_idx], state)
                break
            # otherwise remove the first case from cases and continue
            else
                splice!(case_indices, 1)
            end
        end
    end

    return parents, children
end

"""
Performs fitness-proportional selection on the population according to the parameters in the state, and returns
new sets of parents and children (where the children are new copies of the selected parents).
"""
function fitness_proportional_selection!(population::Array{Individual}, state::State)
    nparents = state.params["NUM_PARENTS"]
    parents = Array{Individual}(nparents)
    children = Array{Individual}(nparents)

    sort!(population, by = x -> x.fitness)

    for i=1:nparents
        parents[i] = population[1]
        children[i] = clone(population[1], state)
        splice!(population, 1)
    end

    # shuffle children so that crossover partners get assigned randomly
    shuffle!(children)

    return parents, children
end


"""
Selects and returns an individual from the given population using fitness-based tournament selection.
Also returns the winner's index in the population.
"""
function tournament(population::Array{Individual}, tournament_size::Int64, replacement::Bool)
    competitors = sample(population, tournament_size, replace=replacement)
    fitness_values = [comp.fitness for comp in competitors]
    idx = indmin(fitness_values)
    return competitors[idx], idx
end


# === SURVIVOR SELECTION FUNCTIONS ===

"""
Select survivors from the currently oversized population, returning the new population of length POPULATION_SIZE.
"""
function select_survivors(population, state)
    popsize = state.params["POPULATION_SIZE"]

    if state.params["TOURNAMENT_SURVIVORS"]
        new_population = Array{Individual}(popsize)

        for i=1:popsize
            winner, idx = tournament(population, state.params["SURVIVORS_TOURNAMENT_SIZE"],
                state.params["REPLACEMENT_SURVIVORS"])

            # the winner gets to stay in the population for the next tournament only when
            # replacement is being used
            if state.params["REPLACEMENT_SURVIVORS"]
                new_population[i] = clone(winner, state)
            else
                splice!(population, idx)
                new_population[i] = winner
            end
        end

        return new_population
    else
        # keep the POPULATION_SIZE best individuals
        sort!(population, by = x -> x.fitness)
        return population[1:popsize]
    end
end


# === MUTATION FUNCTIONS ===

"""
Linear crossover operator for two individuals. Algorithm 5.1 from Brameier, Banzhaf: Linear Genetic
Programming.
"""
function cross!(ind1::Individual, ind2::Individual, state::State)
    # order programs by length so that ind1 is the shorter one
    if length(ind1.program) < length(ind2.program)
        tmp = ind1
        ind1 = ind2
        ind2 = tmp
    end

    # crossover points
    i1 = rand(1:length(ind1.program))
    i2 = rand(1:length(ind2.program))

    # select crossover points again until they are close enough to each other
    i = 0
    while abs(i1 - i2) > min(length(ind1.program) - 1, state.params["MAX_CROSSOVER_POINT_DISTANCE"]) && i < 10
        i1 = rand(1:length(ind1.program))
        i2 = rand(1:length(ind2.program))
        i += 1
    end
    if i == 10
        return
    end

    # lengths of segments to swap
    s1len = s2len = 0
    s1len = rand(1:min(length(ind1.program) - i1 + 1, state.params["MAX_SEGMENT_LENGTH"]))
    s2len = rand(1:min(length(ind2.program) - i2 + 1, state.params["MAX_SEGMENT_LENGTH"]))

    # segments to swap
    s1 = ind1.program[i1:(i1 + s1len - 1)]
    s2 = ind2.program[i2:(i2 + s2len - 1)]

    # select segments to swap again until they are short enough and s1 is no longer than s2
    i = 0
    while (abs(length(s1) - length(s2)) > state.params["MAX_SEGMENT_LENGTH_DIFF"] ||
        length(s1) > length(s2)) && i < 10

        s1len = rand(1:min(length(ind1.program) - i1 + 1, state.params["MAX_SEGMENT_LENGTH"]))
        s2len = rand(1:min(length(ind2.program) - i2 + 1, state.params["MAX_SEGMENT_LENGTH"]))
        s1 = ind1.program[i1:(i1 + s1len - 1)]
        s2 = ind2.program[i2:(i2 + s2len - 1)]
        i += 1
    end

    # assure s1 is no longer than s2
    if i == 10 || length(s1) > length(s2)
        return
    end

    if length(ind2.program) - (length(s2) - length(s1)) < state.params["MIN_LENGTH"] ||
        length(ind1.program) + (length(s2) - length(s1)) > state.params["MAX_LENGTH"]

        # set the lengths of segments to be equal (if possible)
        if rand() < 0.5
            s1len = s2len
        else
            s2len = s1len
        end
        if i1 + s1len > length(ind1.program)
            s1len = s2len = length(ind1.program) - i1
        end

        s1 = ind1.program[i1:(i1 + s1len - 1)]
        s2 = ind2.program[i2:(i2 + s2len - 1)]
    end

    # exchange segments
    deleteat!(ind1.program, i1:(i1 + length(s1) - 1))
    deleteat!(ind2.program, i2:(i2 + length(s2) - 1))
    newprogram1 = ind1.program[1:(i1 - 1)]
    newprogram2 = ind2.program[1:(i2 - 1)]
    append!(newprogram1, s2)
    append!(newprogram2, s1)
    append!(newprogram1, ind1.program[(i1 + length(s1)):end])
    append!(newprogram2, ind2.program[(i2 + length(s2)):end])
    ind1.program = newprogram1
    ind2.program = newprogram2

end


"""
Mutate the given individual. Algorithm 6.1 without the restriction of effective insertions
and deletions from Brameier, Banzhaf: Linear Genetic Programming.
"""
function mut!(ind::Individual, state::State)
    for j = 1:state.params["NUM_MUTATION_OPERATIONS"]
        proglen = length(ind.program)

        # type of variation: insertion or deletion
        insertion = rand() < state.params["INSERTION_RATE"] ? true : false

        # mutation point
        i = rand(1:proglen)

        if proglen < state.params["MAX_LENGTH"] && (insertion || proglen == state.params["MIN_LENGTH"])
            # insert random instruction with an effective output register at program position i
            instr = random_instruction(state)
            splice!(ind.program, i:(i-1), [instr])
        elseif proglen > state.params["MIN_LENGTH"] && (!insertion || proglen == state.params["MAX_LENGTH"])
            splice!(ind.program, i)
        end
    end
end


"""
Effectively mutate the given individual. Algorithm 6.1 with effective insertions and deletions
(effmut2) from Brameier, Banzhaf: Linear Genetic Programming. A modification for inserting/deleting
more than one instruction at a time is included.
"""
function effmut2!(ind::Individual, state::State)
    for j = 1:state.params["NUM_MUTATION_OPERATIONS"]
        find_effective_program!(ind, state)
        proglen = length(ind.program)

        # type of variation: insertion or deletion
        insertion = rand() < state.params["INSERTION_RATE"] ? true : false

        if proglen < state.params["MAX_LENGTH"] && (insertion || proglen == state.params["MIN_LENGTH"])
            # mutation point
            i = rand(1:proglen)

            # find effective calculation registers at position i
            eff_calc_registers = find_effective_calc_registers(ind, state, i)

            # insert random instruction with an effective output register at program position i
            instr = random_instruction(state, eff_calc_registers)
            splice!(ind.program, i:(i-1), [instr])

        elseif proglen > state.params["MIN_LENGTH"] && (!insertion || proglen == state.params["MAX_LENGTH"])
            # randomly choose an effective instruction to remove
            if length(ind.effective_program) > 0
                instr = rand(ind.effective_program)
                i = findfirst(ind.program, instr)
                splice!(ind.program, i)
            end
        end
    end
end


"""
Effectively mutate the given individual in a fitness-neutral way. Algorithm 6.3 (neutreffmut) from Brameier,
Banzhaf: Linear Genetic Programming.
"""
function neutreffmut!(ind::Individual, dataset::Array{Float64}, state::State)

    i = 0
    # do not change the original individual in case the mutation was unsuccessful
    orig = clone(ind, state)

    for it = 1:state.params["MAX_NEUTREFFMUT_ITERATIONS"]
        effmut2!(ind, state)
        fitness!(ind, dataset, state)
        if ind.fitness < orig.fitness
            return true
        else
            # neutreffmut failed, restore original properties of the individual
            ind.output = orig.output
            ind.program = orig.program
            ind.effective_program = orig.effective_program
            ind.effective_features = orig.effective_features
            ind.fitness = orig.fitness
            ind.validation_fitness = orig.validation_fitness
            return false
        end
    end
end


"""
Micro mutation algorithm with no restrictions on which instructions can be mutated. Algorithm 6.2 from
Brameier, Banzhaf: Linear Genetic Programming.
"""
function micromut!(ind::Individual, state::State)
    # select instruction to mutate
    instr = rand(ind.program)

    # select mutation type
    r = rand()
    register = r < state.params["REGISTER_MUTATION_RATE"] ? true : false
    operator = state.params["REGISTER_MUTATION_RATE"] <= r <=
        (state.params["REGISTER_MUTATION_RATE"] + state.params["OPERATOR_MUTATION_RATE"]) ? true : false
    constant = register == false && operator == false ? true : false

    input_and_calc = cat(1, state.input_idx, state.calc_idx)

    if register
        if isa(instr, UnaryInstruction)
            reg = rand([instr.out, instr.in1])
        else # binary instruction
            reg = rand([instr.out, instr.in1, instr.in2])
        end
        if reg == instr.out
            instr.out = rand(state.calc_idx)
        else
            if rand() < state.params["CONSTANTS_RATE"]
                reg = rand(state.const_idx)
            else
                reg = rand(input_and_calc)
            end
        end
    elseif operator
        op = rand(state.operators)
        # change instruction type if necessary
        if isa(op, BinaryOperator) && isa(instr, UnaryInstruction)
            if rand() < state.params["CONSTANTS_RATE"]
                in2 = rand(state.const_idx)
            else
                in2 = rand(input_and_calc)
            end
            instr = BinaryInstruction(op, instr.out, instr.in1, in2)
        elseif isa(op, UnaryOperator) && isa(instr, BinaryInstruction)
            instr = UnaryInstruction(op, instr.out, instr.in1)
        else
            instr.op = op
        end
    else # constant
        j = 0
        # randomly select instruction with a constant, max length(ind.program) tries
        while (!is_const(instr.in1, state) && !isa(instr, BinaryInstruction)) ||
            (isa(instr, BinaryInstruction) && !is_const(instr.in1, state) && !is_const(instr.in2, state)) &&
            j < length(ind.program)

            instr = rand(ind.program)
            j += 1
        end

        if j < length(ind.program)
            # to mutate a constant, add a new constant register with the mutated value
            # in order to not change the constant for other instructions
            if is_const(instr.in1, state)
                new_data = state.registers[instr.in1] + randn()*state.params["CONSTANT_MUTATION_SIGMA"]
                push!(state.registers, new_data)
                push!(state.const_idx, length(state.registers))
                instr.in1 = state.const_idx[end]
            else # in2 is constant
                new_data = state.registers[instr.in2] + randn()*state.params["CONSTANT_MUTATION_SIGMA"]
                push!(state.registers, new_data)
                push!(state.const_idx, length(state.registers))
                instr.in2 = state.const_idx[end]
            end
        end
    end
end


"""
Micro mutation algorithm that only operates on effective instructions. Algorithm 6.2 (effective version)
from Brameier, Banzhaf: Linear Genetic Programming.
"""
function effmicromut!(ind::Individual, state::State)
    find_effective_program!(ind, state)

    # if there are no effective instructions, don't do anything
    if length(ind.effective_program) == 0
        return
    end

    # select effective instruction to mutate
    instr = rand(ind.effective_program)
    # select mutation type
    r = rand()

    register = r < state.params["REGISTER_MUTATION_RATE"] ? true : false
    operator = state.params["REGISTER_MUTATION_RATE"] <= r <=
        (state.params["REGISTER_MUTATION_RATE"] + state.params["OPERATOR_MUTATION_RATE"]) ? true : false
    constant = register == false && operator == false ? true : false

    input_and_calc = cat(1, state.input_idx, state.calc_idx)

    if register
        if isa(instr, UnaryInstruction)
            reg = rand([instr.out, instr.in1])
        else # binary instruction
            reg = rand([instr.out, instr.in1, instr.in2])
        end
        if reg == instr.out
            # find effective calculation registers at this program position
            i = findfirst(ind.program, instr)
            eff_calc_registers = find_effective_calc_registers(ind, state, i)
            # select effective register to use as output
            instr.out = length(eff_calc_registers) > 0 ? rand(eff_calc_registers) : instr.out
        else
            if rand() < state.params["CONSTANTS_RATE"]
                reg = rand(state.const_idx)
            else
                reg = rand(input_and_calc)
            end
        end
    elseif operator
        op = rand(state.operators)
        # change instruction type if necessary
        if isa(op, BinaryOperator) && isa(instr, UnaryInstruction)
            if rand() < state.params["CONSTANTS_RATE"]
                in2 = rand(state.const_idx)
            else
                in2 = rand(input_and_calc)
            end
            instr = BinaryInstruction(op, instr.out, instr.in1, in2)
        elseif isa(op, UnaryOperator) && isa(instr, BinaryInstruction)
            instr = UnaryInstruction(op, instr.out, instr.in1)
        else
            instr.op = op
        end
    else # constant
        j = 0
        # randomly select effective instruction with a constant, max length(ind.effective_program) tries
        while ((!is_const(instr.in1, state) && !isa(instr, BinaryInstruction)) ||
            (isa(instr, BinaryInstruction) && !is_const(instr.in1, state) && !is_const(instr.in2, state))) &&
            j < length(ind.effective_program)

            instr = rand(ind.effective_program)
            j += 1
        end

        if j < length(ind.effective_program)
            # to mutate a constant, add a new constant register with the mutated value
            # in order to not change the constant for other instructions
            if is_const(instr.in1, state)
                new_data = state.registers[instr.in1] + randn()*state.params["CONSTANT_MUTATION_SIGMA"]
                push!(state.registers, new_data)
                push!(state.const_idx, length(state.registers))
                instr.in1 = state.const_idx[end]
            else # in2 is constant
                new_data = state.registers[instr.in2] + randn()*state.params["CONSTANT_MUTATION_SIGMA"]
                push!(state.registers, new_data)
                push!(state.const_idx, length(state.registers))
                instr.in2 = state.const_idx[end]
            end
        end
    end
end


"""
Inserts a randomly generated instruction into a random position in the individual's program.
"""
function insert_random_instruction!(ind::Individual, state::State)
    if length(ind.program) == 0
        return
    end
    i = rand(1:length(ind.program))
    if length(ind.program) < state.params["MAX_LENGTH"]
        insert!(ind.program, i, random_instruction(state))
   end
end


# === HELPER FUNCTIONS ===

"Finds and returns the individual with the lowest (validation) fitness value."
function best_individual(population::Array{Individual}, validation::Bool=false)
    best = population[1]
    for ind in population
        if validation
            best = ind.validation_fitness < best.fitness ? ind : best
        else
            best = ind.fitness < best.fitness ? ind : best
        end
    end
    return best
end


"""
Computes and returns the behavioural diversity (percent of distinct output vectors) for the given population.
Assumes that the outputs and validation_outputs arrays are up to date for each individual.
"""
function diversity(population::Array{Individual})
    output_vectors = Set{Array{Float64}}()
    for ind in population
        push!(output_vectors, ind.outputs)
    end
    return length(output_vectors)/length(population)
end


"""
Compute population statistics for the given population and generation, and store them in the
given data frame.
"""
function compute_statistics!(population::Array{Individual}, best::Individual, gen::Int64, run_data::DataFrame)
    popsize = length(population)

    fitness_values = [ind.fitness for ind in population]
    std_dev = std(fitness_values)
    avg = avg_len = avg_eff_len = avg_eff_features = 0.0
    for ind in population
        avg += ind.fitness
        avg_len += length(ind.program)
        avg_eff_len += length(ind.effective_program)
        avg_eff_features += length(ind.effective_features)
    end

    avg /= popsize
    avg_len /= popsize
    avg_eff_len /= popsize
    avg_eff_features /= popsize

    # save statistics
    run_data[:generation][gen] = gen
    run_data[:best_fit][gen] = best.fitness
    run_data[:avg_fit][gen] = avg
    run_data[:avg_len][gen] = avg_len
    run_data[:std_dev][gen] = std_dev
    run_data[:avg_eff_len][gen] = avg_eff_len
    run_data[:avg_eff_feat][gen] = avg_eff_features
    run_data[:diversity][gen] = diversity(population)
end


"Stochastically inserts random individuals into the population."
function insert_random_individuals!(population::Array{Individual}, best::Individual, stddev::Float64,
    training::Array{Float64}, state::State)

    if rand() < state.params["RANDOM_INSERTION_RATE"] || (best.fitness > 0 && stddev <
        state.params["STDDEV_THRESHOLD"])
        for j=1:state.params["NUM_RANDOM_INSERTIONS"]
            ind = state.params["EFFECTIVE_INITIALIZATION"] ?
                effective_random_individual(state) :
                random_individual(state)
            fitness!(ind, training, state)
            push!(population, ind)
        end
    end
end


"""
Computes the sensitivity of the given individual on the given dataset.
"""
function sensitivity_and_specificity(ind::Individual, dataset::Array{Float64}, state::State)

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    correct = 0
    incorrect = 0

    find_effective_program!(ind, state)

    if length(ind.effective_program) == 0
        return 0.0, 0.0
    else
        N = size(dataset, 2)
        actual = dataset[state.params["OUTPUT_COLUMN"], 1:N]
        for i = 1:N
            @inbounds state.registers[state.input_idx] = dataset[state.params["INPUT_COLUMNS"], i]
            @inbounds state.registers[state.calc_idx] = ones(length(state.calc_idx))
            # run program and collect results
            run!(ind, state.registers)
            predicted = round(logistic(state.registers[ind.output]))

            if actual[i] == 1.0 # diseased
                predicted == 1.0 && (TP += 1)
                predicted == 0.0 && (FN += 1)
            else # healthy
                predicted == 1.0 && (FP += 1)
                predicted == 0.0 && (TN += 1)
            end
            predicted == actual[i] && (correct += 1)
            predicted != actual[i] && (incorrect += 1)
        end
    end

    # println("TP: $(TP), FP: $(FP), TN: $(TN), FN: $(FN), total: $(TP+FP+TN+FN)")
    # println("P: $(TP+FP), N: $(TN+FN)")
    # println("correct: $(correct), incorrect: $(incorrect)")
    # println("MCE = $(incorrect / N)")

    return TP + FN == 0.0 ? 0.0 : (TP/(TP + FN)), # sensitivity
           TN + FP == 0.0 ? 0.0 : (TN/(TN + FP))  # specificity
end


end
