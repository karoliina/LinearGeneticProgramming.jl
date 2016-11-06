module LGPState
using LGPOperator
import Base.show
export State, is_input, is_calc, is_const, init_state, sample_datasets

"Collection of the common elements of a LGP run."
type State
    registers::Array{Float64} # input, calculation and constant registers
    input_idx::Array{Int64} # indices of input registers
    calc_idx::Array{Int64} # indices of input registers
    const_idx::Array{Int64} # indices of input registers
    operators::Array{Operator}
    params_filename::AbstractString
    run_label::AbstractString
    batch_label::AbstractString
    dataset::Array{Any} # Any because first row can include column names
    params::Dict{AbstractString, Any}
end


function is_input(register::Int64, state)
    return register in state.input_idx
end


function is_calc(register::Int64, state)
    return register in state.calc_idx
end


function is_const(register::Int64, state)
    return register in state.const_idx
end


function show(io::IO, state::State)
    print(io, "LGP State initialized from $(state.params_filename) with label $(state.run_label)")
end


"""
Initializes the program state by reading parameters from the given file and constructing
the operators, registers and the params dictionary accordingly. The given filename must be
a Julia file that defines a {AbstractString, Any} dictionary.
"""
function init_state(params_filename::AbstractString, run_label::AbstractString="",
                    batch_label::AbstractString="")
    # read params dict from the given file
    params = include(params_filename)

    # seed the random number generator
    srand(params["RANDOM_SEED"])

    # initialize registers
    input_idx = [1:length(params["INPUT_COLUMNS"]);]
    calc_idx = [input_idx[end] + 1:input_idx[end] + params["NUM_CALC_REGISTERS"];]
    const_idx = [calc_idx[end] + 1:calc_idx[end] + length(params["CONSTANTS"]);]
    registers = Array{Float64}(length(input_idx) + length(calc_idx) + length(const_idx))
    registers[const_idx] = params["CONSTANTS"]

    # create operators
    operators = Operator[]
    for op in params["OPERATORS"]
        op_type = supported_operator(op)
        if op_type == 1
            push!(operators, UnaryOperator(op))
        elseif op_type == 2
            push!(operators, BinaryOperator(op))
        else
            error("Operator $op is not supported!")
        end
    end

    # load dataset
    dataset = readcsv(params["DATASET"])

    return State(registers, input_idx, calc_idx, const_idx, operators, params_filename, run_label,
                 batch_label, dataset, params)
end


"""
Calls the appropriate function for sampling the training, validation and testing datasets from the dataset.
Returns arrays containing the training, validation and testing set constructed from the whole dataset. If the
USE_VALIDATION_DATASET parameter is false, only returns arrays of training and testing sets.
"""
function sample_datasets(state::State)
    if sort(unique(state.dataset[2:end,state.params["OUTPUT_COLUMN"]])) == [0, 1]
        return sample_datasets_classification(state)
    else
        return sample_datasets_random(state)
    end
end


"""
Samples datasets for binary classification tasks. The "sampling" is completely non-random, so the state's
random seed has no effect on the end results.

Each of the returned dataset is transposed so that the features appear as rows and the samples appear as
columns, enabling fast, column-wise accessing of each sample.

As this method uses the values of the output column to divide the dataset into cases and controls, it only
works for binary classification datasets where the output column consists of 0's and 1's.
"""
function sample_datasets_classification(state::State)
    # arrays of training, validation and testing datasets, containing as many elements as there are folds
    nfolds = state.params["FOLDS"]
    use_validation = state.params["USE_VALIDATION_DATASET"]

    training = Array{Array{Float64,2}}(nfolds)
    testing = Array{Array{Float64,2}}(nfolds)

    if use_validation
        validation = Array{Array{Float64,2}}(nfolds)
    end

    # construct folds from the data file
    column_names = state.dataset[1,:]
    dataset = Array{Float64,2}(state.dataset[2:end,:])
    out = state.params["OUTPUT_COLUMN"]

    # calculate the number of cases and controls in order to find out the partition size
    # sort the rows of the dataset by the output column values, so that cases (output column = 1)
    # are before controls (output column = 0)
    dataset = sortrows(dataset, by=x->1-x[out])
    ncases = 0
    ncontrols = 0
    for i=1:size(dataset, 1)
        ncases += dataset[i,out] == 1.0 ? 1 : 0
        ncontrols += dataset[i,out] == 0.0 ? 1 : 0
    end

    half_psize = convert(Int64, min(floor(ncases / nfolds), floor(ncontrols / nfolds)))
    cases_start = 1
    controls_start = indmin(dataset[:,out])

    # construct the nfolds different partitions. for partition 1, use the half_psize first cases
    # (rows 1:half_psize) and the half_psize first controls (rows
    # controls_start:(controls_start+half_psize-1)), etc.
    partitions = Array{Array{Float64}}(nfolds)
    for fold=1:nfolds
        partitions[fold] = cat(1, dataset[cases_start:(cases_start + half_psize - 1),:],
                               dataset[controls_start:(controls_start + half_psize - 1),:])
        cases_start += half_psize
        controls_start += half_psize
    end

    # copy partitions into training/validation/testing arrays in the correct order
    for i=1:nfolds
        testing[i] = transpose(partitions[i])
        if use_validation
            validation[i] = transpose(partitions[i % nfolds + 1])
            # training set is the largest one
            training_set = partitions[(i+1) % nfolds + 1]
            for j=1:nfolds-3
                training_set = cat(1, training_set, partitions[(i+j+1) % nfolds + 1])
            end
            training[i] = transpose(training_set)
        else
            # training set consists of all the remaining partitions
            training_set = partitions[i % nfolds + 1]
            for j=1:nfolds-2
                training_set = cat(1, training_set, partitions[(i+j) % nfolds + 1])
            end
            training[i] = transpose(training_set)
        end
    end

    # end

    if use_validation
        return training, validation, testing
    else
        return training, testing
    end
end


"""
Samples datasets by randomly shuffling the rows of the dataset (unless the number of folds is 1) and dividing
the shuffled dataset into N_folds parts.
"""
function sample_datasets_random(state::State)
    # arrays of training, validation and testing datasets, containing as many elements as there are folds
    nfolds = state.params["FOLDS"]
    use_validation = state.params["USE_VALIDATION_DATASET"]

    training = Array{Array{Float64,2}}(nfolds)
    testing = Array{Array{Float64,2}}(nfolds)

    if use_validation
        validation = Array{Array{Float64,2}}(nfolds)
    end

    # construct folds from the data file
    column_names = state.dataset[1,:]
    dataset = Array{Float64,2}(state.dataset[2:end,:])

    if nfolds == 1
        training[1] = transpose(dataset)
        testing[1] = transpose(dataset)
        if use_validation
            validation[1] = transpose(dataset)
            return training, validation, testing
        else
            return training, testing
        end
    else
        println("not implemented yet!") # TODO
    end
end

end # module
