module LGPUtils

using LGPOperator, LGPState, LGPInstruction, LGPIndividual, LGPFunctions
export deserialize_individual, expr, run_all, run_multiple_individuals
export sensitivity_and_specificity, sensitivity_and_specificity_multiple, sensitivity_specificity_directory
export compute_roc_data_directory

"""
Reads a serialized individual and its associated State from the given files. Returns the individual
as well as the state instance.
"""
function deserialize_individual(ind_filename::AbstractString, state_filename::AbstractString)
    ind = open(deserialize, ind_filename, "r")
    state = open(deserialize, state_filename, "r")
    return ind, state
end


"""
Computes the true and false positive rates for the given invidual on the given dataset, and saves them in a
CSV file.
"""
function compute_roc_data(ind_filename::AbstractString,
                          state_filename::AbstractString,
                          dataset_filename::AbstractString,
                          output_filename_prefix::AbstractString)

    ind, state = deserialize_individual(ind_filename, state_filename)
    find_effective_program!(ind, state)

    # load dataset
    dataset = readcsv(dataset_filename)
    # remove first line containing column names
    dataset = Array{Float64}(dataset[2:end,:])
    # transpose dataset for accessing by column
    dataset = transpose(dataset)

    thresholds = [x for x in 0.0:0.01:1.0]
    npoints = length(thresholds)
    tpr = Array{Float64}(npoints)
    fpr = Array{Float64}(npoints)
    N = size(dataset, 2)

    for i=1:npoints
        TP = FP = TN = FN = 0

        if length(ind.effective_program) > 0
            desired = dataset[state.params["OUTPUT_COLUMN"], 1:N]
            predicted = Array{Float64}(N)
            for j = 1:N
                @inbounds state.registers[state.input_idx] = dataset[state.params["INPUT_COLUMNS"], j]
                @inbounds state.registers[state.calc_idx] = ones(length(state.calc_idx))
                # run program and collect results
                run!(ind, state.registers)
                predicted[j] = abs(logistic(state.registers[ind.output]) - desired[j]) < thresholds[i] ? 0 : 1

                desired[j] == 1.0 && predicted[j] == 1.0 && (TP += 1)
                desired[j] == 1.0 && predicted[j] == 0.0 && (FN += 1)
                desired[j] == 0.0 && predicted[j] == 1.0 && (FP += 1)
                desired[j] == 0.0 && predicted[j] == 0.0 && (TN += 1)
            end
        end

        tpr[i] = TP/(TP + FN)
        fpr[i] = FP/(TN + FP)
    end

    fit = fitness!(ind, dataset, state)
    output_filename = @sprintf "%s_%.0f_roc.csv" output_filename_prefix 100*fit
    f = open(output_filename, "w")
    write(f, "TPR,FPR\n")
    for i=1:npoints
        write(f, "$(tpr[i]),$(fpr[i])\n")
    end
    close(f)

    return tpr, fpr
end


"""
Finds all serialized individual/state combinations in the subdirectories of the given directory, and returns
arrays containing their relative paths and filenames.
"""
function find_individuals(directory_path::AbstractString)
    subdirectories = []
    for x in readdir(directory_path)
        subdir = joinpath(directory_path, x)
        if isdir(subdir)
            push!(subdirectories, subdir)
        end
    end

    ind_files = []
    state_files = []
    for dir in subdirectories
        files = readdir(dir)
        ind_paths = [joinpath(dir, y) for y in filter(x -> ismatch(r"\s*_best.jls", x), files)]
        ind_files = cat(1, ind_files, ind_paths)
        state_paths = [joinpath(dir, y) for y in filter(x -> ismatch(r"\s*_state.jls", x), files)]
        state_files = cat(1, state_files, state_paths)
    end

    if length(ind_files) != length(state_files)
        error("There are $(length(ind_files)) serialized individuals and $(length(state_files)) serialized states!")
    end

    return ind_files, state_files
end


"""
Calls compute_roc_data() for all serialized individual/state combinations found in subdirectories of the given
directory (not including the directory itself), creating files roc_i.csv, where i runs from 1 to the number of
individuals found.
"""
function compute_roc_data_directory(directory_path::AbstractString, dataset_filename::AbstractString)
    ind_files, state_files = find_individuals(directory_path)

    for i=1:length(ind_files)
        println("Computing ROC data for individual $(ind_files[i])...")
        compute_roc_data(ind_files[i], state_files[i], dataset_filename,
                         "$(dirname(ind_files[i]))/$(basename(ind_files[i])[1:end-9])")
    end
end


"""
Deserializes an Individual instance from the given file and state, and computes the fitness of the
individual using all rows of the given dataset and the given fitness function.
"""
function run_all(ind_filename::AbstractString, state_filename::AbstractString,
    dataset_filename::AbstractString, fitness_function::Function)

    ind, state = deserialize_individual(ind_filename, state_filename)

    # load dataset
    dataset = readcsv(dataset_filename)
    # remove first line containing column names
    dataset = Array{Float64}(dataset[2:end,:])
    # transpose dataset for accessing by column
    dataset = transpose(dataset)

    # set fitness function
    state.params["FITNESS_FUNC"] = fitness_function

    # compute and return fitness
    return fitness!(ind, dataset, state)
end


"""
Deserializes an Individual instance from the given file and state, and computes the sensitivity and
specificity of the individual using all rows of the given dataset.
"""
function sensitivity_and_specificity(ind_filename::AbstractString,
                                     state_filename::AbstractString,
                                     dataset_filename::AbstractString)

    ind, state = deserialize_individual(ind_filename, state_filename)

    # load dataset
    dataset = readcsv(dataset_filename)
    # remove first line containing column names
    dataset = Array{Float64}(dataset[2:end,:])
    # transpose dataset for accessing by column
    dataset = transpose(dataset)

    # compute and return sensitivity and specificity
    return LGPFunctions.sensitivity_and_specificity(ind, dataset, state)
end


"""
Computes and returns the sensitivity and specificity for all of the given serialized individuals
and their corresponding states. All values will be calculated on the same dataset, whose filename is
given.
"""
function sensitivity_and_specificity_multiple(ind_filenames::Array{Any},
                                              state_filenames::Array{Any},
                                              dataset_filename::AbstractString)

    results = Array{Tuple{Float64,Float64}}(length(ind_filenames))

    for i=1:length(ind_filenames)
        results[i] = sensitivity_and_specificity(ind_filenames[i], state_filenames[i],
                                                 dataset_filename)
    end

    return results
end


"""
Computes and returns the sensitivity and specificity for all of the deserialized
individuals found in subdirectories of the given directory (not including the
directory itself), using the given dataset.
"""
function sensitivity_specificity_directory(directory_path::AbstractString,
                                           dataset_filename::AbstractString)

    ind_files, state_files = find_individuals(directory_path)
    values = sensitivity_and_specificity_multiple(ind_files, state_files, dataset_filename)

    sensitivity = Dict{AbstractString,Float64}()
    specificity = Dict{AbstractString,Float64}()
    for i=1:length(ind_files)
        sensitivity[ind_files[i]] = values[i][1]
        specificity[ind_files[i]] = values[i][2]
    end

    # how to print the results in sorted order:
    # for key in sort(collect(keys(dict)))
    #     println("$(key) => $(dict[key])")
    # end

    # how to find avg, min, max of the returned results:
    sens_vals = [x[2] for x in sensitivity]
    sens_min, sens_max = extrema(sens_vals)
    sens_avg = mean(sens_vals)
    println("sensitivity (avg, min, max):")
    println(sens_avg)
    println(sens_min)
    println(sens_max)

    spec_vals = [x[2] for x in specificity]
    spec_min, spec_max = extrema(spec_vals)
    spec_avg = mean(spec_vals)
    println("specificity (avg, min, max):")
    println(spec_avg)
    println(spec_min)
    println(spec_max)

    return sensitivity, specificity
end


"""
Runs run_all for all the individuals contained in the arrays of serialized individual filenames and
the corresponding serialized state filenames. The same dataset and fitness function are used for all
individuals.
Returns an array containing the return values of run_all.
"""
function run_multiple_individuals(ind_filenames::Array{Any},
                                  state_filenames::Array{Any},
                                  dataset_filename::AbstractString,
                                  fitness_function::Function)

    results = Array{Float64}(length(ind_filenames))

    for i=1:length(ind_filenames)
        results[i] = run_all(ind_filenames[i], state_filenames[i], dataset_filename,
                             fitness_function)  
    end

    return results # fitness values for each individual in ind_files
end


"""
Computes the fitness for all individuals find in subdirectories of the given directory, using the given
dataset and fitness function.
"""
function run_all_directory(directory_path::AbstractString,
                           dataset_filename::AbstractString,
                           fitness_function::Function)

    ind_files, state_files = find_individuals(directory_path)
    values = run_multiple_individuals(ind_files, state_files, dataset_filename, fitness_function)
    fitness_vals = Array{Tuple{AbstractString, Float64}}(length(ind_files))
    for i=1:length(ind_files)
        fitness_vals[i] = (ind_files[i], values[i])
    end

    # sort by increasing fitness (better first)
    fitness_vals = sort(fitness_vals, by=x -> x[2])
    for i=1:10
        println("$(fitness_vals[i][1]),$(fitness_vals[i][2])")
    end

    return fitness_vals
end


"""
Finds the n best individuals in the subdirectories of the given search directory, and places their serialized
individual and state files to the given destination directory (which will be created, if it doesn't yet
exist). Also runs compute_roc_data() for each of these individuals on the same dataset.
"""
function collect_best_individuals(search_directory::AbstractString,
                                  destination_directory::AbstractString,
                                  dataset_filename::AbstractString,
                                  fitness_function::Function,
                                  n::Int64)

    ind_files, state_files = find_individuals(search_directory)
    fitness_values = run_multiple_individuals(ind_files, state_files, dataset_filename, fitness_function)
    zipped = collect(zip(ind_files, state_files, fitness_values))
    sorted = sort(zipped, by=x->x[3])

    # create the destination directory if it doesn't exist yet
    if !isdir(destination_directory)
        mkdir(destination_directory)
    end

    for x in sorted[1:n]
        path_parts = splitdir(x[1])
        subdir = splitdir(path_parts[1])[end]
        file_prefix = joinpath(destination_directory, subdir)
        fold_label = split(path_parts[end], "_")[1]
        cp(x[1], join([file_prefix, basename(x[1])], "_"))
        cp(x[2], join([file_prefix, basename(x[2])], "_"))
        compute_roc_data(x[1], x[2], dataset_filename, join([file_prefix, fold_label, round(x[3], 2), "roc.csv"], "_"))
    end
end

"""
Converts the individual's effective program into a mathematical expression.
"""
function expr(ind::Individual, state::State)
    # assume that the last instruction of the effective program has the output register as its
    # destination register
    instr = ind.effective_program[end]
    len = length(ind.effective_program)
    expression = expand_register(instr.out, ind.effective_program, len, state)
    return "r$(instr.out) = $(expression)"
end


"""
Expands the definition of the given register using the given program starting from the given position
upwards. Returns a string representation of the register's definition.
"""
function expand_register(reg::Int64, program::Array{Instruction}, pos::Int64, state::State)
    is_const(reg, state) && return @sprintf "%.3f" state.registers[reg]

    pos < 1 && return "r$(reg)" # terminate recursion

    # search for register in the program above it
    i = pos
    while i >= 1
        if program[i].out == reg
            in1 = expand_register(program[i].in1, program, i - 1, state)
            if isa(program[i], BinaryInstruction)
                in2 = expand_register(program[i].in2, program, i - 1, state)
                if pos == length(program)
                    return "$(in1) $(program[i].op.func) $(in2)"
                else
                    return "($(in1) $(program[i].op.func) $(in2))"
                end
            else
                return "$(program[i].op.func)($(in1))"
            end
        end
        i -= 1
    end

    # reached first instruction without finding an expansion for reg. don't return reg.data,
    # since contents of calculation registers may have changed from when the individual's
    # fitness was last computed
    return "r$(reg)"
end


end # module
