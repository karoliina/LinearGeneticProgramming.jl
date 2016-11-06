module LGP

println("Importing dependencies...")
using ArgParse, DataFrames, Formatting, Lumberjack
using LGPOperator, LGPState, LGPInstruction, LGPIndividual, LGPFunctions, LGPUtils, LGPEvolution

# alias Lumberjack to enable lj.info("foo") - using info("foo") causes a warning
const lj = Lumberjack

# configure Lumberjack - set minimum displayed level, enable colorized logs (requires --color=yes)
# levels are debug, info, warn, error
add_truck(LumberjackTruck(STDOUT, "debug", Dict{Any, Any}(:is_colorized => true, :uppercase => true)), "console")

println("Dependencies imported")

function parse_commandline()
    s = ArgParseSettings()::ArgParseSettings

    @add_arg_table s begin
        "params_filename"
            help = "the parameter file name"
            required = true
        "run_label"
            help = "the run label"
            required = false
        "batch_label"
            help = "the batch label"
            required = false
        "--no_print"
            help = "do not print statistics of each generation as program runs"
            action = :store_true
        "--no_save"
            help = "do not save the results of the run"
            action = :store_true
    end

    return parse_args(s)
end


function save_results(r::Int64, best::Individual, best_validation_wrapper::Nullable{Individual},
                      run_data::DataFrame, state::State, last_generation::Int64, run_time::Float64)

    use_validation = state.params["USE_VALIDATION_DATASET"]
    if use_validation && !isnull(best_validation_wrapper)
        best_validation = get(best_validation_wrapper)
    end

    # serialize and write the two best individuals, and write the state, the parameters file
    # and the run statistics into the runs/ directory
    if !isdir("runs")
        mkdir("runs")
    end
    if state.batch_label != ""
        batchdir = "runs/$(state.batch_label)"
    else
        batchdir = "runs"
    end
    if !isdir(batchdir)
        mkdir(batchdir)
    end
    dir = "$(batchdir)/$(state.run_label)"
    if !isdir(dir)
        mkdir(dir)
    end

    open(f -> serialize(f, best), "$(dir)/fold$(r)_best.jls", "w")
    if use_validation
        open(f -> serialize(f, best_validation), "$(dir)/fold$(r)_best_validation.jls", "w")
    end
    open(f -> serialize(f, state), "$(dir)/fold$(r)_state.jls", "w")
    cp("$(state.params_filename)", "$(dir)/fold$(r)_params.jl",
        remove_destination=true)

    generations = state.params["MAX_GENERATIONS"]
    run_data = last_generation <= generations ? run_data[1:generations,:] : run_data
    writetable("$(dir)/fold$(r)_run_data.csv", run_data)

    # write some of the results to the "compilation" results file, if it has been provided
    if haskey(state.params, "RESULTS_FILE")
        results_filename = "$(batchdir)/$(state.run_label)_$(state.params["RESULTS_FILE"])"
        already_exists = isfile(results_filename)
        f = open(results_filename, "a")
        if !already_exists
            print(f, "run label,random seed,fold,best fitness,validation used,run time,")
            println(f, "number of effective features,effective features,output vector")
        end

        fe = FormatExpr("{:s},{:d},{:d},{:f},{:s},{:f},{:d},{:s},{:s}")
        if use_validation
            sorted = sort!(best_validation.effective_features)
            eff_features = join(sorted, " ")
            output_vector = [round(x, 4) for x in best_validation.outputs]
            printfmtln(f, fe, state.run_label, state.params["RANDOM_SEED"], r, best_validation.fitness,
                       "true", run_time, length(best_validation.effective_features), eff_features,
                       join(output_vector, " "))
        else
            sorted = sort!(best.effective_features)
            eff_features = join(sorted, " ")
            output_vector = [round(x, 4) for x in best.outputs]
            printfmtln(f, fe, state.run_label, state.params["RANDOM_SEED"], r, best.fitness,
                       "false", run_time, length(best.effective_features), eff_features,
                       join(output_vector, " "))
        end
        close(f)
    end

    branches_used = any(is_branch_operator, state.operators)

    io = IOBuffer()
    write(io, "RUN #$(r): BEST INDIVIDUAL (TRAINING)\n")
    write(io, "==========================\n")
    show_eff(io, best)
    write(io, "\n")
    show_features(io, best)
    write(io, "\n\n")

    # only compute and print mathematical expression if branches weren't used - the algorithm doesn't work
    # with branch instructions
    if !branches_used
        write(io, "Solution:\n\t$(expr(best, state))\n\n")
    end

    if use_validation
        write(io, "RUN #$(r): BEST INDIVIDUAL (VALIDATION)\n")
        write(io, "==========================\n")
        show_eff(io, best_validation)
        write(io, "\n")
        show_features(io, best)
        write(io, "\n")

        if !branches_used
            write(io, "Solution:\n\t$(expr(best_validation, state))\n")
        end
    end

    str = takebuf_string(io)
    println("\n$(str)")

    f = open("$(dir)/fold$(r)_best_individuals.txt", "w")
    write(f, str)
    close(f)
end


function main()
    tic(); # set timer

    args = parse_commandline()
    # if no run label given, construct one from the current date and time
    if args["run_label"] == nothing
        now = Dates.now()
        run_label = "run_$(Dates.format(now, "yyyy-mm-ddTHH-MM-SS"))"
    else
        run_label = args["run_label"]
    end
    batch_label = args["batch_label"] == nothing ? "" : args["batch_label"]

    # initialize the state here, so that params can be accessed before actual runs start
    println("Initializing state...")
    params_filename = args["params_filename"]
    state = init_state(params_filename, run_label, batch_label)

    # construct the (arrays of) training, validation and testing datasets
    use_validation = state.params["USE_VALIDATION_DATASET"]
    if use_validation
        training, validation, testing = sample_datasets(state)
    else
        training, testing = sample_datasets(state)
    end

    # do as many runs as there are cross-validation folds
    nruns = state.params["FOLDS"]

    # collect the best testing and validation fitness values for each run
    testing_fitness = Array{Float64}(nruns)
    if use_validation
        validation_fitness = Array{Float64}(nruns)
    end

    best = Array{Individual}(nruns)
    best_validation = Array{Individual}(nruns)
    run_data = Array{DataFrame}(nruns)
    state = Array{State}(nruns)
    last_generation = Array{Int64}(nruns)
    run_time = Array{Float64}(nruns)
    rcalls = Array{Future}(nruns)

    nthreads = nprocs()-1
    println("$(nthreads) workers available")
    next_thread = 1
    for r=1:nruns
        println("Starting run $(r)/$(nruns) on worker $(next_thread + 1)")

        if nthreads > 0
            if use_validation
                rcalls[r] = remotecall(perform_run, next_thread + 1, r, params_filename, run_label, batch_label,
                                       args, training, validation, testing)
            else
                rcalls[r] = remotecall(perform_run, next_thread + 1, r, params_filename, run_label, batch_label,
                                       args, training, testing)
            end
            next_thread = (next_thread % nthreads) + 1
        else
            if use_validation
                best[r], best_validation[r], run_data[r], state[r], last_generation[r], run_time[r] =
                    perform_run(r, params_filename, run_label, batch_label, args, training, validation, testing)
                testing_fitness[r] = best[r].fitness
                validation_fitness[r] = best_validation[r].fitness
            else
                best[r], run_data[r], state[r], last_generation[r], run_time[r] =
                    perform_run(r, params_filename, run_label, batch_label, args, training, testing)
                testing_fitness[r] = best[r].fitness
            end
        end
    end

    if nthreads > 0
        for r=1:nruns
            println("Fetching results of run $(r)/$(nruns)")
            if use_validation
                best[r], best_validation[r], run_data[r], state[r], last_generation[r], run_time[r] =
                    fetch(rcalls[r])
                testing_fitness[r] = best[r].fitness
                validation_fitness[r] = best_validation[r].fitness
            else
                best[r], run_data[r], state[r], last_generation[r], run_time[r] = fetch(rcalls[r])
                testing_fitness[r] = best[r].fitness
            end
            println("Fetch done for run $(r)/$(nruns)")
        end
    end


    if !args["no_save"]
        println("Saving results...")
        for r=1:nruns
            if use_validation
                save_results(r, best[r], Nullable{Individual}(best_validation[r]), run_data[r], state[r],
                             last_generation[r], run_time[r])
            else
                save_results(r, best[r], Nullable{Individual}(), run_data[r], state[r], last_generation[r],
                             run_time[r])
            end
        end
    end

    t = toc(); # show elapsed time
    println("Computation took $(Int64(floor(t/60))) minutes $(Int64(round(t % 60))) seconds")

    mean_value = mean(testing_fitness)
    stddev_value = std(testing_fitness)
    @printf "Testing fitness values for each run (avg = %.4f, std dev = %.4f):\n" mean_value stddev_value
    for f in testing_fitness
        @printf "%.4f " f
    end
    println()

    if use_validation
        mean_value = mean(testing_fitness)
        stddev_value = std(testing_fitness)
        @printf "Validation fitness values for each run (avg = %.4f, std dev = %.4f):\n" mean_value stddev_value
        for f in validation_fitness
            @printf "%.4f " f
        end
        println()
    end

end # main function


if !isinteractive()
    main()
end

end # module
