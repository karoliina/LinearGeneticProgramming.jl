module LGPEvolution

using DataFrames, Formatting, Lumberjack
using LGPIndividual, LGPFunctions, LGPState

export perform_run

# alias Lumberjack to enable lj.info("foo") - using info("foo") causes a warning
const lj = Lumberjack

# whether to clear screen - make this a command line arg?
const clear_screen = false

function perform_run(r::Int64, params_filename::AbstractString, run_label::AbstractString, batch_label::AbstractString,
                     args::Dict{AbstractString,Any}, training::Array{Array{Float64,2}}, testing::Array{Array{Float64,2}})
    return perform_run(r, params_filename, run_label, batch_label, args, training, Array{Array{Float64,2}}(), testing)
end


function perform_run(r::Int64, params_filename::AbstractString, run_label::AbstractString, batch_label::AbstractString,
                     args::Dict{AbstractString,Any}, training::Array{Array{Float64,2}}, validation::Array{Array{Float64,2}},
                     testing::Array{Array{Float64,2}})
    # get elapsed time
    tic();

    # initialize the state at the start of each run, since constant registers may have been mutated
    println("Run $(r): Initializing state...")
    state = init_state(args["params_filename"], run_label, batch_label)

    use_validation = state.params["USE_VALIDATION_DATASET"]
    print_progress = !args["no_print"] && nprocs() < 2

    # re-seed the random number generator at the start of each run
    srand(state.params["RANDOM_SEED"])

    # create initial population
    println("Run $(r): Generating initial population...")
    initfunc = state.params["EFFECTIVE_INITIALIZATION"] ? effective_random_individual : random_individual
    population = Individual[initfunc(state) for i=1:state.params["POPULATION_SIZE"]]

    # store some statistics of each generation in a DataFrame
    generations = state.params["MAX_GENERATIONS"]
    columns = [:generation, :best_fit, :avg_fit, :avg_len, :std_dev, :avg_eff_len, :avg_eff_feat, :diversity]
    run_data = DataFrame(Any[zeros(generations) for i=1:length(columns)], columns)

    # calculate initial fitness values
    popsize = length(population)
    eff_feats = zeros(Int64, popsize)
    println("Run $(r): Computing initial fitness for $(popsize) individuals...")
    for i in eachindex(population)
        fitness!(population[i], training[r], state)
        eff_feats[i] = length(population[i].effective_features)
    end

    lj.info(
        "Effective features: max = $(maximum(eff_feats)), min = $(minimum(eff_feats)), avg = $(mean(eff_feats))")

    # start main loop
    if print_progress
        first_row_format = FormatExpr("{:11s} {:11s} {:11s} {:11s} {:11s} {:11s} {:11s} {:20s}\n")
        format = FormatExpr(" {:>4}/{:<4} {:11.4e} {:11.4e} {:11.4e} {:11.4e} {:11.4e} {:11.4e}  {:20s}\n")
    end
    best = best_individual(population)
    if print_progress
        clear_screen && run(`clear`)
        printfmt(first_row_format, "Generation", "Best fit", "Avg fit", "Std dev", "Len", "Eff len", "Eff feat",
            "Run label")
    else
        println("Run $(r): Evolving individuals...")
    end

    gen = 1
    avg = Inf # temporary value for loop condition
    while gen <= generations

        # compute and print statistics for this generation
        compute_statistics!(population, best, gen, run_data)
        if print_progress
            printfmt(format, gen, generations, best.fitness, run_data[:avg_fit][gen], run_data[:std_dev][gen],
                run_data[:avg_len][gen], run_data[:avg_eff_len][gen], run_data[:avg_eff_feat][gen],
                "$(state.run_label)_fold$(r)")
        end

        # if the solution has been found, end the loop
        if best.fitness <= state.params["TERMINATION_THRESHOLD"]
            break
        end

        # force diversity by inserting random individuals
        insert_random_individuals!(population, best, run_data[:std_dev][gen], training[r], state)

        # select parents
        parents, children = select_parents!(population, state, training[r])

        # modify the children (= copies of the selected parents) with variation operators
        for i=1:2:length(children)
            # cross over this pair of children with a probability
            if rand() < state.params["CROSSOVER_RATE"]
                cross!(children[i], children[i + 1], state)
            end

            # mutate each of the children in this pair with a probability
            for j=i:i+1
                if rand() < state.params["MACRO_MUTATION_RATE"]
                    # macro mutation
                    mutation = state.params["MACRO_MUTATION_OPERATOR"]
                    if mutation == neutreffmut!
                        mutation(children[j], training[r], state)
                    else
                        mutation(children[j], state)
                    end
                elseif rand() < state.params["MICRO_MUTATION_RATE"]
                    # micro mutation
                    mutation = state.params["MICRO_MUTATION_OPERATOR"]
                    mutation(children[j], state)
                end
            end
        end

        # compute fitness values of children and check if any of the children is the new
        # best individual
        for i=1:length(children)
            fitness!(children[i], training[r], state)
            if children[i].fitness < best.fitness
                if print_progress
                    @printf "\tNEW BEST! Old = %.4f, new = %.4f\n" best.fitness children[i].fitness
                    lj.info(sprint(show_features, children[i]))
                end
                best = children[i]
                if use_validation
                    # compute the validation fitness of the new best individual
                    fitness!(best, validation[r], state, true)
                end
            end
        end

        # no need to check if current best individual was mutated - only copies of parents are mutated

        # add the children and the original winners (parents) back to the population
        # the parents array is empty if the parent selection method creates copies of the individuals,
        # otherwise the diversity would be unnecessarily decreased
        append!(population, parents)
        append!(population, children)

        # select survivors
        population = select_survivors(population, state)

        gen += 1
    end # main loop

    # test the program with the minimum training or validation error using the testing dataset
    best = best_individual(population)
    if use_validation
        best_validation = best_individual(population, true)
        if best == best_validation
            # make a copy to avoid changing best.fitness
            best_validation = clone(best_validation, state)
        end
        fitness!(best_validation, testing[r], state)
        return best, best_validation, run_data, state, gen, toq()
    else
        fitness!(best, testing[r], state)
        return best, run_data, state, gen, toq()
    end
end

end # module
