Dict{AbstractString, Any}([
    # random number generator
    ("RANDOM_SEED", 1),

    # CSV file to which the run label, random seed, fitness of best individual (fitness of best_validation if
    # USE_VALIDATION_DATASET is true), number of effective features in this individual's program, and a
    # sorted, space-separated list of the effective features in the individual's program will be written.
    ("RESULTS_FILE", "results.csv"),

    # input data
    # the program constructs the cross validation folds from the dataset
    ("DATASET", "data/even5parity.csv"),
    ("FOLDS", 1),
    ("INPUT_COLUMNS", [2:6;]),
    ("OUTPUT_COLUMN", 1),
    # if USE_VALIDATION_DATASET is true, the dataset will be divided to FOLDS partitions, one of which
    # will be the testing set, one the validation set and the rest the training set. if it is false,
    # one partition is the testing set and the rest the training set.
    ("USE_VALIDATION_DATASET", false),

    # fitness function - the given fitness function must be a method in LGPFunctions
    ("FITNESS_FUNC", LGPFunctions.mean_SE),
    ("CLASSIFICATION_ERROR_WEIGHT", 1.0),

    # representation
    ("EFFECTIVE_INITIALIZATION", false),
    # rate at which branch instructions are inserted during *effective* initialization (if at least one branch
    # instruction is included in the instruction set)
    ("BRANCH_INITIALIZATION_RATE", 0.3),

    ("INIT_MIN_LENGTH", 10),
    ("INIT_MAX_LENGTH", 30),
    ("MIN_LENGTH", 1),
    ("MAX_LENGTH", 200),

    # operators - must be Julia functions available to the program
    ("OPERATORS", [&, |, ~]),

    # constant registers
    ("CONSTANTS_RATE", 0.5),
    ("CONSTANTS", [0,1]),

    # calculation registers
    ("NUM_CALC_REGISTERS", 11),

    # population
    ("POPULATION_SIZE", 100),
    ("MAX_GENERATIONS", 50),
    ("STDDEV_THRESHOLD", 1e-5),

    # termination condition for best fitness
    ("TERMINATION_THRESHOLD", 1e-8),

    # parent selection
    ("PARENT_SELECTION_FUNC", LGPFunctions.tournament_selection!), # must be a function

    # tournament selection
    ("REPLACEMENT_PARENTS", true),
    # if REPLACEMENT is false, POPULATION_SIZE - NUM_PARENTS - 1 must be >= TOURNAMENT_SIZE
    ("NUM_PARENTS", 100),
    ("PARENTS_TOURNAMENT_SIZE", 4),

    # survivor selection
    ("TOURNAMENT_SURVIVORS", false),
    ("REPLACEMENT_SURVIVORS", false),
    ("SURVIVORS_TOURNAMENT_SIZE", 6),

    # macro mutations
    ("MACRO_MUTATION_OPERATOR", LGPFunctions.effmut2!), # must be a function
    ("MACRO_MUTATION_RATE", 0.7),
    ("CROSSOVER_RATE", 0.9),
    ("RANDOM_INSERTION_RATE", 0.1),
    ("INSERTION_RATE", 0.6),
    ("NUM_MUTATION_OPERATIONS", 1),
    ("NUM_RANDOM_INSERTIONS", 2),
    ("MAX_SEGMENT_LENGTH", 6),
    ("MAX_CROSSOVER_POINT_DISTANCE", 5),
    ("MAX_SEGMENT_LENGTH_DIFF", 3),
    ("MAX_NEUTREFFMUT_ITERATIONS", 3),

    # micro mutations
    ("MICRO_MUTATION_OPERATOR", LGPFunctions.effmicromut!), # must be a function
    ("MICRO_MUTATION_RATE", 0.5),
    ("REGISTER_MUTATION_RATE", 0.5),
    ("OPERATOR_MUTATION_RATE", 0.3),
    ("CONSTANT_MUTATION_SIGMA", 1),
])
