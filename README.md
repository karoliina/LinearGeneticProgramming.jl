# LinearGeneticProgramming.jl
A Linear Genetic Programming system written in Julia. Most recently tested with Julia 0.5.

Requires the external Julia libraries (install with `Pkg.add("LibraryName")`):

* ArgParse
* DataFrames
* Formatting
* Lumberjack

First edit the parameter file to your liking - an example file, `even5parity_params.jl` together with the associated data set for the 5-bit even parity problem, is provided.

Usage: `julia LGP.jl [--no_print] [--no_save] params_filename run_label batch_label`, where the run and batch labels are optional.
