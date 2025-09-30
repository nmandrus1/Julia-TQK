# TQK

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> TQK

It is authored by Niels Mandrus.

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "TQK"
```
which auto-activate the project and enable local path handling from DrWatson.

## Notes

<!-- |> Dependencies `Conda` and `PyCall` must be installed in that order (hopefully is done automatically) -->
|> For model comparision, data dimension = 2

## Experiment setup

There are 3 scripts

- `01_generate_data.jl` -- Generate the classical/quantum data sets of varying complexity (configuration defined in script file)
- `02_train_kernels.jl` -- Trains quantum kernels/searches for PualiKernel on datasets & outputs corresponding kernel matrices
- `03_train_svm.jl` -- Uses generated kernels to train SVMs on train/test splits 


