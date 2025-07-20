using DataFrames, DrWatson, CSV

function load_breast_cancer(;return_X_y::Bool=false)
    BC_COLUMN_NAMES = [:ID, :Diagnosis, :radius1, :texture1, :perimeter1,
                   :area1, :smoothness1, :compactness1, :concavity1,
                   :concave_points1, :symmetry1, :fractal_dimension1,
                   :radius2, :texture2, :perimeter2, :area2, :smoothness2,
                   :compactness2, :concavity2, :concave_points2, :symmetry2,
                   :fractal_dimension2, :radius3, :texture3, :perimeter3,
                   :area3, :smoothness3, :compactness3, :concavity3,
                   :concave_points3, :symmetry3, :fractal_dimension3]

    data_file = datadir("UCI", "BreastCancer", "wdbc.data")
    df = DataFrame(CSV.File(data_file; header=BC_COLUMN_NAMES))

    # Remove ID 
    select!(df, Not(:ID))
    transform!(df, :Diagnosis => ByRow(y -> y .== "M" ? 1 : -1) => :Diagnosis)

    if return_X_y
        # convert to Matrix
        y = Matrix(select(df,:Diagnosis))
        X = Matrix(select(df, Not(:Diagnosis)))
        return X, y
    end

    return df
end
