using Pkg
#Pkg.instantiate()
using MLDatasets, DataFrames, CSV
using Statistics, StatsBase
using ColorSchemes
include("unsupervised.jl")
include("metrics.jl")
include("utils.jl")

function homework(metric::Function=euclidean_distance)
    #------------------------------------------------------------------
    # LOAD AND PREPARE THE DATA
    #------------------------------------------------------------------
    # load and normalize the original data
    original = CSV.read("boston_housing.csv", DataFrame)
    X = Matrix(original[:, 1:end])
    X = Matrix(X)
    dt = fit(UnitRangeTransform, X, dims=1)
    original = StatsBase.transform(dt, X)
    @info "Computing kmeans"
    centers, labels, km_loss = kmeans_clustering(original, k=3, metric=metric)
    @info size(centers)
end