using MLDatasets, DataFrames
using Statistics, StatsBase
include("unsupervised.jl")

function test()
    # Load Iris dataset
    iris = Iris()
    # Normalize data in [0, 1]
    X = Matrix(iris.features[!, [:sepallength, :sepalwidth]])
    dt = fit(UnitRangeTransform, X, dims=1)
    X = StatsBase.transform(dt, X)
    centers, clusters = mountain_clustering(X, 0.5, 1.0, 10, euclidean_distance)
    display(centers)
end