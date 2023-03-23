using MLDatasets, DataFrames
include("unsupervised.jl")

function test()
    # Load Iris dataset
    iris = Iris()
    centers, clusters = mountain_clustering(Matrix(iris.features[!, [:sepallength, :sepalwidth]]), 5.0, 5.0, 10, euclidean_distance)
end