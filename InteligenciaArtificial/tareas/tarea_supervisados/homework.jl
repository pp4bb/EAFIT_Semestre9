using Pkg
Pkg.instantiate()
run(`pip install -r requirements.txt`)
using MLDatasets, DataFrames, CSV
using Statistics, StatsBase
using ColorSchemes
using PyCall
#-------------------------------------------------------------------
umap=pyimport("umap")
LinearRegression = pyimport("sklearn.linear_model").LinearRegression
SVM = pyimport("sklearn.svm").SVC
TREE = pyimport("sklearn.tree").DecisionTreeClassifier
metrics = pyimport("sklearn.metrics")
#-------------------------------------------------------------------
include("unsupervised.jl")
include("metrics.jl")
include("utils.jl")

#------------------------------------------------------------------
# LOAD AND PREPARE THE DATA
#------------------------------------------------------------------
# load and normalize the original data
original = CSV.read("boston_housing.csv", DataFrame)
X = Matrix(original[:, 1:end])
X = Matrix(X)
dt = fit(UnitRangeTransform, X, dims=1)
original = StatsBase.transform(dt, X)
# load and normalize the expanded data
expanded = CSV.read("boston_housing_expanded.csv", DataFrame) # Already normalized
expanded = Matrix(expanded[:, 1:end])
# apply the UMAP algorithm to the original data to obtain the embedding
# n_components was set to 3 to be able to plot the data
embedding = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.4).fit_transform(original)
# store the data in a vector
#data = [expanded, original, embedding]
data = [Matrix{Float64}(D) for D in [expanded, original, embedding]]

"""
    split_train_test_validation(X::Matrix{Float64}, n::Int64)
Splits the given dataset into train, test and validation sets.

# Arguments
- `X::Matrix{Float64}`: Dataset to split.
- `n::Int64`: Number of elements to use for the train set.

# Returns
- `X_train::Matrix{Float64}`: Train set.
- `X_test::Matrix{Float64}`: Test set.
- `X_validation::Matrix{Float64}`: Validation set.
"""
function split_train_test_validation(X, n)
    X_train = X[1:n, :]
    remaining = size(X, 1) - n
    X_test = X[n+1:n+remaining//2, :]
    X_validation = X[n+remaining//2+1:end, :]
    return X_train, X_test, X_validation
end


"""
    vc_dimension(m::Int64, n::Int64; algorithm::String="linear", degree::Int64=1)
Calculates the VC dimension of the given algorithm.


# Arguments
- `X::Matrix{Float64}`: Dataset to cluster.
- `algorithm::String="linear"`: Algorithm to use
- `degree::Int64=1`: Degree of the polynomial kernel to use.

# Returns
- `vc_dimension::Int64`: VC dimension of the given algorithm.
"""
function vc_dimension(X; algorithm::String="linear", degree::Int64=1)::Int64
    m,n = size(X)
    if algorithm == "linear" || algorithm == "linear_svm"
        return n + 1
    elseif algorithm == "polinomial_svm"
        return n + 1 + degree
    elseif algorithm == "radial_svm"
        return 2^n
    else
        return 2^n
    end
end


"""
    homework(datasets::Vector{Matrix{Float64}}; metric::Function=euclidean_distance)
Runs the homework for the given datasets.

# Arguments
- `datasets::Vector{Matrix{Float64}}`: Vector of datasets to cluster.
- `metric::Function=euclidean_distance`: Distance metric to use.
"""
function homework(datasets::Vector{Matrix{Float64}}; metric::Function=euclidean_distance)
    classification = Vector()

    for i in eachindex(datasets)
        m, n = size(datasets[i])
        epsilons = [0.01, 0.05, 0.1]
        deltas = [0.01, 0.05, 0.1]
        @info "Computing kmeans for dataset $(i)"
        kmeans_results = kmeans_clustering(datasets[i], k=3, metric=metric)
        labels = kmeans_results[2]
        #Function that calculates the optimal training set size
        optimal_training_set_size(ϵ, δ, vc_dim) -> (1/ϵ) * (log(vc_dim)+log(1/δ))
        #Function that calculates the optimal training set size for DecisionTreeClassifier
        optimal_training_set_size_tree(ϵ, δ, depth, m) -> log(2)/(2*ϵ^2)*((2^depth-1)*(1+log2(m)))+1+log(1/δ)
        #VC dimensions
        vc_linear = vc_dimension(datasets[i], algorithm="linear")
        vc_lsvm = vc_dimension(datasets[i], algorithm="linear_svm")
        vc_psvm = vc_dimension(datasets[i], algorithm="polinomial_svm", degree=3)
        vc_rsvm = vc_dimension(datasets[i], algorithm="radial_svm")
        #Iterate over the epsilons and deltas
        for i in eachindex(epsilons)
            #-----------------------
            # LINEAR REGRESSION
            #-----------------------
            N = Vector()
            Y = Vector()
            SCORE = Vector()
            @info "Computing linear classifier for dataset $(i)"
            ϵ = epsilons[i]
            δ = deltas[i]
            optimal_size = Int(optimal_training_set_size(ϵ, δ, vc_linear))
            push!(N, optimal_size)
            x_train, x_test, x_validation = split_train_test_validation(datasets[i], optimal_size)
            y_train, y_test, y_validation = split_train_test_validation(labels, optimal_size)
            linear_model = LinearRegression().fit(x_train, y_train)
            y_pred = linear_model.predict(x_test)
            score = linear_model.score(x_validation, y_validation)
            push!(Y, y_pred)
            push!(SCORE, r2)
            linear_results = [N, Y, SCORE]
            #-----------------------------
            # LINEAR SVM
            #-----------------------------
            N = Vector()
            Y = Vector()
            SCORE = Vector()
            @info "Computing linear SVM for dataset $(i)"
            optimal_size = Int(optimal_training_set_size(ϵ, δ, vc_lsvm))
            push!(N, optimal_size)
            x_train, x_test, x_validation = split_train_test_validation(datasets[i], optimal_size)
            y_train, y_test, y_validation = split_train_test_validation(labels, optimal_size)
            linear_svm_model = SVC(kernel="linear").fit(x_train, y_train)
            y_pred = linear_svm_model.predict(x_test)
            score = linear_svm_model.score(x_validation, y_validation)
            push!(Y, y_pred)
            push!(SCORE, score)
            linear_svm_results = [N, Y, SCORE]
            #---------------------------------
            # POLYNOMIAL SVM
            #---------------------------------
            N = Vector()
            Y = Vector()
            SCORE = Vector()
            @info "Computing polynomial SVM for dataset $(i)"
            optimal_size = Int(optimal_training_set_size(ϵ, δ, vc_psvm))
            push!(N, optimal_size)
            x_train, x_test, x_validation = split_train_test_validation(datasets[i], optimal_size)
            y_train, y_test, y_validation = split_train_test_validation(labels, optimal_size)
            polynomial_svm_model = SVC(kernel="poly", degree=2).fit(x_train, y_train)
            y_pred = polynomial_svm_model.predict(x_test)
            score = polynomial_svm_model.score(x_validation, y_validation)
            push!(Y, y_pred)
            push!(SCORE, score)
            polynomial_svm_results = [N, Y, SCORE]
            #-------------------------------------
            # RADIAL BASIS FUNCTION SVM
            #-------------------------------------
            N = Vector()
            Y = Vector()
            SCORE = Vector()
            @info "Computing radial basis SVM for dataset $(i)"
            optimal_size = Int(optimal_training_set_size(ϵ, δ, vc_rsvm))
            push!(N, optimal_size)
            x_train, x_test, x_validation = split_train_test_validation(datasets[i], optimal_size)
            y_train, y_test, y_validation = split_train_test_validation(labels, optimal_size)
            radial_svm_model = SVC(kernel="rbf").fit(x_train, y_train)
            y_pred = radial_svm_model.predict(x_test)
            score = radial_svm_model.score(x_validation, y_validation)
            push!(Y, y_pred)
            push!(SCORE, score)
            radial_svm_results = [N, Y, SCORE]
            #-------------------------------------
            # DECISION TREE
            #-------------------------------------
            N = Vector()
            Y = Vector()
            SCORE = Vector()
            @info "Computing decision tree for dataset $(i)"
            optimal_size = Int(optimal_training_set_size_tree(ϵ, δ, 3, m))



        end

    end

end

function run()
    
end