using Pkg
#Pkg.instantiate()
#run(`pip install -r requirements.txt`)
using MLDatasets, DataFrames, CSV
using Statistics, StatsBase
using ColorSchemes
using PyCall
#-------------------------------------------------------------------
umap=pyimport("umap")
LinearRegression = pyimport("sklearn.linear_model").LinearRegression
SVM = pyimport("sklearn.svm").SVC
DecisionTreeClassifier = pyimport("sklearn.tree").DecisionTreeClassifier
metrics = pyimport("sklearn.metrics")
train_test_split = pyimport("sklearn.model_selection").train_test_split
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
function split_train_test_validation(X, y, n)
    n = n / size(X, 1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=n, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=1)
    #=
    X_train = X[1:n, :]
    remaining = size(X, 1) - n
    n2 = Int(round(n+remaining/2))
    X_test = X[n+1:n2, :]
    X_validation = X[n2+1:end, :]
    =#
    return x_train, x_test, x_val, y_train, y_test, y_val
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
    matrixes = Vector()
    epsilons = [0.01, 0.05, 0.1]
    deltas = [0.01, 0.05, 0.1]
    
    for i in eachindex(datasets)
        m, n = size(datasets[i])
        size_matrix = zeros(5,3)
        score_matrix = zeros(5,3)
        @info "Computing kmeans for dataset $(i)"
        kmeans_results = kmeans_clustering(datasets[i], k=3, metric=metric)
        labels = kmeans_results[2]
        #Function that calculates the optimal training set size
        optimal_training_set_size(ϵ, δ, vc_dim) = (1/ϵ) * (log(vc_dim)+log(1/δ))
        #Function that calculates the optimal training set size for DecisionTreeClassifier
        optimal_training_set_size_tree(ϵ, δ, depth, f) = log(2)/(2*ϵ^2)*((2^depth-1)*(1+log2(f)))+1+log(1/δ)
        #VC dimensions
        vc_linear = vc_dimension(datasets[i], algorithm="linear")
        vc_lsvm = vc_dimension(datasets[i], algorithm="linear_svm")
        vc_psvm = vc_dimension(datasets[i], algorithm="polinomial_svm", degree=3)
        vc_rsvm = vc_dimension(datasets[i], algorithm="radial_svm")
        results = Vector()
        #Iterate over the epsilons and deltas
        for k in eachindex(epsilons)
            for j in eachindex(deltas)
                ϵ = epsilons[k]
                δ = deltas[j]
                #-----------------------
                # LINEAR REGRESSION
                #-----------------------
                @info "Computing linear classifier for dataset $(i), ϵ=$(ϵ), δ=$(δ)"
                optimal_size = round(optimal_training_set_size(ϵ, δ, vc_linear))
                oz = optimal_size
                if optimal_size > m
                    @info "optimal size is $(optimal_size) but the dataset has $(m) elements, using the whole dataset"
                    # Split the dataset into train 60%, test 20% and validation 20%
                    optimal_size = Int(round(m * 0.6))
                end
                x_train, x_test, x_validation, y_train, y_test, y_validation = split_train_test_validation(datasets[i], labels,optimal_size)
                linear_model = LinearRegression().fit(x_train, y_train)
                y_pred = linear_model.predict(x_test)
                score = linear_model.score(x_validation, y_validation)
                linear_results = [optimal_size, y_pred, score]
                if ϵ == δ
                    size_matrix[1,k] = oz
                    score_matrix[1,k] = score
                end
                #-----------------------------
                # LINEAR SVM
                #-----------------------------
                @info "Computing linear SVM for dataset $(i), ϵ=$(ϵ), δ=$(δ)"
                optimal_size = round(optimal_training_set_size(ϵ, δ, vc_lsvm))
                oz = optimal_size
                if optimal_size > m
                    @info "optimal size is $(optimal_size) but the dataset has $(m) elements, using the whole dataset"
                    # Split the dataset into train 60%, test 20% and validation 20%
                    optimal_size = Int(round(m * 0.6))
                end
                x_train, x_test, x_validation, y_train, y_test, y_validation = split_train_test_validation(datasets[i], labels,optimal_size)
                @info unique(y_train)
                linear_svm_model = SVM(kernel="linear").fit(x_train, y_train)
                y_pred = linear_svm_model.predict(x_test)
                score = linear_svm_model.score(x_validation, y_validation)
                linear_svm_results = [optimal_size, y_pred, score]
                if ϵ == δ
                    size_matrix[2,k] = oz
                    score_matrix[2,k] = score
                end
                #---------------------------------
                # POLYNOMIAL SVM
                #---------------------------------
                @info "Computing polynomial SVM for dataset $(i), ϵ=$(ϵ), δ=$(δ)"
                optimal_size = round(optimal_training_set_size(ϵ, δ, vc_psvm))
                oz = optimal_size
                if optimal_size > m
                    @info "optimal size is $(optimal_size) but the dataset has $(m) elements, using the whole dataset"
                    # Split the dataset into train 60%, test 20% and validation 20%
                    optimal_size = Int(round(m * 0.6))
                end
                x_train, x_test, x_validation, y_train, y_test, y_validation = split_train_test_validation(datasets[i], labels,optimal_size)
                polynomial_svm_model = SVM(kernel="poly", degree=2).fit(x_train, y_train)
                y_pred = polynomial_svm_model.predict(x_test)
                score = polynomial_svm_model.score(x_validation, y_validation)
                polynomial_svm_results = [optimal_size, y_pred, score]
                if ϵ == δ
                    size_matrix[3,k] = oz
                    score_matrix[3,k] = score
                end
                #-------------------------------------
                # RADIAL BASIS FUNCTION SVM
                #-------------------------------------
                @info "Computing radial basis SVM for dataset $(i), ϵ=$(ϵ), δ=$(δ)"
                optimal_size = round(optimal_training_set_size(ϵ, δ, vc_rsvm))
                oz = optimal_size
                if optimal_size > m
                    @info "optimal size is $(optimal_size) but the dataset has $(m) elements, using the whole dataset"
                    # Split the dataset into train 60%, test 20% and validation 20%
                    optimal_size = Int(round(m * 0.6))
                end
                x_train, x_test, x_validation, y_train, y_test, y_validation = split_train_test_validation(datasets[i], labels,optimal_size)
                radial_svm_model = SVM(kernel="rbf").fit(x_train, y_train)
                y_pred = radial_svm_model.predict(x_test)
                score = radial_svm_model.score(x_validation, y_validation)
                radial_svm_results = [optimal_size, y_pred, score]
                if ϵ == δ
                    size_matrix[4,k] = oz
                    score_matrix[4,k] = score
                end
                #-------------------------------------
                # DECISION TREE
                #-------------------------------------
                @info "Computing decision tree for dataset $(i), ϵ=$(ϵ), δ=$(δ)"
                optimal_size = round(optimal_training_set_size_tree(ϵ, δ, 3, m))
                oz = optimal_size
                if optimal_size > m
                    @info "optimal size is $(optimal_size) but the dataset has $(m) elements, using the whole dataset"
                    # Split the dataset into train 60%, test 20% and validation 20%
                    optimal_size = Int(round(m * 0.6))
                end
                x_train, x_test, x_validation, y_train, y_test, y_validation = split_train_test_validation(datasets[i], labels,optimal_size)
                decision_tree_model = DecisionTreeClassifier().fit(x_train, y_train)
                y_pred = decision_tree_model.predict(x_test)
                score = decision_tree_model.score(x_validation, y_validation)
                decision_tree_results = [optimal_size, y_pred, score]
                if ϵ == δ
                    size_matrix[5,k] = oz
                    score_matrix[5,k] = score
                end
                #-------------------------------------
                # STORE RESULTS
                #-------------------------------------
                r = [ϵ, δ, linear_results, linear_svm_results, polynomial_svm_results, radial_svm_results, decision_tree_results]
                push!(results, r)
            end
        end
        push!(classification, results)
        push!(matrixes, [size_matrix, score_matrix])
    end
    return classification, matrixes
end

function run()
    classification, matrixes = homework(data)
    for i in eachindex(matrixes)
        #print two new lines
        print("\n\n")
        size_matrix, score_matrix = matrixes[i]
        @info "Dataset $(i)"
        @info "Size matrix"
        display(size_matrix)
        @info "Score matrix"
        display(score_matrix)
    end
end