#=
    Scenario reduction methods
=#
abstract type AbstractScenariosReducer end
abstract type AbstractDimensionReducer end
abstract type AbstractClusteringMethod end

"""
    mutable struct ManualReducer <: AbstractScenariosReducer
        h::Union{UnitRange{Int64}, Int64}
        y::Union{UnitRange{Int64}, Int64}
        s::Union{UnitRange{Int64}, Int64}
    end

A mutable structure representing a manual scenario reduction. Inherits from the `AbstractScenariosReducer` abstract type.

# Fields
- `h::Union{UnitRange{Int64}, Int64}`: The hours to be included in the reduced scenario.
- `y::Union{UnitRange{Int64}, Int64}`: The years to be included in the reduced scenario.
- `s::Union{UnitRange{Int64}, Int64}`: The scenarios to be included in the reduced scenario.

# Constructor

    ManualReducer(; h = 1:8760, y = 2:3, s = 1:1) = new(h, y, s)
"""
mutable struct ManualReducer <: AbstractScenariosReducer
    h::Union{UnitRange{Int64}, Int64}
    y::Union{UnitRange{Int64}, Int64}
    s::Union{UnitRange{Int64}, Int64}
    ManualReducer(; h = 1:8760, y = 2:3, s = 1:1) = new(h, y, s)
end


"""
    function reduce(reducer::ManualReducer, ω::Scenarios)

Reduces the given `Scenarios` object using the specified `ManualReducer`.

# Arguments
- `reducer::ManualReducer`: The reducer used to specify the reduction process.
- `ω::Scenarios`: The `Scenarios` object to be reduced.

# Returns
- A new reduced `Scenarios` object.
- A probability vector representing the likelihood of each scenario.

# Example

```julia
manual_reducer = ManualReducer(h = 1:8760, y = 2:3, s = 1:1)
reduced_scenarios, probabilities = reduce(manual_reducer, scenarios)
```
"""
function reduce(reducer::ManualReducer, ω::Scenarios)
    # Parameters
    h, y, s = reducer.h, reducer.y, reducer.s
    # Initialize
    demands, generations, storages, converters, grids = similar(ω.demands), similar(ω.generations), similar(ω.storages), similar(ω.converters), similar(ω.grids)
    # Demands
    for (k, a) in enumerate(ω.demands)
        demands[k] = (t = a.t[h, y, s], power = a.power[h, y, s])
    end
    # Generations
    for (k, a) in enumerate(ω.generations)
        generations[k] = (t = a.t[h, y, s], power = a.power[h, y, s], cost = a.cost[y, s])
    end
    # Storages
    for (k, a) in enumerate(ω.storages)
        storages[k] = (cost = a.cost[y, s],)
    end
    # Converters
    for (k, a) in enumerate(ω.converters)
        converters[k] = (cost = a.cost[y, s],)
    end
    # Grids
    for (k, a) in enumerate(ω.grids)
        grids[k] = (cost_in =  a.cost_in[h, y, s], cost_out =  a.cost_out[h, y, s], cost_exceed = a.cost_exceed[y, s])
    end

    return Scenarios(demands, generations, storages, converters, grids), ones(length(s)) / length(s)
end

"""
    mutable struct SAAReducer <: AbstractScenariosReducer
        nsample::Int64
    end

A mutable structure representing a Sample Average Approximation (SAA) scenario reduction. Inherits from the `AbstractScenariosReducer` abstract type.

# Fields
- `nsample::Int64`: The number of samples to be included in the reduced scenario.

# Constructor

    SAAReducer(; nsample = 100) = new(nsample)
"""
mutable struct SAAReducer <: AbstractScenariosReducer
    nsample::Int64

    SAAReducer(; nsample = 100) = new(nsample)
end

"""
    function reduce(reducer::SAAReducer, ω::Scenarios; y::Int64 = 1, s::Int64 = 1)

Reduces the given `Scenarios` object using the specified `SAAReducer`, implementing the Sample Average Approximation scenario reduction method.

# Arguments
- `reducer::SAAReducer`: The reducer used to specify the reduction process.
- `ω::Scenarios`: The `Scenarios` object to be reduced.

# Keyword Arguments
- `y::Int64 = 1`: The initial year to be considered.
- `s::Int64 = 1`: The initial scenario to be considered.

# Returns
- A new reduced `Scenarios` object.
- A probability vector representing the likelihood of each scenario.

# Example

```julia
saa_reducer = SAAReducer(nsample = 100)
reduced_scenarios, probabilities = reduce(saa_reducer, scenarios)
```
"""
function reduce(reducer::SAAReducer, ω::Scenarios; y::Int64 = 1, s::Int64 = 1)
    # Parameters
    _, ny, ns = size(ω.demands[1].power)
    # Monte carlo indices
    idx = zip(rand(y:ny, reducer.nsample), rand(s:ns, reducer.nsample))
    # Initialize
    demands, generations, storages, converters, grids = similar(ω.demands), similar(ω.generations), similar(ω.storages), similar(ω.converters), similar(ω.grids)
    # Demands
    for (k, a) in enumerate(ω.demands)
        demands[k] = (t = reshape(hcat([a.t[:, y, s] for (y,s) in idx]...),:,1,reducer.nsample), power = reshape(hcat([a.power[:, y, s] for (y,s) in idx]...),:,1,reducer.nsample))
    end
    # Generations
    for (k, a) in enumerate(ω.generations)
        generations[k] = (t = reshape(hcat([a.t[:, y, s] for (y,s) in idx]...),:,1,reducer.nsample), power = reshape(hcat([a.power[:, y, s] for (y,s) in idx]...),:,1,reducer.nsample), cost = repeat(a.cost[y:y, s:s],1,reducer.nsample))
    end
    # Storages
    for (k, a) in enumerate(ω.storages)
        storages[k] = (cost =  repeat(a.cost[y:y, s:s],1,reducer.nsample),)
    end
    # Converters
    for (k, a) in enumerate(ω.converters)
        converters[k] = (cost =  repeat(a.cost[y:y, s:s],1,reducer.nsample),)
    end
    # Grids
    for (k, a) in enumerate(ω.grids)
        grids[k] = (cost_in = reshape(hcat([a.cost_in[:, y, s] for (y,s) in idx]...),:,1,reducer.nsample), cost_out = reshape(hcat([a.cost_out[:, y, s] for (y,s) in idx]...),:,1,reducer.nsample))
    end

    return Scenarios(demands, generations, storages, converters, grids), ones(reducer.nsample) / reducer.nsample
end

"""
    mutable struct MeanValueReducer <: AbstractScenariosReducer
    end

A mutable structure representing an Expected Value (Mean Value) scenario reduction. Inherits from the `AbstractScenariosReducer` abstract type.

# Constructor

    MeanValueReducer() = new()

"""
mutable struct MeanValueReducer <: AbstractScenariosReducer
    MeanValueReducer() = new()
end

"""
    function reduce(reducer::MeanValueReducer, ω::Scenarios; y::Int64 = 1, s::Int64 = 1)

Reduces the given `Scenarios` object using the specified `MeanValueReducer`, implementing the Expected Value scenario reduction method.

# Arguments
- `reducer::MeanValueReducer`: The reducer used to specify the reduction process.
- `ω::Scenarios`: The `Scenarios` object to be reduced.

# Keyword Arguments
- `y::Int64 = 1`: The initial year to be considered.
- `s::Int64 = 1`: The initial scenario to be considered.

# Returns
- A new reduced `Scenarios` object.
- A probability vector representing the likelihood of each scenario (in this case, a single element array containing `1`).

# Example

```julia
mean_value_reducer = MeanValueReducer()
reduced_scenarios, probabilities = reduce(mean_value_reducer, scenarios)
```
"""
function reduce(reducer::MeanValueReducer, ω::Scenarios; y::Int64 = 1, s::Int64 = 1)
    # Mean value
    # Initialize
    demands, generations, storages, converters, grids = similar(ω.demands), similar(ω.generations), similar(ω.storages), similar(ω.converters), similar(ω.grids)
    # Demands
    for (k, a) in enumerate(ω.demands)
        demands[k] = (t =a.t[:, y:y, s:s], power = mean(a.power, dims=[2,3]))
    end
    # Generations
    for (k, a) in enumerate(ω.generations)
        generations[k] = (t = a.t[:, y:y, s:s], power =  mean(a.power, dims=[2,3]), cost = a.cost[y:y, s:s])
    end
    # Storages
    for (k, a) in enumerate(ω.storages)
        storages[k] = (cost =  a.cost[y:y, s:s],)
    end
    # Converters
    for (k, a) in enumerate(ω.converters)
        converters[k] = (cost =  a.cost[y:y, s:s],)
    end
    # Grids
    for (k, a) in enumerate(ω.grids)
        grids[k] = (cost_in = mean(a.cost_in, dims=[2,3]), cost_out =  mean(a.cost_out, dims=[2,3]))
    end

    return Scenarios(demands, generations, storages, converters, grids), [1.]
end

"""
    mutable struct FeatureBasedReducer <: AbstractScenariosReducer
        transformation::Union{UnionAll, Nothing}
        reduction::Union{AbstractDimensionReducer, Nothing}
        clustering::AbstractClusteringMethod
    end

A mutable structure representing a Clustering scenario reduction using feature-based methods. Inherits from the `AbstractScenariosReducer` abstract type.

# Fields
- `transformation`: A transformation method to apply before clustering (default: `UnitRangeTransform`)
- `reduction`: A dimensionality reduction method to apply before clustering (default: `StatsReduction()`)
- `clustering`: The clustering method to use (default: `KmedoidsClustering()`)

# Constructor

    FeatureBasedReducer(; transformation = UnitRangeTransform,
                        reduction = StatsReduction(),
                        clustering = KmedoidsClustering()) = new(transformation, reduction, clustering)


"""
mutable struct FeatureBasedReducer <: AbstractScenariosReducer
    transformation::Union{UnionAll, Nothing}
    reduction::Union{AbstractDimensionReducer, Nothing}
    clustering::AbstractClusteringMethod

    FeatureBasedReducer(; transformation = UnitRangeTransform,
                        reduction = StatsReduction(),
                        clustering = KmedoidsClustering()) = new(transformation, reduction, clustering)
end

"""
    function reduce(reducer::FeatureBasedReducer, ω::Scenarios; y::Int64 = 1, s::Int64 = 1)

Reduces the given `Scenarios` object using the specified `FeatureBasedReducer`, implementing a clustering-based scenario reduction method.

# Arguments
- `reducer::FeatureBasedReducer`: The reducer used to specify the reduction process, including transformation, dimension reduction, and clustering methods.
- `ω::Scenarios`: The `Scenarios` object to be reduced.

# Keyword Arguments
- `y::Int64 = 1`: The initial year to be considered.
- `s::Int64 = 1`: The initial scenario to be considered.

# Returns
- A new reduced `Scenarios` object.
- A probability vector representing the likelihood of each scenario in the reduced set.
- An array of assignments indicating which cluster each original scenario belongs to.

# Example

```julia
feature_based_reducer = FeatureBasedReducer(transformation = UnitRangeTransform,
                                            reduction = StatsReduction(),
                                            clustering = KmedoidsClustering())
reduced_scenarios, probabilities, assignments = reduce(feature_based_reducer, scenarios)
```
"""
function reduce(reducer::FeatureBasedReducer, ω::Scenarios; y::Int64 = 1, s::Int64 = 1)
    # Parameters
    nh, ny, ns = size(ω.demands[1].power)
    # Initialize
    demands, generations, storages, converters, grids = similar(ω.demands), similar(ω.generations), similar(ω.storages), similar(ω.converters), similar(ω.grids)
    # Formatting
    t_d, t_g = [reshape(a.t[:,2:end,:], nh, :) for a in ω.demands], [reshape(a.t[:,2:end,:], nh, :) for a in ω.generations]
    data_d, data_g, data_gd = [reshape(a.power[:,2:end,:], nh, :) for a in ω.demands], [reshape(a.power[:,2:end,:], nh, :) for a in ω.generations], [reshape(a.cost_in[:,2:end,:], nh, :) for a in ω.grids]
    # Transformation
    norm = replace!.([Genesys.StatsBase.standardize(reducer.transformation, d, dims = 1) for d in vcat(data_d, data_g, data_gd)], NaN => 0.)
    # Dimension reduction
    embedding = replace!(dimension_reduction(reducer.reduction, norm), NaN => 0.)
    # Clustering
    medoids, counts, assignments = clustering(reducer.clustering, embedding)
    # Building reduced scenario
    for (k, a) in enumerate(ω.demands)
        demands[k] = (t = reshape(t_d[k][:,medoids], nh, 1, :), power = reshape(data_d[k][:,medoids], nh, 1, :))
    end
    # Generations
    for (k, a) in enumerate(ω.generations)
        generations[k] = (t = reshape(t_g[k][:,medoids], nh, 1, :), power = reshape(data_g[k][:,medoids], nh, 1, :), cost =  repeat(a.cost[y:y, s:s], 1, length(medoids)))
    end
    # Storages
    for (k, a) in enumerate(ω.storages)
        storages[k] = (cost =  repeat(a.cost[y:y, s:s], 1, length(medoids)),)
    end
    # Converters
    for (k, a) in enumerate(ω.converters)
        converters[k] = (cost =  repeat(a.cost[y:y, s:s], 1, length(medoids)),)
    end
    # Grids
    for (k, a) in enumerate(ω.grids)
        grids[k] = (cost_in = reshape(data_gd[k][:,medoids], nh, 1, :), cost_out = reshape(reshape(a.cost_out[:,2:end,:], nh, :)[:,1], nh, 1, :), cost_exceed = reshape(reshape(a.cost_exceed[2:end,:], 1, :)[:,1], 1, : ))
    end

    return Scenarios(demands, generations, storages, converters, grids), counts / sum(counts), assignments
end

# Transformation
StatsBase.standardize(DT::Nothing, X; dims=nothing, kwargs...) = X

"""
    mutable struct PCAReduction <: AbstractDimensionReducer
        n_components::Int64
    end

A mutable structure representing Principal Component Analysis (PCA) dimensionality reduction. Inherits from the `AbstractDimensionReducer` abstract type.

# Fields
- `n_components`: The number of principal components to use for dimensionality reduction (default: 2)

# Constructor

    PCAReduction(; n_components = 2) = new(n_components)

"""
mutable struct PCAReduction <: AbstractDimensionReducer
    n_components::Int64

    PCAReduction(; n_components = 2) = new(n_components)
end


"""
    function dimension_reduction(reducer::PCAReduction, data::Array{Array{Float64,2}}; aggregated::Bool=false)

Performs dimension reduction using PCA (Principal Component Analysis) on the input data.

# Arguments
- `reducer::PCAReduction`: The PCA reduction instance specifying the number of components to be used for reduction.
- `data::Array{Array{Float64,2}}`: A vector of `d x n` matrices with `d` dimensions and `n` observations.

# Keyword Arguments
- `aggregated::Bool=false`: If `true`, PCA is applied on the aggregated data, otherwise, it is applied individually on each matrix in `data`.

# Returns
- The reduced data after applying PCA.

# Example

```julia
pca_reduction = PCAReduction(n_components = 3)
reduced_data = dimension_reduction(pca_reduction, data)
```
"""
function dimension_reduction(reducer::PCAReduction, data::Array{Array{Float64,2}}; aggregated::Bool=false)
    # data is a vector of d x n matrix with d dimension and n observation
    if aggregated
        m = MultivariateStats.fit(PCA, vcat(data...), maxoutdim = reducer.n_components)
        return MultivariateStats.transform(m, data)
    else
        M = [MultivariateStats.fit(PCA, d, maxoutdim = reducer.n_components) for d in data]
        return vcat([MultivariateStats.transform(M[k], data[k]) for k in 1:length(data)]...)
    end
end

"""
    struct StatsReduction <: AbstractDimensionReducer

A structure representing statistics-based dimensionality reduction. Inherits from the `AbstractDimensionReducer` abstract type.

# Constructor

    StatsReduction()
"""
struct StatsReduction <: AbstractDimensionReducer end

"""
    function dimension_reduction(reducer::StatsReduction, data::Array{Array{Float64,2}}; aggregated::Bool=false)

Performs dimension reduction using descriptive statistics on the input data.

# Arguments
- `reducer::StatsReduction`: The StatsReduction instance.
- `data::Array{Array{Float64,2}}`: A vector of `d x n` matrices with `d` dimensions and `n` observations.

# Keyword Arguments
- `aggregated::Bool=false`: If `true`, calculates the statistics on the aggregated data, otherwise, calculates them individually for each matrix in `data`.

# Returns
- The reduced data containing sum, max, mean, variance, kurtosis, and skewness for each matrix in `data`.

# Example

```julia
stats_reduction = StatsReduction()
reduced_data = dimension_reduction(stats_reduction, data)
```
"""
function dimension_reduction(reducer::StatsReduction, data::Array{Array{Float64,2}}; aggregated::Bool=false)
    # data is a vector of d x n matrix with d dimension and n observation
    if aggregated
        d = vcat(data...)
        # Sum
        s = sum(d, dims = 1)
        # Max
        max = maximum(d, dims = 1)
        # 4 moments
        m = mean(d, dims = 1)
        v = var(d, dims = 1)
        kurt = permutedims([kurtosis(d[:,j]) for j in 1:size(d, 2)])
        skew = permutedims([skewness(d[:,j]) for j in 1:size(d, 2)])
    else
        # Sum
        s = vcat([sum(d, dims = 1) for d in data]...)
        # Max
        max = vcat([maximum(d, dims = 1) for d in data]...)
        # 4 moments
        m = vcat([mean(d, dims = 1) for d in data]...)
        v = vcat([var(d, dims = 1) for d in data]...)
        kurt = vcat([permutedims([kurtosis(d[:,j]) for j in 1:size(d, 2)]) for d in data]...)
        skew = vcat([permutedims([skewness(d[:,j]) for j in 1:size(d, 2)]) for d in data]...)
    end
    # Return aggregated values
    return vcat(s, max, m, v, kurt, skew)
    # return vcat(m, v, kurt, skew)
end

# No reduction
dimension_reduction(reducer::Nothing, data::Array{Array{Float64,2}}) = vcat(data...)

"""
    mutable struct KmedoidsClustering <: AbstractClusteringMethod

A structure representing K-medoids clustering method. Inherits from the `AbstractClusteringMethod` abstract type.

# Fields
- `n_clusters::Int64`: Number of clusters to form.
- `distance::Distances.SemiMetric`: Distance metric to use for calculating the dissimilarity between points.
- `log::Bool`: Whether to print logs during the clustering process.

# Constructor

    KmedoidsClustering(; n_clusters = 20, distance = Distances.Euclidean(), log = true)

"""
mutable struct KmedoidsClustering <: AbstractClusteringMethod
    n_clusters::Int64
    distance::Distances.SemiMetric
    log::Bool

    KmedoidsClustering(; n_clusters = 20, distance = Distances.Euclidean(), log = true) = new(n_clusters, distance, log)
end


"""
    function clustering(method::KmedoidsClustering, embedding::AbstractArray{Float64,2})

Performs K-medoids clustering on the given embedding.

# Arguments
- `method::KmedoidsClustering`: The KmedoidsClustering instance.
- `embedding::AbstractArray{Float64,2}`: A `d x n` matrix with `d` dimensions and `n` observations.

# Returns
- `medoids`: The indices of the medoids.
- `counts`: The number of elements in each cluster.
- `assignments`: The cluster assignments for each observation.

# Example

```julia
kmedoids_clustering = KmedoidsClustering(n_clusters=5, distance=Euclidean())
medoids, counts, assignments = clustering(kmedoids_clustering, embedding)
```
"""
function clustering(method::KmedoidsClustering, embedding::AbstractArray{Float64,2})
    # data is a d x n matrix with d dimension and n observation
    # Distance matrix
    D = pairwise(method.distance, embedding, dims = 2)
    # Clustering
    results = kmedoids(D, method.n_clusters, display = method.log ? :iter : :none)

    return results.medoids, results.counts, results.assignments
end
