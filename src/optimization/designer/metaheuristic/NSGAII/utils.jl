
"""
compute_SA

A function taking a set of solution as an entry and compute the Sobol first index and the Pearson correlation.
    A [@ref NSGAIISensitivityAnalysis] is then returned
# Parameters:
- `track_real`: A list of feasible solutions

"""
function compute_SA(track_real::Vector{Any})
    param = [x.val_param for x in track_real]
    obj = [x.critere for x in track_real]
    X = zeros(length(param[1]),length(param))
    Y_reg = zeros(length(obj[1]),length(obj))
    for i in 1:length(param)
        X[:,i] = param[i]
        Y_reg[:,i] = obj[i]
    end

    Y_EASI = obj

    tmp_EASI = []
    for i in 1:length(obj[1])
        push!(tmp_EASI, gsa(X, [y[i] for y in Y_EASI], EASI()).S1)
    end
 
    res_EASI = transpose(hcat(tmp_EASI...))
    res_reg = gsa(X, Y_reg, RegressionGSA(true))

    return NSGAIISensitivityAnalysis(res_EASI,  res_reg.pearson)

end



function plot_results(pop, cross)

    nb_gene = length(cross.cross_bgx)
    ############## Croisement evolution
    figure("Croisement")
    PyPlot.plot(1:nb_gene, cross.cross_bgx, label="BGX")
    PyPlot.plot(1:nb_gene, cross.cross_blx, label="BLX")
    PyPlot.plot(1:nb_gene, cross.cross_sbx, label="SBX")
    legend()


    ################# critère
    figure("front Julia")

    PyPlot.scatter([a.critere[1] for a in  pop],[a.critere[2] for a in pop])
 
end