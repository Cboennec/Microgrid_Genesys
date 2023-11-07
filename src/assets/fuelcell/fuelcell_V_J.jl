  
mutable struct deg_params
    a_slope::Float64
    b_slope::Float64
    c_slope::Float64
    power_slope::Float64
    b::Float64
    adjustment_coef::Float64
    start_stop_coef::Float64
end


function create_deg_params(datas::Vector{DataFrames.DataFrame}, Js::Vector{Float64}, V_J::DataFrames.DataFrame, J_ref::Float64, objective_lifetime::Float64; power = 1/2)

    P_max = datas[3]

    as, b = fit_all_curves(datas, Js)

    a_slope, b_slope, c_slope = fit_dot(Js, as, power)

    V_ini_ref =  interpolation(V_J.J,  V_J.V, J_ref, true)

    ΔV_tot = V_ini_ref * 0.1

    V_deg_ref = interpolation(P_max.J,  P_max.V, J_ref, true)

    current_lifetime = ΔV_tot / (V_deg_ref * 1e-6)

    adaptation_coefficient = current_lifetime * 1e-6/objective_lifetime

    start_stop_coef = 0.0000196 #0.00196% as stated in Pucheng Pei, Qianfei Chang, Tian Tang,
   # A quick evaluating method for automotive fuel cell lifetime (https://doi.org/10.1016/j.ijhydene.2008.04.048)
    
    return deg_params(a_slope, b_slope, c_slope, power, b, adaptation_coefficient, start_stop_coef)

end
  
  mutable struct FuelCell_V_J <: AbstractFuelCell
    # Paramètres
    α_p::Float64
    η_H2_E::Float64
    η_H2_H::Float64
    lifetime::Int64
    nHoursMax::Float64
    SoH_threshold::Float64
    update_by_year::Int # The frequency (per year) of SoH and V(I) update
    K::Float64 # Defined as a constant Latent Heat Value * masse molaire * stoechiometric coefficient / 2Faraday constant  
    k_aux::Float64
    bounds::NamedTuple{(:lb, :ub), Tuple{Float64, Float64}}
    # Initial conditions
    soh_ini::Float64
    V_J_ini::DataFrames.DataFrame # Initial V(I)
    powerMax_ini::Float64
    deg_params::deg_params
    min_J::Float64
    couplage::Bool
    # Variables

    N_cell::Int64 #The number of assembled cells
    surface::Float64 #The number of assembled cells

    J_ref::Float64 # The nominal current

    V_J::AbstractArray{Float64,2}

    powerMax::AbstractArray{Float64,3}
    powerMin::AbstractArray{Float64,3}
    η::AbstractArray{Float64,3}
    carrier::Vector{EnergyCarrier}
    soh::AbstractArray{Float64,3}

  
    # Eco
    cost::AbstractArray{Float64,2}
    # Inner constructor
    FuelCell_V_J(; α_p = 8/100, #Min power 
            η_H2_E = 0.4,
            η_H2_H = 0.4,
            lifetime = 14,
            nHoursMax = 10000.,
            SoH_threshold = 0.9,
            update_by_year = 12,
            K = (33.33 *  2.016 * 1.2 * 3600)  / (2*96485.3321),  #PCI * M_H2 * λ * 3600/(2*F)
            k_aux = 0.15,
            bounds = (lb = 0., ub = 50.),
            soh_ini = 1.,
            V_J_ini = DataFrames.DataFrame(CSV.File(joinpath("Examples","data","V_J_PAC.csv"), delim = ",", header = [Symbol("J"),Symbol("V")], types=Dict(:J=>Float64, :V=>Float64))),
            powerMax_ini = 1e-6,
            deg_params = deg_params(), 
            V_max = 0.8,
            couplage = true
            ) =
            new(α_p, η_H2_E, η_H2_H, lifetime, nHoursMax, SoH_threshold, update_by_year, K, k_aux, bounds, soh_ini, V_J_ini, powerMax_ini, deg_params, V_max, couplage)
  end
  
  

  
  ### Preallocation
  function preallocate!(fc::FuelCell_V_J, nh::Int64, ny::Int64, ns::Int64)
      fc.powerMax = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; fc.powerMax[1,1,:] .= fc.powerMax_ini
      fc.powerMin = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; fc.powerMin[1,1,:] .= fc.powerMax_ini
      fc.η = convert(SharedArray,zeros(nh+1, ny+1, ns))
      fc.carrier = [Electricity(), Heat(), Hydrogen()]
      fc.carrier[1].power = convert(SharedArray,zeros(nh, ny, ns))
      fc.carrier[2].power = convert(SharedArray,zeros(nh, ny, ns))
      fc.carrier[3].power = convert(SharedArray,zeros(nh, ny, ns))
      fc.soh = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; fc.soh[1,1,:] .= fc.soh_ini
      fc.cost = convert(SharedArray,zeros(ny, ns))

      fc.V_J = zeros(3, length(fc.V_J_ini.J)) #J, V, P
       
      return fc
  end
    


### Operation dynamic
function compute_operation_dynamics!(h::Int64, y::Int64, s::Int64, fc::FuelCell_V_J, decision::Float64, Δh::Int64)
    #Apply minimum power
    #fc.α_p * (fc.powerMax[h,y,s]/(1-fc.k_aux)) <= decision ? power_E = min(decision, fc.powerMax[h,y,s]) : power_E = 0. 
    fc.powerMin[h,y,s] <= decision ? power_E = min(decision, fc.powerMax[h,y,s]) : power_E = 0. 
        
    if power_E > 0
        #Compute the power needed to feed the auxiliaries and deliver the right power
        P_tot = floor(power_E / (1 - fc.k_aux); digits=6)

        #Find the corresponding current from an interpolation from P(I) curve 
        
        j = interpolation(fc.V_J[3,:], fc.V_J[1, :], P_tot, true)
       
        i = j * fc.surface
        
        η_E = power_E / (fc.K * i * fc.N_cell)

        fc.η[h,y,s] = η_E

        η_H = 0.8 - η_E

        fc.carrier[1].power[h,y,s], fc.carrier[2].power[h,y,s], fc.carrier[3].power[h,y,s] = (power_E), (power_E * η_H / η_E),  - (power_E / η_E) 
    
    else 
        
        fc.carrier[1].power[h,y,s], fc.carrier[2].power[h,y,s], fc.carrier[3].power[h,y,s] = 0, 0, 0
    end

    if (h%convert(Int64,floor(8760/fc.update_by_year))) == 0 && fc.couplage
        interval = (h-convert(Int64,floor(8760/fc.update_by_year))+1):h

        powers = fc.carrier[1].power[interval,y,s]

        #println("Powers : ", powers)
        fc.soh[h+1,y,s] = update_FC(fc, powers)

        fc.powerMax[h+1,y,s] = maximum(fc.V_J[3,:]) * (1-fc.k_aux)

        fc.powerMin[h+1,y,s] = compute_min_power(fc)
        
    else 
        fc.soh[h+1,y,s] = fc.soh[h,y,s] 
        fc.powerMax[h+1,y,s] = fc.powerMax[h,y,s] 
        fc.powerMin[h+1,y,s] = fc.powerMin[h,y,s] 
    end

    return fc
end

#Update V_J.V, then V_J.P then this should affect the maximum power and finally the SoH
function update_FC(fc::FuelCell_V_J, powers::Vector{Float64})

    coef_b = fc.deg_params.b

    start_stop_count = get_start_stops(powers)


    current_densities = []
    
    # get the sequence of intensities
    for p in powers
        if p .> 1e-6
            push!(current_densities, interpolation(fc.V_J[3,:], fc.V_J[1,:], p, true))
        end
    end

    for j in current_densities
       coef_a = get_slope_deg(j, fc.deg_params.power_slope, fc.deg_params.a_slope, fc.deg_params.b_slope, fc.deg_params.c_slope)
 
       ΔV = (fc.V_J[1,:] * coef_a).+coef_b 

       ΔV *= fc.deg_params.adjustment_coef

       fc.V_J[2,:] .-= ΔV
    end

    start_stop_count = get_start_stops(powers)

    V_nom = interpolation(fc.V_J[1,:], fc.V_J[2,:], fc.J_ref, true)

    fc.V_J[2,:] .-= fc.deg_params.start_stop_coef * V_nom * start_stop_count 

    fc.V_J[3,:] = fc.V_J[2,:] .* (fc.V_J[1,:] * fc.surface * fc.N_cell) 

    V_nom_ini = interpolation(fc.V_J_ini.J, fc.V_J_ini.V, fc.J_ref, true)
    V_nom = interpolation(fc.V_J[1,:], fc.V_J[2,:], fc.J_ref, true)


    plt = PyPlot.subplot()
    
    PyPlot.plot(fc.V_J[1,:], fc.V_J[2,:])
    plt.set_ylabel("Tension (V)")
    plt.set_xlabel("Current density (A/cm²)")

    return V_nom/V_nom_ini

end

    #For rule base
    function compute_operation_dynamics(fc::FuelCell_V_J, h::Int64, y::Int64, s::Int64, decision::Float64, Δh::Int64)
        #fc.α_p * (fc.powerMax[h,y,s]/(1-fc.k_aux)) <= decision ? power_E = min(decision, fc.powerMax[h,y,s]) : power_E = 0. 
        fc.powerMin[h,y,s] <= decision ? power_E = min(decision, fc.powerMax[h,y,s]) : power_E = 0. 
        #power_E = min(decision, fc.powerMax[h,y,s])

        if power_E > 0
            

            #Compute the power needed to feed the auxiliaries and deliver the right power
            P_tot = floor(power_E / (1 - fc.k_aux); digits=6)
    
            #Find the corresponding current from an interpolation from P(I) curve 
            j = interpolation(fc.V_J[3,:], fc.V_J[1,:], P_tot, true )
            i = j * fc.surface

            η_E = power_E / (fc.K * i * fc.N_cell)

            fc.η[h,y,s] = η_E
    
            η_H = 0.8 - η_E
        
            fc.carrier[1].power[h,y,s], fc.carrier[2].power[h,y,s], fc.carrier[3].power[h,y,s] = (power_E), (power_E * η_H / η_E),  - (power_E / η_E) 
            
        else 
            fc.carrier[1].power[h,y,s], fc.carrier[2].power[h,y,s], fc.carrier[3].power[h,y,s] = 0, 0, 0
        end
    
      
    if (h%convert(Int64,floor(8760/fc.update_by_year))) == 0 && fc.couplage
        interval = (h-convert(Int64,floor(8760/fc.update_by_year))+1):h

        powers = fc.carrier[1].power[interval,y,s]

        #println("Powers : ", powers)
        fc.soh[h+1,y,s] = update_FC(fc, powers)

        fc.powerMax[h+1,y,s] = maximum(fc.V_J[3,:]) * (1-fc.k_aux)

        fc.powerMin[h+1,y,s] = compute_min_power(fc)
        
    else 
        fc.soh[h+1,y,s] = fc.soh[h,y,s] 
        fc.powerMax[h+1,y,s] = fc.powerMax[h,y,s] 
        fc.powerMin[h+1,y,s] = fc.powerMin[h,y,s] 
    end
        
        return  fc.carrier[1].power[h,y,s], fc.carrier[2].power[h,y,s], fc.carrier[3].power[h,y,s] 
    end


  
    function initialize_investments!(s::Int64, fc::FuelCell_V_J, decision::NamedTuple{(:surface, :N_cell), Tuple{Float64, Int64}})

        fc.V_J = zeros(3, length(fc.V_J_ini.J)) #J, V, P

        fc.surface = decision.surface
        fc.N_cell = decision.N_cell
        
        fc.soh[1,1,s] = fc.soh_ini

        for (i,a) in enumerate([fc.V_J_ini.J, fc.V_J_ini.V, fc.V_J_ini.V .* fc.V_J_ini.J * fc.surface * fc.N_cell])
            fc.V_J[i,:] = a 
        end

        fc.powerMax[1,1,s] = maximum(fc.V_J[3,:]) * (1-fc.k_aux)

        fc.powerMin[1,1,s] = compute_min_power(fc)

        #We chose a value for which we have data on the degradation behaviour   
        fc.J_ref = 0.62

    end
  
  ### Investment dynamic
    function compute_investment_dynamics!(y::Int64, s::Int64, fc::FuelCell_V_J,  decision::NamedTuple{(:surface, :N_cell), Tuple{Float64, Int64}})
            
      fc.powerMax[1,y+1,s], fc.powerMin[1,y+1,s], fc.soh[1,y+1,s], fc.V_J = compute_investment_dynamics(fc, (powerMax = fc.powerMax[end,y,s], powerMin = fc.powerMin[end,y,s], soh = fc.soh[end,y,s]), decision)
    end


    function compute_investment_dynamics(fc::FuelCell_V_J, state::NamedTuple{(:powerMax, :powerMin, :soh), Tuple{Float64, Float64, Float64}},  decision::NamedTuple{(:surface, :N_cell), Tuple{Float64, Int64}})
        if decision.N_cell > 1e-2 

            V_J = zeros(3, length(fc.V_J_ini.J)) #J, V, P

            fc.surface = decision.surface
            fc.N_cell = decision.N_cell
            
    
            for (i,a) in enumerate([fc.V_J_ini.J, fc.V_J_ini.V, fc.V_J_ini.V .* fc.V_J_ini.J * fc.surface * fc.N_cell])
                V_J[i,:] = a 
            end
    
            fc.J_ref = 0.62
    
            powerMax_next = maximum(fc.V_J[3,:]) * (1-fc.k_aux)

            powerMin_next = compute_min_power(fc)
    
            soh_next = fc.soh_ini

        else
            powerMax_next = state.powerMax
            powerMin_next = state.powerMin

            soh_next = state.soh
            V_J = fc.V_J
        end

        return powerMax_next, powerMin_next, soh_next, V_J
    end
     

   
  


function fit_all_curves(data, Js)

    n_data = length(data)
    n_data_point = [length(data[i].J) for i in 1:n_data]


    m2 = Model(Gurobi.Optimizer)
    #set_optimizer_attribute(m2, "TimeLimit", 100)
    
    @variable(m2, a[1:n_data] >= 0)
    @variable(m2, b >= 0)

    @variable(m2, errors[1:n_data])
    @constraint(m2, [d in 1:n_data], errors[d] >= sum( (a[d]*data[d].J[i]+b - data[d].V[i])^2 for i in 1:n_data_point[d]))

    #Minimize the squared error
    @objective(m2, Min, sum(errors[d] for d in 1:n_data))


    optimize!(m2)
    plt = PyPlot.subplot()
    for d in 1:n_data
        PyPlot.scatter(data[d].J, data[d].V, label = string(Js[d] , " A/cm² : data"))
        PyPlot.plot(data[d].J, data[d].J .* value.(m2[:a])[d] .+ value(m2[:b]), label = string(Js[d] , " A/cm² : model"))
    end
    plt.set_xlabel("Current density (A/cm²)", fontsize=20)
    plt.set_ylabel("μV/h", fontsize=20)
    PyPlot.legend(fontsize = 15)

    return value.(m2[:a]), value(m2[:b])
end


function fit_dot(Js, as, power)

    n_data_point = length(Js)

    m2 = Model(Gurobi.Optimizer)
    set_optimizer_attribute(m2, "TimeLimit", 100)
    
    @variable(m2, a)
    @variable(m2, b)
    @variable(m2, c)


    @variable(m2, errors[1:n_data_point])
    @constraint(m2, [i in 1:n_data_point], errors[i] == (a*Js[i]^(power)+(b * Js[i]) + c) - as[i])

    #Minimize the squared error
    @objective(m2, Min, sum((errors[i]^2) for i in 1:n_data_point))


    optimize!(m2)

    p = PyPlot.subplot()

   

    interval =0:0.01:1
    PyPlot.plot(interval, [value(m2[:a])*x^(power)+(value(m2[:b]) * x) + value(m2[:c]) for x in interval], label = string("ax^", power, " + bx + c"))
 
    p.set_xlabel("Current density (A/cm²)", fontsize=20)
    p.set_ylabel("Slope", fontsize=20)
    PyPlot.legend(fontsize = 15)
     
    
        PyPlot.scatter(Js, as, label = "Fitted slopes coefficients")
        PyPlot.legend(fontsize = 15)
    

    return value(m2[:a]), value(m2[:b]), value(m2[:c])
end



function get_slope_deg(j, power, a_slope, b_slope, c_slope)
    
    slope = a_slope * j^(power) + b_slope * j + c_slope 
    return slope
    
end




