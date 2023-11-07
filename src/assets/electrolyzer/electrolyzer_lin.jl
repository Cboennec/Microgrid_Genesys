mutable struct Electrolyzer_lin <: AbstractElectrolyzer
    # Paramètres
    α_p::Float64
    η_E_H2::Float64
    η_E_H::Float64
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
    J_min::Float64
    couplage::Bool
    # Variables

    a_η::Float64 # the slope for the fucntion η(P)
    b_η::Float64 # the ordinate at the origin for the function η(P)

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
    Electrolyzer_lin(; α_p = 8/100, #Min power 
            η_E_H2 = 0.5,
            η_H_H2 = 0.3,
            lifetime = 14,
            nHoursMax = 20000.,
            SoH_threshold = 0.8,
            update_by_year = 12,
            K = (33.33 *  2.016 * 3600)  / (2*96485.3321),  #PCI * M_H2 * λ * 3600/(2*F)
            k_aux = 0.10,
            bounds = (lb = 0., ub = 50.),
            soh_ini = 1.,
            V_J_ini = DataFrames.DataFrame(CSV.File(joinpath("Examples","data","V_J_Elyz.csv"), delim = ",", header = [Symbol("J"),Symbol("V")], types=Dict(:J=>Float64, :V=>Float64))),
            powerMax_ini = 1e-6,
            J_min = 0.1,
            couplage = true
            ) =
            new(α_p, η_E_H2, η_H_H2, lifetime, nHoursMax, SoH_threshold, update_by_year, K, k_aux, bounds, soh_ini, V_J_ini, powerMax_ini, J_min, couplage)
  end
  
  

  
  ### Preallocation
  function preallocate!(elyz::Electrolyzer_lin, nh::Int64, ny::Int64, ns::Int64)
      elyz.powerMax = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; elyz.powerMax[1,1,:] .= elyz.powerMax_ini
      elyz.powerMin = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; elyz.powerMin[1,1,:] .= elyz.powerMax_ini
      elyz.carrier = [Electricity(), Heat(), Hydrogen()]
      elyz.η = convert(SharedArray,zeros(nh+1, ny+1, ns))
      elyz.carrier[1].power = convert(SharedArray,zeros(nh, ny, ns))
      elyz.carrier[2].power = convert(SharedArray,zeros(nh, ny, ns))
      elyz.carrier[3].power = convert(SharedArray,zeros(nh, ny, ns))
      elyz.soh = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; elyz.soh[1,1,:] .= elyz.soh_ini
      elyz.cost = convert(SharedArray,zeros(ny, ns))

      elyz.V_J = zeros(3, length(elyz.V_J_ini.J)) #J, V, P
       
      return elyz
  end
    

### Operation dynamic
function compute_operation_dynamics!(h::Int64, y::Int64, s::Int64, elyz::Electrolyzer_lin, decision::Float64, Δh::Int64)
    #Apply minimum power
    elyz.powerMin[h,y,s]/(1 + elyz.k_aux)  <= -decision ? power_E = max(decision, -elyz.powerMax[h,y,s]) : power_E = 0. 
    
    
    if power_E < 0
        η_E_H2 = elyz.a_η * -power_E + elyz.b_η

        elyz.η[h,y,s] = η_E_H2

        η_E_H = 0.8 - η_E_H2
       
        elyz.carrier[1].power[h,y,s], elyz.carrier[2].power[h,y,s], elyz.carrier[3].power[h,y,s] = (power_E), - power_E * η_E_H,  - power_E * η_E_H2
    else 
        elyz.carrier[1].power[h,y,s], elyz.carrier[2].power[h,y,s], elyz.carrier[3].power[h,y,s] = 0, 0, 0
    end

    if (h%convert(Int64,floor(8760/elyz.update_by_year))) != 0
        elyz.soh[h+1,y,s] = elyz.soh[h,y,s] 
        elyz.powerMax[h+1,y,s] = elyz.powerMax[h,y,s] 
        elyz.powerMin[h+1,y,s] = elyz.powerMin[h,y,s] 

    else 
        interval = (h-convert(Int64,floor(8760/elyz.update_by_year))+1):h

        powers = elyz.carrier[1].power[interval,y,s]

        #println("Powers : ", powers)
        elyz.soh[h+1,y,s] = update_elyz(elyz, powers)

        elyz.powerMax[h+1,y,s] = maximum(elyz.V_J[3,:]) * (1+elyz.k_aux)
        elyz.powerMin[h+1,y,s] = compute_min_power(elyz)

    end

    return elyz
end



### Operation dynamic
function compute_operation_dynamics(elyz::Electrolyzer_lin, h::Int64, y::Int64, s::Int64, decision::Float64, Δh::Int64)
    #Apply minimum power
    elyz.powerMin[h,y,s]/(1 + elyz.k_aux) <= -decision ? power_E = max(decision, -elyz.powerMax[h,y,s]) : power_E = 0. 
    
    if power_E < 0
       
        η_E_H2 = elyz.a_η * -power_E + elyz.b_η

        elyz.η[h,y,s] = η_E_H2

        η_E_H = 0.8 - η_E_H2
       
        elyz.carrier[1].power[h,y,s], elyz.carrier[2].power[h,y,s], elyz.carrier[3].power[h,y,s] = (power_E), - power_E * η_E_H,  - power_E * η_E_H2

    else 
        elyz.carrier[1].power[h,y,s], elyz.carrier[2].power[h,y,s], elyz.carrier[3].power[h,y,s] = 0, 0, 0
    end


    if (h%convert(Int64,floor(8760/elyz.update_by_year))) != 0
        elyz.soh[h+1,y,s] = elyz.soh[h,y,s] 
        elyz.powerMax[h+1,y,s] = elyz.powerMax[h,y,s] 
        elyz.powerMin[h+1,y,s] = elyz.powerMin[h,y,s] 

    else 
        interval = (h-convert(Int64,floor(8760/elyz.update_by_year))+1):h

        powers = elyz.carrier[1].power[interval,y,s]

        elyz.soh[h+1,y,s] = update_elyz(elyz, powers)

        elyz.powerMax[h+1,y,s] = maximum(elyz.V_J[3,:]) * (1+elyz.k_aux)
        elyz.powerMin[h+1,y,s] = compute_min_power(elyz)
    end
   
 
    return elyz.carrier[1].power[h,y,s], elyz.carrier[2].power[h,y,s], elyz.carrier[3].power[h,y,s]
end



#Update V_J.V, then V_J.P then this should affect the maximum power and finally the SoH
function update_elyz(elyz::Electrolyzer_lin, powers::Vector{Float64})

    hours_funct = sum(-powers .> 1e-6)
   
    elyz.V_J[2,:] .+= hours_funct * 1e-5 # 10 μV/h

    elyz.V_J[3,:] = elyz.V_J[2,:] .* (elyz.V_J[1,:] * elyz.surface * elyz.N_cell) 

    if elyz.couplage
        P_max = maximum(elyz.V_J[3,:])
        P_min = compute_min_power(elyz)

        η_P_min = get_η_E(P_min, elyz)
        η_P_max = get_η_E(P_max, elyz)

        elyz.a_η = (η_P_max - η_P_min) / (P_max - P_min)
        elyz.b_η = η_P_min - elyz.a_η * P_min
    end

    
    V_nom = interpolation(elyz.V_J[1,:], elyz.V_J[2,:], elyz.J_ref, true)
    V_nom_ini = interpolation(elyz.V_J_ini.J, elyz.V_J_ini.V, elyz.J_ref, true)

    plt = PyPlot.subplot()    
    PyPlot.plot(elyz.V_J[1,:], elyz.V_J[2,:])
    plt.set_ylabel("Tension (V)")
    plt.set_xlabel("Current density (A/cm²)")

    return 1 - (V_nom-V_nom_ini)/V_nom_ini # 1 - survoltage / voltage_ini

   
end

  
    function initialize_investments!(s::Int64, elyz::Electrolyzer_lin, decision::NamedTuple{(:surface, :N_cell), Tuple{Float64, Int64}})

        elyz.V_J = zeros(3, length(elyz.V_J_ini.J)) #J, V, P

        elyz.surface = decision.surface
        elyz.N_cell = decision.N_cell
        
        elyz.soh[1,1,s] = elyz.soh_ini

        for (i,a) in enumerate([elyz.V_J_ini.J, elyz.V_J_ini.V, elyz.V_J_ini.V .* elyz.V_J_ini.J * elyz.surface * elyz.N_cell])
            elyz.V_J[i,:] = a 
        end

        P_max = maximum(elyz.V_J[3,:])
        P_min = compute_min_power(elyz)
    
        η_P_min = get_η_E(P_min, elyz)
        η_P_max = get_η_E(P_max, elyz)
    
        elyz.a_η = (η_P_max - η_P_min) / (P_max - P_min)
        elyz.b_η = η_P_min - elyz.a_η * P_min

        elyz.powerMax[1,1,s] = P_max
        elyz.powerMin[1,1,s] = P_min

        #We chose a value for which we have data on the degradation behaviour   
        elyz.J_ref = 1.


    end
  
  ### Investment dynamic
    function compute_investment_dynamics!(y::Int64, s::Int64, elyz::Electrolyzer_lin,  decision::NamedTuple{(:surface, :N_cell), Tuple{Float64, Int64}})
            
        elyz.powerMax[1,y+1,s], elyz.powerMin[1,y+1,s], elyz.soh[1,y+1,s], elyz.V_J = compute_investment_dynamics(elyz, (powerMax = elyz.powerMax[end,y,s], powerMin = elyz.powerMin[end,y,s], soh = elyz.soh[end,y,s]), decision)
    end


    function compute_investment_dynamics(elyz::Electrolyzer_lin, state::NamedTuple{(:powerMax, :powerMin, :soh), Tuple{Float64, Float64, Float64}},  decision::NamedTuple{(:surface, :N_cell), Tuple{Float64, Int64}})
        if decision.N_cell > 1e-2 

            V_J = zeros(3, length(elyz.V_J_ini.J)) #J, V, P

            elyz.surface = decision.surface
            elyz.N_cell = decision.N_cell
            
    
            for (i,a) in enumerate([elyz.V_J_ini.J, elyz.V_J_ini.V, elyz.V_J_ini.V .* elyz.V_J_ini.J * elyz.surface * elyz.N_cell])
                V_J[i,:] = a 
            end
    
            P_max = maximum(elyz.V_J[3,:])
            P_min = compute_min_power(elyz)
        
            η_P_min = get_η_E(P_min, elyz)
            η_P_max = get_η_E(P_max, elyz)
        
            elyz.a_η = (η_P_max - η_P_min) / (P_max - P_min)
            elyz.b_η = η_P_min - elyz.a_η * P_min

            elyz.J_ref = 1.
            
            powerMax_next = P_max
            powerMin_next = P_min

            soh_next = elyz.soh_ini

        else
            powerMax_next = state.powerMax
            powerMin_next = state.powerMin

            soh_next = state.soh
            V_J = elyz.V_J
        end

        return powerMax_next, powerMin_next, soh_next, V_J
    end
      