
  
  mutable struct FuelCell_lin <: AbstractFuelCell
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
    V_max::Float64
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
    FuelCell_lin(; α_p = 8/100, #Min power 
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
  function preallocate!(fc::FuelCell_lin, nh::Int64, ny::Int64, ns::Int64)
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
function compute_operation_dynamics!(h::Int64, y::Int64, s::Int64, fc::FuelCell_lin, decision::Float64, Δh::Int64)
    #Apply minimum power
    #fc.α_p * (fc.powerMax[h,y,s]/(1-fc.k_aux)) <= decision ? power_E = min(decision, fc.powerMax[h,y,s]) : power_E = 0. 
    fc.powerMin[h,y,s] <= decision ? power_E = min(decision, fc.powerMax[h,y,s]) : power_E = 0. 
        
    if power_E > 0
        
        η_E = fc.a_η * power_E + fc.b_η 

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
function update_FC(fc::FuelCell_lin, powers::Vector{Float64})

    coef_b = fc.deg_params.b

    n_hours_active = sum(powers .> 1e-6)
    
    coef_a = get_slope_deg(fc.J_ref, fc.deg_params.power_slope, fc.deg_params.a_slope, fc.deg_params.b_slope, fc.deg_params.c_slope)

    ΔV = (fc.V_J[1,:] * coef_a) .+ coef_b

    ΔV *= fc.deg_params.adjustment_coef

    ΔV *= n_hours_active
   
    fc.V_J[2,:] .-= ΔV
    
    V_nom = interpolation(fc.V_J[1,:], fc.V_J[2,:], fc.J_ref, true)

    fc.V_J[3,:] = fc.V_J[2,:] .* (fc.V_J[1,:] * fc.surface * fc.N_cell) 

    P_max = maximum(fc.V_J[3,:]) * (1-fc.k_aux)
    P_min= compute_min_power(fc)

    η_P_min = get_η_E(P_min, fc)
    η_P_max = get_η_E(P_max, fc)

    fc.a_η = (η_P_max - η_P_min) / (P_max - P_min)
    fc.b_η = η_P_min - fc.a_η * P_min

    V_nom_ini = interpolation(fc.V_J_ini.J, fc.V_J_ini.V, fc.J_ref, true)
    V_nom = interpolation(fc.V_J[1,:], fc.V_J[2,:], fc.J_ref, true)
    
    plt = PyPlot.subplot()
    
    PyPlot.plot(fc.V_J[1,:], fc.V_J[2,:])
    plt.set_ylabel("Tension (V)")
    plt.set_xlabel("Current density (A/cm²)")
    # plt = PyPlot.subplot()

    # x_val = vcat([x for x in P_min:1:P_max], P_max)
    # PyPlot.plot(x_val, x_val .* fc.a_η .+ fc.b_η)
    # plt.set_ylabel("η")
    # plt.set_xlabel("Power (kW)")

    return V_nom/V_nom_ini

end

    #For rule base
    function compute_operation_dynamics(fc::FuelCell_lin, h::Int64, y::Int64, s::Int64, decision::Float64, Δh::Int64)
        #fc.α_p * (fc.powerMax[h,y,s]/(1-fc.k_aux)) <= decision ? power_E = min(decision, fc.powerMax[h,y,s]) : power_E = 0. 
        fc.powerMin[h,y,s] <= decision ? power_E = min(decision, fc.powerMax[h,y,s]) : power_E = 0. 
        #power_E = min(decision, fc.powerMax[h,y,s])

        if power_E > 0
            
            η_E = fc.a_η * power_E + fc.b_η 

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


  
    function initialize_investments!(s::Int64, fc::FuelCell_lin, decision::NamedTuple{(:surface, :N_cell), Tuple{Float64, Int64}})

        fc.V_J = zeros(3, length(fc.V_J_ini.J)) #J, V, P

        fc.surface = decision.surface
        fc.N_cell = decision.N_cell
        
        fc.soh[1,1,s] = fc.soh_ini

        for (i,a) in enumerate([fc.V_J_ini.J, fc.V_J_ini.V, fc.V_J_ini.V .* fc.V_J_ini.J * fc.surface * fc.N_cell])
            fc.V_J[i,:] = a 
        end


        
        P_max = maximum(fc.V_J[3,:]) * (1-fc.k_aux)
        fc.powerMax[1,1,s] = P_max

        P_min= compute_min_power(fc)
        fc.powerMin[1,1,s] = P_min 

        η_P_min = get_η_E(P_min, fc)
        η_P_max = get_η_E(P_max, fc)

        fc.a_η = (η_P_max - η_P_min) / (P_max - P_min)
        fc.b_η = η_P_min - fc.a_η * P_min

        #We chose a value for which we have data on the degradation behaviour   
        fc.J_ref = 0.62

    end
  
  ### Investment dynamic
    function compute_investment_dynamics!(y::Int64, s::Int64, fc::FuelCell_lin,  decision::NamedTuple{(:surface, :N_cell), Tuple{Float64, Int64}})
            
      fc.powerMax[1,y+1,s], fc.powerMin[1,y+1,s], fc.soh[1,y+1,s], fc.V_J = compute_investment_dynamics(fc, (powerMax = fc.powerMax[end,y,s], powerMin = fc.powerMin[end,y,s], soh = fc.soh[end,y,s]), decision)
    end


    function compute_investment_dynamics(fc::FuelCell_lin, state::NamedTuple{(:powerMax, :powerMin, :soh), Tuple{Float64, Float64, Float64}},  decision::NamedTuple{(:surface, :N_cell), Tuple{Float64, Int64}})
        if decision.N_cell > 1e-2 

            V_J = zeros(3, length(fc.V_J_ini.J)) #J, V, P

            fc.surface = decision.surface
            fc.N_cell = decision.N_cell
            
    
            for (i,a) in enumerate([fc.V_J_ini.J, fc.V_J_ini.V, fc.V_J_ini.V .* fc.V_J_ini.J * fc.surface * fc.N_cell])
                V_J[i,:] = a 
            end
    
            fc.J_ref = 0.62
    
            P_max = maximum(fc.V_J[3,:]) * (1-fc.k_aux)
            powerMax_next = P_max
    
            P_min= compute_min_power(fc)
            powerMin_next = P_min 
    
            η_P_min = get_η_E(P_min, fc)
            η_P_max = get_η_E(P_max, fc)
    
            fc.a_η = (η_P_max - η_P_min) / (P_max - P_min)
            fc.b_η = η_P_min - fc.a_η * P_min
    
            soh_next = fc.soh_ini

        else
            powerMax_next = state.powerMax
            powerMin_next = state.powerMin

            soh_next = state.soh
            V_J = fc.V_J
        end

        return powerMax_next, powerMin_next, soh_next, V_J
    end
     

   
  
  


