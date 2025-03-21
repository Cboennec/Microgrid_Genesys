#=
    Rule based controller
=#
mutable struct RBCOptions
    policy_selection::Int64

    RBCOptions(; policy_selection = 1) = new(policy_selection)
end

mutable struct RBC <: AbstractController
    options::RBCOptions
    decisions::NamedTuple
    history::AbstractScenarios

    RBC(; options = RBCOptions()) = new(options)
end

### Policies
function π_1(h::Int64, y::Int64, s::Int64, mg::Microgrid, controller::RBC)
    # Utils to simplify the writting
    Δh = mg.parameters.Δh
    liion, tes, h2tank = mg.storages[1], mg.storages[2], mg.storages[3]
    heater, elyz, fc = mg.converters[1], mg.converters[2], mg.converters[3]

    # Net power elec
    p_net_E = mg.demands[1].carrier.power[h,y,s] - mg.generations[1].carrier.power[h,y,s]

    # Liion
    _, _, u_liion = compute_operation_dynamics(liion, h, y, s, p_net_E, Δh)

    if p_net_E < 0.
        # Elyz
        _, u_elyz_E, elyz_H, elyz_H2 = compute_operation_dynamics(elyz, (powerMax = elyz.powerMax[y,s], soh = elyz.soh[h,y,s]), p_net_E - u_liion, Δh)
        # H2 tank
        _, u_h2tank = compute_operation_dynamics(h2tank, (Erated = h2tank.Erated[y,s], soc = h2tank.soc[h,y,s]), -elyz_H2, Δh)
        # Test H2
        elyz_H2 == - u_h2tank ? nothing : u_elyz_E = elyz_H = elyz_H2 = u_h2tank = 0.
        # FC
        u_fc_E, fc_H, fc_H2 = 0., 0., 0.
        # Heater
        u_heater_E, heater_H = compute_operation_dynamics(heater, (powerMax = heater.powerMax[y,s],), p_net_E - u_liion - u_elyz_E, Δh)
    else
        # FC
        _, u_fc_E, fc_H, fc_H2 = compute_operation_dynamics(fc, (powerMax = fc.powerMax[y,s], soh = fc.soh[h,y,s]), p_net_E - u_liion, Δh)
        # H2 tank
        _, u_h2tank = compute_operation_dynamics(h2tank, (Erated = h2tank.Erated[y,s], soc = h2tank.soc[h,y,s]), -fc_H2, Δh)
        # Test H2
        fc_H2 == - u_h2tank ? nothing : u_fc_E = fc_H = fc_H2 = u_h2tank = 0.
        # Elyz
        u_elyz_E, elyz_H, elyz_H2 = 0., 0., 0.
        # Heater
        u_heater_E, heater_H = 0., 0.
    end

    # Net heating power post H2
    p_net_H = mg.demands[2].carrier.power[h,y,s] - fc_H - elyz_H - heater_H

    # TES
    _, u_tes = compute_operation_dynamics(tes, (Erated = tes.Erated[y,s], soc = tes.soc[h,y,s]), p_net_H, Δh)

    # Heater
    if p_net_H < 0.
        _u_heater_E = 0.
    else
        _u_heater_E, _ = compute_operation_dynamics(heater, (powerMax = heater.powerMax[y,s],), - (p_net_H - u_tes) / heater.η_E_H, Δh)
    end

    # Store values
    controller.decisions.storages[1][h,y,s] = u_liion
    controller.decisions.storages[2][h,y,s] = u_tes
    controller.decisions.storages[3][h,y,s] = u_h2tank
    controller.decisions.converters[1][h,y,s]  = u_heater_E + _u_heater_E
    controller.decisions.converters[2][h,y,s] = u_elyz_E
    controller.decisions.converters[3][h,y,s] = u_fc_E
end

function π_2(h::Int64, y::Int64, s::Int64, mg::Microgrid, controller::RBC)
    @inbounds controller.decisions.storages[1][h,y,s] = mg.demands[1].carrier.power[h,y,s] - mg.generations[1].carrier.power[h,y,s]
end

function π_3(h::Int64, y::Int64, s::Int64, mg::Microgrid, controller::RBC)
    # Compute the heater electrical power based on the simple model
    controller.decisions.converters[1][h,y,s] = - max(min(mg.demands[2].carrier.power[h,y,s] / mg.converters[1].η_E_H, mg.converters[1].powerMax[y,s]), 0.)
    # Compute the liion decision from the power balance
    controller.decisions.storages[1][h,y,s] = mg.demands[1].carrier.power[h,y,s] - mg.generations[1].carrier.power[h,y,s] - controller.decisions.converters[1][h,y,s]
end

#Hydrogen batterie + Liion 
function π_4(h::Int64, y::Int64, s::Int64, mg::Microgrid, controller::RBC)
    Δh = mg.parameters.Δh
    liion, h2tank = mg.storages[1], mg.storages[2]
    elyz, fc = mg.converters[1], mg.converters[2]

    # Net power elec
    p_net_E = mg.demands[1].carrier.power[h,y,s] - mg.generations[1].carrier.power[h,y,s]

    #On charge ou décharge la batterie autant que possible

    u_liion = compute_operation_dynamics(liion, (Erated = liion.Erated[y,s], soc = liion.soc[h,y,s], soh = liion.soh[h,y,s]), p_net_E, Δh)

    if p_net_E < 0.
        # Elyz
        _, u_elyz_E, elyz_H, elyz_H2 = compute_operation_dynamics(elyz, (powerMax = elyz.powerMax[y,s], soh = elyz.soh[h,y,s]), p_net_E - u_liion, Δh)
        # H2 tank
        _, u_h2tank = compute_operation_dynamics(h2tank, (Erated = h2tank.Erated[y,s], soc = h2tank.soc[h,y,s]), -elyz_H2, Δh)
        # Test H2
        elyz_H2 == - u_h2tank ? nothing : u_elyz_E = elyz_H = elyz_H2 = u_h2tank = 0.
        # FC
        u_fc_E, fc_H, fc_H2 = 0., 0., 0.
    else
        # FC
        _, u_fc_E, fc_H, fc_H2 = compute_operation_dynamics(fc, (powerMax = fc.powerMax[y,s], soh = fc.soh[h,y,s]), p_net_E - u_liion, Δh)
        # H2 tank
        _, u_h2tank = compute_operation_dynamics(h2tank, (Erated = h2tank.Erated[y,s], soc = h2tank.soc[h,y,s]), -fc_H2, Δh)
        # Test H2
        fc_H2 == - u_h2tank ? nothing : u_fc_E = fc_H = fc_H2 = u_h2tank = 0.
        # Elyz
        u_elyz_E, elyz_H, elyz_H2 = 0., 0., 0.
    end


    controller.decisions.storages[1][h,y,s] = u_liion
    controller.decisions.storages[2][h,y,s] = u_h2tank
    controller.decisions.converters[1][h,y,s] = u_elyz_E
    controller.decisions.converters[2][h,y,s] = u_fc_E

end



### PV, 
# Bat, tank
# FC, ELYZ, Heater
function π_5(h::Int64, y::Int64, s::Int64, mg::Microgrid, controller::RBC)
    # Utils to simplify the writting
    Δh = mg.parameters.Δh
    liion, h2tank = mg.storages[1], mg.storages[2]
    heater, elyz, fc = mg.converters[1], mg.converters[2], mg.converters[3]

    # Net power elec
    p_net_E = mg.demands[1].carrier.power[h,y,s] - mg.generations[1].carrier.power[h,y,s]

    # priority to the battery
    u_liion = compute_operation_dynamics(liion, (Erated = liion.Erated[y,s], soc = liion.soc[h,y,s], soh = liion.soh[h,y,s]), p_net_E, Δh)

    if p_net_E < 0.
        # Elyz
        u_elyz_E, elyz_H, elyz_H2 = compute_operation_dynamics(elyz, h, y, s , p_net_E - u_liion, Δh)
        # H2 tank
        _, u_h2tank = compute_operation_dynamics(h2tank, (Erated = h2tank.Erated[y,s], soc = h2tank.soc[h,y,s]), -elyz_H2, Δh)
        # Did we charge the H2Tank
        elyz_H2 == - u_h2tank ? nothing : u_elyz_E = elyz_H = elyz_H2 = u_h2tank = 0.
        # FC
        u_fc_E, fc_H, fc_H2 = 0., 0., 0.
        # Rest goes to the Heater
        u_heater_E, heater_H = compute_operation_dynamics(heater, (powerMax = heater.powerMax[y,s],), p_net_E - u_liion - u_elyz_E, Δh)
    else
        # FC
        u_fc_E, fc_H, fc_H2 = compute_operation_dynamics(fc, h, y, s, p_net_E - u_liion, Δh)
        # H2 tank
        _, u_h2tank = compute_operation_dynamics(h2tank, (Erated = h2tank.Erated[y,s], soc = h2tank.soc[h,y,s]), -fc_H2, Δh)
        # Test H2
        fc_H2 == - u_h2tank ? nothing : u_fc_E = fc_H = fc_H2 = u_h2tank = 0.
        # Elyz
        u_elyz_E, elyz_H, elyz_H2 = 0., 0., 0.
        # Heater
        u_heater_E, heater_H = 0., 0.
    end

    # Net heating power post H2
    p_net_H = mg.demands[2].carrier.power[h,y,s] - fc_H - elyz_H - heater_H

    #
    # Heater
    if p_net_H < 0.
        _u_heater_E = 0.
    else
        _u_heater_E, _ = compute_operation_dynamics(heater, (powerMax = heater.powerMax[y,s],), - (p_net_H) / heater.η_E_H, Δh)
    end

    # Store values
    controller.decisions.storages[1][h,y,s] = u_liion
    controller.decisions.storages[2][h,y,s] = u_h2tank
    controller.decisions.converters[1][h,y,s]  = u_heater_E + _u_heater_E
    controller.decisions.converters[2][h,y,s] = u_elyz_E
    controller.decisions.converters[3][h,y,s] = u_fc_E
end




### PV, 
# Bat, tank
# FC, ELYZ, Heater 
# The H2 is used to fill the Liion
# Their is no heating demand, the heater is used for curtailment
function π_6(h::Int64, y::Int64, s::Int64, mg::Microgrid, controller::RBC)
    # Utils to simplify the writting
    Δh = mg.parameters.Δh
    liion, h2tank = mg.storages[1], mg.storages[2]
    heater, elyz, fc = mg.converters[1], mg.converters[2], mg.converters[3]

    # Net power elec
    p_net_E = mg.demands[1].carrier.power[h,y,s] - mg.generations[1].carrier.power[h,y,s]

    # priority to the battery
    u_liion = compute_operation_dynamics(liion, h, y, s, p_net_E, Δh)

    if p_net_E < 0.
        # Elyz
        u_elyz_E, elyz_H, elyz_H2 = compute_operation_dynamics(elyz, h, y, s , p_net_E - u_liion, Δh)
        # H2 tank
        _, u_h2tank = compute_operation_dynamics(h2tank, (Erated = h2tank.Erated[y,s], soc = h2tank.soc[h,y,s]), -elyz_H2, Δh)
        # Did we charge the H2Tank
        elyz_H2 == - u_h2tank ? nothing : u_elyz_E = elyz_H = elyz_H2 = u_h2tank = 0.
        # FC
        u_fc_E, fc_H, fc_H2 = 0., 0., 0.
        # Rest goes to the Heater
        u_heater_E, heater_H = compute_operation_dynamics(heater, (powerMax = heater.powerMax[y,s],), p_net_E - u_liion - u_elyz_E, Δh)
    else
        # If the battery is getting low and the fuelcell is not activated due to minor demand we activate and use the excess to charge the battery with it
        if p_net_E - u_liion < fc.eff_model.powerMin[h,y,s] && p_net_E - u_liion != 0 #&& liion.soc[h,y,s] < 0.6 #&& h2tank.soc[h,y,s] > 0.3 
            p_adjust_E = fc.eff_model.powerMin[h,y,s]
            u_liion = p_net_E - fc.eff_model.powerMin[h,y,s]
        else
            p_adjust_E = p_net_E - u_liion
        end

        # FC
        u_fc_E, fc_H, fc_H2 = compute_operation_dynamics(fc, h, y, s, p_adjust_E, Δh)
        # H2 tank
        _, u_h2tank = compute_operation_dynamics(h2tank, (Erated = h2tank.Erated[y,s], soc = h2tank.soc[h,y,s]), -fc_H2, Δh)
        # Test H2
        fc_H2 == - u_h2tank ? nothing : u_fc_E = fc_H = fc_H2 = u_h2tank = 0.
        # Elyz
        u_elyz_E, elyz_H, elyz_H2 = 0., 0., 0.
        # Heater
        u_heater_E, heater_H = 0., 0.
    end

    # Net heating power post H2
    p_net_H = - fc_H - elyz_H - heater_H

    #
    # Heater
    if p_net_H < 0.
        _u_heater_E = 0.
    else
        _u_heater_E, _ = compute_operation_dynamics(heater, (powerMax = heater.powerMax[y,s],), - (p_net_H) / heater.η_E_H, Δh)
    end



    # Store values
    controller.decisions.storages[1][h,y,s] = u_liion
    controller.decisions.storages[2][h,y,s] = u_h2tank
    controller.decisions.converters[1][h,y,s]  = u_heater_E + _u_heater_E
    controller.decisions.converters[2][h,y,s] = u_elyz_E
    controller.decisions.converters[3][h,y,s] = u_fc_E
end

### PV, 
# Bat, tank
# FC, ELYZ 
# The H2 is used to fill the Liion
# Their is no heating demand, the heater is used for curtailment
function π_7(h::Int64, y::Int64, s::Int64, mg::Microgrid, controller::RBC)
    # Utils to simplify the writting
    Δh = mg.parameters.Δh
    liion, h2tank = mg.storages[1], mg.storages[2]
    elyz, fc = mg.converters[1], mg.converters[2]

    # Net power elec
    p_net_E = mg.demands[1].carrier.power[h,y,s] - mg.generations[1].carrier.power[h,y,s]

    # priority to the battery
    u_liion, _ = compute_operation_dynamics(liion, h, y, s, p_net_E, Δh)

    if p_net_E < 0.

        # If there is a rest but its not enough to activate the elyz
        if p_net_E - u_liion < elyz.eff_model.powerMax[h,y,s] * elyz.min_part_load && p_net_E - u_liion > 0 && p_net_E >= elyz.eff_model.powerMin[h,y,s]
            p_adjust_E = elyz.eff_model.powerMin[h,y,s]
            u_liion = p_net_E - p_adjust_E
        else
            p_adjust_E = p_net_E - u_liion
        end


        # Elyz
        u_elyz_E, elyz_H, elyz_H2 = compute_operation_dynamics(elyz, h, y, s , p_adjust_E, Δh)
        # H2 tank
        _, u_h2tank = compute_operation_dynamics(h2tank, (Erated = h2tank.Erated[y,s], soc = h2tank.soc[h,y,s]), -elyz_H2, Δh)
        # Did we charge the H2Tank
        elyz_H2 == - u_h2tank ? nothing : u_elyz_E = elyz_H = elyz_H2 = u_h2tank = 0.
        # FC
        u_fc_E, fc_H, fc_H2 = 0., 0., 0.

       
    else
         # If there is a rest but its not enough to activate the fc
        if p_net_E - u_liion < fc.eff_model.powerMin[h,y,s] && p_net_E - u_liion > 0 && p_net_E >= fc.eff_model.powerMin[h,y,s]
            p_adjust_E = fc.eff_model.powerMin[h,y,s]
            u_liion = p_net_E - p_adjust_E
        else
            p_adjust_E = p_net_E - u_liion
        end
      
        # FC
        u_fc_E, fc_H, fc_H2 = compute_operation_dynamics(fc, h, y, s, p_adjust_E, Δh)
        # H2 tank
        _, u_h2tank = compute_operation_dynamics(h2tank, (Erated = h2tank.Erated[y,s], soc = h2tank.soc[h,y,s]), -fc_H2, Δh)
        # Test H2
        fc_H2 == - u_h2tank ? nothing : u_fc_E = fc_H = fc_H2 = u_h2tank = 0.
        # Elyz
        u_elyz_E, elyz_H, elyz_H2 = 0., 0., 0.
    end


  
    # Store values
    controller.decisions.storages[1][h,y,s] = u_liion
    controller.decisions.storages[2][h,y,s] = u_h2tank
    controller.decisions.converters[1][h,y,s] = u_elyz_E
    controller.decisions.converters[2][h,y,s] = u_fc_E
end


### PV, 
# Bat, tank
# FC, ELYZ 
function π_8(h::Int64, y::Int64, s::Int64, mg::Microgrid, controller::RBC)
    # Utils to simplify the writting
    Δh = mg.parameters.Δh
    liion, h2tank = mg.storages[1], mg.storages[2]
    elyz, fc = mg.converters[1], mg.converters[2]

    # Net power elec
    p_net_E = mg.demands[1].carrier.power[h,y,s] - mg.generations[1].carrier.power[h,y,s]

    # priority to the battery
    u_liion, _ = compute_operation_dynamics(liion, h, y, s, p_net_E, Δh)

    rest_after_bat = p_net_E - u_liion

    if rest_after_bat > 0 # Il manque de l'energie à fournir
        u_fc_E, _, u_fc_H2 = compute_operation_dynamics(fc, h, y, s, rest_after_bat, Δh)
        rest_after_fc = rest_after_bat - u_fc_E

        if u_fc_E == 0
            # La pile ne peut pas prendre le reste alors on lui donne son min 
            u_fc_E = fc.eff_model.powerMin[h,y,s] 
            u_liion = p_net_E - u_fc_E
        end

        _, u_h2tank = compute_operation_dynamics(h2tank, (Erated = h2tank.Erated[y,s], soc = h2tank.soc[h,y,s]), -u_fc_H2, Δh)
        # Test H2
        u_fc_H2 == - u_h2tank ? nothing : u_fc_E = fc_H = u_fc_H2 = u_h2tank = 0.
        # Elyz
        u_elyz_E, elyz_H, elyz_H2 = 0., 0., 0.
       
    else # Il y'a un surplus d'énergie
        u_elyz_E, _, u_elyz_H2 = compute_operation_dynamics(elyz, h, y, s, rest_after_bat, Δh)
        rest_after_fc = rest_after_bat - u_elyz_E

        if u_elyz_E == 0
            # L'electrolyzer ne peut pas prendre le reste alors on lui donne son min 
            min_elyz = ceil(elyz.eff_model.powerMax[h,y,s] * elyz.min_part_load, digits = 6)

            u_elyz_E, _, u_elyz_H2 = compute_operation_dynamics(elyz, h, y, s, -min_elyz, Δh)

            u_liion = p_net_E + min_elyz
        end

        # H2 tank
        _, u_h2tank = compute_operation_dynamics(h2tank, (Erated = h2tank.Erated[y,s], soc = h2tank.soc[h,y,s]), -u_elyz_H2, Δh)
        # Did we charge the H2Tank
        u_elyz_H2 == - u_h2tank ? nothing : u_elyz_E = elyz_H = elyz_H2 = u_h2tank = 0.
        u_fc_E, fc_H, u_fc_H2 = 0., 0., 0.

    end


      
    # Store values
    controller.decisions.storages[1][h,y,s] = u_liion
    controller.decisions.storages[2][h,y,s] = u_h2tank
    controller.decisions.converters[1][h,y,s] = u_elyz_E
    controller.decisions.converters[2][h,y,s] = u_fc_E
end


### Offline
function initialize_controller!(mg::Microgrid, controller::RBC, ω::AbstractScenarios)
    # Preallocation
    preallocate!(mg, controller)

    return controller
end

### Online
function compute_operation_decisions!(h::Int64, y::Int64, s::Int64, mg::Microgrid, controller::RBC)
    # Chose policy
    if controller.options.policy_selection == 1
        return π_1(h, y, s, mg, controller)
    elseif controller.options.policy_selection == 2
        return π_2(h, y, s, mg, controller)
    elseif controller.options.policy_selection == 3
        return π_3(h, y, s, mg, controller)
    elseif controller.options.policy_selection == 4
        return π_4(h, y, s, mg, controller)
    elseif controller.options.policy_selection == 5
        return π_5(h, y, s, mg, controller)
    elseif controller.options.policy_selection == 6
        return π_6(h, y, s, mg, controller)
    elseif controller.options.policy_selection == 7
        return π_7(h, y, s, mg, controller)
    elseif controller.options.policy_selection == 8
        return π_8(h, y, s, mg, controller)
    else
        error("Policy not defined !")
    end
end
