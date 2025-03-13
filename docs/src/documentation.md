# Structures and Functions

## Microgrid
```@docs
Genesys.Microgrid
Genesys.GlobalParameters
```



## Storages

### Batteries
```@docs
Genesys.AbstractLiion
Genesys.Liion
```

#### Battery Efficiency Models
```@docs
Genesys.AbstractLiionEffModel
Genesys.FixedLiionEfficiency
Genesys.PolynomialLiionEfficiency
```

#### Battery Aging Models
```@docs
Genesys.AbstractLiionAgingModel
Genesys.EnergyThroughputLiion
Genesys.FixedLifetimeLiion
Genesys.RainflowLiion
Genesys.SemiEmpiricalLiion
```

### Thermal Storage
```@docs
Genesys.ThermalStorage
```

### Hydrogen
```@docs
Genesys.H2Tank
```

## Generation
```@docs
Genesys.Solar
```

## Carrier
```@docs
Genesys.EnergyCarrier
Genesys.Electricity
Genesys.Heat
Genesys.Hydrogen
```

## Converter
```@docs
Genesys.Electrolyzer
Genesys.FuelCell
Genesys.Heater
```
## Demand
```@docs
Genesys.Demand
```

## grid
```@docs
Genesys.Grid
```









## Scenario

```@docs
Genesys.Scenarios
```

### Scenarios Reducer

```@docs
Genesys.ManualReducer
Genesys.SAAReducer
Genesys.MeanValueReducer
Genesys.FeatureBasedReducer
```

### Dimention Reducer
```@docs
Genesys.PCAReduction
Genesys.StatsReduction
```

### Transformation
```@docs
Genesys.UnitRangeTransform
Genesys.ZScoreTransform
```

### Custering
```@docs
Genesys.KmedoidsClustering
```


### Generator
```@docs
Genesys.MarkovGenerator
Genesys.AnticipativeGenerator
```





### Base generic function


#### Generate
```@docs
Genesys.generate
```

#### Reduce
```@docs
Genesys.reduce
```
