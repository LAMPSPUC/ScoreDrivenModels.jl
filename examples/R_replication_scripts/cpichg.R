library("GAS")

GASSpec <- UniGASSpec(Dist = "std", ScalingType = "Identity", GASPar = list(location = TRUE, scale = TRUE, shape = FALSE))
cpichg <- read.csv("../../test/data/cpichg.csv", header = FALSE)
Fit <- UniGASFit(GASSpec, cpichg$V1)