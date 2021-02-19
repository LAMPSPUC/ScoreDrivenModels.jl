library("rugarch")
bg96 <- read.csv("./test/data/BG96.csv", header = FALSE)

spec = ugarchspec(mean.model = list(armaOrder = c(0, 0), include.mean = FALSE, archm = FALSE,
archpow = 1, arfima = FALSE, external.regressors = NULL, archex = FALSE))
fit = ugarchfit(data = bg96$V1, spec = spec)
coef(fit)