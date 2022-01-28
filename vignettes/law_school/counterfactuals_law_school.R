library("mlr3")
library("mlr3pipelines")
library("mlr3learners")
library("mlr3fairness")
library("randomForest")
source("vignettes/law_school/generate_law_school.R")

# "Real" counterfactuals
x = generate_law_school(1000)
xs = generate_cf_law_school_sex(x)
xr = generate_cf_law_school_race(x)

table(x$pass)
table(x$pass, xs$pass)
table(x$pass, xr$pass)

data = data.table(x)
rf = randomForest(pass ~ ., data = data)
# Create a predictor object
predictor = iml::Predictor$new(rf, type = "prob", data = data)

# Compute changes under true counterfactual
mean(predictor$predict(data.table(xr))[,1] - predictor$predict(data.table(x))[,1])
mean(predictor$predict(data.table(xs))[,1] - predictor$predict(data.table(x))[,1])


# Find counterfactuals for x_interest
cf_classif = CFClassif$new(predictor, protected = "sex", n_generations = 10L)
cfactuals = cf_classif$find_counterfactuals(
    x_interest = data[150L, ], desired_class = "2", desired_prob = c(0.5, 1)
)

xtra = data.table(xs[150,])[, role := "true_counterfactual"]
plot_counterfactuals(cfactuals, data, xtra)

# Average change in prediction
mean(predictor$predict(cfactuals$data)[,1] - predictor$predict(data[150L,])[,1])



