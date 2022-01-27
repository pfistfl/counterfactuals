library("mlr3")
library("mlr3pipelines")
library("mlr3learners")
library("mlr3fairness")
library("randomForest")


data = tsk("adult_train")$data()[1:1000, ]
rf = randomForest(target ~ ., data = data)
# Create a predictor object
predictor = iml::Predictor$new(rf, type = "prob", data = data)
# Find counterfactuals for x_interest
cf_classif = CFClassif$new(predictor, protected = "race", n_generations = 10L)
cfactuals = cf_classif$find_counterfactuals(
    x_interest = data[150L, ], desired_class = "White", desired_prob = c(0.5, 1)
)

plot_counterfactuals(cfactuals, data)




