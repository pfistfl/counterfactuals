rm(list = ls())
library("mlr3")
library("mlr3pipelines")
library("mlr3learners")
library("mlr3fairness")
library("gower")
library("randomForest")
library("data.table")
library("dplyr")
library("ggplot2")
devtools::load_all()
source("paper/experiments/metrics.R")
SEED = 202201L
generations = 175L


## Fit random forest predictor
data = tsk("adult_train")$data()[1:1000, ]
# combine categories of protected attribute with few observations
data$race = factor(data$race, levels = c("Amer-Indian-Eskimo", "Asian-Pac-Islander", 
  "Black", "Other", "White"), labels = c("Other", "Other", "Black", "Other", "White"))
set.seed(SEED)
rf = randomForest(target ~ ., data = data)
# Create a predictor object
predictor = iml::Predictor$new(rf, type = "prob", data = data)
vars = predictor$data$feature.names
# Trainingdata with protected status
X_search = predictor$data$X
row_id = c(X_search[, "race"] == "White") 
X_search = data.table(X_search[row_id,])

# Generate counterfactuals for class shift from race black to white
# Multiple counterfactuals 
idxs = which(data$race == "Black")
x_interest = data[idxs[1],]

# Analyse a single observations
set.seed(SEED)
cf_classif = CFClassif$new(predictor, protected = "race", n_generations = generations, quiet = TRUE)
cfactuals = cf_classif$find_counterfactuals(
  x_interest = x_interest, desired_class = "White", desired_prob = c(0.5, 1)
)
paretoplot1 = plot_paretofront(cfactuals) 
paretoplot1 = paretoplot1 + 
  xlab(expression(o[valid])) + 
  ylab(expression(o[close])) +
  labs(color=expression(o[plaus])) 
# ggsave(filename = "paper/experiments/adult/paretofront_adult.pdf", plot = paretoplot1, width = 4, height = 2.5)

set.seed(SEED)
ptnse = plot_counterfactuals(cfactuals, data, attribute = "race")
ptnse
# ggsave(filename = "tsne_adult.pdf", width = 6, height = 3.5)

# Average change in prediction
mean(predictor$predict(cfactuals$data)[,1] - predictor$predict(data[150L,])[,1])
