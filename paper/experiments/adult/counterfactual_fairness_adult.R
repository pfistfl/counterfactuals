library("mlr3")
library("mlr3pipelines")
library("mlr3learners")
library("mlr3fairness")
library("randomForest")

set.seed(1234)
# setup random forest
data = tsk("adult_train")$data()[1:1000, ]
data$race = factor(data$race, levels = c("Amer-Indian-Eskimo", "Asian-Pac-Islander", 
  "Black", "Other", "White"), labels = c("Other", "Other", "Black", "Other", "White"))
rf = randomForest(target ~ ., data = data)
# Create a predictor object
predictor = iml::Predictor$new(rf, type = "prob", data = data)

# Find counterfactuals for x_interest
x_interest = data[150L, ]
cf_classif = CFClassif$new(predictor, protected = "race", n_generations = 10L, epsilon = 0)
cfactuals = cf_classif$find_counterfactuals(
  x_interest = x_interest, desired_class = "White", desired_prob = c(0.5, 1)
)
cfactuals$data

# Average change in prediction
mean(predictor$predict(cfactuals$data)[,1] - predictor$predict(x_interest)[,1])

# TSNE plot
pltt = plot_counterfactuals(cfactuals, data, attribute = "race")
ggsave(pltt, filename = "tsne_adult.pdf", width = 6, height = 4)

