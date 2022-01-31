rm(list = ls())
library("mlr3")
library("mlr3pipelines")
library("mlr3learners")
library("mlr3fairness")
library("randomForest")
library("data.table")
library("dplyr")
devtools::load_all()
source("paper/experiments/metrics.R")
SEED = 202201L
generations = 30L


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
gen_cf_classif = function(data, idx, vars, return_cfactuals = FALSE) {
  cf_classif = CFClassif$new(predictor, protected = "race", n_generations = generations, quiet = TRUE) #TODO: use epsilon = 0
  cfactuals = cf_classif$find_counterfactuals(
    x_interest = data[idx, ], desired_class = "White", desired_prob = c(0.5, 1)
  )
  cfactuals$subset_to_valid()
  
  # Whatif 
  nnidx = gower_topn(data[idx, ], X_search, n = 1L)$index
  whatif_cfactual = X_search[nnidx]

  out = rbind(
    cfactuals$x_interest[, role := "x_interest"],
    whatif_cfactual[, role := "nearest_train"],
    cfactuals$data[, role := "gen_cf"]
  )
  if (return_cfactuals) return(cfactuals)
  probs = predictor$predict(out)
  out[, prob0 := probs[,1]]
  out[, prob1 := probs[,2]]
  out$idx = idx
  return(out)
}

set.seed(SEED)
res = map(idxs, function(row) {gen_cf_classif(data, row, vars)})
dt = rbindlist(res)
fwrite(dt, file = "paper/experiments/adult/cf_exp_b_w.csv")

# Compute distance between different points 
# dt = fread(file = "paper/experiments/adult/cf_exp_b_w.csv")
xint  = dt[role == "x_interest", ]
cfobs = dt[role == "nearest_train",]
dtg = as.data.table(data)
# fac_cols = names(which(sapply(dtg, is.factor)))
# if (length(fac_cols) > 0L) {
#   dtg[,(fac_cols) := lapply(.SD, as.character), .SDcols = fac_cols]
# }
dint = rbindlist(map(seq_len(nrow(xint)), function(i) {
  xc = xint[i,]
  # Distance of true counterfactual to counterfactuals --> unknown
  d_int_true = NA
  # Mean distance of counterfactuals to x_interest
  dx = dt[, gow := gower(xc, .SD, vars), .SDcols = vars][idx == xc$idx,]
  d_int_gen  = mean(dx[role == "gen_cf" & sex == xc$sex,]$gow)
  # Distance of closest White training data point to x_interest
  d_int_nn = dx[role == "nearest_train",]$gow
  # Mean distance of all other datapoints to x_interest
  d_int_rnd  = mean(dtg[, gow := gower(xc, .SD, vars), .SDcols = vars]$gow)

  # Compare true to others --> unknown
  d_true_gen = NA
  return(list(d_int_true, d_int_gen, d_int_nn, d_int_rnd, d_true_gen, xc$idx))
}))

colnames(dint) = c("d(x, xdash)", "d(x, xstar)", "d(x, xnn)", "d(xdash, xrnd)", "d(xdash, xstar)", "idx")
tab = data.table(t(round(colMeans(dint), 3)))
tab$dataset = "adult"
knitr::kable(tab[, c(6,1:5)], format = "latex")


# Compute fairness
source = dt[role == "x_interest", c("prob0", "idx")]
mdt = merge(source, dt[role == "gen_cf", c("prob0", "idx")], by = "idx")

out = mdt[, .(gen = mean(prob0.x - prob0.y)), by = idx]
out$nntrain = source$prob0 - dt[role == "nearest_train",]$prob0

library(ggplot2)
dout = melt(out, id.vars = "idx")[, value := abs(value)]
p = ggplot(dout) +
  geom_violin(aes(x = variable, y = value)) +
  geom_point(aes(x = variable, y = value)) +
  geom_path(aes(x = variable, y = value), color = "darkgrey") +
  theme_bw() +
  xlab("") +
  ylab("x - x'") +
  scale_x_discrete(labels = c("Generated", "Nearest"))

ggsave("paper/experiments/adult/comp_adult_icuf.pdf", p, width = 4, height = 4)

colMeans(abs(out))


# Analyse a single observations
set.seed(SEED)
cfactuals = gen_cf_classif(data, idxs[1], vars, return_cfactuals = TRUE)
cfactuals$subset_to_valid()

ptnse = plot_counterfactuals(cfactuals, data, attribute = "race")
ggsave(filename = "paper/experiments/adult/tsne_adult.pdf", width = 6, height = 3.5)

# Average change in prediction
mean(predictor$predict(cfactuals$data)[,1] - predictor$predict(data[150L,])[,1])
