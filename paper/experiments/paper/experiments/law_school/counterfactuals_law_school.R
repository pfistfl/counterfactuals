library("mlr3")
library("mlr3pipelines")
library("mlr3learners")
library("mlr3fairness")
library("randomForest")
library("data.table")
devtools::load_all()
source("paper/experiments/law_school/generate_law_school.R")
source("paper/experiments/metrics.R")
SEED = 4321L

# "Real" counterfactuals
x = generate_law_school(1000, seed = SEED)
xs = generate_cf_law_school_sex(x, seed = SEED)
xr = generate_cf_law_school_race(x, seed = SEED, target_race = "White")

table(x$pass)
# Sex does not seem to have a great effect in this example!
table(x$pass, xs$pass) 
# Race does seem to have an effect in this example!
table(x$pass, xr$pass)


## Fit the predictor
data = data.table(x)
fwrite(data, file = "paper/experiments/law_school/gen_data.csv")
rf = randomForest(pass ~ ., data = data)
# Create a predictor object
predictor = iml::Predictor$new(rf, type = "prob", data = data)

# Compute changes under true counterfactual
mean(predictor$predict(data.table(xr))[,1] - predictor$predict(data.table(x))[,1])
mean(predictor$predict(data.table(xs))[,1] - predictor$predict(data.table(x))[,1])


# Example: Find counterfactuals for x_interest
# cf_classif = CFClassif$new(predictor, protected = "race", n_generations = 10L)
# cfactuals = cf_classif$find_counterfactuals(
#     x_interest = data[150L, ], desired_class = "White", desired_prob = c(0.5, 1)
# )



# Experiment: Race Black -> White  (This runs for multiple minutes)
set.seed(SEED)
vars = vars_law_school
idxs = which(data$race == "Black")
gen_cf_classif = function(data, xr, idx, vars) {
    cf_classif = CFClassif$new(predictor, protected = "race", n_generations = 30L)
    cfactuals = cf_classif$find_counterfactuals(
        x_interest = data[idx, ], desired_class = "White", desired_prob = c(0.5, 1)
    )
    xri = data.table(xr)[idx, colnames(cfactuals$data), with = FALSE]
    out = rbind(
        cfactuals$x_interest[, role := "x_interest"],
        cfactuals$data[, role := "gen_cf"],
        xri[, role := "true_cf"],
        find_nn(cfactuals$x_interest, data, "White", "race", vars)[, role := "x_nearest"][, colnames(xri), with = FALSE]
    )
    probs = predictor$predict(out)
    out[, prob0 := probs[,1]]
    out[, prob1 := probs[,2]]
    out$idx = idx
    return(out)
}

res = map(idxs, function(row) {gen_cf_classif(data, xr, row, vars)})
dt = rbindlist(res)
fwrite(dt, file = "paper/experiments/law_school/cf_exp_b_w.csv")



vars = vars_law_school
dt = fread("paper/experiments/law_school/cf_exp_b_w.csv")
dtg = fread("paper/experiments/law_school/gen_data.csv")

# Compute distance between different points 
xint  = dt[role == "x_interest", ]
xtrue = dt[role == "true_cf", ]
dint = rbindlist(map(seq_len(nrow(xint)), function(i) {
    xc = xint[i,]
    # Compare interest to others
    dx = dt[, gow := gower(xc, .SD, vars), .SDcols = vars][idx == xc$idx,]
    d_int_true = mean(dx[role == "true_cf",]$gow)
    d_int_gen  = mean(dx[role == "gen_cf" & sex == xc$sex,]$gow)
    d_int_rnd  = mean(dtg[, gow := gower(xc, .SD, vars), .SDcols = vars]$gow)
    # Compare true to others
    xtr = xtrue[i,]
    dtr = dt[, gow := gower(xtr, .SD, vars), .SDcols = vars][idx == xc$idx,]
    d_true_gen  = mean(dtr[role == "gen_cf" & sex == xtr$sex,]$gow)
    return(list(d_int_true, d_int_gen, d_int_rnd, d_true_gen, xc$idx))
}))
colnames(dint) = c("d(x, xdagger)", "d(x, xstar)", "d(xdagger, xrnd)", "d(xdagger, xstar)", "idx")
tab = data.table(t(round(colMeans(dint), 3)))
tab$dataset = "law school"
knitr::kable(tab[, c(6,1:4)], format = "latex")


# Compute fairness
source = dt[role == "x_interest", c("prob0", "idx")]
mdt = merge(source, dt[role == "gen_cf", c("prob0", "idx")], by = "idx")

out = mdt[, .(gen = mean(prob0.x - prob0.y)), by = idx]
out$true = source$prob0 - dt[role == "true_cf",]$prob0

library(ggplot2)
dout = melt(out, id.vars = "idx")[, value := abs(value)]
p = ggplot(dout) + 
  geom_violin(aes(x = variable, y = value)) + 
  geom_point(aes(x = variable, y = value)) +
  geom_path(aes(x = variable, y = value, group = idx), color = "darkgrey") +
  theme_bw() +
  xlab("") + 
  ylab("x - x'") +
  scale_x_discrete(labels = c("Generated", "True"))

ggsave("comp_icuf.pdf", p)

colMeans(abs(out))


xtra = data.table(xs[150,])[, role := "true_counterfactual"]
plot_counterfactuals(cfactuals, data, xtra)

# Average change in prediction
mean(predictor$predict(cfactuals$data)[,1] - predictor$predict(data[150L,])[,1])



