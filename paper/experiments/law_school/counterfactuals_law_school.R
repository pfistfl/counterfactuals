library("mlr3")
library("mlr3pipelines")
library("mlr3learners")
library("mlr3fairness")
library("randomForest")
library("data.table")
library("ggplot2")
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


## Fit the predictor (and save data)
data = data.table(x)
fwrite(data, file = "paper/experiments/law_school/gen_data.csv")
rf = randomForest(pass ~ ., data = data)
# Create a predictor object
predictor = iml::Predictor$new(rf, type = "prob", data = data)

# Compute changes under true counterfactual
mean(predictor$predict(data.table(xr))[,1] - predictor$predict(data.table(x))[,1])
mean(predictor$predict(data.table(xs))[,1] - predictor$predict(data.table(x))[,1])


# Figure 2: Generate counterfactuals for a single instance
# Example: Find counterfactuals for x_interest (Black -> White)
set.seed(SEED)
id = 85L 
cf_classif = CFClassif$new(predictor, protected = "race", n_generations = 50L, fixed_features = "sex")
cfactuals = cf_classif$find_counterfactuals(
    x_interest = data[id, ], desired_class = "White", desired_prob = c(0.5, 1)
)
cfactuals$subset_to_valid()
xtra = data.table(xr[id,])[, role := "true_cf"]
set.seed(SEED)
p = plot_counterfactuals(cfactuals, data, extra_points = xtra, attribute = "race", perplexity = 100)
p
ggsave("paper/experiments/law_school/tsne_bw.pdf", p, width = 6.2, height = 5.8, scale = 1)


# Experiment: Generate counterfactuals for multiple instances:
# Race Black -> White  (This runs for multiple minutes)
set.seed(SEED)
vars = vars_law_school
idxs = which(data$race == "Black")
gen_cf_classif = function(data, xr, idx, vars) {
    cf_classif = CFClassif$new(predictor, protected = "race", n_generations = 30L)
    cfactuals = cf_classif$find_counterfactuals(
        x_interest = data[idx, ], desired_class = "White", desired_prob = c(0.5, 1)
    )
    cfactuals$subset_to_valid()
    xri = data.table(xr)[idx, colnames(cfactuals$data), with = FALSE]
    out = rbind(
        cfactuals$x_interest[, role := "x_interest"],
        cfactuals$data[, role := "gen_cf"],
        xri[, role := "true_cf"],
        copy(cfactuals$x_interest)[, role := "a_flipped"][, race := "White"],
        find_nn(cfactuals$x_interest, data, "White", "race", vars)[, role := "x_nearest"][, colnames(xri), with = FALSE]
    )
    probs = predictor$predict(out)
    out[, prob0 := probs[,1]]
    out[, prob1 := probs[,2]]
    out$idx = idx
    return(out)
}

# Comparison to NN baseline and flipping baseline.

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
    dx = dt[, gow := t(gower(xc, .SD, vars)), .SDcols = vars][idx == xc$idx,]
    d_int_true = mean(dx[role == "true_cf",]$gow)
    d_int_gen  = mean(dx[role == "gen_cf" & sex == xc$sex,]$gow)
    d_int_near = mean(dx[role == "x_nearest",]$gow)
    # d_int_flip = mean(dx[role == "a_flipped",]$gow)
    d_int_rnd  = mean(dtg[, gow := t(gower(xc, .SD, vars)), .SDcols = vars]$gow)

    # Compare true to others
    xtr = xtrue[i,]
    dtr = dt[, gow := t(gower(xtr, .SD, vars)), .SDcols = vars][idx == xc$idx,]
    d_true_gen  = min(dtr[role == "gen_cf" & sex == xtr$sex,]$gow)
    return(list(d_int_true, d_int_gen, d_int_rnd, d_true_gen, d_int_near, xc$idx))
}))
colnames(dint) = c("d(x, xdagger)", "d(x, xstar)", "d(x, xrnd)", "d(xdagger, xstar)", "d(x, xnn)", "idx")
tab = data.table(t(round(colMeans(dint), 3)))
tab$dataset = "law school"
knitr::kable(tab[, c(1,2,3,5)], format = "latex")


# Figure 5: Fairness violin plot
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
  ylab("icuf") +
  scale_x_discrete(labels = c("generated", "true")) +
  theme(
      axis.text.x = element_text(size = 14),
      axis.text.y = element_text(size = 14),
      axis.title = element_text(size = 16)
  )


ggsave("comp_icuf.pdf", p, scale = .7)

p = ggplot(out) + 
  geom_point(aes(x = gen, y = true)) +
  theme_bw() +
  xlab("icuf(true)") + 
  ylab("icuf(generated)'") +
  geom_abline(slope = 1, color = "darkblue") +
  scale_x_discrete(labels = c("gen_cf", "true_cf"))
p
ggsave("comp_icuf_2.pdf", p, scale = .7)

 # Reported gcuf: 
colMeans(abs(out))

# Average change in prediction
mean(predictor$predict(cfactuals$data)[,1] - predictor$predict(data[150L,])[,1])

# Compare predictions between baselines
dd = dcast(dt, idx ~ role, value.var = "prob0", fun.aggregate = mean)
diffs = dd[, lapply(.SD, function(x) abs(x - x_interest)), .SDcols = colnames(dd)[c(2,3,4,6)]]
colnames(diff) = c("d_fl")

knitr::kable(diffs[, lapply(.SD, mean)], format = "latex", digits = 3)
knitr::kable(diffs[, lapply(.SD, function(x) {x = sd(x) / (length(x) - 1L)})], format = "latex", digits = 3)


