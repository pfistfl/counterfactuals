library("data.table")
library("mlr3misc")

gower = function(x, y, vars) {
    y = y[, vars, with = FALSE]
    x = x[, vars, with = FALSE]
    gower_dist(x,y)
}

l0 = function(x, y, vars) {
    checkmate::assert(nrow(x) == 1L)
    y = y[, vars, with = FALSE]
    x = x[, vars, with = FALSE]
    x = rbindlist(map(seq_len(nrow(y)), function(i) x))
}


d_true_gen = function(x, fun = l0, vars) {
    fun(
        x[role == "true_cf",],
        x[role == "gen_cf", ],
        vars
    )
}

d_int_gen = function(x, fun = l0, vars) {
    fun(
        x[role == "x_interest",],
        x[role == "gen_cf", ],
        vars
    )
}

d_true_int = function(x, fun = l0, vars) {
    fun(
        x[role == "true_cf",],
        x[role == "x_interest", ],
        vars
    )
}

d_rnd_int = function(x, fun = l0, vars) {
    fun(
        x[role == "x_interest", ],
        dtg,
        vars
    )
}

vars_law_school = c("lsat", "gpa", "dem")
vars_adult = c("age", "capital_gain", "capital_loss", "education", "education_num", "hours_per_week", "martial_status", "occupation", "relationship","workclass")

find_nn = function(xint, data, class = "White", protected = "race", vars) {
    if (!is.data.table(data)) setDT(data)
    y = data[data[[protected]] == class,]
    dsts = gower::gower_dist(xint[, vars, with = FALSE], y[, vars, with = FALSE])
    y[which.min(dsts)]
}