generate_law_school = function(n, race = NULL, sex = NULL, seed = 123L) {
    # read in coefficients
    attach(readRDS("paper/experiments/law_school/la_law_coef.rds"), pos = 2L, name = "coefs")
    on.exit(detach("coefs"))

    # Set seed 
    set.seed(seed)
    # Produce race. This has to be sampled in order to get the same seed.
    ll = gen_race_sex(race_sex, n)
    if (is.null(race)) {
        race = as.character(ll$race)
    }
    if (is.null(sex)) {
        sex  = as.integer(ll$sex)
    }
    a = binarize_a(race, sex)
    u = rnorm(n)
    gpa = rnorm(n, ugpa0 + eta_u_ugpa*u + a %*% eta_a_ugpa, sigma_g)
    lsat = rpois(n, exp(lsat0 + eta_u_lsat*u + eta_a_lsat*a))
    zfya = rnorm(eta_u_fya*u + a %*% eta_a_fya, 2) + fya0
    fya = rbinom(n, 1, exp(zfya) / (1+exp(zfya)))
    dem = rnorm(n)
    return(data.frame(race = factor(race), sex = factor(sex), gpa, lsat, dem, pass = factor(fya)))
}

gen_race_sex = function(race_sex, n) {
    # Sample sex
    sex = sample(c(1,2), n, prob = colSums(race_sex), replace = TRUE)
    # Sample race
    race = sapply(sex, function(s) {
        racep = race_sex[,s] / sum(race_sex[,s])
        sample(rownames(race_sex), 1, replace = TRUE, prob = racep)
    })
    list(race = race, sex = sex)
}

binarize_a = function(race, sex) {
    race = as.character(race)
    sex = as.integer(sex)
    # Create data frame
    a = data.frame(
        amerind = as.numeric(race == "Amerindian"),
        asian   = as.numeric(race == "Asian"),
        black   = as.numeric(race == "Black"),
        hisp    = as.numeric(race == "Hispanic"),
        mexican = as.numeric(race == "Mexican"),
        other   = as.numeric(race == "Other"),
        puerto  = as.numeric(race == "Puertorican"),
        white   = as.numeric(race == "White"),
        female  = as.numeric(sex == 1),
        male    = as.numeric(sex == 2)
    )
    as.matrix(a)
}

generate_cf_law_school_sex = function(data, seed = 123L) {
    sexes = levels(data$sex)
    X = generate_law_school(nrow(data), sex = 3 - as.integer(data$sex), seed = seed)
    levels(X$sex) = sexes
    return(X)
}

generate_cf_law_school_race = function(data, seed = 123L, target_race = NULL) {
    set.seed(seed)
    races = levels(data$race)
    race = sapply(data$race, function(ro) sample(setdiff(races, ro), 1))
    if (!is.null(target_race)) race = rep(target_race, nrow(data))
    X = generate_law_school(nrow(data), race = race, seed = seed)
    levels(X$race) = races
    return(X)
}
