# Code taken from https://github.com/mkusner/counterfactual-fairness/blob/master/law_school_classifiers.R
library(dplyr)
library(caret)
library(rstan)

raw_data = read.csv("paper/experiments/law_school/law_data.csv")
law = dplyr::select(raw_data, race, sex, LSAT, UGPA, region_first, ZFYA, sander_index, first_pf) 
law = law[law$region_first != "PO",]
law$region_first = factor(law$region_first)

law$amerind = as.numeric(law$race == "Amerindian")
law$asian   = as.numeric(law$race == "Asian")
law$black   = as.numeric(law$race == "Black")
law$hisp    = as.numeric(law$race == "Hispanic")
law$mexican = as.numeric(law$race == "Mexican")
law$other   = as.numeric(law$race == "Other")
law$puerto  = as.numeric(law$race == "Puertorican")
law$white   = as.numeric(law$race == "White")
law$female    = as.numeric(law$sex == 1)
law$male      = as.numeric(law$sex == 2)

sense_cols = c("amerind", "asian", "black", "hisp", "mexican", "other", "puerto", "white", "male", "female")

set.seed(0)
trainIndex = createDataPartition(law$first_pf, p = .8, 
                                  list = FALSE, 
                                  times = 1)
lawTrain = law[trainIndex,]
lawTest  = law[-trainIndex,]

#n = nrow(df2)
n = nrow(lawTrain)
ne = nrow(lawTest)


lawTrain$LSAT = round(lawTrain$LSAT)
lawTest$LSAT = round(lawTest$LSAT)

# don't fit model transductively
# ------------------------------
law_stan_train = list(N = n, K = length(sense_cols), a = data.matrix(lawTrain[,sense_cols]), 
                          ugpa = lawTrain[,c("UGPA")], lsat = lawTrain[,c("LSAT")], zfya = lawTrain[,c("ZFYA")])


fit_law_train = stan(file = 'paper/experiments/law_school_train.stan', data = law_stan_train, iter = 2000, chains = 1, verbose = TRUE)
# Extract information
la_law_train = extract(fit_law_train, permuted = TRUE)

saveRDS(la_law_train, "vignettes/la_law_train.rds")


## Read saved extracted values
la_law_train = readRDS("paper/experiments/law_school/la_law_train.rds")
coefficients = list(
    ugpa0      = mean(la_law_train$ugpa0),
    eta_u_ugpa = mean(la_law_train$eta_u_ugpa),
    eta_a_ugpa = colMeans(la_law_train$eta_a_ugpa),
    lsat0      = mean(la_law_train$lsat0),
    eta_u_lsat = mean(la_law_train$eta_u_lsat),
    eta_a_lsat = colMeans(la_law_train$eta_a_lsat),
    fya0      = -1.0, # NA, estimating  here
    eta_u_fya = mean(la_law_train$eta_u_zfya),
    eta_a_fya = colMeans(la_law_train$eta_a_zfya),
    sigma_g = mean(la_law_train$sigma_g),
    race_sex = prop.table(table(lawTrain$race, lawTrain$sex))
)

saveRDS(coefficients, "paper/experiments/law_school/la_law_coef.rds")
