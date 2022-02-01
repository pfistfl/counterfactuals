make_fitness_function_cf = function(predictor, predictor_protected, x_interest, pred_column, weights, k, fixed_features, param_set) {
  function(xdt) {
    # Add values of fixed_features just for prediction
    if (!is.null(fixed_features)) {
      xdt[, (fixed_features) := x_interest[, fixed_features, with = FALSE]]
    }
    xdt = xdt[, names(x_interest), with = FALSE]
    factor_cols = names(which(sapply(predictor$data$X, is.factor)))
    for (factor_col in factor_cols) {
      fact_col_pred = predictor$data$X[[factor_col]]
      value =  factor(xdt[[factor_col]], levels = levels(fact_col_pred), ordered = is.ordered(fact_col_pred))
      set(xdt, j = factor_col, value = value)
    }
    int_cols = names(which(sapply(predictor$data$X, is.integer)))
    if (length(int_cols) > 0L) {
      xdt[,(int_cols) := lapply(.SD, as.integer), .SDcols = int_cols]
    }
    ranges = param_set$upper - param_set$lower
    prob_prot = 1 - as.data.table(predictor_protected$predict_newdata(xdt))[[pred_column]]
    dist_x_interest = as.vector(StatMatch::gower.dist(x_interest, xdt, rngs = ranges, KR.corr = FALSE))
    dist_train = gower_topn(x = xdt, y = predictor$data$X, n = k)$distance
    if (!is.null(weights)) {
      dist_train = apply(dist_train, 2L, weighted.mean, w = weights)
    } else {
      dist_train = apply(dist_train, 2L, mean)
    }
    data.table(cbind(prob_prot, dist_x_interest, dist_train))
  }
}



plot_counterfactuals = function(cfactuals, data, attribute = NULL, extra_points = NULL, ...) {
  library("ggplot2")
  require_namespaces("Rtsne")
  setDT(data)
  data[, role := "data"]
  
  cdf = cfactuals$data[, role := "gen_cf"]
  idf = cfactuals$x_interest[, role := "x"]
  df = rbind(idf, cdf, data[, colnames(cdf), with = FALSE], extra_points[, colnames(cdf), with = FALSE])

  X = model.matrix( ~ ., data = df[, setdiff(colnames(df), c("role", "attribute")), with = FALSE])
  dups = duplicated(X)
  X = unique(X)
  X = Rtsne::normalize_input(X)
  rtdf = Rtsne::Rtsne(X, ...)$Y
  edf = data.table(cbind(data.frame(rtdf), df[which(!dups), "role", with = FALSE]))
  if (!is.null(attribute)) edf = cbind(edf,  df[which(!dups), attribute, with = FALSE])
  
  if (is.null(extra_points)) {
    points = c(16,18,15)
    scales = c(1.2, 3,5)
  } else {
    points = c(16,18,3,15)
    scales = c(1.2, 3, 5, 5)
  }
  ggplot(edf, aes(x = X1, y = X2, color = role, shape = role, size = role), alpha = .7) +
    geom_point(aes_string(color = eval(attribute)), alpha = .85) +
    geom_point(data=edf[role %in% "gen_cf",],
      pch = 23,
      size=4,
      stroke = .7,
      colour = "#333333") +
    theme_minimal() +
    scale_shape_manual(values = points) +
    scale_size_manual(values = 1.8*scales, guide = "none") +
    scale_colour_brewer(palette = "Set1") +
    geom_point(data=edf[role %in% extra_points$role,], pch = 3, stroke = 3.5, size=7, colour = "black") +
    theme(
      legend.title = element_blank(),
      axis.text.x=element_blank(),
      axis.text.y=element_blank(),
      axis.ticks=element_blank(),
      axis.title.x=element_blank(),
      axis.title.y=element_blank(),
      legend.position = "bottom",
      legend.spacing.x = unit(.1, 'cm')
    )
}

fit_prot_predictor = function(data, new_target) {
    require_namespaces(c("mlr3learners", "mlr3pipelines"))
    newtsk = convert_prot_task(data, new_target)
    lrn = lrn("classif.ranger", predict_type = "prob")
    prot_predictor = as_learner(ppls("robustify", task = newtsk, learner = lrn) %>>% lrn)
    prot_predictor$train(newtsk)
    return(prot_predictor)
}

convert_prot_task = function(data, new_target) {
    TaskClassif$new("pprot", data, new_target)
}


make_cf_train_pop_initializer =  function(ps, x_interest, max_changed, protected, desired_class, 
  predictor, fitness_function, mu) {
    function(param_set, n) { 
      make_f_design = function(X, flex_cols, x_interest, sdevs_num_feats) {
      function(ps, n) {
        id_desired = c(predictor$data$X[, protected, with = FALSE] == desired_class)
        X_sub = predictor$data$X[id_desired,]
        fitness_vals = fitness_function(X_sub)
        if (nrow(X_sub) > mu) {
          is_dom = bbotk::is_dominated(t(fitness_vals))
          X_nondom = unique(X_sub[!is_dom])
          if (nrow(X_nondom) > mu) {
            X_nondom = X_nondom[sample.int(nrow(X_nondom), mu)]
          } else {
            # sample dominated training samples
            X_dom = X_sub[sample.int(nrow(unique(X_sub[is_dom])), mu - nrow(X_nondom))]
            X_nondom = rbindlist(list(X_nondom, X_dom))
          }
        } else {
          X_nondom = X_sub
        }
        
        param_set = make_param_set(X, lower = NULL, upper = NULL)
        mydesign = SamplerUnif$new(param_set)$sample(n)
        mydesign$data[, protected] = desired_class
        mydesign$data = reset_columns(mydesign$data, p_use_orig = 0.5, max_changed = 1e15, x_interest = x_interest)
        mydesign$data[sample.int(nrow(mydesign$data), nrow(X_nondom))] = X_nondom
        mydesign
      }
    }
    f_design = make_f_design(predictor$data$X, flex_cols, x_interest, sdevs_num_feats)
  
  my_design = f_design(param_set, n)
  x_interest_reorderd = x_interest[, names(my_design$data), with = FALSE]
  
  
  factor_cols = names(x_interest_reorderd)[sapply(x_interest_reorderd, is.factor)]
  if (length(factor_cols) > 0L) {
    x_interest_reorderd[, (factor_cols) := lapply(.SD, as.character), .SDcols = factor_cols]
  }
  
  # If more changes than allowed, randomly reset some features such that constraint holds
  if (!is.null(max_changed)) {
    for (i in seq_len(nrow(my_design$data))) {
      n_changes = count_changes(my_design$data[i, ], x_interest_reorderd)
      if (n_changes > max_changed) {
        idx_diff = which(my_design$data[i, ] != x_interest_reorderd)
        idx_reset = sample(idx_diff, size = n_changes - max_changed)
        set(my_design$data, i, j = idx_reset, value = x_interest_reorderd[, ..idx_reset])
      }
    }
  }
  
  my_design
}
}


