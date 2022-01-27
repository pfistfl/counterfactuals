generate_law_school = function(n) {
    u = rnorm(1)

    ugpa =
    lsat = 
    zfya = 
    
  ugpa0      ~ normal(0, 1);
  eta_u_ugpa ~ normal(0, 1);
  lsat0     ~ normal(0, 1);
  eta_u_lsat ~ normal(0, 1);
  eta_u_zfya ~ normal(0, 1);

  eta_a_ugpa ~ normal(zero_K, one_K);
  eta_a_lsat ~ normal(zero_K, one_K);
  eta_a_zfya ~ normal(zero_K, one_K);

  sigma_g_Sq ~ inv_gamma(1, 1);

  // have data about these
  ugpa ~ normal(ugpa0 + eta_u_ugpa * u + a * eta_a_ugpa, sigma_g);
  lsat ~ poisson(exp(lsat0 + eta_u_lsat * u + a * eta_a_lsat));
  zfya ~ normal(eta_u_zfya * u + a * eta_a_zfya, 1);
}