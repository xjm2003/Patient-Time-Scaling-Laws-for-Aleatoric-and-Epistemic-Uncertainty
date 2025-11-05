from typing import Optional
import pandas as pd
import stan
def sim_lmm_data(
    seed=123,
    N=1000,
    T=10,
    beta=(1.0, -1.0, 2.0),
    sigma=1,
    Sigma_b=None,
    z=None
):
    rng = np.random.default_rng(seed)
    k = len(beta)
    if Sigma_b is None:
        Sigma_b = np.eye(k)
    M = N * T
    X = rng.normal(size=(M, k)).astype(float)
    if z is None:
        Z = X.copy()
    else:
        Z = z
    #Z = np.zeros_like(X)
    #Z[:, 0] = 1.0
    #group = np.repeat(np.arange(N), T)
    L = np.linalg.cholesky(Sigma_b)
    z_latent = rng.normal(size=(N, k))
    b = z_latent @ L.T
    random_part = (Z * b[group]).sum(axis=1)
    mu = X @ np.asarray(beta) + random_part
    y = rng.normal(mu, sigma)
    out = {"y": y, "group": group.astype(int), "mu": mu}
    for j in range(k):
        out[f"X{j+1}"] = X[:, j]
    for j in range(k):
        out[f"Z{j+1}"] = Z[:, j]
    df = pd.DataFrame(out)
    return df
stan_code = r"""
data {
  int<lower=1> N;
  int<lower=1> P;
  int<lower=1> K;
  vector[N] y;
  array[N] int<lower=1, upper=P> group;
  matrix[N, K] X;
  matrix[N, K] Z;
}
parameters {
  vector[K] beta;
  vector<lower=0>[K] tau_b;
  cholesky_factor_corr[K] L_Omega_b;
  matrix[K, P] z_b;
  real<lower=0> sigma;
}
transformed parameters {
  matrix[K, P] b = diag_pre_multiply(tau_b, L_Omega_b) * z_b;
}
model {
  beta ~ normal(0, 5);
  tau_b ~ exponential(1);
  L_Omega_b ~ lkj_corr_cholesky(2);
  to_vector(z_b) ~ normal(0, 1);
  sigma ~ exponential(1);
  matrix[N, K] B = (b[, group])';
  vector[N] eta = X * beta + rows_dot_product(Z, B);
  y ~ normal(eta, sigma);
}
generated quantities {
  corr_matrix[K] Omega_b = multiply_lower_tri_self_transpose(L_Omega_b);
  cov_matrix[K] Sigma_b = quad_form_diag(Omega_b, tau_b);
}
"""
def prepare_lmm_stan_data(
    df: pd.DataFrame,
    y_col: str = "y",
    group_col: str = "group",
    x_prefix: str = "X",
    z_prefix: Optional[str] = "Z",
    dropna: bool = True,
):
    if dropna:
        need_cols = [y_col, group_col] + [c for c in df.columns if c.startswith(x_prefix)]
        df = df.dropna(subset=need_cols).copy()
    y = df[y_col].to_numpy(dtype=float).reshape(-1)
    group_raw = df[group_col].to_numpy()
    group_codes, uniques = pd.factorize(group_raw, sort=True)
    group_1_based = (group_codes + 1).astype(int)
    P = int(len(uniques))
    x_cols = [c for c in df.columns if c.startswith(x_prefix)]
    X = df[x_cols].to_numpy(dtype=float)
    M_obs, K = X.shape
    Z = None
    z_cols = []
    if z_prefix is not None:
        z_cols = [c for c in df.columns if c.startswith(z_prefix)]
        if z_cols:
            Z = df[z_cols].to_numpy(dtype=float)
    if Z is None:
        Z = X
    data_stan = {
        "N": int(M_obs),
        "P": int(P),
        "K": int(K),
        "y": y.astype(float),
        "group": group_1_based.astype(int),
        "X": X.astype(float),
        "Z": Z.astype(float),
    }
    meta = {"group_levels": list(uniques), "x_cols": x_cols, "z_cols": z_cols if z_cols else None}
    return data_stan, meta
def fit_lmm_with_pystan(
    df: pd.DataFrame,
    y_col: str = "y",
    group_col: str = "group",
    x_prefix: str = "X",
    z_prefix: Optional[str] = "Z",
    num_chains: int = 1,
    num_warmup: int = 200,
    num_samples: int = 200,
):
    data_stan, meta = prepare_lmm_stan_data(
        df, y_col=y_col, group_col=group_col, x_prefix=x_prefix, z_prefix=z_prefix
    )
    posterior = stan.build(stan_code, data=data_stan)
    fit = posterior.sample(
        num_chains=num_chains,
        num_warmup=num_warmup,
        num_samples=num_samples
    )
    K = data_stan["K"]
    beta_mean = fit["beta"].mean(axis=1)
    sigma_mean = fit["sigma"].mean()
    return fit, {"beta_mean": beta_mean, "sigma_mean": sigma_mean,
                 "dims": (data_stan["N"], data_stan["P"], K), "meta": meta}
def reconstruct_b(fit):
    tau = fit["tau_b"]
    L   = fit["L_Omega_b"]
    z   = fit["z_b"]
    b_kpd = np.einsum("jid,ipd->jpd", L * tau[:, None, :], z)
    return b_kpd
def posterior_blocks(fit, i1):
    beta = fit["beta"]
    b = reconstruct_b(fit)
    bi = b[:, i1 - 1, :]
    covb  = np.cov(beta, rowvar=True, ddof=1)
    covbi = np.cov(bi,    rowvar=True, ddof=1)
    beta_c = beta - beta.mean(axis=1, keepdims=True)
    bi_c = bi - bi.mean(axis=1, keepdims=True)
    covbbi = (beta_c @ bi_c.T) / (beta.shape[1] - 1)
    return covb, covbi, covbbi
def pred_var_decomp(fit, df, i1, x_prefix="X", z_prefix="Z", group_col="group", include_noise=True):
    covb, covbi, covbbi = posterior_blocks(fit, i1)
    gi0 = i1 - 1
    sub = df[df[group_col] == gi0].iloc[0]
    x_cols = [c for c in df.columns if c.startswith(x_prefix)]
    z_cols = [c for c in df.columns if c.startswith(z_prefix)] or x_cols
    x_i = sub[x_cols].to_numpy(float)
    z_i = sub[z_cols].to_numpy(float)
    term_param = float(x_i @ covb  @ x_i)
    term_rand  = float(z_i @ covbi @ z_i)
    term_cross = float(2.0 * x_i @ covbbi @ z_i)
    var_mean   = term_param + term_rand + term_cross
    out = {"param": term_param, "random": term_rand, "cross": term_cross, "var_mean": var_mean}
    if include_noise:
        sigma2 = float((fit["sigma"]**2).mean())
        out.update({"noise": sigma2, "var_y": var_mean + sigma2})
    return out

if __name__ == "__main__":
    import os
    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    figs_dir = "figs"
    os.makedirs(figs_dir, exist_ok=True)

    num_chains  = 2
    num_warmup  = 400
    num_samples = 800
    base_seed   = 123
    R_repeats   = 10

    N_list = [50, 100, 200, 400]
    T_list = [3, 5, 10, 20]
    N_fixed_list = [50, 200]
    T_fixed_list = [3, 10]
    M_fixed_list = [1000, 2000, 4000]

    cache = {}
    Sigma_b = np.diag([10.0, 1e-8, 1e-8])


    def metrics_once(N, T, seed, i1=1):
        df = sim_lmm_data(seed=seed, N=N, T=T, Sigma_b=Sigma_b)
        fit, _ = fit_lmm_with_pystan(
            df, y_col="y", group_col="group", x_prefix="X", z_prefix="Z",
            num_chains=num_chains, num_warmup=num_warmup, num_samples=num_samples
        )
        covb, covbi, covbbi = posterior_blocks(fit, i1=i1)
        lamb  = float(np.linalg.eigvalsh(covb).max())
        lambi = float(np.linalg.eigvalsh(covbi).max())
        cross = pred_var_decomp(fit, df, i1=i1, include_noise=False)
        xcovbbiz = float(cross["cross"] / 2.0)
        return lamb, lambi, xcovbbiz

    def metrics_mean_se(N, T, R=R_repeats, i1=1):
        key = (int(N), int(T), int(R))
        if key in cache:
            return cache[key]
        valsb, valsbi, valsX = [], [], []
        for r in range(R):
            seed = base_seed + N * 100000 + T * 1000 + r
            lamb, lambi, xcovbbiz = metrics_once(N, T, seed, i1=i1)
            valsb.append(lamb); valsbi.append(lambi); valsX.append(xcovbbiz)
        def mean_se(a):
            a = np.asarray(a, float)
            m = a.mean()
            se = a.std(ddof=1) / math.sqrt(len(a)) if len(a) > 1 else 0.0
            return float(m), float(se)
        mb, seb   = mean_se(valsb)
        mbi, sebi = mean_se(valsbi)
        mX, seX   = mean_se(valsX)
        out = {"meanb": mb, "seb": seb, "meanbi": mbi, "sebi": sebi, "meanX": mX, "seX": seX, "M": N*T}
        cache[key] = out
        print(f"[N={N:4d}, T={T:3d}, M={N*T:6d}]  covb={mb:.6f}±{seb:.6f}  covbi={mbi:.6f}±{sebi:.6f}  x'covbbi z={mX:.6f}±{seX:.6f}")
        return out

    rows = []
    for N in N_list:
        for T in T_list:
            res = metrics_mean_se(N, T, R=R_repeats, i1=1)
            rows.append({
                "N": N, "T": T, "M": res["M"],
                "covb_mean": res["meanb"], "covb_se": res["seb"],
                "covbi_mean": res["meanbi"], "covbi_se": res["sebi"],
                "xcovbbiz_mean": res["meanX"], "xcovbbiz_se": res["seX"],
                "R": R_repeats, "chains": num_chains, "warmup": num_warmup, "samples": num_samples
            })
    res_df = pd.DataFrame(rows).sort_values(["N", "T"]).reset_index(drop=True)
    res_df.to_csv("results_summary.csv", index=False)

    def plot_with_err(x, y, yerr, label):
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=label)

    def plot_block_N_fixed(metric_key_mean, metric_key_se, title, outfile):
        plt.figure(figsize=(7,5))
        for N0 in N_fixed_list:
            xs, ys, es = [], [], []
            for T in T_list:
                q = res_df[(res_df["N"]==N0) & (res_df["T"]==T)].iloc[0]
                xs.append(T); ys.append(q[metric_key_mean]); es.append(q[metric_key_se])
            plot_with_err(xs, ys, es, f"N={N0}")
        plt.xlabel("T"); plt.ylabel(metric_key_mean)
        plt.title(f"{title} (N fixed)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, outfile), dpi=200)

    def plot_block_T_fixed(metric_key_mean, metric_key_se, title, outfile):
        plt.figure(figsize=(7,5))
        for T0 in T_fixed_list:
            xs, ys, es = [], [], []
            for N in N_list:
                q = res_df[(res_df["T"]==T0) & (res_df["N"]==N)].iloc[0]
                xs.append(N); ys.append(q[metric_key_mean]); es.append(q[metric_key_se])
            plot_with_err(xs, ys, es, f"T={T0}")
        plt.xlabel("N"); plt.ylabel(metric_key_mean)
        plt.title(f"{title} (T fixed)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, outfile), dpi=200)

    def factor_pairs(M):
        pairs = []
        for N in N_list:
            if M % N == 0:
                T = M // N
                if T in T_list:
                    pairs.append((N, T))
        return sorted(pairs)

    def plot_block_M_fixed(metric_key_mean, metric_key_se, title, outfile):
        plt.figure(figsize=(7,5))
        for M0 in M_fixed_list:
            pts = factor_pairs(M0)
            if not pts:
                continue
            xs, ys, es = [], [], []
            for (N,T) in pts:
                q = res_df[(res_df["N"]==N) & (res_df["T"]==T)].iloc[0]
                xs.append(N); ys.append(q[metric_key_mean]); es.append(q[metric_key_se])
            order = np.argsort(xs)
            xs = [xs[i] for i in order]; ys = [ys[i] for i in order]; es = [es[i] for i in order]
            plot_with_err(xs, ys, es, f"M={M0}")
        plt.xlabel("N  (with N·T constant)"); plt.ylabel(metric_key_mean)
        plt.title(f"{title} (M fixed)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, outfile), dpi=200)

    plot_block_N_fixed("covb_mean",   "covb_se",   "||covb||2",         "covb_vs_T_Nfixed_mean.png")
    plot_block_N_fixed("covbi_mean",  "covbi_se",  "||covbi||2",        "covbi_vs_T_Nfixed_mean.png")
    plot_block_N_fixed("xcovbbiz_mean","xcovbbiz_se","x'covbbi z",      "xcovbbiz_vs_T_Nfixed_mean.png")

    plot_block_T_fixed("covb_mean",   "covb_se",   "||covb||2",         "covb_vs_N_Tfixed_mean.png")
    plot_block_T_fixed("covbi_mean",  "covbi_se",  "||covbi||2",        "covbi_vs_N_Tfixed_mean.png")
    plot_block_T_fixed("xcovbbiz_mean","xcovbbiz_se","x'covbbi z",      "xcovbbiz_vs_N_Tfixed_mean.png")

    plot_block_M_fixed("covb_mean",   "covb_se",   "||covb||2",         "covb_vs_N_Mfixed_mean.png")
    plot_block_M_fixed("covbi_mean",  "covbi_se",  "||covbi||2",        "covbi_vs_N_Mfixed_mean.png")
    plot_block_M_fixed("xcovbbiz_mean","xcovbbiz_se","x'covbbi z",      "xcovbbiz_vs_N_Mfixed_mean.png")

    print("Saved results_summary.csv and figures to:", figs_dir)
