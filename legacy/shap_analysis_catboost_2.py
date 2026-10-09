import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings("ignore")

# ==== 1. 分子描述符计算函数 ====
def calculate_descriptors(mol):
    try:
        return {
            'MinAbsEStateIndex': Descriptors.MinAbsEStateIndex(mol),
            'MaxPartialCharge': Descriptors.MaxPartialCharge(mol),
            'BCUT2D_MWHI': Descriptors.BCUT2D_MWHI(mol),
            'BCUT2D_CHGHI': Descriptors.BCUT2D_CHGHI(mol),
            'BCUT2D_LOGPHI': Descriptors.BCUT2D_LOGPHI(mol),
            'BCUT2D_LOGPLOW': Descriptors.BCUT2D_LOGPLOW(mol),
            'AvgIpc': Descriptors.AvgIpc(mol),
            'BalabanJ': Descriptors.BalabanJ(mol),
            'Ipc': Descriptors.Ipc(mol),
            'PEOE_VSA10': Descriptors.PEOE_VSA10(mol),
            'SMR_VSA4': Descriptors.SMR_VSA4(mol),
            'SMR_VSA6': Descriptors.SMR_VSA6(mol),
            'SlogP_VSA2': Descriptors.SlogP_VSA2(mol),
            'TPSA': Descriptors.TPSA(mol),
            'EState_VSA5': Descriptors.EState_VSA5(mol),
            'EState_VSA8': Descriptors.EState_VSA8(mol),
            'VSA_EState2': Descriptors.VSA_EState2(mol),
            'VSA_EState7': Descriptors.VSA_EState7(mol),
            'VSA_EState9': Descriptors.VSA_EState9(mol),
            'fr_NH0': Descriptors.fr_NH0(mol)
        }
    except Exception:
        return None

# ==== 2. 数据读取与描述符计算 ====
print("Loading data and calculating descriptors...")
df = pd.read_csv("training_set.csv")
df = df.dropna(subset=["SMILES", "pLD50"])

descriptors = []
valid_indices = []

for i, smi in enumerate(df["SMILES"]):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        desc = calculate_descriptors(mol)
        if desc is not None:
            descriptors.append(desc)
            valid_indices.append(i)

X = pd.DataFrame(descriptors)
y = df.loc[valid_indices, "pLD50"].values

# 若有全零/常数列，后续相关性计算要注意；此处不强制删除，留到相关性处处理

# ==== 3. 特征标准化 & 模型训练 ====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Training CatBoost model...")
model = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=8, verbose=0, random_seed=42)
model.fit(X_scaled, y)

# ==== 4. SHAP 计算 ====
print("Calculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled)  # (n_samples, n_features) for regression

# ==== 5. 生成 SHAP 汇总 CSV ====
print("Summarizing SHAP importances to CSV...")

# 计算指标
shap_vals = np.array(shap_values)  # (N, F)
feature_names = X.columns.tolist()

mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
mean_shap = np.mean(shap_vals, axis=0)

# 与“原始特征值”（未缩放）之间的方向性（Spearman 相关）
spearman_corr = []
spearman_p = []
direction = []
for j, fname in enumerate(feature_names):
    xj = X[fname].values
    # 若该特征方差为0，相关性定义为NaN/无方向
    if np.nanstd(xj) < 1e-12 or np.nanstd(shap_vals[:, j]) < 1e-12:
        spearman_corr.append(np.nan)
        spearman_p.append(np.nan)
        direction.append("NA")
    else:
        r, p = spearmanr(xj, shap_vals[:, j], nan_policy='omit')
        spearman_corr.append(r)
        spearman_p.append(p)
        if np.isnan(r):
            direction.append("NA")
        else:
            direction.append("positive" if r > 0 else "negative")

# 原始特征分布统计
value_mean = X.mean(axis=0).values
value_std = X.std(axis=0, ddof=0).values

summary_df = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap,
    "mean_shap": mean_shap,
    "spearman_corr_feature_vs_shap": spearman_corr,
    "spearman_pvalue": spearman_p,
    "direction": direction,
    "value_mean": value_mean,
    "value_std": value_std
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

# 增加排名列
summary_df.insert(0, "rank", np.arange(1, len(summary_df) + 1))

# 保存 CSV
summary_csv_path = "shap_summary.csv"
summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

# ==== 6. （可选）SHAP 柱状图（保留你的原图）====
print("Generating SHAP feature importance bar plot...")
plt.figure()
shap.summary_plot(shap_vals, X, plot_type="bar", max_display=20, show=False)
plt.savefig("shap_feature_importance_bar_plot.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Done. SHAP summary saved to '{summary_csv_path}', and bar plot saved to 'shap_feature_importance_bar_plot.png'.")

