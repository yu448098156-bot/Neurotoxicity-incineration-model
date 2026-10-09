# -*- coding: utf-8 -*-
# 依赖：pandas numpy rdkit scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1) 固定的20个描述符
# =========================
FEATURES = [
    'MinAbsEStateIndex', 'MaxPartialCharge', 'BCUT2D_MWHI', 'BCUT2D_CHGHI',
    'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'AvgIpc', 'BalabanJ', 'Ipc', 'PEOE_VSA10',
    'SMR_VSA4', 'SMR_VSA6', 'SlogP_VSA2', 'TPSA', 'EState_VSA5', 'EState_VSA8',
    'VSA_EState2', 'VSA_EState7', 'VSA_EState9', 'fr_NH0'
]

def calculate_descriptors(mol):
    """计算固定的20个RDKit描述符"""
    if mol is None:
        return None
    d = {
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
    return d

# =========================
# 2) 读取与预处理
# =========================
def load_and_process_data(file_path):
    """读取CSV，解析SMILES，计算描述符；返回列：FEATURES + ['pLD50']"""
    data = pd.read_csv(file_path)
    if 'SMILES' not in data.columns or 'pLD50' not in data.columns:
        raise ValueError("输入文件必须包含列：'SMILES' 和 'pLD50'")

    valid_rows, invalid_smiles, missing_targets = [], [], []

    for idx, (smiles, target) in enumerate(zip(data['SMILES'], data['pLD50'])):
        smi = str(smiles).strip()
        if pd.isna(target):
            missing_targets.append(idx + 1)
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_smiles.append((idx + 1, smi))
            continue
        desc = calculate_descriptors(mol)
        desc['pLD50'] = float(target)
        valid_rows.append(desc)

    if invalid_smiles:
        print(f"⚠️ 以下 {len(invalid_smiles)} 个SMILES无法解析：")
        for row, smi in invalid_smiles[:20]:
            print(f"  - 行 {row}: {smi}")
        if len(invalid_smiles) > 20:
            print("  ...")

    if missing_targets:
        print(f"⚠️ 以下 {len(missing_targets)} 行 pLD50 缺失，已跳过（示例显示前10条）：{missing_targets[:10]}{' ...' if len(missing_targets)>10 else ''}")

    df = pd.DataFrame(valid_rows)
    cols = [c for c in FEATURES if c in df.columns] + ['pLD50']
    return df[cols]

train_df = load_and_process_data('training_set.csv')
test_df  = load_and_process_data('test_set.csv')

X_train = train_df.drop(columns=['pLD50']).values
y_train = train_df['pLD50'].values
X_test  = test_df.drop(columns=['pLD50']).values
y_test  = test_df['pLD50'].values

# =========================
# 3) 标准化（可选） & 训练
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

base_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
base_model.fit(X_train_scaled, y_train)
y_pred_test = base_model.predict(X_test_scaled)

# =========================
# 4) 指标计算
# =========================
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

metrics_train = compute_metrics(y_train, base_model.predict(X_train_scaled))
metrics_test  = compute_metrics(y_test,  y_pred_test)

def r2_adjusted(y_true, y_pred, n, p):
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

r2_adj_train = r2_adjusted(y_train, base_model.predict(X_train_scaled), len(y_train), X_train.shape[1])

# =========================
# 5) 正确的LOO（避免泄漏）
# =========================
def loo_predictions(X, y, model):
    """每折：对训练子集标准化 + clone模型拟合；返回与y同长度的LOO预测"""
    loo = LeaveOneOut()
    y_pred = np.zeros_like(y, dtype=float)
    for tr_idx, te_idx in loo.split(X):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr = y[tr_idx]
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        m = clone(model)
        m.fit(X_tr_s, y_tr)
        y_pred[te_idx] = m.predict(X_te_s)
    return y_pred

y_pred_loo = loo_predictions(X_train, y_train, base_model)

def q2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

q2_loo_value  = q2_score(y_train, y_pred_loo)
q2_test_value = q2_score(y_test,  y_pred_test)

# 适用域覆盖率（阈值 = k * RMSE_test）
k = 1.5
threshold = k * metrics_test['RMSE']
ad_train_coverage = np.mean(np.abs(y_train - base_model.predict(X_train_scaled)) < threshold) * 100
ad_test_coverage  = np.mean(np.abs(y_test  - y_pred_test) < threshold) * 100

print(f"Train -> RMSE: {metrics_train['RMSE']:.3f}, MAE: {metrics_train['MAE']:.3f}, R²: {metrics_train['R2']:.3f}")
print(f"Test  -> RMSE: {metrics_test['RMSE']:.3f}, MAE: {metrics_test['MAE']:.3f}, R²: {metrics_test['R2']:.3f}")
print(f"R² Adjusted (Train): {r2_adj_train:.3f}")
print(f"q² LOO: {q2_loo_value:.3f}")
print(f"q² Test: {q2_test_value:.3f}")
print(f"AD coverage (Train, |err|<{threshold:.3f}): {ad_train_coverage:.1f}%")
print(f"AD coverage (Test, |err|<{threshold:.3f}): {ad_test_coverage:.1f}%")

# =========================
# 6) 指标与逐样本预测导出CSV
# =========================
summary = {
    "R2_Train": [metrics_train['R2']],
    "R2_Test":  [metrics_test['R2']],
    "R2_Adjusted_Train": [r2_adj_train],
    "RMSE_Train": [metrics_train['RMSE']],
    "MAE_Train":  [metrics_train['MAE']],
    "RMSE_Test":  [metrics_test['RMSE']],
    "MAE_Test":   [metrics_test['MAE']],
    "q2_LOO":     [q2_loo_value],
    "q2_Test":    [q2_test_value],
    "AD_threshold": [threshold],
    "AD_coverage_Train_%": [ad_train_coverage],
    "AD_coverage_Test_%":  [ad_test_coverage],
}
summary_df = pd.DataFrame(summary)
summary_df.to_csv("metrics_summary.csv", index=False)
print("✅ 指标已保存：metrics_summary.csv")

per_sample_train = pd.DataFrame({
    "y_train_true": y_train,
    "y_train_pred_LOO": y_pred_loo,                           # LOO预测
    "y_train_pred_inbag": base_model.predict(X_train_scaled)  # 训练内预测（对比用）
})
per_sample_train.to_csv("predictions_train.csv", index=False)
print("✅ 训练集逐样本预测已保存：predictions_train.csv")

per_sample_test = pd.DataFrame({
    "y_test_true": y_test,
    "y_test_pred": y_pred_test
})
per_sample_test.to_csv("predictions_test.csv", index=False)
print("✅ 测试集逐样本预测已保存：predictions_test.csv")

# =========================
# 7) 可视化（合并：LOO+Test）
# =========================
sns.set_context("notebook", font_scale=1.2)
sns.set_style("whitegrid")

def plot_q2_combined_scatter(y_train_true, y_train_pred_loo, y_test_true, y_test_pred,
                             q2_loo_value, q2_test_value, title="q² LOO and q² Test Combined Scatter"):
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # 黑色实线边框
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.8)

    # 散点（训练：LOO；测试：独立集）
    plt.scatter(y_train_pred_loo, y_train_true, c='blue', alpha=0.55, s=90, edgecolor='black', label='Train (q² LOO)')
    plt.scatter(y_test_pred, y_test_true,   c='red',  alpha=0.55, s=90, edgecolor='black', label='Test (q² Test)')

    # 轴范围与理想线
    all_true = np.concatenate([y_train_true, y_test_true])
    all_pred = np.concatenate([y_train_pred_loo, y_test_pred])
    lo = min(all_true.min(), all_pred.min())
    hi = max(all_true.max(), all_pred.max())
    plt.plot([lo, hi], [lo, hi], 'k--', linewidth=1.8, label='Ideal Prediction')

    # 回归线（all_pred -> all_true）
    X_lr = all_pred.reshape(-1, 1)
    lr = LinearRegression().fit(X_lr, all_true)
    x_plot = np.linspace(lo, hi, 200).reshape(-1, 1)
    y_plot = lr.predict(x_plot)
    plt.plot(x_plot, y_plot, color='green', linestyle='-', linewidth=2.0, label='Regression Line')

    # 注释（加粗）
    plt.text(0.05, 0.95, f'q² LOO = {q2_loo_value:.3f}', transform=ax.transAxes,
             fontsize=13, color='blue', fontweight='bold', va='top')
    plt.text(0.05, 0.90, f'q² Test = {q2_test_value:.3f}', transform=ax.transAxes,
             fontsize=13, color='red',  fontweight='bold', va='top')

    plt.xlabel('Predicted pLD50', fontsize=13, fontweight='bold')
    plt.ylabel('Experimental pLD50', fontsize=13, fontweight='bold')
    plt.title(title, fontsize=15, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_q2_combined_scatter(
    y_train_true=y_train,
    y_train_pred_loo=y_pred_loo,
    y_test_true=y_test,
    y_test_pred=y_pred_test,
    q2_loo_value=q2_loo_value,
    q2_test_value=q2_test_value
)
