# -*- coding: utf-8 -*-
"""
- 避免 CatBoost 与 NumPy 二进制不兼容导致的导入报错：若导入失败自动回退到 HistGradientBoostingRegressor
- 全局字体使用 Arial，并全部加粗、字号加大
- 坐标轴外框与刻度线加粗，网格线删除
- 输出为 SVG 矢量图，字体样式保持稳定
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors

# ====== CatBoost 可选导入（失败则回退） ======
USE_CATBOOST = True
try:
    import catboost as cb
except Exception as e:
    USE_CATBOOST = False
    FALLBACK_IMPORT_ERROR = e
    from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib as mpl
import matplotlib.pyplot as plt

# ====== 全局 Matplotlib 样式（Arial + 加粗 + 加大；外框/刻度加粗） ======
mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 2.0,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
    "xtick.minor.width": 2.0,
    "ytick.minor.width": 2.0,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "black",

    # 关键：SVG 中将字体转为路径，保证字体样式、大小和加粗效果不变
    "svg.fonttype": "path"
})

# =============== 分子描述符 ===============
def calculate_descriptors(mol):
    if mol is None:
        return None
    descriptors = {
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
    return descriptors

# =============== 数据加载 ===============
def load_and_process_data(file_path):
    data = pd.read_csv(file_path)
    valid_rows, invalid_smiles = [], []

    for idx, smiles in enumerate(data['SMILES']):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_smiles.append((idx, smiles))
            continue

        d = calculate_descriptors(mol)
        d['pLD50'] = data['pLD50'][idx]
        valid_rows.append(d)

    if invalid_smiles:
        print(f"以下 {len(invalid_smiles)} 个SMILES无法解析：")
        for idx, s in invalid_smiles:
            print(f"  行 {idx+1}: {s}")

    return pd.DataFrame(valid_rows)

# =============== 读取训练/测试集 ===============
train_data = load_and_process_data('training_set.csv')
test_data  = load_and_process_data('test_set.csv')

X_train = train_data.drop(columns=['pLD50'])
y_train = train_data['pLD50'].values
X_test  = test_data.drop(columns=['pLD50'])
y_test  = test_data['pLD50'].values

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# =============== 构建模型（CatBoost优先，失败则回退） ===============
def build_model():
    if USE_CATBOOST:
        return cb.CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function='RMSE',
            random_seed=42,
            verbose=False
        )
    else:
        print("CatBoost 导入失败，自动回退到 HistGradientBoostingRegressor。")
        print("导入失败原因：", repr(FALLBACK_IMPORT_ERROR))
        return HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.05,
            max_depth=None,
            l2_regularization=1e-3,
            random_state=42
        )

model = build_model()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# =============== 评估指标 ===============
mse_test = mean_squared_error(y_test, y_pred)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_pred)

def q2_test(y_true, y_pred):
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_tot)

q2_test_value = q2_test(y_test, y_pred)

# =============== 绘图（Arial + 粗体 + 去网格 + 外框/刻度加粗） ===============
def plot_error_colormap_with_fit(y_true, y_pred, rmse, mae, q2,
                                 out_file="prediction_vs_experimental_error_colormap.svg"):
    errors = np.abs(y_true - y_pred)

    # 拟合回归线
    coeffs = np.polyfit(y_true, y_pred, 1)
    poly_eq = np.poly1d(coeffs)
    y_fit = poly_eq(y_true)

    # 为了让线更平滑，按 x 排序
    sort_idx = np.argsort(y_true)
    y_true_sorted = y_true[sort_idx]
    y_fit_sorted = y_fit[sort_idx]

    # 简单预测带（基于残差标准差）
    residuals = y_pred - y_true
    std = np.std(residuals)
    upper = y_fit_sorted + 1.96 * std
    lower = y_fit_sorted - 1.96 * std

    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(
        y_true, y_pred,
        c=errors,
        cmap="coolwarm",
        s=110,
        edgecolor="black",
        linewidth=1.2,
        alpha=0.8
    )

    ax.plot(
        y_true_sorted, y_fit_sorted,
        color="blue",
        linewidth=2.2,
        label="Regression Line"
    )
    ax.plot(
        y_true_sorted, upper,
        "b--",
        alpha=0.8,
        linewidth=2.0,
        label="95% Prediction Band"
    )
    ax.plot(
        y_true_sorted, lower,
        "b--",
        alpha=0.8,
        linewidth=2.0
    )

    # 色条
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Absolute Error", fontsize=14, fontweight="bold")
    cbar.outline.set_linewidth(2.0)
    cbar.ax.tick_params(width=2.0, length=6, labelsize=12)
    for t in cbar.ax.get_yticklabels():
        t.set_fontweight("bold")

    # 理想线
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        linewidth=2.0,
        label="Ideal Prediction"
    )

    # 标签与标题
    ax.set_xlabel("Experimental pLD50", fontsize=16, fontweight="bold")
    ax.set_ylabel("Predicted pLD50", fontsize=16, fontweight="bold")
    ax.set_title("Prediction vs Experimental with Error Color Map", fontsize=18, fontweight="bold")

    # 删除网格线
    ax.grid(False)

    # 外框和刻度加粗
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    ax.tick_params(axis="both", which="both", width=2.0, length=6, labelsize=14)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    # 图例
    leg = ax.legend(frameon=True, fontsize=13, loc="best")
    leg.get_frame().set_linewidth(2.0)
    for txt in leg.get_texts():
        txt.set_fontweight("bold")

    # 指标文本框
    metrics_text = f"RMSE = {rmse:.2f}\nMAE = {mae:.2f}\nq² = {q2:.2f}"
    ax.text(
        0.98, 0.02, metrics_text,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="right",
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            boxstyle="round,pad=0.5",
            linewidth=2.0
        )
    )

    plt.tight_layout()
    plt.savefig(out_file, format="svg", bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"图像已保存：{out_file}")

# =============== 调用绘图 ===============
plot_error_colormap_with_fit(
    y_true=np.array(y_test),
    y_pred=np.array(y_pred),
    rmse=rmse_test,
    mae=mae_test,
    q2=q2_test_value,
    out_file="prediction_vs_experimental_error_colormap.svg"
)


