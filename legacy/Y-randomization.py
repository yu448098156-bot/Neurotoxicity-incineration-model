import pandas as pd  
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import seaborn as sns
import shap
import warnings
import matplotlib.cm as cm

warnings.filterwarnings("ignore")

# ==== 全局字体与样式（Arial + 加粗，去网格）====
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams["axes.labelweight"] = "bold"
mpl.rcParams["axes.titlesize"] = 18
mpl.rcParams["axes.labelsize"] = 16
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["legend.fontsize"] = 14
mpl.rcParams["figure.dpi"] = 100
# 刻度与坐标轴统一黑色
mpl.rcParams["xtick.color"] = "black"
mpl.rcParams["ytick.color"] = "black"
mpl.rcParams["axes.edgecolor"] = "black"

sns.set_theme(context="talk", style="ticks", font="Arial")

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
    except:
        return None

# ==== 2. 数据读取与描述符计算 ====
print("Loading data and calculating descriptors...")
df = pd.read_csv("training_set.csv")
df = df.dropna(subset=["SMILES", "pLD50"])

descriptors = []
valid_indices = []

for i, smi in enumerate(df["SMILES"]):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        desc = calculate_descriptors(mol)
        if desc:
            descriptors.append(desc)
            valid_indices.append(i)

X = pd.DataFrame(descriptors)
y = df.loc[valid_indices, "pLD50"].values

# ==== 3. 特征标准化 & 模型训练 ====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Training CatBoost model...")
model = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=8, verbose=0)
model.fit(X_scaled, y)

# ==== 4. SHAP 值计算 ====
print("Calculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled)

# ==== 5. Y-随机化验证 ====
print("Performing Y-randomization validation...")

original_r2 = r2_score(y, model.predict(X_scaled))
original_q2 = cross_val_score(model, X_scaled, y, cv=5, scoring='r2').mean()

n_iterations = 100
random_r2_list = []
random_q2_list = []

for i in range(n_iterations):
    y_random = np.random.permutation(y)
    model_random = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=8, verbose=0)
    model_random.fit(X_scaled, y_random)
    
    r2_random = r2_score(y_random, model_random.predict(X_scaled))
    q2_random = cross_val_score(model_random, X_scaled, y_random, cv=5, scoring='r2').mean()
    
    random_r2_list.append(r2_random)
    random_q2_list.append(q2_random)

print(f"Original R²: {original_r2:.3f}")
print(f"Original q²: {original_q2:.3f}")
print(f"Mean Random R²: {np.mean(random_r2_list):.3f}")
print(f"Mean Random q²: {np.mean(random_q2_list):.3f}")

# ==== 可视化（R²=绿色，q²=蓝色；竖排组图，7×8 英寸）====

def style_axis(ax):
    ax.grid(False)
    for side in ['top', 'right', 'bottom', 'left']:
        ax.spines[side].set_linewidth(2.8)
        ax.spines[side].set_color('black')
    ax.tick_params(axis='both', which='both',
                   width=2.8, length=7, direction='out',
                   color='black', labelcolor='black')

def tighten_x(ax, data, extra=None, pad_frac=0.02):
    vals = np.array(data if extra is None else list(data) + [extra])
    xmin, xmax = float(np.min(vals)), float(np.max(vals))
    rng = xmax - xmin if xmax > xmin else 1.0
    ax.set_xlim(xmin - pad_frac * rng, xmax + pad_frac * rng)

# === 竖排两张子图：2 行 1 列，整体 7×8 ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))

# 上：R²（绿色）
sns.histplot(random_r2_list, kde=False, bins=20, color='#2ca02c', ax=ax1)  # 深绿柱
sns.kdeplot(random_r2_list, ax=ax1, color='#145a32', linewidth=2.8)        # 更深绿曲线
ax1.axvline(original_r2, color='red', linestyle='--', linewidth=2.8)
ax1.set_title("Y-Randomization: R² Distribution", fontweight='bold', color='black')
ax1.set_xlabel("R² (Randomized Y)", fontweight='bold', color='black')
ax1.set_ylabel("Frequency", fontweight='bold', color='black')
style_axis(ax1)
tighten_x(ax1, random_r2_list, extra=original_r2, pad_frac=0.02)

# —— 压缩一下 y 轴：按最大柱高稍微留 5% 余量 ——
if ax1.patches:
    ymax1 = max(p.get_height() for p in ax1.patches)
    ax1.set_ylim(0, ymax1 * 1.05)

ylim1 = ax1.get_ylim()
xlim1 = ax1.get_xlim()
ax1.text((xlim1[0] + xlim1[1]) / 2, ylim1[1] * 0.9,
         f'Original R² = {original_r2:.2f}',
         color='red', fontsize=16, fontweight='bold',
         ha='center', va='top')

leg1 = ax1.legend(['Original R²'], loc='best')
leg1.get_frame().set_linewidth(2.5)
leg1.get_frame().set_edgecolor('black')

# 下：q²（蓝色）
sns.histplot(random_q2_list, kde=False, bins=20, color='#1f77b4', ax=ax2)  # 深蓝柱
sns.kdeplot(random_q2_list, ax=ax2, color='#0b3d91', linewidth=2.8)        # 更深蓝曲线
ax2.axvline(original_q2, color='red', linestyle='--', linewidth=2.8)
ax2.set_title("Y-Randomization: q² Distribution", fontweight='bold', color='black')
ax2.set_xlabel("q² (Randomized Y)", fontweight='bold', color='black')
ax2.set_ylabel("Frequency", fontweight='bold', color='black')
style_axis(ax2)
tighten_x(ax2, random_q2_list, extra=original_q2, pad_frac=0.02)

# —— 同样压缩一下 q² 图的 y 轴 ——
if ax2.patches:
    ymax2 = max(p.get_height() for p in ax2.patches)
    ax2.set_ylim(0, ymax2 * 1.05)

ylim2 = ax2.get_ylim()
xlim2 = ax2.get_xlim()
ax2.text((xlim2[0] + xlim2[1]) / 2, ylim2[1] * 0.9,
         f'Original q² = {original_q2:.2f}',
         color='red', fontsize=16, fontweight='bold',
         ha='center', va='top')

leg2 = ax2.legend(['Original q²'], loc='best')
leg2.get_frame().set_linewidth(2.5)
leg2.get_frame().set_edgecolor('black')

# 布局稍微紧凑一点，减少上下空白
plt.tight_layout()
plt.savefig("y_randomization_validation.png", dpi=300)
plt.close()
