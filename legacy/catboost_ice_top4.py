import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
import shap
import warnings
import matplotlib.cm as cm

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

# ==== 4. SHAP 值计算（用于特征重要性排序）====
print("Calculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled)

# ==== 5. 输出前4个特征的彩色ICE图，2×2排列====
print("Generating color-coded ICE plots for top 4 features in 2x2 layout...")

shap_importance = np.abs(shap_values).mean(axis=0)
top4_indices = np.argsort(shap_importance)[-4:][::-1]
top4_features = X.columns[top4_indices]

fig, axs = plt.subplots(2, 2, figsize=(15, 12))
axs = axs.flatten()

for i, feature in enumerate(top4_features):
    ax = axs[i]
    feature_idx = X.columns.get_loc(feature)
    X_feat = X_scaled[:, feature_idx]

    unique_vals = np.sort(np.unique(X_feat))
    ice_curves = []

    for sample_idx in range(X_scaled.shape[0]):
        X_temp = np.tile(X_scaled[sample_idx], (len(unique_vals), 1))
        X_temp[:, feature_idx] = unique_vals
        preds = model.predict(X_temp)
        ice_curves.append(preds)

    ice_curves = np.array(ice_curves)
    slopes = ice_curves[:, -1] - ice_curves[:, 0]

    norm = plt.Normalize(slopes.min(), slopes.max())
    colors = cm.bwr(norm(slopes))

    for j in range(ice_curves.shape[0]):
        ax.plot(unique_vals, ice_curves[j], color=colors[j], alpha=0.4)

    pdp = ice_curves.mean(axis=0)
    ax.plot(unique_vals, pdp, color='black', linewidth=3, label='PDP (mean)')

    ax.set_title(f"ICE Plot with Positive/Negative Influence - {feature}", fontsize=12)
    ax.set_xlabel(f"{feature} (scaled)")
    ax.set_ylabel("Prediction")
    ax.legend()

    sm = plt.cm.ScalarMappable(cmap='bwr', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label("Prediction change (end - start)", rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig("top4_ice_colorcoded_2x2.png", dpi=300, bbox_inches='tight')
plt.close()

print("2x2 color-coded ICE plots saved successfully.")



