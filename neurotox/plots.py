"""Separate Matplotlib figures from saved numeric outputs; no retraining."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .io import read_csv

def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts={f.name for f in font_manager.fontManager.ttflist}
    font=next((x for x in ['Arial','Liberation Sans','DejaVu Sans'] if x in fonts),'DejaVu Sans')
    plt.rcParams.update({'font.family':font,'font.size':11,'axes.labelweight':'bold',
                         'axes.linewidth':1.6,'xtick.major.width':1.4,'ytick.major.width':1.4,
                         'svg.fonttype':'none'})
    return plt

def save(fig,name,out):
    out.mkdir(parents=True,exist_ok=True)
    fig.tight_layout()
    fig.savefig(out/f'{name}.svg',bbox_inches='tight')
    fig.savefig(out/f'{name}.png',dpi=300,bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)

def plot_all(comparison:Path,explanation:Path|None,yrandom:Path|None,output:Path):
    plt=setup()
    if (comparison/'model_comparison.csv').exists():
        _,rows=read_csv(comparison/'model_comparison.csv')
        names=[r['model'] for r in rows]
        fig,ax=plt.subplots(figsize=(6.4,4.3))
        ax.bar(names,[float(r['CV_R2_mean']) for r in rows],
               yerr=[float(r['CV_R2_SD_sample']) for r in rows],capsize=3)
        ax.set_ylabel('Five-fold validation R² (mean ± s.d.)')
        ax.tick_params(axis='x',rotation=30)
        save(fig,'model_comparison_CV',output)
        fig,ax=plt.subplots(figsize=(6.4,4.3))
        ax.bar(names,[float(r['R2_test']) for r in rows])
        ax.set_ylabel('Held-out test R²')
        ax.tick_params(axis='x',rotation=30)
        save(fig,'model_comparison_test',output)
    if (comparison/'CatBoost_predictions_test.csv').exists():
        _,rows=read_csv(comparison/'CatBoost_predictions_test.csv')
        y=np.array([float(r['observed_pLD50']) for r in rows]);p=np.array([float(r['predicted_pLD50']) for r in rows])
        from .metrics import regression_metrics
        m=regression_metrics(y,p)
        fig,ax=plt.subplots(figsize=(5.5,4.7))
        scat=ax.scatter(y,p,c=np.abs(y-p),s=32)
        bounds=[min(y.min(),p.min())-.05,max(y.max(),p.max())+.05]
        ax.plot(bounds,bounds,linestyle='--',label='1:1 line')
        coeff=np.polyfit(y,p,1)
        ax.plot(bounds,np.polyval(coeff,bounds),label='Linear fit')
        ax.set_xlabel('Observed pLD50');ax.set_ylabel('Predicted pLD50')
        fig.colorbar(scat,ax=ax,label='Absolute prediction error')
        ax.text(.97,.04,f"R² = {m['R2']:.3f}\nRMSE = {m['RMSE']:.3f}\nMAE = {m['MAE']:.3f}",
                transform=ax.transAxes,ha='right',va='bottom')
        ax.legend(frameon=False,fontsize=9)
        save(fig,'Fig2a_CatBoost_test',output)
    if explanation is not None and (explanation/'SHAP_importance.csv').exists():
        _,rows=read_csv(explanation/'SHAP_importance.csv')
        rows=rows[:10]
        fig,ax=plt.subplots(figsize=(5.4,4.5))
        ax.barh([r['descriptor'] for r in rows],[float(r['mean_absolute_SHAP']) for r in rows])
        ax.invert_yaxis();ax.set_xlabel('Mean absolute SHAP value')
        save(fig,'Fig2c_SHAP_importance',output)
        for feature in ['EState_VSA5','SlogP_VSA2','SMR_VSA4','VSA_EState2']:
            path=explanation/f'ICE_{feature}.csv'
            if not path.exists():continue
            headers,rows=read_csv(path)
            x=np.array([float(r['descriptor_scaled']) for r in rows])
            matrix=np.array([[float(r[h]) for h in headers[4:]] for r in rows])
            delta=matrix[-1]-matrix[0]
            norm=plt.Normalize(delta.min(),delta.max())
            cmap=plt.get_cmap()
            fig,ax=plt.subplots(figsize=(5.2,4.2))
            for j in range(matrix.shape[1]):
                ax.plot(x,matrix[:,j],alpha=.25,linewidth=.6,color=cmap(norm(delta[j])))
            ax.plot(x,matrix.mean(axis=1),linewidth=2.2,label='Mean prediction')
            ax.set_xlabel(f'{feature} (standardized)');ax.set_ylabel('Predicted pLD50')
            fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,
                         label='Prediction change (end − start)')
            ax.legend(frameon=False,fontsize=9)
            save(fig,f'Fig2d_ICE_{feature}',output)
    if yrandom is not None and (yrandom/'Y_randomization.csv').exists():
        _,rows=read_csv(yrandom/'Y_randomization.csv')
        original=next(r for r in rows if r['type']=='original')
        random=[r for r in rows if r['type']=='permuted']
        for key,label in [('train_R2','Training R²'),('CV_R2_mean','Mean five-fold validation R²')]:
            fig,ax=plt.subplots(figsize=(5.4,4.0))
            ax.hist([float(r[key]) for r in random],bins=20)
            ax.axvline(float(original[key]),linestyle='--',label='Unpermuted labels')
            ax.set_xlabel(label);ax.set_ylabel('Permutation count')
            ax.legend(frameon=False,fontsize=9)
            save(fig,f'Fig2b_Yrandom_{key}',output)
    print(f'Figures saved to {output}; every panel is a separate SVG/PNG.')
