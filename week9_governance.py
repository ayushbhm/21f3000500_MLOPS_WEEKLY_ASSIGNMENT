import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scipy.stats import ks_2samp
import shap
from fairlearn.metrics import MetricFrame

# ── TASK 1: Load data + location attribute ──
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')

np.random.seed(42)
X['location'] = np.random.randint(0, 2, size=len(X))

X_feats = iris.feature_names
X_train, X_test, y_train, y_test = train_test_split(X[X_feats], y, test_size=0.2, random_state=42, stratify=y)
loc_train, loc_test = train_test_split(X['location'], test_size=0.2, random_state=42, stratify=y)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Task 1 done. Accuracy:', accuracy_score(y_test, y_pred))

# ── TASK 2: Fairlearn MetricFrame ──
metrics = {
    'accuracy':  lambda yt, yp: accuracy_score(yt, yp),
    'precision': lambda yt, yp: precision_score(yt, yp, average='macro', zero_division=0),
    'recall':    lambda yt, yp: recall_score(yt, yp, average='macro', zero_division=0),
}
mf = MetricFrame(metrics=metrics, y_true=y_test, y_pred=y_pred, sensitive_features=loc_test)
print('\nTask 2 - Overall:\n', mf.overall)
print('\nBy Location:\n', mf.by_group)
print('\nDifference:\n', mf.difference())

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, metric, color in zip(axes, ['accuracy','precision','recall'], ['#2196F3','#4CAF50','#FF5722']):
    vals = mf.by_group[metric]
    bars = ax.bar(['Location 0','Location 1'], vals.values, color=color, alpha=0.75, edgecolor='black', width=0.4)
    ax.axhline(mf.overall[metric], color='red', linestyle='--', label=f'Overall: {mf.overall[metric]:.3f}')
    ax.set_title(f'{metric.capitalize()} by Location', fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.legend()
    for bar, val in zip(bars, vals.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f'{val:.3f}', ha='center')
plt.suptitle('Fairlearn MetricFrame - Performance by Location Group', fontweight='bold')
plt.tight_layout()
plt.savefig('task2_fairness.png', dpi=150, bbox_inches='tight')
print('Task 2 plot saved.')

# ── TASK 3: SHAP (TreeExplainer - fast) ──
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)
print('\nTask 3 - SHAP values shape:', np.array(shap_values).shape)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, name in enumerate(iris.target_names):
    plt.sca(axes[i])
    shap.summary_plot(shap_values[:, :, i], X_test, feature_names=list(X_feats), show=False, plot_size=None)
    axes[i].set_title(f'SHAP - {name}', fontweight='bold')
plt.suptitle('SHAP Summary Plots', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('task3_shap_all.png', dpi=150, bbox_inches='tight')

plt.figure(figsize=(9,5))
shap.summary_plot(shap_values[:, :, 2], X_test, feature_names=list(X_feats), show=False)
plt.title('SHAP - Virginica', fontweight='bold')
plt.tight_layout()
plt.savefig('task3_shap_virginica.png', dpi=150, bbox_inches='tight')
print('Task 3 plots saved.')

mean_abs = pd.Series(np.abs(shap_values[:, :, 2]).mean(axis=0), index=X_feats).sort_values(ascending=False)
print('Virginica feature importance:\n', mean_abs)

# ── TASK 4: Drift Detection ──
ref = X_train.copy()
cur = X_test.copy()
cur['petal length (cm)'] += 2.5
cur['petal width (cm)']  *= 1.8
cur['sepal length (cm)'] += np.random.normal(1.5, 0.5, len(cur))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, feat in zip(axes.flatten(), X_feats):
    ax.hist(ref[feat], bins=20, alpha=0.6, color='steelblue', label='Reference', density=True)
    ax.hist(cur[feat], bins=20, alpha=0.6, color='tomato',    label='Drifted',   density=True)
    ax.set_title(feat, fontweight='bold'); ax.legend()
plt.suptitle('Task 4: Feature Drift - Reference vs Production', fontweight='bold')
plt.tight_layout()
plt.savefig('task4_drift.png', dpi=150, bbox_inches='tight')

print('\nKS Drift Test:')
print(f'{"Feature":<30} {"KS stat":>10} {"p-value":>10} {"Drift?":>8}')
for feat in X_feats:
    s, p = ks_2samp(ref[feat], cur[feat])
    print(f'{feat:<30} {s:>10.4f} {p:>10.4f} {"YES" if p<0.05 else "No":>8}')

print('\nAll tasks complete.')
