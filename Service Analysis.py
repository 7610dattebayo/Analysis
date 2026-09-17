#ライブラリのインポート
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#フォント設定 (日本語対応フォントに変更)
#plt.rcParams["font.family"] = "MS Gothic"

#データの読み込み
df = pd.read_csv("契約予測（スタンダード_契約1カ月目）_読込用.csv", encoding="utf-8")

#ここからDoWhyによる因果関係のモデル化

'''
#データの読み込み
df = pd.read_csv("契約予測（スタンダード_契約1カ月目）_読込用_因果推論用.csv", encoding="utf-8")

from dowhy import CausalModel

treatment = "rensai_follow"     #介入を行いたい特徴量
outcome = "contract_flag"       #目的変数
common_causes = [col for col in df.columns if col not in [treatment, outcome]]  #他の共通原因変数（特徴量）

dowhy_model = CausalModel(
    data=df,
    treatment=treatment,                  
    outcome=outcome,                
    common_causes=common_causes    
)

#因果ダイヤグラムの可視化
#causal_graph = dowhy_model.view_model()
#causal_graph.render(filename="因果ダイヤグラムの可視化", format="png", cleanup=True)

#因果モデルの識別
identified_estimand = dowhy_model.identify_effect(proceed_when_unidentifiable=True)
#print("Identified estimand:")
#print(identified_estimand)
'''
'''
##推定手法：線形回帰

#因果効果の推定
estimate = dowhy_model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
#print(f"Causal Estimate is {estimate.value}")

#結果の確認
refutation1 = dowhy_model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter")
refutation2 = dowhy_model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause")
refutation3 = dowhy_model.refute_estimate(identified_estimand, estimate, method_name="data_subset_refuter")

#print(refutation1)
#print(refutation2)
#print(refutation3)
'''
'''
##推定手法：傾向スコアマッチング
from sklearn.linear_model import LogisticRegression
propensity_model = LogisticRegression(solver="liblinear", max_iter=1000)

#因果効果の推定
estimate = dowhy_model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_matching",
    method_params={
        "propensity_score_model": propensity_model
    }
)

#推定値の確認
#print(f"Causal Estimate is {estimate.value}")

#反証結果の確認
refutation1 = dowhy_model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter")
refutation2 = dowhy_model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause")
refutation3 = dowhy_model.refute_estimate(identified_estimand, estimate, method_name="data_subset_refuter")

#print(refutation1)
#print(refutation2)
#print(refutation3)
'''
'''
##推定手法：ダブルロバスト

#共変量、処置変数、結果変数の設定
X = df.drop(["rensai_follow", "contract_flag"], axis=1)
treatment = df["rensai_follow"]
outcome = df["contract_flag"]

#傾向スコアの推定
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X, treatment)
propensity_score = xgb_model.predict_proba(X)[:, 1]

#結果変数の予測
xgb_model_outcome = xgb.XGBClassifier()
xgb_model_outcome.fit(np.column_stack((X, treatment)), outcome)
predicted_outcome = xgb_model_outcome.predict_proba(np.column_stack((X, treatment)))[:, 1]

#結果変数の予測
outcome_model = xgb.XGBRegressor()
outcome_model.fit(np.column_stack((X, treatment)), outcome)
predicted_outcome_treated = outcome_model.predict(np.column_stack((X, np.ones(len(treatment)))))
predicted_outcome_control = outcome_model.predict(np.column_stack((X, np.zeros(len(treatment)))))

#ダブルロバスト推定の実行
df['propensity_score'] = propensity_score
df['predicted_outcome_treated'] = predicted_outcome_treated
df['predicted_outcome_control'] = predicted_outcome_control

#傾向スコアに基づいた重み付け
weight_treated = treatment / propensity_score
weight_control = (1 - treatment) / (1 - propensity_score)

#ダブルロバスト推定の計算
df['outcome_treated'] = treatment * (outcome - predicted_outcome_treated) / propensity_score
df['outcome_control'] = (1 - treatment) * (outcome - predicted_outcome_control) / (1 - propensity_score)
df['dr_outcome']      = df['outcome_treated'] + df['outcome_control'] + \
                     predicted_outcome_treated - predicted_outcome_control

#平均因果効果の推定
ate = df['dr_outcome'].mean()
print("Average Treatment Effect (ATE):", ate)
'''

#ここまでDoWhyによる因果関係のモデル化


#特徴量の格納（説明変数:B列以降）
train_df = df.iloc[:,1:16]

#予測ターゲットの格納
target_df = df[["contract_flag"]]

#モデル学習のための、訓練データとテストデータを7:3で分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_df, target_df, test_size=0.3, random_state=3)

'''
print('x.shape = {}'.format(train_df.shape))
print('y.shape = {}'.format(target_df .shape))
print('trainX.shape = {}'.format(X_train.shape))
print('trainY.shape = {}'.format(y_train.shape))
print('testX.shape = {}'.format(X_test.shape))
print('testY.shape = {}'.format(y_test.shape))
'''
'''
#Trainデータの基本統計量
filtered_df = X_train[X_train != 0]
describe_df = filtered_df.describe()
#print(describe_df)

#Excelファイルとして出力
describe_df.to_csv("trainデータの基本統計量.csv")
'''

#XGBoostハイパーパラメータ設定
xgb_dtrain = xgb.DMatrix(X_train, label=y_train, feature_names = train_df.columns.to_list())
xgb_dtest = xgb.DMatrix(X_test, label=y_test, feature_names = train_df.columns.to_list())

xgb_params = {
        "objective"         : "binary:logistic",  # 二項分類のロジスティック回帰
        "eval_metric"       : "logloss",          # 二値交差エントロピー
        "booster"           : "gbtree",           # boosterの種類
        "learning_rate"     : 0.1878257143334672,# 学習率
        "max_depth"         : 3,                  # 決定木の深さ
}

#XGBoostの学習
evals_result = {}
xgb_model = xgb.train(params=xgb_params,
                      dtrain=xgb_dtrain,
                      evals=[(xgb_dtrain, "train"),(xgb_dtest, "test")],
                      num_boost_round=76,
                      verbose_eval=False,
                      evals_result = evals_result)

#testデータの予測と評価（ラベル1の確率）
y_test_pred_proba = xgb_model.predict(xgb.DMatrix(X_test))
#print(y_test_pred_proba)

#testデータの予測と評価（確率をラベルに変換）
y_test_pred = np.round(y_test_pred_proba)
#print(y_test_pred)

#testデータに対する予測精度（正解率）の可視化
print("正解率: " + str(round(accuracy_score(y_test,y_test_pred),3)))
print("適合率: " + str(round(precision_score(y_test,y_test_pred, average="macro"),3)))
print("再現率: " + str(round(recall_score(y_test,y_test_pred, average="macro"),3)))

#各ラウンドでの正解率を計算するために、予測結果を取得
train_accuracy = []
test_accuracy = []

for i in range(1, 76 + 1):
    # モデルを各ラウンドで部分的に訓練
    xgb_acc = xgb.train(xgb_params, 
                        xgb_dtrain, 
                        i, 
                        evals=[(xgb_dtrain, "train"),(xgb_dtest, "test")], 
                        verbose_eval=False)
    
    #訓練データとテストデータでの予測
    train_pred = xgb_acc.predict(xgb_dtrain)
    test_pred = xgb_acc.predict(xgb_dtest)
    
    #0.5を閾値として二値分類
    train_pred_binary = (train_pred > 0.5).astype(int)
    test_pred_binary = (test_pred > 0.5).astype(int)
    
    #正解率を計算
    train_acc = accuracy_score(y_train, train_pred_binary)
    test_acc = accuracy_score(y_test, test_pred_binary)
    
    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)

#testデータに対する正解率の推移を可視化
plt.figure(figsize=(12, 8))
plt.subplot(111)
plt.plot(train_accuracy, label="Train Accuracy")
plt.plot(test_accuracy, label="Test Accuracy")
plt.legend()
plt.title("Accuracy Over Boosting Rounds with Logloss")
plt.xlabel("boosting round")
plt.ylabel("accuracy")
plt.savefig("契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_正解率の可視化", bbox_inches='tight')

#testデータに対する損失関数の推移を可視化
plt.figure(figsize=(12, 8))
plt.subplot(111)
plt.plot(evals_result["train"]["logloss"], label="train logloss")
plt.plot(evals_result["test"]["logloss"], label="test logloss")
plt.legend()
plt.title("Loss Function")
plt.xlabel("rounds")
plt.ylabel("logloss")
plt.savefig("契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_損失関数の可視化", bbox_inches='tight')

#testデータに対する予測精度（混同行列）の可視化
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_test_pred)
plt.figure(figsize=(12, 8))
plt.subplot(111)
sns.heatmap(cm, annot=True, cmap="OrRd", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predict")
plt.ylabel("Label")
plt.savefig("契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_混合行列の可視化.png", bbox_inches="tight")

#testデータに対する予測精度に影響を与えた説明変数（特徴量）の可視化
plt.figure(figsize=(12, 8))
plt.subplot(111)
xgb.plot_importance(xgb_model, importance_type="gain")
plt.title("Feature Importance")
plt.xlabel("Contribution_Rate")
plt.ylabel("Features")
plt.savefig("契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_特徴量の可視化.png", bbox_inches="tight")

#SHAP値の計算
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_train)
#print(shap_values)

#全件レコードの期待値
expected_values = explainer.expected_value
#print(expected_values)

'''
#SHAP（大域的説明）

#ラベル1のSHAP値を個別に可視化（bar）
plt.figure()
shap.summary_plot(shap_values, X_train, feature_names=train_df.columns.to_list(), plot_type="bar", max_display=21, show=False)
plt.savefig("契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_SHAP値の可視化（bar）.png", bbox_inches="tight")

#ラベル0とラベル1のSHAP値を個別に可視化（bar）
plt.figure()
if isinstance(shap_values, list) and len(shap_values) == 2:
    shap.summary_plot(shap_values[0], X_train, feature_names=train_df.columns.to_list(), plot_type="bar", max_display=21, title="SHAP values for label 0", show=False)
    plt.savefig("契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_ラベル0のSHAP値の可視化（bar）.png", bbox_inches="tight")
    shap.summary_plot(shap_values[1], X_train, feature_names=train_df.columns.to_list(), plot_type="bar", max_display=21, title="SHAP values for label 1", show=False)
    plt.savefig("契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_ラベル1のSHAP値の可視化（bar）.png", bbox_inches="tight")
else:
    shap.summary_plot(shap_values, X_train, feature_names=train_df.columns.to_list(), plot_type="bar", max_display=21, title="SHAP values for label 0 or label 1", show=False)
    plt.savefig("契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_その他ラベルのSHAP値の可視化（bar）.png", bbox_inches="tight")

#SHAP値の可視化(besswarm)
plt.figure()
shap.summary_plot(shap_values, X_train, feature_names=train_df.columns.to_list(), plot_type="dot", max_display=21, show=False)
plt.savefig("契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_SHAP値の可視化（besswarm）.png", bbox_inches="tight")

#SHAP値の可視化(dependence)（ニュースレター）
plt.figure()
shap.dependence_plot(ind="nl_mail_opens_count",interaction_index="nl_mail_opens_count", shap_values=shap_values, features=X_train, show=False)
plt.savefig("契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_SHAP値の可視化（dependence）_ニュースレター.png", bbox_inches="tight")

#SHAP値の可視化(dependence)（Google）
plt.figure()
shap.dependence_plot(ind="google_visits",interaction_index="google_visits", shap_values=shap_values, features=X_train, show=False)
plt.savefig("契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_SHAP値の可視化（dependence）_Google.png", bbox_inches="tight")

#SHAP値の可視化(dependence)（コメントプラス）
plt.figure()
shap.dependence_plot(ind="comment_views",interaction_index="comment_views", shap_values=shap_values, features=X_train, show=False)
plt.savefig("契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_SHAP値の可視化（dependence）_コメントプラス.png", bbox_inches="tight")

#SHAP値の可視化(dependence)（ニュースの要点）
plt.figure()
shap.dependence_plot(ind="news_summary_count",interaction_index="news_summary_count", shap_values=shap_values, features=X_train, show=False)
plt.savefig("契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_SHAP値の可視化（dependence）_ニュースの要点.png", bbox_inches="tight")

#SHAP値の可視化(dependence)（Yahoo）
plt.figure()
shap.dependence_plot(ind="yahoo_visits",interaction_index="yahoo_visits", shap_values=shap_values, features=X_train, show=False)
plt.savefig("契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_SHAP値の可視化（dependence）_Yahoo.png", bbox_inches="tight")

#SHAP値の可視化(dependence)(プッシュ通知)
plt.figure()
shap.dependence_plot(ind="push_app_count",interaction_index="push_app_count", shap_values=shap_values, features=X_train, show=False)
plt.savefig("契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_SHAP値の可視化（dependence）_プッシュ通知.png", bbox_inches="tight")

#SHAP値の可視化(dependence)（連載フォロー）
plt.figure()
shap.dependence_plot(ind="rensai_follow",interaction_index="rensai_follow", shap_values=shap_values, features=X_train, show=False)
plt.savefig("契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_SHAP値の可視化（dependence）_連載フォロー.png", bbox_inches="tight")

#サブサンプリングのためのランダムサンプルの選択
sample_size = 1000 
if len(X_train) > sample_size:
    X_sample = X_train.sample(sample_size, random_state=3)
    shap_values_sample = explainer.shap_values(X_sample)
else:
    X_sample = X_train
    shap_values_sample = shap_values

#sHAP値の可視化(decision) 
plt.figure()
shap.decision_plot(explainer.expected_value, shap_values_sample, X_sample, show=False)
plt.savefig('契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_SHAP値の可視化（decision）.png', bbox_inches='tight')

#sHAP値の可視化(heatmap)
plt.figure()
shap.summary_plot(shap_values_sample, X_sample, plot_type="heatmap", show=False)
plt.savefig('契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_SHAP値の可視化（heatmap）.png', bbox_inches='tight')
'''
'''
#SHAP(局所的説明)

#SHAP値の可視化(scatter)
plt.figure()
shap.force_plot(explainer.expected_value, shap_values[0, :], X_train[0, :], feature_names=train_df.columns.to_list(), matplotlib=True)
plt.savefig("契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_SHAP値の可視化（scatter）.png", bbox_inches="tight")

#SHAP値の可視化(waterfall) （例として最初のテストサンプルを使用）
shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=X_test[0], feature_names=train_df.columns.to_list()))
plt.savefig('契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_SHAP値の可視化（waterfall）.png', bbox_inches='tight')
'''
'''
#ロジスティック回帰によるSHAP値の統計的検証
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

#SHAP値の計算
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_train)

#標準化
scaler = StandardScaler()
shap_values = scaler.fit_transform(shap_values)

#SHAP値と目的変数をデータフレームにまとめる
feature_names = train_df.columns.to_list()
shap_df = pd.DataFrame(shap_values, columns=feature_names)
shap_df["target"] = y_train
'''
'''
#SHAP値の基本統計量
shap_df_train_csv = shap_df.describe()

#Excelファイルとして出力
shap_df_train_csv.to_csv("shap値の基本統計量_説明変数最適化.csv")
#各特徴量のSHAP値と目的変数の相関を計算
correlations = shap_df.corr().drop("target")

#結果の表示
#print('各特徴量のSHAP値と目的変数の相関:')
#print(correlations)

#ヒートマップのプロット
plt.figure(figsize=(12, 10))
sns.heatmap(correlations, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation between SHAP Values and Target")
plt.savefig("契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_説明変数最適化_SHAP値と目的変数の相関図.png", bbox_inches="tight")
'''
'''
#NaNのチェックと削除
shap_df = shap_df.dropna()

#ロジスティック回帰の実行
X_shap = shap_df.drop("target", axis=1)
y_shap = shap_df["target"]

#ロジスティック回帰で統計的検証
X_logit = sm.add_constant(X_shap)  # 切片を追加
logit_model = sm.Logit(y_shap, X_logit)  # ロジスティック回帰モデル
result = logit_model.fit()

#結果の表示
print(result.summary())

#係数とp値の取得
coef = result.params[1:]  # 切片(const)を除く
p_values = result.pvalues[1:]  # 切片(const)を除く

#プロット: 係数 vs p値
plt.figure(figsize=(20, 12))
plt.scatter(coef, p_values)
plt.axhline(0.05, linestyle="--", color="red", label="p=0.05")
plt.axvline(0, linestyle="--", color="gray")

#ラベル付け
for i, txt in enumerate(feature_names):
    plt.annotate(txt, (coef[i], p_values[i]), fontsize=8, xytext=(5,5), textcoords='offset points')

plt.xlabel("Coefficients")
plt.ylabel("p-value")
plt.yscale("log")  
plt.title("Coefficients vs p-value")
plt.legend()
plt.savefig("契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_ロジスティック回帰によるSHAP値の統計的検証（係数とp値）.png", bbox_inches="tight")

#プロット: 係数
plt.figure()
plt.barh(feature_names, coef)
plt.xlabel("Coefficients")
plt.ylabel("Features")
plt.title("Coefficients")
plt.savefig("契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_ロジスティック回帰によるSHAP値の統計的検証（係数）.png", bbox_inches="tight")

#プロット: p値
plt.figure()
plt.barh(feature_names, p_values)
plt.axvline(0.05, linestyle="--", color="red", label="p=0.05")
plt.xlabel("p-value")
plt.ylabel("Features")
plt.title("p-values")
plt.xscale("log")  # p値のスケールをログにすると見やすい
plt.savefig("契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_ロジスティック回帰によるSHAP値の統計的検証（p値）.png", bbox_inches="tight")

#プロット: 予測確率とSHAp値の分布
for i, feature in enumerate(feature_names):
    plt.figure()
    plt.scatter(shap_values[:, i], y_train)
    log_reg = LogisticRegression()
    log_reg.fit(shap_values[:, i].reshape(-1, 1), y_train)
    x_vals = np.linspace(shap_values[:, i].min(), shap_values[:, i].max(), 100)
    y_vals = log_reg.predict_proba(x_vals.reshape(-1, 1))[:, 1]
    plt.plot(x_vals, y_vals, color='red', label='Logistic regression line')
    plt.xlabel(f"SHAP value for {feature}")
    plt.ylabel("Forecast probability")
    plt.title(f"SHAP values vs Forecast probability")
    plt.savefig(f"{feature}_LogisticRegression.png")  
    plt.close()

'''
'''
#重回帰分析によるSHAP値の統計的検証
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

#SHAP値と目的変数をデータフレームにまとめる
feature_names = train_df.columns.to_list()
shap_df = pd.DataFrame(shap_values, columns=feature_names)
shap_df["target"] = y_train
'''
'''
#SHAP値の基本統計量
shap_df_train_csv = shap_df.describe()

#Excelファイルとして出力
shap_df_train_csv.to_csv("shap値の基本統計量_説明変数最適化.csv")
#各特徴量のSHAP値と目的変数の相関を計算
correlations = shap_df.corr().drop("target")

#結果の表示
#print('各特徴量のSHAP値と目的変数の相関:')
#print(correlations)

#ヒートマップのプロット
plt.figure(figsize=(12, 10))
sns.heatmap(correlations, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation between SHAP Values and Target")
plt.savefig("契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_説明変数最適化_SHAP値と目的変数の相関図.png", bbox_inches="tight")
'''
'''
#NaNのチェックと削除
shap_df = shap_df.dropna()

#重回帰分析の実行
X_shap = shap_df.drop("target", axis=1)
y_shap = shap_df["target"]

#statsmodelsを使用して回帰分析を行い、t値を計算
X_shap = sm.add_constant(X_shap)  # 切片の追加
sm_model = sm.OLS(y_shap, X_shap).fit()

#結果の表示
result = sm_model
print(result.summary())

t_values = result.tvalues[1:]  # 切片を除くt値
coefficients = result.params[1:]  # 切片を除く係数

#係数 vs t値プロット
plt.figure(figsize=(20, 12))
plt.scatter(coefficients, t_values)
for i, txt in enumerate(coefficients.index):
    plt.annotate(txt, (coefficients[i], t_values[i]), fontsize=8)
plt.axhline(y=0, color="grey", linestyle='--')
plt.xlabel("Coefficients")
plt.ylabel("t-values")
plt.title("Coefficients vs t-values")
plt.grid(True)
plt.savefig("契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_説明変数最適化_重回帰分析によるSHAP値の統計的検証（係数とt値）.png", bbox_inches="tight")

#係数のプロット
plt.figure()
#plt.bar(range(len(coefficients)), coefficients)
plt.barh(feature_names, coefficients)
plt.xlabel("Features")
plt.ylabel("Coefficients")
plt.title("Coefficients")
#plt.xticks(range(len(coefficients)), feature_names, rotation=90)
plt.savefig("契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_説明変数最適化_重回帰分析によるSHAP値の統計的検証（係数）.png", bbox_inches="tight")

#t値のプロット
plt.figure()
#plt.bar(range(len(t_values)), t_values)
plt.barh(feature_names, t_values)
#plt.axhline(y=2, color="r", linestyle='--')
plt.axvline(2, linestyle="--", color="red", label="t=2")
plt.xlabel("Features")
plt.ylabel("t-values")
plt.title("t-values")
#plt.xticks(range(len(t_values)), feature_names, rotation=90)
plt.savefig("契約予測（スタンダード_契約1カ月目）_分類モデル_xgboost_説明変数最適化_重回帰分析によるSHAP値の統計的検証（t値）.png", bbox_inches="tight")

#予測確率の取得
y_prob = xgb_model.predict(xgb_dtrain)

#SHAP値と予測確率のプロット
from sklearn.linear_model import LinearRegression
for i, feature in enumerate(feature_names):
    plt.figure()
    plt.scatter(shap_values[:, i], y_prob, label="Data points")
    
    #線形回帰を適用
    lin_reg = LinearRegression()
    lin_reg.fit(shap_values[:, i].reshape(-1, 1), y_prob)
    
    #線形回帰線を描写
    x_vals = np.linspace(shap_values[:, i].min(), shap_values[:, i].max(), 100)
    y_vals = lin_reg.predict(x_vals.reshape(-1, 1))
    plt.plot(x_vals, y_vals, color="red", label="Linear regression line")
    
    #プロットの装飾
    plt.xlabel(f"SHAP value for {feature}")
    plt.ylabel("Forecast probability")
    plt.title(f"SHAP values vs Forecast probability")
    plt.savefig(f"{feature}_LinearRegression.png")
    plt.close()
'''
