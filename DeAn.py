import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

data = pd.read_csv(r"C:\Users\dangb\PycharmProjects\PythonProject\Dataset\bank.csv", sep=',')

plt.figure(figsize=(10, 6))
sns.histplot(data['age'], bins=20, kde=True, color='skyblue')
plt.title("Phân bố độ tuổi khách hàng")
plt.xlabel("Tuổi")
plt.ylabel("Số lượng")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x=data['balance'], color='lightcoral')
plt.title("Boxplot số dư tài khoản (balance)")
plt.xlabel("Số dư")
plt.grid(axis='x')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x=data['job'], order=data['job'].value_counts().index)
plt.title("Phân bố nghề nghiệp khách hàng")
plt.xlabel("Nghề nghiệp")
plt.ylabel("Số lượng")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x=data['marital'])
plt.title("Phân bố tình trạng hôn nhân khách hàng")
plt.xlabel("Tình trạng hôn nhân")
plt.ylabel("Số lượng")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

features = ['age', 'job', 'marital', 'education', 'balance', 'housing', 'loan']
data = data[features]

label_encoders = {}
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
warnings.filterwarnings("ignore", category=UserWarning)

eda_data = data.copy()

conditions = [
    (data['balance'] < 0) | ((data['loan'] == 1) & (data['housing'] == 1)),
    ((data['balance'] >= 0) & (data['balance'] < 1000)),
    (data['balance'] >= 1000)
]
risk_labels = ['High', 'Medium', 'Low']
data['risk_level'] = np.select(conditions, risk_labels, default='Medium')

eda_data = data.copy()

if 'marital' in label_encoders:
    eda_data['marital_label'] = label_encoders['marital'].inverse_transform(eda_data['marital'])

plt.figure(figsize=(10,6))
sns.countplot(data=eda_data, x='marital_label', hue='risk_level', palette='Set2')
plt.title("Phân bố mức độ rủi ro theo tình trạng hôn nhân", fontsize=14)
plt.xlabel("Tình trạng hôn nhân")
plt.ylabel("Số lượng khách hàng")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

if 'education' in label_encoders:
    eda_data['education_label'] = label_encoders['education'].inverse_transform(eda_data['education'])

plt.figure(figsize=(10,6))
sns.countplot(data=eda_data, x='education_label', hue='risk_level', palette='Set3')
plt.title("Phân bố mức độ rủi ro theo trình độ học vấn", fontsize=14)
plt.xlabel("Trình độ học vấn")
plt.ylabel("Số lượng khách hàng")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
numeric_cols = ['age', 'balance', 'loan', 'housing']
corr_matrix = eda_data[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("🔥 Ma trận tương quan giữa các đặc trưng số học", fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(data=eda_data, x='age', hue='risk_level', multiple='stack', palette='Set2', bins=20)
plt.title("Phân phối mức độ rủi ro theo độ tuổi", fontsize=14)
plt.xlabel("Tuổi")
plt.ylabel("Số lượng")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

le_risk = LabelEncoder()
data['risk_level_encoded'] = le_risk.fit_transform(data['risk_level'])

X_dt = data.drop(['risk_level', 'risk_level_encoded'], axis=1)
y_dt = data['risk_level_encoded']

X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X_dt, y_dt, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train_dt, y_train_dt)

y_pred_dt = model.predict(X_test_dt)
print("\nĐộ chính xác mô hình Decision Tree:", accuracy_score(y_test_dt, y_pred_dt))
print("\n Báo cáo phân loại (Decision Tree):\n", classification_report(y_test_dt, y_pred_dt))

cm = confusion_matrix(y_test_dt, y_pred_dt)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_risk.classes_, yticklabels=le_risk.classes_)
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.show()

plt.figure(figsize=(15, 8))
plot_tree(model, feature_names=X_dt.columns, class_names=le_risk.classes_, filled=True)
plt.title("Decision Tree - Phân loại rủi ro tài chính")
plt.show()

investment_mapping = {
    'High': ['Quỹ tiết kiệm an toàn', 'Bảo hiểm nhân thọ'],
    'Medium': ['Trái phiếu chính phủ', 'Chứng chỉ quỹ'],
    'Low': ['Cổ phiếu', 'Bất động sản', 'Quỹ đầu tư mạo hiểm']
}

data['investment_suggestions'] = data['risk_level'].map(investment_mapping)
data['investment_recommendation_rulebased'] = data['investment_suggestions'].apply(lambda x: x[0])

print("\n Gợi ý đầu tư (Rule-Based) - 10 khách hàng đầu tiên:")
print(data[['age', 'job', 'balance', 'risk_level', 'investment_recommendation_rulebased']].head(10))

product_profiles = {
    'Quỹ tiết kiệm an toàn': {'age': 55, 'balance': 100, 'loan': 1, 'housing': 1},
    'Bảo hiểm nhân thọ': {'age': 50, 'balance': 200, 'loan': 1, 'housing': 1},
    'Trái phiếu chính phủ': {'age': 40, 'balance': 500, 'loan': 0, 'housing': 1},
    'Chứng chỉ quỹ': {'age': 35, 'balance': 800, 'loan': 0, 'housing': 1},
    'Cổ phiếu': {'age': 30, 'balance': 1500, 'loan': 0, 'housing': 0},
    'Bất động sản': {'age': 45, 'balance': 2000, 'loan': 0, 'housing': 1},
    'Quỹ đầu tư mạo hiểm': {'age': 28, 'balance': 3000, 'loan': 0, 'housing': 0},
}

def content_based_recommendation(row):
    similarities = {}
    for product, profile in product_profiles.items():
        vec1 = np.array([row['age'], row['balance'], row['loan'], row['housing']])
        vec2 = np.array([profile['age'], profile['balance'], profile['loan'], profile['housing']])
        sim = cosine_similarity([vec1], [vec2])[0][0]
        similarities[product] = sim
    return max(similarities, key=similarities.get)

data['investment_recommendation_content'] = data.apply(content_based_recommendation, axis=1)

print("\n Gợi ý đầu tư (Content-Based) - 10 khách hàng đầu tiên:")
print(data[['age', 'balance', 'risk_level', 'investment_recommendation_content']].head(10))

weights = {
    'rule_based': 0.8,
    'content_based': 0.9
}

data['score_fusion'] = data.apply(lambda row: {
    row['investment_recommendation_rulebased']: weights['rule_based'],
    row['investment_recommendation_content']: weights['content_based']
}, axis=1)

data['final_recommendation'] = data['score_fusion'].apply(lambda scores: max(scores, key=scores.get))

print("\n Gợi ý đầu tư (Score-Level Fusion) - 10 khách hàng đầu tiên:")
print(data[['age', 'balance', 'risk_level', 'final_recommendation']].head(10))

data[['age', 'balance', 'risk_level', 'final_recommendation']].to_excel('goi_y_dau_tu_score_level_fusion.xlsx', index=False)
print("\n Kết quả Score-Level Fusion đã lưu vào file: goi_y_dau_tu_score_level_fusion.xlsx")

data.to_csv("output_final_recommendations.csv", index=False)
print("\n Đã lưu file kết quả vào 'output_final_recommendations.csv'")

def classify_risk_level(age, balance, loan, housing):
    if balance < 0 or (loan == 1 and housing == 1):
        return 'High'
    elif 0 <= balance < 1000:
        return 'Medium'
    else:
        return 'Low'

def get_rule_based_recommendation(risk_level):
    return investment_mapping.get(risk_level, ["Không xác định"])[0]

def get_content_based_recommendation(customer):
    similarities = {}
    for product, profile in product_profiles.items():
        vec1 = np.array([customer['age'], customer['balance'], customer['loan'], customer['housing']])
        vec2 = np.array([profile['age'], profile['balance'], profile['loan'], profile['housing']])
        sim = cosine_similarity([vec1], [vec2])[0][0]
        similarities[product] = sim
    return max(similarities, key=similarities.get)

report_dict = classification_report(y_test_dt, y_pred_dt, target_names=le_risk.classes_, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()
df_metrics = df_report.loc[le_risk.classes_, ['precision', 'recall', 'f1-score']]

plt.figure(figsize=(10,6))
df_metrics.plot(kind='bar')
plt.title("Biểu đồ Precision - Recall - F1-score theo từng lớp rủi ro (Decision Tree)")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))
sns.countplot(data=data, x='risk_level', hue='risk_level', palette='Set2', legend=False)
plt.title("Phân bố các mức độ rủi ro")
plt.xlabel("Mức độ rủi ro")
plt.ylabel("Số lượng khách hàng")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

fusion_recommendation_counts = data['final_recommendation'].value_counts()
plt.figure(figsize=(6,6))
fusion_recommendation_counts.plot.pie(
    autopct='%1.1f%%',
    startangle=90,
    colors=sns.color_palette('Set3')
)
plt.title("Tỷ lệ các danh mục đầu tư được gợi ý (Score-Level Fusion)")
plt.ylabel("")
plt.tight_layout()
plt.show()

X_rf = data.drop(['risk_level', 'risk_level_encoded', 'investment_suggestions',
                  'investment_recommendation_rulebased', 'investment_recommendation_content',
                  'score_fusion', 'final_recommendation'], axis=1)
y_rf = data['risk_level_encoded']

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_rf = scaler.fit_transform(X_train_rf)
X_test_rf = scaler.transform(X_test_rf)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_rf, y_train_rf)

y_pred_rf = rf.predict(X_test_rf)

y_pred_label_rf = le_risk.inverse_transform(y_pred_rf)

print("\n[Classification Report - Random Forest]:")
print(classification_report(y_test_rf, y_pred_rf, target_names=le_risk.classes_))

df_result_rf = pd.DataFrame(X_test_rf, columns=X_rf.columns)
df_result_rf['RiskLevel'] = y_pred_label_rf

df_result_rf['InvestmentSuggestion'] = df_result_rf['RiskLevel'].map(lambda x: investment_mapping[x][0])

print("\n[Danh sách gợi ý đầu tư cho 10 khách hàng đầu tiên (Random Forest)]:")
print(df_result_rf[['RiskLevel', 'InvestmentSuggestion']].head(10))

df_result_rf[['RiskLevel', 'InvestmentSuggestion']].to_excel('goi_y_dau_tu_model2.xlsx', index=False)
print("\n Kết quả Random Forest đã lưu vào file: goi_y_dau_tu_model2.xlsx")

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test_rf, y_pred_rf), annot=True, fmt='d', cmap='Blues',
            xticklabels=le_risk.classes_, yticklabels=le_risk.classes_)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x=y_pred_label_rf, hue=y_pred_label_rf, palette='Set2', legend=False)
plt.title("Phân bố mức độ rủi ro dự đoán (Random Forest)")
plt.xlabel("Risk Level")
plt.ylabel("Số lượng khách hàng")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

report_dict_rf = classification_report(y_test_rf, y_pred_rf, target_names=le_risk.classes_, output_dict=True)
df_report_rf = pd.DataFrame(report_dict_rf).transpose()
df_metrics_rf = df_report_rf.loc[le_risk.classes_, ['precision', 'recall', 'f1-score']]

plt.figure(figsize=(8, 5))
df_metrics_rf.plot(kind='bar')
plt.title("Precision - Recall - F1-score theo từng lớp (Random Forest)")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

investment_counts_rf = df_result_rf['InvestmentSuggestion'].value_counts()

plt.figure(figsize=(6, 6))
investment_counts_rf.plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set3'))
plt.title("Tỷ lệ các danh mục đầu tư được gợi ý (Random Forest)")
plt.ylabel("")
plt.tight_layout()
plt.show()

print("\n Nhập dữ liệu khách hàng mới để đánh giá rủi ro và gợi ý đầu tư:")
age = int(input("Tuổi: "))
job = input("Nghề nghiệp (ví dụ: admin, technician, blue-collar, etc.): ").lower()
marital = input("Tình trạng hôn nhân (single/married/divorced): ").lower()
education = input("Trình độ học vấn (primary/secondary/tertiary/unknown): ").lower()
balance = int(input("Số dư tài khoản: "))
housing = input("Có vay nhà? (yes/no): ").lower()
loan = input("Có vay cá nhân? (yes/no): ").lower()

new_customer = pd.DataFrame([{
    'age': age,
    'job': job,
    'marital': marital,
    'education': education,
    'balance': balance,
    'housing': housing,
    'loan': loan
}])

for col in ['job', 'marital', 'education', 'housing', 'loan']:
    if col in label_encoders:
        known_classes = set(label_encoders[col].classes_)
        new_customer[col] = new_customer[col].apply(lambda x: x if x in known_classes else 'unknown')
        new_customer[col] = label_encoders[col].transform(new_customer[col])

X_new = new_customer[X_dt.columns]

predicted_risk_encoded = model.predict(X_new)[0]
predicted_risk = le_risk.inverse_transform([predicted_risk_encoded])[0]

loan_val = int(new_customer['loan'].values[0])
housing_val = int(new_customer['housing'].values[0])
risk_level_rulebased = classify_risk_level(age, balance, loan_val, housing_val)

investment_recommendation_rulebased = get_rule_based_recommendation(risk_level_rulebased)

investment_recommendation_content = get_content_based_recommendation({
    'age': age,
    'balance': balance,
    'loan': loan_val,
    'housing': housing_val
})

score_fusion = {}
score_fusion[investment_recommendation_rulebased] = score_fusion.get(investment_recommendation_rulebased, 0) + weights['rule_based']
score_fusion[investment_recommendation_content] = score_fusion.get(investment_recommendation_content, 0) + weights['content_based']

final_recommendation = max(score_fusion, key=score_fusion.get)

print("\n KẾT QUẢ ĐÁNH GIÁ KHÁCH HÀNG MỚI:")
print(f"Rule-Based Risk Level: {risk_level_rulebased}")
print(f"Decision Tree Risk Prediction: {predicted_risk}")
print(f"Gợi ý đầu tư (Rule-Based): {investment_recommendation_rulebased}")
print(f"Gợi ý đầu tư (Content-Based): {investment_recommendation_content}")
print("\nĐiểm số của các gợi ý (Score-Level Fusion):", score_fusion)
print("GỢI Ý CUỐI CÙNG (Score-Level Fusion):", final_recommendation)

