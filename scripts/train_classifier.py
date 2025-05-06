import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load enhanced data
df = pd.read_csv('outputs/enhanced_player_stats.csv')
df = df.dropna()


# Create performance labels
# 0 = Poor (<10), 1 = Average (10–30), 2 = Good (>30)
def label_performance(runs):
    if runs < 10:
        return 0
    elif runs <= 30:
        return 1
    else:
        return 2


df['performance'] = df['total_runs'].apply(label_performance)

# Extract year
df['year'] = pd.to_datetime(df['date']).dt.year

# Encode categorical features
le_venue = LabelEncoder()
le_team = LabelEncoder()
df['venue_enc'] = le_venue.fit_transform(df['venue'])
df['team_enc'] = le_team.fit_transform(df['batting_team'])

# Define features and target
features = ['balls_faced', 'strike_rate', 'fours', 'sixes', 'venue_enc', 'team_enc', 'year']
X = df[features]
y = df['performance']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Classification Accuracy: {accuracy:.2f}\n")
print("📄 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Poor', 'Average', 'Good']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Poor', 'Average', 'Good'],
            yticklabels=['Poor', 'Average', 'Good'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("📊 Confusion Matrix - Player Performance")
plt.tight_layout()
plt.savefig("outputs/performance_confusion_matrix.png")
plt.show()
