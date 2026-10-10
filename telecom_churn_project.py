import pandas as pd
from io import StringIO
from sklearn.linear_model import LogisticRegression

# 1. Real-World Telecom Customer Dataset (Sample)
# Monthly_Charges_INR, Contract_Months, Internet_Service (0=No, 1=Yes), Churn (1=Left, 0=Stayed)
telecom_data = """Monthly_Charges,Contract_Months,Internet_Service,Churn
300,2,1,1
800,24,1,0
250,1,0,1
1200,36,1,0
400,6,0,0
900,3,1,1"""

df = pd.read_csv(StringIO(telecom_data))

# 2. Splitting Features and Target
X = df[['Monthly_Charges', 'Contract_Months', 'Internet_Service']]
Y = df['Churn']

# 3. Training the AI Churn Predictor
churn_model = LogisticRegression()
churn_model.fit(X, Y)
print("Status: Telecom Customer Churn AI Model Trained Successfully!")

# 4. Testing a New Scenario: 
# A customer paying 1000 INR/month, with only a 2-month contract, and having Internet Service
new_customer_csv = "Monthly_Charges,Contract_Months,Internet_Service\n1000,2,1"
new_customer_df = pd.read_csv(StringIO(new_customer_csv))

prediction = churn_model.predict(new_customer_df)

print("\n--- AI Customer Churn Prediction ---")
if prediction == 1:
    print("AI Decision: Warning! This customer is highly likely to LEAVE the network.")
else:
    print("AI Decision: This customer is loyal and will STAY with the network.")