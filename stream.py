import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px




df=pd.read_csv("Loan approval prediction.csv")
df_Copy = df.copy()




#sidebar
option = st.sidebar.selectbox("Pick a choice:",['Home','EDA','ML'])
if option == 'Home':
    
   
    df_Copy = df_Copy[(df_Copy['person_age'] <= 65) & (df_Copy['person_age'] >= 20)]
    df_Copy = df_Copy[df_Copy['person_income'] > 200000]
    df_Copy = df_Copy[df_Copy['person_income'] <= 200000]
    df_Copy = df_Copy[(df_Copy['person_emp_length'] <= 31)]
    df_Copy = df_Copy[(df_Copy['loan_percent_income'] <= 0.55)]
    df_Copy =df_Copy[df_Copy['cb_person_cred_hist_length']==20]
    st.title("loan App")
    st.dataframe(df.head(20))

    st.title('Shape of the DataFrame')
    st.subheader('DataFrame Shape:')
    shape = df.shape
    st.write(f"The shape of the DataFrame is: {shape}")

    summary = df.describe()
    st.title('Summary Statistics of the Dataset')
    st.write(summary)

    st.title('Check for Missing Values')
    st.subheader('Missing Values Count:')
    missing_values = df.isnull().sum()
    st.write(missing_values)

    st.title('DataFrame Description for Object Columns')
    st.subheader('Description of Object Columns:')
    object_description = df.describe(include=object)
    st.write(object_description)

elif option == 'EDA':
    st.title("loan  EDA")
    fig, ax = plt.subplots()
    df_Copy['loan_status'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title('Distribution of Loan Status')
    ax.set_xlabel('Loan Status')
    ax.set_ylabel('Count')

    numeric_df = df_Copy.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    st.pyplot(plt)
    st.pyplot(fig)

    fig = plt.figure(figsize=(10, 6))
    sns.countplot(data=df_Copy, x='person_home_ownership')
    plt.title('Distribution of Loan Applicants by Home Ownership')


    st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df_Copy, x='loan_intent', ax=ax)
    ax.set_title('Loan Intent Distribution')
    plt.xticks(rotation=45)

    st.pyplot(fig)
        

    subset_features = ['loan_amnt', 'loan_int_rate', 'person_income', 'person_age', 'loan_status']
    fig = sns.pairplot(df_Copy[subset_features], hue='loan_status')
    st.pyplot(fig)

    st.title('Average Loan Amount by Loan Grade and Status')
    plt.figure(figsize=(12, 6))
    sns.barplot(x='loan_grade', y='loan_amnt', hue='loan_status', data=df_Copy, estimator=np.mean)
    plt.title('Average Loan Amount by Loan Grade and Status')
    plt.xlabel('Loan Grade')
    plt.ylabel('Average Loan Amount')
    plt.legend(title='Loan Status')
    st.pyplot(plt)

    st.title('Relation between Income and Loan Approval')
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='loan_status', y='person_income', data=df_Copy)
    plt.title('Relation between Income and Loan Approval')
    plt.xlabel('Loan Status (1=Approved, 0=Not Approved)')
    plt.ylabel('Applicant Income')
    st.pyplot(plt)

        
    st.subheader('3- Relationship of Some Features with Loan Status')
    columns = ['cb_person_default_on_file', 'person_home_ownership', 'loan_grade', 'loan_intent']
    selected_column = st.selectbox('Select a column to group by:', columns)
    grouped_data = df_Copy.groupby([df_Copy[selected_column], 'loan_status']).size().reset_index(name='count')
    fig = px.bar(
        grouped_data,
        x=selected_column,
        y='count',
        color='loan_status',
        title=f'Number of loan_status and {selected_column}'
    )

    fig.update_xaxes(title_text=selected_column)
    fig.update_yaxes(title_text='Count')
    fig.update_traces(
        text=grouped_data['count'],
        textposition='outside'
    )
    st.plotly_chart(fig)


    numerical_data = df.select_dtypes(include=['number']).columns
    selected_column = st.selectbox('Select a numerical column to display box plot:', numerical_data)
    fig = px.box(df, y=selected_column, title=f"Box Plot of {selected_column}")
    st.plotly_chart(fig, use_container_width=True, key=f"box_plot_{selected_column}")


    numerical_data = df.select_dtypes(include=['number']).columns
    selected_x = st.selectbox('Select a column for X axis:', numerical_data)
    selected_y = st.selectbox('Select a column for Y axis:', numerical_data)
    fig = px.scatter(df, x=selected_x, y=selected_y, title=f"Scatter Plot of {selected_x} vs {selected_y}")
    st.plotly_chart(fig, use_container_width=True, key=f"scatter_plot_{selected_x}_{selected_y}")
elif option == "ML":
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import streamlit as st

    def add_disposable_income(df):
        df['disposable_income'] = df['person_income'] - (df['loan_amnt'] * (1 + df['loan_int_rate']))

    def add_employment_stability(df):
        def employment_stability(emp_length):
            if emp_length < 5:
                return 'Unstable'
            elif 5 <= emp_length <= 10:
                return 'Relatively Stable'
            else:
                return 'Highly Stable'

        df['employment_stability'] = df['person_emp_length'].apply(employment_stability)

    def add_credit_history_stability(df):
        def credit_history_stability(hist_length):
            if hist_length < 5:
                return 'Short'
            elif 5 <= hist_length <= 10:
                return 'Medium'
            else:
                return 'Long'

        df['credit_history_stability'] = df['cb_person_cred_hist_length'].apply(credit_history_stability)

    def add_income_credit_interaction(df):
        df['income_credit_interaction'] = df['person_income'] * df['cb_person_cred_hist_length']

    def scale_numerical_features(df, columns):
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])

    def encode_categorical_features(df, columns):
        label_encoder = LabelEncoder()
        for col in columns:
            df[col] = label_encoder.fit_transform(df[col])

    def preprocess_data(df):
        add_disposable_income(df)
        add_employment_stability(df)
        add_credit_history_stability(df)
        add_income_credit_interaction(df)
        
        columns_to_scale = ['person_age', 'person_income', 'person_emp_length', 
                            'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
                            'cb_person_cred_hist_length']
        columns_to_encode = ['person_home_ownership', 'loan_intent', 'loan_grade', 
                            'cb_person_default_on_file', 'employment_stability', 
                            'credit_history_stability']

        scale_numerical_features(df, columns_to_scale)
        encode_categorical_features(df, columns_to_encode)

    preprocess_data(df_Copy)

    st.title("Loan Analysis Dashboard")

    st.write("Preprocessing completed successfully.")
    

    import pandas as pd
    import pickle
    import streamlit as st
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # تحميل البيانات (تأكد من استبدال هذا بالبيانات الفعلية لديك)
    # df_Copy = pd.read_csv('path_to_your_file.csv')

    # معالجة البيانات (تشفير وتحجيم)
    def preprocess_data(df):
        # تشفير الأعمدة النصية باستخدام LabelEncoder
        label_encoder = LabelEncoder()
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = label_encoder.fit_transform(df[col])
        
        # تحجيم الأعمدة العددية
        columns_to_scale = ['person_age', 'person_income', 'person_emp_length', 
                            'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
                            'cb_person_cred_hist_length', 'disposable_income', 
                            'income_credit_interaction']
        
        # التأكد من وجود الأعمدة في البيانات
        columns_to_scale = [col for col in columns_to_scale if col in df.columns]
        scaler = StandardScaler()
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        
        return df, label_encoder

    # معالجة البيانات
    df_Copy, label_encoder = preprocess_data(df_Copy)

    # تقسيم البيانات إلى السمات (features) والهدف (target)
    X = df_Copy.drop('loan_status', axis=1)  # Assuming 'loan_status' is the target variable
    y = df_Copy['loan_status']

    # تقسيم البيانات إلى تدريب واختبار
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # تدريب نموذج الانحدار اللوجستي
    clf = LogisticRegression(random_state=42, max_iter=500)
    clf.fit(X_train, y_train)

    # حفظ النموذج المدرب باستخدام pickle
    pickle.dump(clf, open('my_model.pkl', 'wb'))

    # واجهة المستخدم في Streamlit
    st.title("Loan Status Prediction")
    st.text("In this app, you can enter multiple features and predict the loan status.")

    # عرض المدخلات لجميع الميزات بشكل ديناميكي
    inputs = {}
    for column in df_Copy.drop('loan_status', axis=1).columns:
        # إذا كانت العمود عددية، يمكن إدخال القيم باستخدام number_input
        if df_Copy[column].dtype in ['int64', 'float64']:
            inputs[column] = st.number_input(f"Enter {column}", value=0.0, step=0.1)
        # إذا كانت العمود فئوية، يمكن إدخال القيم باستخدام selectbox
        elif df_Copy[column].dtype == 'object':
            unique_values = df_Copy[column].unique()
            # تحويل القيم الفئوية إلى قيم رقمية باستخدام LabelEncoder
            encoded_values = label_encoder.fit_transform(unique_values)
            inputs[column] = st.selectbox(f"Select {column}", options=encoded_values)

    # تحويل المدخلات إلى DataFrame
    input_data = pd.DataFrame([inputs])

    # معالجة المدخلات كما في البيانات الأصلية
    input_data, _ = preprocess_data(input_data)

    # تحميل النموذج المدرب
    clf = pickle.load(open('my_model.pkl', 'rb'))

    # إجراء التنبؤ بناءً على المدخلات
    btn = st.button("Submit")
    if btn:
        result = clf.predict(input_data)

        if result == 1:
            st.write("Loan Approved")
        else:
            st.write("Loan Not Approved")


    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
    from imblearn.combine import SMOTEENN
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import LabelEncoder
    import matplotlib.pyplot as plt
    import streamlit as st

    
    def preprocess_data(df):
        label_encoder = LabelEncoder()
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = label_encoder.fit_transform(df[col])
        return df

    df_Copy = preprocess_data(df_Copy)  
    X = df_Copy.drop('loan_status', axis=1)
    y = df_Copy['loan_status']

    smoteenn = SMOTEENN(sampling_strategy=0.25, random_state=42)
    X_resampled, y_resampled = smoteenn.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    model_choice = st.selectbox('Choose the model:', ['Random Forest', 'KNN', 'Logistic Regression', 'SVM', 'Decision Tree'])

    if model_choice == 'Random Forest':
        model = RandomForestClassifier(n_estimators=30, max_depth=15, min_samples_leaf=5, min_samples_split=20, random_state=42)
    elif model_choice == 'KNN':
        model = KNeighborsClassifier(n_neighbors=4, weights='uniform')
    elif model_choice == 'Logistic Regression':
        model = LogisticRegression(random_state=42, max_iter=500, C=100)
    elif model_choice == 'SVM':
        model = SVC(probability=True, random_state=42)
    elif model_choice == 'Decision Tree':
        model = DecisionTreeClassifier(random_state=0, max_depth=1000, min_samples_split=5, min_samples_leaf=15)

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    st.write(f"**Training Accuracy: {train_accuracy:.4f}**")

    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    st.write(f"**Testing Accuracy: {test_accuracy:.4f}**")

    st.write("**Classification Report (Testing Data):**")
    st.text(classification_report(y_test, y_test_pred))

    if hasattr(model, "predict_proba"):
        fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        st.pyplot(plt)

        st.write(f"**AUC: {roc_auc:.4f}**")
    else:
        st.write("**The selected model does not support probability predictions.**")

        
