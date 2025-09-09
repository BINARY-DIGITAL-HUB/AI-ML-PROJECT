import streamlit as st
import pandas as pd
import joblib as jb

# @helper function to make prediction
def predict(features):

    # load model 
    model = jb.load('software_testing_model.jb')

    return model.predict([features])  
    # return True  
    

st.title("Software Automation Testing App")

st.header("Feature Input Options")

# Option to select manual input or CSV upload.
option = st.radio("Choose input method:", ("Manual Input", "Upload CSV"))

# Define feature names with meanings/acronyms.
features_def = [
    ("loc", "McCabe's line count of code"),
    ("v(g)", 'McCabe "cyclomatic complexity"'),
    ("ev(g)", 'McCabe "essential complexity"'),
    ("iv(g)", 'McCabe "design complexity"'),
    ("n", 'Halstead total operators + operands'),
    ("v", 'Halstead "volume"'),
    ("l", 'Halstead "program length"'),
    ("d", 'Halstead "difficulty"'),
    ("i", 'Halstead "intelligence"'),
    ("e", 'Halstead "effort"'),
    ("b", 'Halstead'),
    ("t", "Halstead's time estimator"),
    ("lOCode", "Halstead's line count"),
    ("lOComment", "Halstead's count of lines of comments"),
    ("lOBlank", "Halstead's count of blank lines"),
    ("locCodeAndComment", "Combined lines of code and comments"),
    ("uniq_Op", "Unique operators"),
    ("uniq_Opnd", "Unique operands"),
    ("total_Op", "Total operators"),
    ("total_Opnd", "Total operands"),
    ("branchCount", "Branch count of the flow graph")
]

if option == "Manual Input":
    st.subheader("Enter Features Manually")
    # Create a dictionary to hold feature values.
    features = {}
    for feat, meaning in features_def:
        label = f"{feat} ({meaning})"
        features[feat] = st.number_input(label, min_value=0.0, step=1.0, value=0.0)
        print(list(features.values()))
    if st.button("Predict Defect"):
        result = predict(list(features.values()))
        print(f' Prediction {result}')
        if result:
            st.markdown('<h2 style="color:green;">Defect Detected</h2>', unsafe_allow_html=True)
        else:
            st.markdown('<h2 style="color:blue;">No Defect Detected</h2>', unsafe_allow_html=True)

elif option == "Upload CSV":
    st.subheader("Upload CSV File")
    st.markdown("The CSV file should contain the following columns:")
    st.markdown(", ".join([feat for feat, _ in features_def]))
    
    csv_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)

            # converting type into numberic 
            for col in df.columns: 
                df[col] = df[col].astype('int64')

            # Verify all required columns exist in the CSV.
            required_cols = [feat for feat, _ in features_def]
            if all(col in df.columns for col in required_cols):
                # List to store prediction results.
                results = []
                # Iterate through each row, call predict, and store result.
                for index, row in df.iterrows():
                    feature_dict = {feat: row[feat] for feat in required_cols}
                    print(list(feature_dict.values()))
                    defect = predict(list(feature_dict.values()))
                    results.append({
                        "Test Case": index + 1,
                        "Defect Status": "Defect Detected" if defect else "No Defect Detected"
                    })
                results_df = pd.DataFrame(results)
                st.subheader("Prediction Results")
                st.dataframe(results_df)
            else:
                st.error("CSV file does not contain all the required columns.")
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")
