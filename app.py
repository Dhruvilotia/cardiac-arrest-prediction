import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os

# Set page config
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical theme
st.markdown("""
<style>
    :root {
        --primary-blue: #0A5C8A;
        --secondary-blue: #1A85C1;
        --medical-red: #D32F2F;
        --light-bg: #F4F8FA;
    }
    .stApp {
        background-color: var(--light-bg);
    }
    h1, h2, h3 {
        color: var(--primary-blue);
    }
    .stButton>button {
        background-color: var(--medical-red);
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: var(--primary-blue);
        color: white;
    }
    .risk-card {
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .high-risk {
        background: linear-gradient(135deg, #D32F2F, #FF5252);
    }
    .low-risk {
        background: linear-gradient(135deg, #0A5C8A, #1A85C1);
    }
    .suggestion-box {
        background-color: white;
        border-left: 5px solid var(--primary-blue);
        padding: 1rem;
        margin-top: 1rem;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Fix cursor for interactive elements */
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] *,
    div[data-baseweb="slider"] > div,
    div[data-baseweb="slider"] * {
        cursor: pointer !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "heart_disease_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Data Storage Functions ---
HISTORY_FILE = "prediction_history.csv"

def init_history_file():
    if not os.path.exists(HISTORY_FILE):
        df = pd.DataFrame(columns=[
            "Date_Time", "Patient_ID", "Patient_Name", "Age", "Sex", "Chest_Pain_Type", 
            "Resting_BP", "Cholesterol", "Fasting_Blood_Sugar", "Resting_ECG", 
            "Max_Heart_Rate", "Exercise_Angina", "ST_Depression", "ST_Slope", 
            "Major_Vessels", "Thalassemia", "Prediction_Result"
        ])
        df.to_csv(HISTORY_FILE, index=False)

def save_prediction(patient_id, patient_name, features_dict, prediction_result):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    result_str = "High Risk" if prediction_result == 1 else "Low Risk"
    
    new_record = pd.DataFrame([{
        "Date_Time": now,
        "Patient_ID": patient_id,
        "Patient_Name": patient_name,
        "Age": features_dict['age'],
        "Sex": features_dict['sex'],
        "Chest_Pain_Type": features_dict['cp'],
        "Resting_BP": features_dict['trestbps'],
        "Cholesterol": features_dict['chol'],
        "Fasting_Blood_Sugar": features_dict['fbs'],
        "Resting_ECG": features_dict['restecg'],
        "Max_Heart_Rate": features_dict['thalach'],
        "Exercise_Angina": features_dict['exang'],
        "ST_Depression": features_dict['oldpeak'],
        "ST_Slope": features_dict['slope'],
        "Major_Vessels": features_dict['ca'],
        "Thalassemia": features_dict['thal'],
        "Prediction_Result": result_str
    }])
    
    # Append to CSV
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        df = pd.concat([df, new_record], ignore_index=True)
    else:
        df = new_record
    df.to_csv(HISTORY_FILE, index=False)

def get_patient_history(patient_id=None, patient_name=None):
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()
    
    df = pd.read_csv(HISTORY_FILE)
    if patient_id and str(patient_id).strip() != "":
        return df[df["Patient_ID"].astype(str) == str(patient_id)]
    elif patient_name and str(patient_name).strip() != "":
        # Case insensitive match
        return df[df["Patient_Name"].str.lower() == str(patient_name).lower()]
    return pd.DataFrame()

def get_all_history():
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()
    return pd.read_csv(HISTORY_FILE)

def delete_record(index):
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        if index in df.index:
            df = df.drop(index)
            df.to_csv(HISTORY_FILE, index=False)

def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
        init_history_file()

# Initialize data file
init_history_file()

# --- Sidebar Navigation & Inputs ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3004/3004071.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Prediction System", "Prediction History", "Understanding Medical Terms", "About Us"])
st.sidebar.markdown("---")

if page == "Prediction System":
    st.sidebar.title("Patient Inputs")
    st.sidebar.markdown("Enter patient clinical parameters below:")

    patient_name = st.sidebar.text_input("Patient Name")
    patient_id = st.sidebar.text_input("Patient ID")
    
    st.sidebar.markdown("---")

    age = st.sidebar.slider("Age", 20, 100, 50)
    sex_opts = {"Male": 1, "Female": 0}
    sex_label = st.sidebar.selectbox("Sex", options=list(sex_opts.keys()))
    sex = sex_opts[sex_label]

    cp_opts = {
        "0: Typical Angina": 0, 
        "1: Atypical Angina": 1, 
        "2: Non-anginal Pain": 2, 
        "3: Asymptomatic": 3
    }
    cp_label = st.sidebar.selectbox("Chest Pain Type", options=list(cp_opts.keys()))
    cp = cp_opts[cp_label]

    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.sidebar.slider("Cholesterol (mg/dl)", 126, 564, 200)

    fbs_opts = {
        "No (<= 120 mg/dl)": 0,
        "Yes (> 120 mg/dl)": 1
    }
    fbs_label = st.sidebar.selectbox("Fasting Blood Sugar", options=list(fbs_opts.keys()))
    fbs = fbs_opts[fbs_label]

    restecg_opts = {
        "0: Normal": 0,
        "1: ST-T wave abnormality": 1,
        "2: Left ventricular hypertrophy": 2
    }
    restecg_label = st.sidebar.selectbox("Resting ECG", options=list(restecg_opts.keys()))
    restecg = restecg_opts[restecg_label]

    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 71, 202, 150)

    exang_opts = {"No": 0, "Yes": 1}
    exang_label = st.sidebar.selectbox("Exercise Induced Angina", options=list(exang_opts.keys()))
    exang = exang_opts[exang_label]

    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", 0.0, 6.2, 1.0, 0.1)

    slope_opts = {
        "0: Upsloping": 0,
        "1: Flat": 1,
        "2: Downsloping": 2
    }
    slope_label = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", options=list(slope_opts.keys()))
    slope = slope_opts[slope_label]

    ca = st.sidebar.slider("Number of Major Vessels Colored by Flourosopy", 0, 4, 0)

    thal_opts = {
        "1: Normal": 1,
        "2: Fixed Defect": 2,
        "3: Reversable Defect": 3
    }
    thal_label = st.sidebar.selectbox("Thalassemia", options=list(thal_opts.keys()))
    thal = thal_opts[thal_label]

    # --- Main Page ---
    col1, col2 = st.columns([1.5, 10])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/875/875601.png", width=100)
    with col2:
        st.title("Heart Disease Prediction System")

    st.markdown("### Clinical Assessment Dashboard")
    st.write("This application uses advanced Machine Learning to assess the risk of heart disease based on clinical parameters. Provide the inputs in the sidebar and click the button below to generate a prediction.")

    # Show past history if this is a returning patient
    if (patient_name and patient_name.strip() != "") or (patient_id and patient_id.strip() != ""):
        past_records = get_patient_history(patient_id, patient_name)
        if not past_records.empty:
            st.info(f"**Patient Record Found**: Found {len(past_records)} previous assessment(s) for patient.")
            with st.expander("View Previous Assessments"):
                st.dataframe(past_records[["Date_Time", "Prediction_Result", "Age", "Resting_BP", "Cholesterol", "Max_Heart_Rate"]], use_container_width=True)

    if st.button("Predict Cardiac Risk", use_container_width=True):
        if not patient_name and not patient_id:
            st.warning("Please enter a Patient Name or ID in the sidebar to save this prediction record.")
        
        # Prepare features
        feature_vals = {
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 
            'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang, 
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        features_df = pd.DataFrame(features, columns=list(feature_vals.keys()))
        
        # Predict
        prediction = model.predict(features_df)[0]
        
        # Save Prediction Logic
        if patient_name or patient_id:
            save_prediction(patient_id, patient_name, feature_vals, prediction)
            st.success("✅ Prediction result successfully saved to patient history.")
        
        st.markdown("---")
        res_col, chart_col = st.columns([1, 1])
        
        with res_col:
            st.subheader("Assessment Result")
            if prediction == 1:
                st.markdown(
                    """
                    <div class="risk-card high-risk">
                        <h1 style='color:white; margin-bottom: 0;'>⚠️ High Risk</h1>
                        <p style='font-size: 1.2rem; margin-top: 5px;'>Patient shows indications of cardiovascular disease.</p>
                    </div>
                    """, unsafe_allow_html=True
                )
                
                st.markdown("""
                <div class="suggestion-box">
                    <h4>🩺 Recommended Actions</h4>
                    <ul>
                        <li>Immediate consultation with a cardiologist.</li>
                        <li>Conduct further diagnostic tests (e.g., Angiography, Echocardiogram).</li>
                        <li>Strict monitoring of Blood Pressure and Cholesterol.</li>
                        <li>Implement lifestyle and dietary interventions immediately.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(
                    """
                    <div class="risk-card low-risk">
                        <h1 style='color:white; margin-bottom: 0;'>✅ Low Risk</h1>
                        <p style='font-size: 1.2rem; margin-top: 5px;'>Patient currently shows low probability of cardiovascular disease.</p>
                    </div>
                    """, unsafe_allow_html=True
                )
                
                st.markdown("""
                <div class="suggestion-box">
                    <h4>🌱 Health Maintenance</h4>
                    <ul>
                        <li>Continue regular cardiovascular exercise (150 mins/week).</li>
                        <li>Maintain a balanced, heart-healthy diet.</li>
                        <li>Schedule routine annual check-ups.</li>
                        <li>Monitor stress levels and ensure adequate sleep.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
        with chart_col:
            st.subheader("Patient Vitals Overview")
            categories = ['Testing BP (sys)', 'Cholesterol', 'Max Heart Rate']
            patient_values = [
                min((trestbps / 120) * 100, 150),
                min((chol / 200) * 100, 150),
                min((thalach / 150) * 100, 150)
            ]
            optimal_values = [100, 100, 100]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=categories,
                y=patient_values,
                name='Patient',
                marker_color='#D32F2F' if prediction == 1 else '#0A5C8A'
            ))
            fig.add_trace(go.Scatter(
                x=categories,
                y=optimal_values,
                name='Optimal Boundary',
                mode='lines+markers',
                line=dict(color='green', width=2, dash='dash')
            ))
            fig.update_layout(
                barmode='group',
                plot_bgcolor='rgba(0,0,0,0)',
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis=dict(title="Relative Index (100 = Optimal)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
                
        st.markdown("### Key Patient Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Blood Pressure", f"{trestbps} mmHg", "High" if trestbps > 130 else "Normal", delta_color="inverse")
        m2.metric("Cholesterol", f"{chol} mg/dl", "High" if chol > 200 else "Normal", delta_color="inverse")
        m3.metric("Max Heart Rate", f"{thalach} bpm", "Low" if thalach < 100 else "Normal")
        m4.metric("ST Depression", f"{oldpeak}", "Warning" if oldpeak > 1.0 else "Normal", delta_color="inverse")

elif page == "Prediction History":
    st.title("Prediction History")
    st.write("View and manage previously generated patient prediction profiles.")
    
    st.markdown("---")
    
    df = get_all_history()
    
    if df.empty:
        st.info("No prediction records found. Make a prediction in the Prediction System page to save data.")
    else:
        # Search & Filter functionality
        c1, c2, c3 = st.columns(3)
        search_query = c1.text_input("🔍 Search by Patient Name or ID")
        res_filter = c2.selectbox("Filter by Result", ["All", "High Risk", "Low Risk"])
        
        # Apply filters
        df_filtered = df.copy()
        if search_query:
            mask = (
                df_filtered["Patient_Name"].astype(str).str.contains(search_query, case=False, na=False) |
                df_filtered["Patient_ID"].astype(str).str.contains(search_query, case=False, na=False)
            )
            df_filtered = df_filtered[mask]
            
        if res_filter != "All":
            df_filtered = df_filtered[df_filtered["Prediction_Result"] == res_filter]
            
        # Display Data Grid
        st.markdown("### 📋 Prediction Records")
        
        # Style dataframe to colorize predictions
        def highlight_results(s):
            if s.name == 'Prediction_Result':
                return ['background-color: #ffebee; color: #c62828' if v == 'High Risk' else 'background-color: #e8f5e9; color: #2e7d32' for v in s]
            return ['' for v in s]
        
        st.dataframe(
            df_filtered.style.apply(highlight_results),
            use_container_width=True,
            column_order=["Date_Time", "Patient_ID", "Patient_Name", "Prediction_Result", "Age", "Sex", "Resting_BP", "Cholesterol"],
            hide_index=True
        )
        
        # Actions Interface
        st.markdown("### ⚙️ Actions")
        
        action_col1, action_col2 = st.columns([2, 1])
        
        with action_col1:
            st.markdown("#### View Details")
            record_idx = st.selectbox("Select a Record Date/Time to View Details or Delete:", options=df_filtered.index, format_func=lambda x: f"{df_filtered.loc[x, 'Date_Time']} - {df_filtered.loc[x, 'Patient_Name']}")
            
            if record_idx is not None:
                if st.button("👁️ View Full Details", key="view_btn"):
                    with st.expander("Record Details", expanded=True):
                        st.json(df_filtered.loc[record_idx].to_dict())
                
                if st.button("🗑️ Delete Selected Record", type="primary", key="del_btn"):
                    delete_record(record_idx)
                    st.success("Record deleted successfully.")
                    st.rerun()
                    
        with action_col2:
            st.markdown("#### Data Management")
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download History as CSV",
                data=csv,
                file_name='prediction_history.csv',
                mime='text/csv',
                use_container_width=True
            )
            
            st.markdown("   ") # Spacer
            if st.button("🚨 Clear All History", type="primary", use_container_width=True):
                clear_history()
                st.success("All history has been cleared.")
                st.rerun()

elif page == "Understanding Medical Terms":

    st.title("Understanding Medical Terms")
    st.write("This page explains the clinical parameters used in our prediction model in simple, easy-to-understand language.")
    
    st.markdown("---")
    
    terms = {
        "Chest Pain Type": "Describes the type of chest pain a patient feels. Certain chest pain types may indicate heart problems.",
        "Resting Blood Pressure": "Blood pressure measured when the patient is relaxed. High values may increase risk of heart disease.",
        "Cholesterol": "Fat present in blood. High cholesterol can block arteries.",
        "Fasting Blood Sugar": "Blood sugar level measured after fasting for several hours.",
        "Resting ECG": "Test that measures electrical activity of the heart while resting.",
        "Maximum Heart Rate Achieved": "Highest heart rate reached during exercise testing.",
        "Exercise Induced Angina": "Chest pain that occurs during physical activity.",
        "ST Depression": "Change in ECG during exercise that may indicate reduced blood flow to the heart.",
        "Slope of ST Segment": "Shape of the ECG signal during peak exercise.",
        "Number of Major Vessels": "Number of coronary arteries with blockage detected by imaging.",
        "Thalassemia": "A blood disorder that can affect oxygen transport in the body."
    }
    
    for term, explanation in terms.items():
        st.markdown(f"**{term}**")
        st.markdown(f"Explanation: *{explanation}*")
        st.markdown("<br>", unsafe_allow_html=True)

elif page == "About Us":
    st.title("About Us")
    st.markdown("---")
    
    st.header("Description of the Heart Disease Prediction System")
    st.write("Heart Sense AI is an advanced predictive tool designed to evaluate cardiovascular health. By analyzing key clinical parameters, it provides an instant assessment of a patient's risk for heart disease.")
    
    st.header("Purpose of the project")
    st.write("The primary goal of this project is to create an accessible, user-friendly tool that helps individuals and healthcare professionals quickly identify potential heart risks, enabling early intervention and preventive care.")
    
    st.header("How machine learning helps detect heart disease risk")
    st.write("Machine learning algorithms analyze historical medical data to identify complex patterns and correlations between various health metrics (like age, cholesterol, and blood pressure) and heart disease. When new patient data is inputted, the trained model uses these learned patterns to accurately predict the likelihood of cardiovascular issues.")
    
    st.header("Information about the dataset used")
    st.write("This system is trained on a comprehensive clinical dataset containing hundreds of patient records. Each record includes multiple vital signs, test results, and a final diagnosis indicating the presence or absence of heart disease, ensuring the model's predictions are rooted in real-world medical data.")
    
    st.header("Benefits for patients and healthcare awareness")
    st.write("- **Early Detection**: Identifies risks before severe symptoms appear.")
    st.write("- **Accessible Screening**: Provides a quick preliminary assessment from anywhere.")
    st.write("- **Health Literacy**: Educates users about vital heart health metrics through clear explanations.")
    st.write("- **Proactive Care**: Encourages users to seek timely medical advice and adopt healthier lifestyles.")
    
    st.header("Developer information")
    st.write("This project was developed by an AI enthusiast passionate about leveraging machine learning to solve real-world healthcare challenges. The aim is to bridge the gap between complex medical diagnostics and accessible technology.")
