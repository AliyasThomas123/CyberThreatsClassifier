import pickle,joblib
import pandas as pd
from datetime import datetime
import streamlit as st
from PIL import Image
from sklearn.preprocessing import LabelEncoder ,StandardScaler
from sklearn.metrics import accuracy_score , classification_report
FIELDS_TO_BE_REMOVED = ['MitreTechniques', 'Usage',         
'ActionGrouped',       
'ActionGranular',       
'EmailClusterId',       
'ThreatFamily',         
'ResourceType' ,        
'Roles',                
'AntispamDirection',   
'SuspicionLevel',       
'LastVerdict',
'IncidentGrade'          
]

ecoded_class = {'Timestamp': {'Afternoon': 0, 'Evening': 1, 'Morning': 2, 'Night': 3}, 'Category': {'Collection': 0, 'CommandAndControl': 1, 'CredentialAccess': 2, 'CredentialStealing': 3, 'DefenseEvasion': 4, 'Discovery': 5, 'Execution': 6, 'Exfiltration': 7, 'Exploit': 8, 'Impact': 9, 'InitialAccess': 10, 'LateralMovement': 11, 'Malware': 12, 'Persistence': 13, 'PrivilegeEscalation': 14, 'Ransomware': 15, 'SuspiciousActivity': 16, 'UnwantedSoftware': 17, 'Weaponization': 18, 'WebExploit': 19}, 'IncidentGrade': {'BenignPositive': 0, 'FalsePositive': 1, 'TruePositive': 2}, 'EntityType': {'ActiveDirectoryDomain': 0, 'AmazonResource': 1, 'AzureResource': 2, 'Blob': 3, 'BlobContainer': 4, 'CloudApplication': 5, 'CloudLogonRequest': 6, 'CloudLogonSession': 7, 'Container': 8, 'ContainerImage': 9, 'ContainerRegistry': 10, 'File': 11, 'GenericEntity': 12, 'GoogleCloudResource': 13, 'IoTDevice': 14, 'Ip': 15, 'KubernetesCluster': 16, 'KubernetesNamespace': 17, 'KubernetesPod': 18, 'Machine': 19, 'MailCluster': 20, 'MailMessage': 21, 'Mailbox': 22, 'MailboxConfiguration': 23, 'Malware': 24, 'Nic': 25, 'OAuthApplication': 26, 'Process': 27, 'RegistryKey': 28, 'RegistryValue': 29, 'SecurityGroup': 30, 'Url': 31, 'User': 32}, 'EvidenceRole': {'Impacted': 0, 'Related': 1}}
model_path = r'./best_decision_tree_model.pkl'
scalar =joblib.load("scaler.pkl") 
class_mapper ={
    
    
       2: 'TruePositive' ,
       1: 'FalsePositive',
       0: 'BenignPositive'
    
}

def process_timestamp(date_str):
    timestamp = datetime.strptime(date_str , "%Y-%m-%dT%H:%M:%S.%fZ")
    hour = timestamp.hour
    if hour < 6:
        return "Night"  # Night
    elif hour < 12:
        return 'Morning'  # Morning
    elif hour < 18:
        return 'Afternoon'  # Afternoon
    else:
        return 'Evening' # Evening

def make_prediction(df):
    
    final_df = df
    ev = df['IncidentGrade']
    
    
    df.drop(columns=[col for col in FIELDS_TO_BE_REMOVED if col in df.columns],axis=1,inplace=True)
   
    for col in df.columns:        
       encoded_col_value = ecoded_class.get(col,'')
       if encoded_col_value :
           df[col]=df[col].map(encoded_col_value)
    
    df= scalar.transform(df)
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    pred = model.predict(df)
    print("PRED",pred)
    pred_class = [class_mapper[x] for x in pred]
    acc =accuracy_score(ev , pred_class)
    st.success(acc)
    final_df['IncidentGrade']  = pred_class
    return {"df" : final_df,"class":pred_class}



def color_classes(row):
    if row['IncidentGrade'] == 'TruePositive':
        return ['background-color: green'] * len(row)
    elif row['IncidentGrade'] == 'TrueNegative':
        return ['background-color: blue'] * len(row)
    elif row['IncidentGrade'] == 'BenignPositive':
        return ['background-color: orange'] * len(row)
    else:
        return [''] * len(row)  # No color for 

# Apply the color mapping to the entire DataFrame





# Sidebar content


# Main panel
st.title("CyberSec")
st.write("CyberSec Incident Classifier")
image = Image.open('./cyber.png')

# Resize the image while maintaining aspect ratio
new_height = 200  # Adjust the height as needed
aspect_ratio = image.width / image.height
new_width = int(new_height * aspect_ratio)
resized_image = image.resize((new_width, new_height))

# Display the resized image in the sidebar
st.sidebar.image(resized_image)

uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx", "txt" ,"json"])

# Predict button at center
if st.sidebar.button("Predict"):
    if uploaded_file is not None:
        st.sidebar.success("File uploaded successfully!")
        if uploaded_file.name.endswith(".json"):
            import json
            data = json.load(uploaded_file)
            df= pd.DataFrame(data)
        if uploaded_file.name.endswith(".csv"):
           df = pd.read_csv(uploaded_file)
        df= pd.DataFrame(data)
        
        df['Timestamp']=df['Timestamp'].apply(process_timestamp)
       
        st.write("Running prediction...")
        response= make_prediction(df)
        if response :
            #st.success(response['df'])
            df  = response['df']
            class_counts = df['IncidentGrade'].value_counts()
            total_records = len(df)
            predicted_percentages = class_counts / total_records * 100
            styled_df = df.style.apply(color_classes , axis=1)
            with st.container():
                

            
               
                col1, col2, col3 ,col4 = st.columns(4)

                # Card for True Positive
                col1.metric(label="True Positive", value=f"{predicted_percentages.get('TruePositive', 0):.2f}%", 
                            delta=f"{class_counts.get('TruePositive', 0)} records")

                # Card for True Negative
                col2.metric(label="True Negative", value=f"{predicted_percentages.get('TrueNegative', 0):.2f}%", 
                            delta=f"{class_counts.get('TrueNegative', 0)} records")

                # Card for Benign Positive
                col3.metric(label="Benign Positive", value=f"{predicted_percentages.get('BenignPositive', 0):.2f}%", 
                            delta=f"{class_counts.get('BenignPositive', 0)} records")
                col4.metric(label=" Total Records", value=f" {len(df):.2f}%", 
                            delta=f" {class_counts.get('BenignPositive', 0)} records")
                


                
                st.dataframe(styled_df ,use_container_width=False)

        
    else:
        st.sidebar.warning("Please upload a file first.")
