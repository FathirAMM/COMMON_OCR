import streamlit as st
from PIL import Image
import numpy as np
from passporteye import read_mrz
from paddleocr import PaddleOCR
import re
from fuzzywuzzy import fuzz


# Setup the page
st.set_page_config(page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# st.markdown("""
#     <style>
#     /* Main background style */
#     body {
#         background-color: #000; /* Black background */
#     }
#     /* Main font style */
#     html, body, [class*="css"] {
#         font-family: 'Arial', sans-serif;
#         color: #ffdd00; /* Yellow text for general readability */
#     }
#     /* Header styles */
#     h1, h2 {
#         color: #ffdd00; /* Yellow for headers */
#     }
#     /* Button styles */
#     .stButton>button {
#         color: #000; /* Black text on buttons */
#         background-color: #ffdd00; /* Yellow background for buttons */
#         border-radius: 5px;
#         padding: 8px 16px; /* Comfortable padding */
#         font-size: 16px; /* Readable font size */
#     }
#     /* Background color for tabs and uploader */
#     .stTabs, .stFileUploader {
#         background-color: #333; /* Dark grey for slight contrast */
#     }
#     /* Customizing the tab labels */
#     .stTab>button {
#         font-size: 16px;
#         color: #ffdd00; /* Yellow text for tabs */
#         font-weight: bold; /* Bold font for tabs */
#     }
#     /* Form label visibility */
#     .stForm label {
#         color: #ffdd00; /* Yellow for form labels */
#         font-weight: bold; /* Bold for better visibility */
#     }
#     /* Adjusting input field visibility */
#     .stTextInput>div>div>input, .stSelectbox>div>select {
#         color: #000; /* Black text for maximum contrast */
#         background-color: #ffdd00; /* Yellow background for input fields */
#         border: 1px solid #ffdd00; /* Yellow border for distinction */
#     }
#     /* Enhance select box visibility */
#     .stSelectbox>div>select {
#         font-size: 16px; /* Larger font for easier selection visibility */
#     }
#     </style>
# """, unsafe_allow_html=True)

Display the application title in a creative format
st.markdown("""
#  🤖 **AI-Powered Document Info Extractor**
""")

# st.markdown("""
#     <h1 style='text-align: center; color:  #FFFF00;'>🤖 <strong>AI-Powered Document Information Extractor</strong></h1>
# """, unsafe_allow_html=True)

#st.write("<p style='text-align: center; color: #FFFF00;'>🔍 One-stop solution for extracting information from various documents efficiently. Navigate through the tabs to start processing your documents !</p>", unsafe_allow_html=True)

st.write(" 🔍 One-stop solution for extracting information from various documents efficiently. Navigate through the tabs to start processing your documents !")





# Initialize OCR
ocr = PaddleOCR(lang='en', use_gpu=False)

# Functions for extracting MRZ and OCR data for passports and licenses
def extract_mrz_data(image):
    mrz = read_mrz(image, save_roi=True)
    if mrz is None:
        return {}
    mrz_data = mrz.to_dict()
    # Rename keys for clarity
    mrz_data['nic_number'] = mrz_data.pop('personal_number', '')
    mrz_data['passport_number'] = mrz_data.pop('number', '')
    mrz_data['mrz_code'] = mrz_data.pop('raw_text', '')
    return mrz_data

def extract_text_from_image(image):
    image_array = np.array(image)
    result = ocr.ocr(image_array, rec=True)
    return " ".join([line[1][0] for line in result[0]])

def process_ocr_results(ocr_results):
    extracted_info = {
        "Driving Licence No": None,
        "National Identification Card No": None,
        "Name": None,
        "Address": [],
        "Data Of Birth": None,
        "Date Of Issue": None,
        "Date Of Expiry": None,
        "Blood Group": None
    }

    for page in ocr_results:
        for line in page:
            text = line[1][0]
            if re.search(r'^5\.B\d+', text):
                extracted_info["Driving Licence No"] = re.sub(r'^5\.', '', text)
            elif re.search(r'^\d{11}$', text):
                extracted_info["National Identification Card No"] = text
            elif re.search(r'\.2\s+(.*)', text):
                extracted_info["Name"] = text
            elif re.search(r'^8\.', text):
                extracted_info["Address"].append(text.split('.', 1)[1].strip())
            elif re.search(r'^3\.\d{2}\.\d{2}\.\d{4}', text):
                extracted_info["Data Of Birth"] = text.split('.', 1)[1].strip()
            elif re.search(r'^4a\.\d{2}\.\d{2}\.\d{4}', text):
                extracted_info["Date Of Issue"] = text.split('.', 1)[1].strip()
            elif re.search(r'^4b\.\d{2}\.\d{2}\.\d{4}', text):
                extracted_info["Date Of Expiry"] = text.split('.', 1)[1].strip()
            elif re.search(r'^Blood Group', text, re.IGNORECASE):
                extracted_info["Blood Group"] = text.split(None, 2)[-1]
    return extracted_info

def load_and_process_image(image_file):
    image = Image.open(image_file).convert('RGB')
    image_np = np.array(image)
    result = ocr.ocr(image_np, rec=True)
    return image, process_ocr_results(result)

# Function to extract vehicle details using OCR
def extract_key_value(ocr_results, key_name, line_param, value_index, fuzz_score_threshold=80, threshold=10):
    mid_height_results = []
    for coordinates, (text, _) in ocr_results:
        mid_height = (coordinates[0][1] + coordinates[3][1]) / 2
        mid_height_results.append(((coordinates[0], coordinates[3]), (text, _), mid_height))
    sorted_results = sorted(mid_height_results, key=lambda x: x[2])
    key_match = None
    for (_, _), (text, _), mid_height in sorted_results:
        if fuzz.partial_ratio(key_name.replace(" ", ""), text) >= fuzz_score_threshold:
            key_match = text
            break

    if key_match is None:
        return None

    key_mid_height = None
    for (_, _), (text, _), mid_height in sorted_results:
        if text == key_match:
            key_mid_height = mid_height
            break

    if key_mid_height is None:
        return None

    values = []
    for (_, _), (text, _), mid_height in sorted_results:
        if line_param == 'same_line' and abs(mid_height - key_mid_height) <= threshold:
            values.append(text)
        elif line_param == 'next_line' and mid_height > key_mid_height + threshold:
            values.append(text)

    # Handling value_index which can be an int or a list
    if isinstance(value_index, list):
        return [values[i] for i in value_index if 0 <= i < len(values)]
    elif 0 <= value_index < len(values):
        return values[value_index]
    else:
        return None


def extract_details_from_image(image):
    image_array = np.array(image)  # Convert to NumPy array
    result = ocr.ocr(image_array, rec=True)
    ocr_results = result[0] 

    # Define key-value pairs with fuzz_score_threshold and threshold
    key_value_pairs = [
        ("Registration No.", "next_line", 0, 80, 30),
        ("Chassis No.", "next_line", 1, 80, 30),
        ("Current Owner/Address/ID.No.", "next_line", [0, 1], 80, 60),
        ("Conditions/Special Notes", "next_line", 0, 80, 10),
        ("Absolute Owner", "next_line", [0, 1, 2], 80, 20),
        ("Engine No", "next_line", 0, 80, 20),
        ("Cylinder Capacity (cc)", "next_line", 1, 80, 20),
        ("Class of Vehicle", "next_line", 0, 80, 20),
        ("Taxation Class", "next_line", 1, 80, 20),
        ("Status when Registered", "next_line", 0, 80, 10),
        ("Make", "next_line", 0, 80, 10),
        ("Model", "next_line", 1, 80, 20),
        ("Wheel Base", "next_line", 0, 80, 10),
        ("Type of Body", "next_line", 0, 80, 20)
    ]

    extracted_details = {}

    for key_name, value_in, value_at, fuzz_score_threshold, threshold in key_value_pairs:
        result = extract_key_value(ocr_results, key_name, value_in, value_at, fuzz_score_threshold, threshold)
        extracted_details[key_name] = result

    return extracted_details

# Streamlit tab setup
tab1, tab2, tab3 = st.tabs(["🪪 Driving License Information Extractor", "🛂 Passport Information Extractor", "🚗 Vehicle CR Book Information Extractor"])

with tab1:
    st.title('🪪 Driving License Information Extractor ')

    col1, col2 = st.columns([2, 3])
    with col1:
        st.write("Upload an image of a driving license:")
        uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key='license_uploader')
        if uploaded_image is not None:
            image, extracted_info = load_and_process_image(uploaded_image)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            if st.button('Process Image', key='process_license_image'):
                st.session_state["extracted_info"] = extracted_info


    with col2:
        st.write("Extracted Details")
        form = st.form(key='license_details_form')
        labels = [
            "Driving Licence No", "National Identification Card No", "Name", "Address", "Data Of Birth", "Date Of Issue", "Date Of Expiry", "Blood Group"
        ]
        for label in labels:
            default_value = st.session_state.get("extracted_info", {}).get(label, '')
            if isinstance(default_value, list):
                default_value = ', '.join(default_value)
            form.text_input(label, default_value)
        form.form_submit_button('Submit')

with tab2:
    st.title("🛂 Passport Information Extractor ")
    col1, col2 = st.columns([2, 3])
    with col1:
        st.write("Upload an image of the passport:")
        uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key='passport_uploader')
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            if st.button('Process Image', key='process_passport_image'):
                with st.spinner("Extracting data from passport..."):
                    mrz_data = extract_mrz_data(uploaded_image)
                    extracted_text = extract_text_from_image(image)
                    if "passport" in extracted_text.lower():
                        st.warning("🔍 Passport found! 🎉")
                    for label, value in mrz_data.items():
                        st.session_state[label] = value

    with col2:
        st.write("Extracted Passport Details")
        form = st.form(key='passport_details_form')
        labels = ["names", "surname", "nationality", "nic_number", "passport_number", "date_of_birth", "expiration_date", "sex", "type", "mrz_code"]
        for label in labels:
            default_value = st.session_state.get(label, '')
            form.text_input(label.capitalize(), default_value)
        form.form_submit_button('Submit')

with tab3:
    st.title("🚗 Vehicle CR Book Information Extractor ")
    col1, col2 = st.columns([2, 3])
    with col1:
        st.write("Upload an image of a vehicle CR book:")
        uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key='vehicle_uploader')
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            if st.button('Process Image', key='process_vehicle_image'):
                with st.spinner("Processing image..."):
                    extracted_details = extract_details_from_image(image)
                    for label, value in extracted_details.items():
                        st.session_state[label] = value if value else ''

    with col2:
        st.write("Extracted Details")
        form = st.form(key='vehicle_details_form')
        labels = [
            "Registration No.", "Chassis No.", "Current Owner/Address/ID.No.", "Conditions/Special Notes",
            "Absolute Owner", "Engine No", "Cylinder Capacity (cc)", "Class of Vehicle",
            "Taxation Class", "Status when Registered", "Make", "Model", "Wheel Base", "Type of Body"
        ]
        for label in labels:
            default_value = st.session_state.get(label, '')
            form.text_input(label, default_value)
        form.form_submit_button('Submit')

