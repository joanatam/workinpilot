import base64
import os
import re as regex0

import fitz  # or the chosen library
import nltk
import requests
import spacy
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from streamlit_extras.stylable_container import stylable_container

# Streamlit set_page_config method has a 'initial_sidebar_state' argument that controls sidebar state.
st.set_page_config(initial_sidebar_state="auto")

# Initialize NLTK data
nltk.download(r'punkt')
nltk.download(r'punkt_tab')
nltk.download(r'stopwords')
nlp = spacy.load(r'en_core_web_sm')


text_selected = False
file_list = ["No file selected","",""]
folder_path='/Users/personal/Documents'


# Nextcloud WebDAV URL and credentials
#webdav_url = 'https://yournextcloud.domain/remote.php/dav/files/yourusername/'
webdav_url = 'http://localhost:8088/remote.php/dav/files/nextcloud_admin'
username = 'nextcloud_admin'
password = 'seK#rit12'
remote_path = 'resume_library'

if 'wurl' not in st.session_state:
    st.session_state['wurl'] = webdav_url
if 'wuser' not in st.session_state:
    st.session_state['wuser'] = username
if 'wpass' not in st.session_state:
    st.session_state['wpass'] = password
if 'wpath' not in st.session_state:
    st.session_state['wpath'] = remote_path


if 'folderpath' not in st.session_state:
    st.session_state['folderpath'] = folder_path
if 'filelist' not in st.session_state:
    st.session_state.filelist = file_list


# Initialize a session state variable that tracks the sidebar state (either 'expanded' or 'collapsed').
if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded'



def Uploader():
    st.write("Uploader")
    st.sidebar.caption = "Uploader"

def Parser():
    st.write("Parser")
    st.sidebar.caption = "Parser"

def NLP_Parser():
    st.write("NLP Parser")
    st.sidebar.caption = "NLP Parser"

def Previewer():
    st.write("Previewer")
    preview_file = st.selectbox('Select a PDF', st.session_state.filelist)
    if st.sidebar.caption == "Nothing Selected":
        st.error("Nothing Selected")
        st.stop()
    else:
        st.sidebar.caption = "Previewer"

# Show title and description of the app.
st.title('Work In Pilot [Admin]')
st.sidebar.title("Help")
st.sidebar.subheader("ADMIN VERSION")
st.sidebar.markdown('Upload, preview or parse resume PDF file \n You can use this page to upload or parse a PDF document text or preview a resume PDF.')
st.sidebar.caption ="Nothing Selected"
#sidebar_selection = st.sidebar.selectbox("Recent Selections", options=file_list)

#if st.sidebar.sidebar_selection not in st.session_state:
#    st.sidebar.sidebar_selection = st.sidebar.selectbox("Recent Selections", options=file_list)
sidebar_selection = st.sidebar.selectbox("Recent Selections", options=st.session_state['filelist']) #options=file_list)

if 'sidebar_combo' not in st.session_state:
    st.session_state['sidebar_combo'] = sidebar_selection

##DEBUG st.write(st.session_state)
sidebar_selection.join(file_list)

pg = st.navigation([Uploader, Parser, NLP_Parser, Previewer])
pg.run()

##DEBUG  st.write(st.session_state)

# format horiz
display_format = "Upload"
display_container = "Embed"
if st.sidebar.caption == "Previewer":
    col1, col2 = st.columns(2)
    with col1:
        display_format = st.radio(label="Display", options=["Upload", "Parse", "NLP Parse", "Preview"], horizontal=True)
    with col2:
        display_container = st.radio(label="Window Type", options=["Embed", "Frame", "Popup"], horizontal=True)


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file in a Pythonic way.
    """
    try:
        # save file history
        # st.session_state.filelist.insert(0, pdf_path)
        #
        with fitz.open(pdf_path) as pdf_document:
            text = ""
            for page in pdf_document:
                text += page.get_text()
            return text
    except FileNotFoundError:
        return f"Error: File not found at '{pdf_path}'"
    except Exception as e:
        return f"An error occurred: {e}"

#
# MAIN
#


def file_selector(folder_path=st.session_state['folderpath']):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a PDF to parse', filenames)

    return os.path.join(folder_path, selected_filename)

#DEBUG st.write(st.sidebar.caption)
uploaded = None
fqp = None
if 'pfile' not in st.session_state:
    st.session_state['pfile'] = fqp

if st.sidebar.caption == "Parser":
    fqp = file_selector()
    st.write('Selected `%s`' % os.path.realpath(fqp))
    with stylable_container(
        key="Parse_Data",
        css_styles="""
        button{
            float: right;
        }
        """
    ):
        if st.button('Parse Text'):
            text_selected = True
elif st.sidebar.caption == "NLP Parser":
    fqp = file_selector()
    st.write('Selected `%s`' % os.path.realpath(fqp))
    with stylable_container(
        key="NLP_Parse_Data",
        css_styles="""
        button{
            float: right;
        }
        """
    ):
        if st.button('Natural Language AI Parse'):
            text_selected = True

elif st.sidebar.caption == "Uploader":
    uploaded = st.file_uploader(label="Upload the document you want to parse", type="pdf")
    with stylable_container(
        key="Preview_Data",
        css_styles="""
        button{
            float: right;
        }
        """
    ):
        if st.button('Upload to WorkInPilot',type="primary",) and uploaded == False:
            st.error(f"Please select a PDF from dropdown list")
            uploaded = st.file_uploader(label="Upload your resume", type="pdf")

if uploaded is None and text_selected == False:
    st.stop()
base64_pdf = bytes()
if uploaded:
    base64_pdf = base64.b64encode(uploaded.read()).decode("utf-8")
    bytes_data = uploaded.getvalue()
    parsable_text = bytes_data

    # Authentication and headers
    uname = st.session_state['wuser']
    passw = st.session_state['wpass']
    webdavurl = st.session_state['wurl']
    rpath = st.session_state['wpath']
    #
    auth = (uname, passw)
    headers = {'Content-Type': 'application/octet-stream'}
    # Upload the file
    response = requests.put(webdavurl + os.sep + rpath + os.sep + uploaded.name,
                            data=parsable_text, auth=auth, headers=headers)

    # Check the response
    if response.status_code == 201:
        print('[201] File uploaded successfully.')
        st.write('[201 ]File uploaded successfully.')
    elif response.status_code == 204:
        print('[204] File updated successfully.')
        st.write('[204] File updated successfully.')
    else:
        print(f'Error uploading file: {response.status_code} - {response.text}')
        st.write(f'Error uploading file: {response.status_code} - {response.text}')

def extract_contact_info(text):
    contact_info = {}
    # Extract name
    name_pattern = r'^([A-Za-z\s]+)'
    match = regex0.search(name_pattern, text, regex0.MULTILINE)
    if match:
        contact_info['name'] = match.group(0).strip()

    # Extract email
    email_pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
    match = regex0.search(email_pattern, text)
    if match:
        contact_info['email'] = match.group(0)

    # Extract phone number
    phone_pattern = r'\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4}'
    match = regex0.search(phone_pattern, text)
    if match:
        contact_info['phone'] = match.group(0)

    return contact_info


def extract_work_experience(text):
    work_experience = []
    # import pdb
    # pdb.set_trace()
    # Split text into sections
    if r"EXPERIENCE" in text:
        # pdb.set_trace()
        sections = text.split(r"EXPERIENCE")

        sections2 = sections[1].split("\n")
        sections = sections2
        for section in sections:
            # Check if section is work experience
            #if 'work experience' in section.lower() or 'experience' in section.lower():
            # if r"\tEXPERIENCE" in section or  \
            #     'work experience' in section.lower() or 'experience' in section.lower():
            #pdb.set_trace()
            # Extract job title, company, dates, description
            job_title_pattern = r'([A-Za-z\s]+)\s*\n'
            company_pattern = r'([A-Za-z\s]+)\s*\n'
            dates_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*-\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
            description_pattern = r'([A-Za-z\s]+)'
            job_title_match = regex0.search(job_title_pattern, section, regex0.IGNORECASE)
            company_match = regex0.search(company_pattern, section, regex0.IGNORECASE)
            dates_match = regex0.search(dates_pattern, section, regex0.IGNORECASE)
            description_match = regex0.search(description_pattern, section, regex0.IGNORECASE)
            if job_title_match:
                work_experience.append({
                    'job_title': job_title_match.group(0).strip()
                })
            if company_match:
                work_experience.append({
                    'company': company_match.group(0).strip()
                })
            if dates_match:
                work_experience.append({
                    'dates': dates_match.group(0)
                })
            if description_match:
                work_experience.append({
                    'description': description_match.group(0)
                })

            #if job_title_match and company_match and dates_match and description_match:
            #    work_experience.append({
            #        'job_title': job_title_match.group(0).strip(),
            #        'company': company_match.group(0).strip(),
            #        'dates': dates_match.group(0),
            #        'description': description_match.group(0)
            #    })
    return work_experience


def extract_education(text):
    education = []
    # Split text into sections
    if 'EDUCATION' in text:

        sections = text.split('EDUCATION')
        section2 = sections[1].split('\n')
        sections = section2
        for section in sections:
            # Check if section is education
            # if 'education' in section.lower():
            # Extract degree, university, dates
            degree_pattern = r'([A-Za-z\s]+)\s*\n'
            university_pattern = r'([A-Za-z\s]+)\s*\n'
            dates_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*-\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
            degree_match = regex0.search(degree_pattern, section, regex0.IGNORECASE)
            university_match = regex0.search(university_pattern, section, regex0.IGNORECASE)
            dates_match = regex0.search(dates_pattern, section, regex0.IGNORECASE)
            if degree_match or university_match or dates_match:
                education.append({
                    'degree': degree_match.group(0).strip(),
                    'university': university_match.group(0).strip(),
                    'dates': dates_match.group(0)
                })
    return education


def extract_skills(text):
    skills = set()

    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    for token in tokens:
        if token.lower() not in stop_words:
            skills.add(token)
    return skills


if display_container == "Embed":
    if display_format == "Upload" and uploaded:
        pdf_display = (
            f'<embed src="data:application/pdf;base64,{base64_pdf}" '
            'width="800" height="1000" type="application/pdf"></embed>'
        )
        st.markdown(pdf_display, unsafe_allow_html=True)
        if uploaded.name not in st.session_state.filelist:
            st.session_state.filelist.insert(0, uploaded.name)
        st.session_state.sidebar_combo = uploaded.name
    elif st.sidebar.caption == "NLP Parser" and fqp:
        if fqp[-4:] == ".pdf":
            txt = extract_text_from_pdf(os.path.realpath(fqp))
            #st.write(txt)

            contact_info = extract_contact_info(txt)
            work_experience = extract_work_experience(txt)
            education = extract_education(txt)
            skills = extract_skills(txt)
            st.write('Contact Information:')
            st.write(contact_info)
            st.write('Work Experience:')
            for experience in work_experience:
                st.write(experience)
            st.write('Education:')
            for edu in education:
                st.write(edu)
            st.write('Skills:')
            # import pdb
            # pdb.set_trace()
            st.write(skills)

            if fqp not in st.session_state.filelist:
                st.session_state.filelist.insert(0, fqp)
            st.session_state.sidebar_combo = fqp
        else:
            st.error(f"Please select a PDF in first dropdown: {fqp}")
            st.stop()

    else:
        # preview page
        if fqp[-4:] == ".pdf":
            txt = extract_text_from_pdf(os.path.realpath(fqp))
            st.write(txt)

            if fqp not in st.session_state.filelist:
                st.session_state.filelist.insert(0, fqp)
            st.session_state.sidebar_combo = fqp
        else:
            st.error(f"Please select a PDF in first dropdown: {fqp}")
            st.stop()

elif display_container == "Frame":
    if display_format == "Upload" and uploaded:
        pdf_display = (
            f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
            'width="800" height="1000" type="application/pdf"></iframe>'
        )
        st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        if fqp[-4:] == ".pdf":
            txt = extract_text_from_pdf(os.path.realpath(fqp))
            st.write(txt)
        else:
            st.error(f"Please select a PDF in first dropdown: {fqp}")
            st.stop()
else:
    st.error(f"Unknown display window type: {display_container}")


