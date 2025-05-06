import base64
import os
import re as regex0
from urllib.parse import urlparse

# find document sections
import pdfplumber
import torch

# validate english words
import langid
from wordfreq import zipf_frequency

# for docx templates
from docxtpl import DocxTemplate, RichText
import tempfile

import easywebdav
import socket

import fitz  # pip install pymupdf or the chosen library
import nltk
import requests
import spacy
import spacy_transformers
import streamlit as st
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk import WordNetLemmatizer
from streamlit_extras.stylable_container import stylable_container
from dotenv import load_dotenv

# Streamlit set_page_config method has a 'initial_sidebar_state' argument that controls sidebar state.
st.set_page_config(initial_sidebar_state="auto")

if 'nltk_init' not in st.session_state:
    # Initialize NLTK data - need only do this once
    nltk.download(r'punkt')
    nltk.download(r'punkt_tab')
    nltk.download(r'stopwords')
    nlp = spacy.load(r'en_core_web_sm')
    # nlp = spacy.load(r'en_core_web_md')
    nltk.download('wordnet')
    st.session_state ['nltk_init'] = True

# load .env vars into environment
load_dotenv()

torch.classes.__path__ = [] # add this line to manually set it to empty.

text_selected = False
file_list = ["No file selected","",""]
# @TODO fix this for Linux/NextCloud?:
folder_path=os.getenv("HOME", '/Users/personal/Documents')


# Nextcloud WebDAV URL and credentials

hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)
#st.write(f"Hostname: {hostname} IP Address: {ip_address}")
st.write("WorkinPilot")

webdav_url = os.getenv("webdav_url")

username = os.getenv('username')
password = os.getenv('password')
remote_path = os.getenv('remote_path')
client_baseurl =os.getenv('client_baseurl', "http://localhost:8088/remote.php/dav") + "/files/" + username


if 'wdurl' not in st.session_state:
    st.session_state['wdurl'] = webdav_url
if 'wduser' not in st.session_state:
    st.session_state['wduser'] = username
if 'wdpass' not in st.session_state:
    st.session_state['wdpass'] = password
if 'wdpath' not in st.session_state:
    st.session_state['wdpath'] = remote_path
if 'wdfqpath' not in st.session_state:
    st.session_state['wdfqpath'] = webdav_url + os.sep + remote_path + os.sep
if 'client_baseurl' not in st.session_state:
    st.session_state["client_baseurl"] = client_baseurl
    

if 'folderpath' not in st.session_state:
    st.session_state['folderpath'] = folder_path
if 'filelist' not in st.session_state:
    st.session_state.filelist = file_list


# Initialize a session state variable that tracks the sidebar state (either 'expanded' or 'collapsed').
if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded'



def Uploader():
    st.write("Upload Resume From Local Files")
    st.sidebar.caption = "Uploader"

def Parser():
    st.write("Parse Resume into Text")
    st.sidebar.caption = "Parser"

def NLP_Parser():
    st.write("NLP Parse Resume into JSON Objects")
    st.sidebar.caption = "NLP Parser"

def Previewer():
    st.write("Preview Document Sections")
    st.sidebar.caption = "Previewer"

def Smart_Parser():
    st.write("Smart Parser - Apply AI techniques to resume sections")
    st.sidebar.caption = "Smart Parser"

    #preview_file = st.selectbox('Select a PDF', st.session_state.filelist)
    # if st.sidebar.caption == "Nothing Selected":
    #     st.error("Nothing Selected")
    #     st.stop()
    # else:
    #     st.sidebar.caption = "Previewer"

# Show title and description of the app.
st.title('Resume Workbench')
st.sidebar.title("Help")
st.sidebar.markdown('Upload, preview or parse resume PDF file \n You can use this page to upload or parse a PDF document text or preview a resume PDF.')
#st.sidebar.caption ="Nothing Selected"

sidebar_use_inference = st.sidebar.checkbox("Infer resume sections when parsing?")

sidebar_selection = st.sidebar.selectbox("Recent Selections", options=st.session_state['filelist']) #options=file_list)

if 'sidebar_combo' not in st.session_state:
    st.session_state['sidebar_combo'] = sidebar_selection

sidebar_selection.join(file_list)

pg = st.navigation([Uploader, Parser, NLP_Parser, Previewer, Smart_Parser])
pg.run()

# format horiz
display_format = "Upload"
display_container = "Embed"
# if st.sidebar.caption == "Previewer":
#     col1, col2 = st.columns(2)
#     with col1:
#         display_format = st.radio(label="Display", options=["Upload", "Parse", "NLP Parse", "Preview"], horizontal=True)
#     with col2:
#         display_container = st.radio(label="Window Type", options=["Embed", "Frame", "Popup"], horizontal=True)


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file in a Pythonic way.
    """
    try:
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
def webdav_file_selector():
    parsed_url = urlparse(st.session_state["wdurl"])
    server_address = parsed_url.geturl()
    client = easywebdav.connect(server_address,
                                username=st.session_state["wduser"],
                                password=st.session_state["wdpass"],
                                protocol=parsed_url.scheme
                                )

    client.baseurl = st.session_state["client_baseurl"]
    fpath = st.session_state["wdpath"]

    #import pdb
    #pdb.set_trace()

    if client.exists(fpath) == False:
        try:
            client.mkdir(fpath)
        except Exception as e:
            st.warning("Could not find resume library.  Please contact site administrator.",icon="⚠️")
            st.warning(e)
            st.stop()

    
    try:
        files = client.ls(fpath)
        pdfs = [x.name for x in files if x.size > 0 and 'pdf' in x.name]
        selected_filename = st.selectbox('Select a PDF to parse', pdfs)
        return selected_filename
    except easywebdav.WebdavException as e: #.client.ConnectionError as e:
        print(f"Connection error: {e}")
        st.write(f"Connection error: {e}")
    except requests.RequestException as e: #easywebdav..client.HTTPError as e:
        print(f"HTTP request error: {e}")
        st.write(f"HTTP request error: {e}")
    return ""


def file_selector(folder_path=st.session_state['folderpath']):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a PDF to parse', filenames)
    return os.path.join(folder_path, selected_filename)


uploaded = None
fqp = None

if st.sidebar.caption == "Parser":
    fqp = webdav_file_selector()
    st.write('Selected `%s`' % fqp)
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
    fqp = webdav_file_selector()
    st.write('Selected `%s`' % fqp)
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
        key="Upload_Data",
        css_styles="""
        button{
            float: right;
        }
        """
    ):
        if st.button('Upload to WorkInPilot',type="primary",) and uploaded == False:
            st.error(f"Please select a PDF from dropdown list")
            uploaded = st.file_uploader(label="Upload your resume", type="pdf")
elif st.sidebar.caption == "Previewer":
    fqp = webdav_file_selector()
    st.write('Selected `%s`' % fqp)
    with stylable_container(
        key="Preview_Data",
        css_styles="""
        button{
            float: right;
        }
        """
    ):
        if st.button('Preview Document Sections'):
            text_selected = True

elif st.sidebar.caption == "Smart Parser":
    fqp = webdav_file_selector()
    st.write('Selected `%s`' % fqp)
    with stylable_container(
        key="Smart_Parse_Data",
        css_styles="""
        button{
            float: right;
        }
        """
    ):
        if st.button('Smart Parse'):
            text_selected = True



if uploaded is None and text_selected == False:
    st.stop()
base64_pdf = bytes()
if uploaded:
    base64_pdf = base64.b64encode(uploaded.read()).decode("utf-8")
    bytes_data = uploaded.getvalue()
    parsable_text = bytes_data

    # Authentication and headers
    uname = st.session_state['wduser']
    passw = st.session_state['wdpass']
    webdavurl = st.session_state['client_baseurl']
    rpath = st.session_state['wdpath']
    #
    auth = (uname, passw)
    headers = {'Content-Type': 'application/octet-stream'}
    # Upload the file
    put_file = webdavurl + os.sep + rpath + os.sep + uploaded.name
    #import pdb
    #pdb.set_trace()
    response = requests.put(put_file,
                            data=parsable_text, 
                            auth=auth, 
                            headers=headers)

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


def save_to_temp_file(fn, byts):
    with fn as temp_file:
        temp_file.write(byts)
        temp_file.flush()  # Ensure data is written to disk

def retrieve_from_temp_file(fn,delete=True):
    with open(fn, mode='rb') as temp_file:
        temp_file.seek(0)
        retrieved_data = temp_file.read()
        if delete:
            temp_file.close()
            os.unlink(temp_file.name)
        return retrieved_data

def get_temp_file_name():
    return tempfile.NamedTemporaryFile(mode='wb', delete=False)


def extract_contact_info(text):
    contact_info = {}

    first_lines = text[0:200].split("\n") if len(text) > 200 else text.split("\n")

    # Extract name
    name_pattern = r'^([A-Za-z\s]+)'
    match = regex0.search(name_pattern, text, regex0.MULTILINE)
    if match:
        temp = match.group(0).split("\n")[0]
        contact_info['name'] = temp.strip()

    # address pattern
    # address_regex = r"^\d{1,5}\s([\w\s]+,)?\s?[A-Za-z]+\s[A-Za-z]{2}\s\d{5}(-\d{4})?$"
    address_pattern = r"^\d{1,5}\s([\w\s]+)\s[A-Za-z]{2}\s"
    match = regex0.search(address_pattern, text, regex0.MULTILINE)
    if match:
        # temp = match.group(0).split("\n")[0]
        # contact_info['address'] = temp.strip()
        contact_info['address'] = text[match.start():match.end()]

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

    # new
    phone_number_pattern = (
        r"^(\+\d{1,3})?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$"
    )
    match = regex0.findall(phone_number_pattern, text)
    if match:
        contact_info['phone_numbers'] = match.group(0)

    # new - social media
    # expect lowercase token with lastname in it
    if 'name' in contact_info and len(contact_info['name']) > 0:
        firstinitial = contact_info['name'][1:].lower()
        lastname = contact_info['name'].split(" ")[-1].lower()
        for line in first_lines:
            stuff0 = line.strip().split(" ")[-1]
            stuff = stuff0.strip().split("/")[-1]
            if lastname in stuff and regex0.search(r"^[a-z]+$",stuff):
                contact_info['social_media_id'] = stuff
                break


    # TEST ONLY generate new contact info sheet from our template
    # check that template is available
    template_file = "./templates/mgr_resume_tmpl.docx"
    if os.path.exists(template_file):
        rt = RichText()
        if 'name' in contact_info and len(contact_info['name']) > 0:
            first_name = contact_info['name'].split(" ")[0]
            rt.add(first_name, size=60)
            rt_embedded = RichText("our friend ", size=32, color="#6B83A4", bold=True)
            rt_embedded.add(rt)

        doc = DocxTemplate("./templates/mgr_resume_tmpl.docx")
        context = {'contact_info': contact_info }
        doc.render(context)

        temp_file = get_temp_file_name()
        temp_file_dict = {'name': 'mgr_resume.docx',
                      'temp_path': os.path.realpath(temp_file.name)
                          }

        doc.save(temp_file.name)

        bytbuff = retrieve_from_temp_file(fn=temp_file.name, delete=True)

        # can this go to webdav?
        rpath = st.session_state['wdpath']
        parsed_url = urlparse(st.session_state["wdurl"])
        server_address = parsed_url.geturl()
        client = easywebdav.connect(server_address,
                                    username=st.session_state["wduser"],
                                    password=st.session_state["wdpass"],
                                    protocol=parsed_url.scheme)
        #client.baseurl = st.session_state["wdurl"]
    
        client.baseurl = st.session_state["client_baseurl"]
        fpath = st.session_state["wdpath"]

        webdavurl = st.session_state['wdurl']
        reconstructed_url = (webdavurl + os.sep + rpath +
                             os.sep + "generated_from_template" + os.sep + temp_file_dict['name']
                             )
        # # Authentication and headers
        auth = (st.session_state['wduser'], st.session_state['wdpass'])
        headers = {'Content-Type': 'application/octet-stream'}
        try:
            resp = client.session.put(reconstructed_url,
                                      data=bytbuff,allow_redirects=False,
                                      auth=auth, headers=headers )
        except easywebdav.WebdavException as e: #.client.ConnectionError as e:
            st.write(f"Connection error: {e}")
        except requests.RequestException as e: #easywebdav..client.HTTPError as e:
            st.write(f"HTTP request error: {e}")

    return contact_info


def extract_education_details(text):
    # Use regular expressions to extract education details
    education = {}
    education_details = []

    masterspattern = regex0.compile(r'Masters? ')
    mbapattern = regex0.compile(r'(M\.?B\.?A\.?)', regex0.IGNORECASE)
    mbamatches = regex0.findall(mbapattern, text) #, regex0.IGNORECASE)
    bapattern = regex0.compile(r"(B\.?A\.?)")
    bspattern = regex0.compile(r"(B\.?S\.?)")
    mastersmatches = regex0.findall(masterspattern, text)
    bamatches = regex0.findall(bapattern, text) #, regex0.IGNORECASE)
    bsmatches = regex0.findall(bspattern, text) #, regex0.IGNORECASE)
    mba_match = ""
    ba_match = ""
    bs_match = ""

    if len(mbamatches) > 0:
        mba_match = mbamatches[len(mbamatches)-1]
    if len(bamatches) > 0:
        ba_match = bamatches[len(bamatches)-1]
    if len(bsmatches) > 0:
        bs_match = bsmatches[len(bsmatches)-1]
    if len(mastersmatches) > 0:
        masters_match = mastersmatches[len(mastersmatches)-1]

    findme = 'university'
    if findme in text.lower():
        idx = text.lower().index(findme)
        part0 = text[:idx:]
        parts = part0.split('\n')
        prev = parts[len(parts)-1].strip()
        prefix = prev.split(',')
        st = text[text.lower().index(findme):]
        lines = st.split('\n')
        flines = st.split('\n')
        lines = flines[0].split(',')
        if len(lines) == 1:
            lines = flines
        txtout = lines[0]
        while len(lines) < 3:
            lines.append("")
        education_details.append({
            'school': prefix[len(prefix)-1]+" "+txtout.strip(),
            'description':lines[1].strip(),
            'details': lines[2].strip(),
            'degree': mba_match if len(mba_match) > 0 else prefix[0]
        })
        education['university'] = education_details[-1]
    findme = 'bachelor'
    if findme in text.lower():
        idx = text.lower().index(findme)
        part0 = text[:idx:]
        parts = part0.split('\n')
        prev = parts[len(parts)-1]
        st = text[idx:]
        flines = st.split('\n')
        lines = flines[0].split(',')
        if len(lines) == 1:
            lines = flines
        txtout = lines[0]
        if len(ba_match) == 0:
            ba_match = "Bachelors"
        while len(lines) < 3:
            lines.append("")
        education_details.append({
            # 'school':(prev+" "+txtout).strip(),
            # 'description':lines[1].strip(),
            # 'details': lines[2].strip(),
            # 'degree': ba_match
            'school': lines[1].strip(),
            'description': (prev + " " + txtout).strip(),
            'details': lines[2].strip(),
            'degree': ba_match if len(ba_match) > 0 else bs_match
        })
        education['bachelor'] = education_details[-1]
    else:
        findme = 'college'
        if findme in text.lower():
            idx = text.lower().index(findme)
            part0 = text[:idx:]
            parts = part0.split('\n')
            prev = parts[len(parts)-1]
            st = text[idx:]
            lines = st.split('\n')
            flines = st.split('\n')
            lines = flines[0].split(',')
            if len(lines) == 1:
                lines = flines
            txtout = lines[0]
            while len(lines) < 3:
                lines.append("")
            education_details.append({
                'school':prev+" "+txtout,
                'description':lines[1].strip(),
                'details': lines[2].strip(),
                'degree': ba_match if len(ba_match) > 0 else bs_match
            })
            education['college'] = education_details[-1]

    # return education_details
    return education

def case_insensitive_split_re(mtext, delimiter):
    return regex0.split(delimiter, mtext, flags=regex0.IGNORECASE)

def extract_work_experience(text, override=False):
    work_experience = []

    # Split text into sections
    if r"experience" in text.lower() or override:
        #sections = text.split(r"EXPERIENCE")
        sections = case_insensitive_split_re(text, 'experience')
        # if len(sections) > 0:
        #     paragraphs = sections[1].split('\n')
        # else:
        #     paragraphs = [sections[0]]
        paragraphs = sections

        if override:
            for section in sections:
                work_experience.append({
                    'description': section.strip()
                })
                # Check if section is work experience
                # Extract job title, company, dates, description
                # job_title_pattern = r'([A-Za-z\s]+)\s*\n'
                # company_pattern = r'([A-Za-z\s]+)\s*\n'
                # dates_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*-\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
                # description_pattern = r'([A-Za-z\s]+)'
                # job_title_match = regex0.search(job_title_pattern, section, regex0.IGNORECASE)
                # company_match = regex0.search(company_pattern, section, regex0.IGNORECASE)
                # dates_match = regex0.search(dates_pattern, section, regex0.IGNORECASE)
                # description_match = regex0.search(description_pattern, section, regex0.IGNORECASE)
                # if job_title_match:
                #     work_experience.append({
                #         'job_title': job_title_match.group(0).strip()
                #     })
                # if company_match:
                #     work_experience.append({
                #         'company': company_match.group(0).strip()
                #     })
                # if dates_match:
                #     work_experience.append({
                #         'dates': dates_match.group(0)
                #     })
                # if description_match:
                #     work_experience.append({
                #         'description': (description_match.group(0)).strip()
                #     })

        else:
            glued_sent = ""
            for sentnc in paragraphs:

                if len(sentnc.strip()) < 5:
                    continue
                elif "." in sentnc:
                    glued_sent += sentnc.split('.')[0]
                    work_experience.append({
                        'description': glued_sent.strip()
                    })
                    glued_sent = ""
                    continue
                else:
                    glued_sent += " "
                    glued_sent += sentnc.strip()
                    continue

                # Check if section is work experience
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
                        'description': (description_match.group(0)).strip()
                    })

                #if job_title_match and company_match and dates_match and description_match:
                #    work_experience.append({
                #        'job_title': job_title_match.group(0).strip(),
                #        'company': company_match.group(0).strip(),
                #        'dates': dates_match.group(0),
                #        'description': description_match.group(0)
                #    })
    return work_experience


def extract_education_section(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)


    # Initialize the education section
    education_section = []

    # Loop through each sentence
    for sentence in sentences:
        # Tokenize the sentence into words
        words = word_tokenize(sentence)

        # Remove stopwords and punctuation
        stop_words = set(stopwords.words('english'))

        #import pdb
        #pdb.set_trace()

        words = [word for word in words if word.lower() not in stop_words and word.isalpha()]

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        # Check if the sentence contains education-related keywords
        education_keywords = ['education', 'degree', 'university', 'college','school', 'graduated', 'graduation', 'certificate']
        if any(word.lower() in education_keywords for word in words):
            education_section.append(sentence)

    # Join the education section sentences into a single string
    education_section =''.join(education_section)

    return education_section


def extract_education(text):
    education = []
    # Split text into sections
    if 'EDUCATION' in text:

        sections = text.split('EDUCATION')
        if len(sections) > 0:
            section2 = sections[1].split('\n')
        else:
            section2 = [sections[0]]
        sections = section2
        for section in sections:
            # Check if section is education
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


def extract_bytes_from_webdav_pdf(fqp):
    parsed_url = urlparse(st.session_state["wdurl"])
    server_address = parsed_url.geturl()
    client = easywebdav.connect(server_address,
                                username=st.session_state["wduser"],
                                password=st.session_state["wdpass"],
                                protocol=parsed_url.scheme
                                )
    #client.baseurl = st.session_state["wdurl"]
    client.baseurl = st.session_state["client_baseurl"]
    fpath = st.session_state["wdpath"]
    byte_data = bytes()
    reconstructed_url = parsed_url.scheme + "://" + parsed_url.netloc + fqp
    try:
        with client.session.get(reconstructed_url, stream=True) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=8192):
                # append each chunk of bytes
                byte_data += chunk
    except Exception as e:
        st.write(f"An error occurred: {e}")
    return byte_data


def extract_text_from_webdav_pdf(fqp):
    byte_data = extract_bytes_from_webdav_pdf(fqp)
    txt = ""
    try:
        tmpdoc = fitz.Document(stream=byte_data)
        for page in tmpdoc:
            txt += page.get_text()
        return txt
    except Exception as e:
        return f"An error occurred: {e}"


def parse_pdf(file_path):

    high_sections = []
    paragraphs = []
    full_text = ""

    # Open the PDF file with pdfplumber
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # Extract text with layout analysis
            text = page.extract_text()
            full_text += text

            if text:

                # look for pattern of uppercase string(s) on 1 line
                # assume all-uppercase on 1 line = section header
                # deal w greediness
                list0 = regex0.findall(r'[\n][A-Z]{5,}[\s[A-Z]{5,}]?[\n]?',text)
                list1 = regex0.findall(r'\n[A-Z]{4,}[\n]{1}',text)
                list2 = regex0.findall(r'\n[A-Z]{4,}[" "]{1}[A-Z]{4,}',text)
                list3 = regex0.findall(r'\s[A-Z]{5,}[\n]{1}',text)
                list4 = regex0.findall(r'EDUCATION',text)
                #st.write(list3)
                #list4 = regex0.findall(r'\s[A-Z]{5,}[\s]{1}',text)
                for section_name in list0:
                    s = section_name.split('\n')[1].strip().replace('\n','')
                    if s not in high_sections:
                        high_sections.append(s)
                for section_name in list1:
                    s = section_name.split('\n')[1].strip().replace('\n','')
                    if s not in high_sections:
                        high_sections.append(s)
                for section_name in list2:
                    s = section_name.replace('\n','').strip()
                    if s not in high_sections:
                        high_sections.append(s)
                for section_name in list3:
                    s = section_name.strip().replace('\n','').strip()
                    if s not in " ".join(high_sections):
                        high_sections.append(s)
                for section_name in list4:
                    s = section_name.strip().replace('\n','').strip()
                    if s not in " ".join(high_sections):
                        high_sections.append(s)

                # Split text by newlines or other delimiters to identify paragraphs
                # page_paragraphs = text.split('\n')
                page_paragraphs = text.split('. ')

                for paragraph in page_paragraphs:
                    # Here you can implement logic to detect sections or paragraph boundaries
                    # For example, assuming sections have a specific style or keyword:
                    if paragraph.strip().endswith('\n\n'):  # ':'):
                        high_sections.append(paragraph.strip())
                    else:
                        paragraphs.append(paragraph.strip())

    if len(high_sections) < 1 and sidebar_use_inference:
        st.markdown(":orange[Using Inference to Detect Sections ...]")
        common_sections = ['Summary',
                           'Skills', 'Experience',
                           'Education', 'Projects']
        for asection in common_sections:
            if asection in full_text:
                high_sections.append(asection)


    return high_sections, paragraphs, full_text


def display_results(sections, paragraphs, txt):

    sect_indices = {}

    langid.set_languages(['en'])  # ISO 639-1 codes

    st.divider()
    st.write("**Section Detection**")
    for a in sections:
        if isinstance(a, list):
            for b in a:
                eword = (b.split('\n'))[1].strip()
                lang,score = langid.classify(eword)
                if ' ' in eword:
                    syn = zipf_frequency(eword.split()[0], 'en', wordlist='small')
                else:
                    syn = zipf_frequency(eword, 'en', wordlist='small')
                if syn <= 0.0:
                    continue
                sect_indices[eword] = [txt.find(eword)]
                if score > 9.0:
                    st.write(" - ",eword," ",syn)
                else:
                    st.write(" - ", eword, " [wordfreq: ",syn," score: ",str(score), "]")
        else:
            eword = a.strip()
            lang, score = langid.classify(eword)
            if ' ' in eword:
                syn = zipf_frequency(eword.split()[0], 'en', wordlist='small')
            else:
                syn = zipf_frequency(eword, 'en', wordlist='small')
            if syn <= 0.0:
                continue
            sect_indices[eword] = [txt.find(eword)]
            if score > 9.0:
                st.write(" - ",eword," ",syn)
            else:
                st.write(" - ", eword, " [wordfreq: ", syn, " score: ", str(score), "]")

    # eof
    sect_indices['EOF'] = [len(txt),len(txt)]
    st_list = sorted(list(sect_indices.values()))

    j_list = sorted(list(sect_indices.items()), key=lambda x: x[1][0])
    for idx, key in enumerate(sect_indices):
        if idx > len(sect_indices) - 2:
            break
        sect_indices[j_list[idx][0]].append(st_list[idx + 1][0])

    if len(sections) > 0 and len(sect_indices)>0:
        st.divider()
        st.markdown(":page_facing_up: :blue-background[ Text Block Dump ]")
    else:
        st.markdown(":no_entry_sign: :red[No Sections Detected]")

    ks = list(sect_indices.keys())
    vs = list(sect_indices.values())
    for idx in range (len(sect_indices)-1):
        k = ks[idx]
        v = sect_indices[k]


        st.write("Section[",k,"]")

        section_text = txt[v[0]+len(k)+1:v[1]]

        generic = {}
        generic['type'] = 'Section'
        generic['name'] = k
        generic['content'] = section_text
        st.write(generic)


        paras = []
        if k == "EDUCATION":
            paras = section_text.split('\n')
        else:
            ps = section_text.split('. ')
            for para in ps:
                paras = para.split('\n')

        for para in paras:
            if (len(para.strip())>0):
                if '\n' in para:
                    lines = para.replace('\n',' ')
                    st.write(lines)
                else:
                    st.write(f" - {para}")

    st.write(f"*Document EOF at line [{sect_indices['EOF'][1]}]*")

    st.divider()
    st.write("**Natural Paragraphs Detected**")
    for para in paragraphs:
        st.write(f" - {para}")

    # create new template
    # doc = DocxTemplate("./templates/contactinfo_test_tpl.docx")
    # context = {'name': "World company"}
    # doc.render(context)
    # doc.save("./output/contactinfo_test.docx")


def smart_display_results(sections, paragraphs, txt):

    sect_indices = {}

    langid.set_languages(['en'])  # ISO 639-1 codes

    st.divider()
    st.write("**Section Detection**")
    for a in sections:
        if isinstance(a, list):
            for b in a:
                eword = (b.split('\n'))[1].strip()
                lang,score = langid.classify(eword)
                if ' ' in eword:
                    syn = zipf_frequency(eword.split()[0], 'en', wordlist='small')
                else:
                    syn = zipf_frequency(eword, 'en', wordlist='small')
                if syn <= 0.0:
                    continue
                sect_indices[eword] = [txt.find(eword)]
                if score > 9.0:
                    st.write(" - ",eword," ",syn)
                else:
                    st.write(" - ", eword, " [wordfreq: ",syn," score: ",str(score), "]")
        else:
            eword = a.strip()
            lang, score = langid.classify(eword)
            if ' ' in eword:
                syn = zipf_frequency(eword.split()[0], 'en', wordlist='small')
            else:
                syn = zipf_frequency(eword, 'en', wordlist='small')
            if syn <= 0.0:
                continue
            sect_indices[eword] = [txt.find(eword)]
            if score > 9.0:
                st.write(" - ",eword," ",syn)
            else:
                st.write(" - ", eword, " [wordfreq: ", syn, " score: ", str(score), "]")

    # eof
    sect_indices['EOF'] = [len(txt),len(txt)]
    st_list = sorted(list(sect_indices.values()))

    j_list = sorted(list(sect_indices.items()), key=lambda x: x[1][0])
    for idx, key in enumerate(sect_indices):
        if idx > len(sect_indices) - 2:
            break
        sect_indices[j_list[idx][0]].append(st_list[idx + 1][0])

    if len(sections) > 0 and len(sect_indices)>0:
        st.divider()
        # st.markdown(":page_facing_up: :blue-background[ Text Block Dump ]")
    else:
        st.markdown(":no_entry_sign: :red[No Sections Detected]")

    ks = list(sect_indices.keys())
    vs = list(sect_indices.values())

    st.write("Section[Contact Info]")
    # smart_contact_info = extract_contact_info(txt[0:sect_indices[ks[0]][0]])
    # st.write(smart_contact_info)
    generic = {}
    generic['type'] = 'Section'
    generic['name'] = 'Contact Info'
    top_sect = min(vs)[0]
    generic['raw_content'] = txt[:top_sect]
    generic['parsed_content'] = dict(extract_contact_info(txt[:top_sect]))
    st.write(generic)

    for idx in range (len(sect_indices)-1):
        k = ks[idx]
        v = sect_indices[k]

        st.write("Section[",k,"]")

        section_text = txt[v[0]+len(k)+1:v[1]]

        generic = {}
        generic['type'] = 'Section'
        generic['name'] = k
        generic['raw_content'] = section_text
        if "EDUCATION" in k.upper():
            generic['parsed_content'] = dict(extract_education_details(section_text))
        st.write(generic)

        # if k.upper() == "EDUCATION":
        #     st.write(extract_education_details(section_text))
        # elif "SKILL" in k.upper():
        #     st.write(extract_skills(section_text))
        # elif "EXPERIENCE" in k.upper():
        #     st.write(extract_work_experience(section_text, override=True))
        # else:
        #     st.write(section_text)

    st.write(f"*Document EOF at line [{sect_indices['EOF'][1]}]*")


def find_local_file(filename):
    if filename:
        fqp_parts = filename.split("/")
        file_name = fqp_parts[-1]

        if os.path.exists("/Users/personal/Documents/"+file_name):
            return os.path.realpath("/Users/personal/Documents/"+file_name)
        elif os.path.exists("/Users/personal/Downloads/"+file_name):
            return os.path.realpath("/Users/personal/Downloads/"+file_name)
        elif os.path.exists("/tmp/"+file_name):
            return os.path.realpath("/tmp/"+file_name)
        else:
            # make one in temp
            new_file = "/tmp/"+file_name
            # using one in webdav env
            old_file = filename
            bytes_data = extract_bytes_from_webdav_pdf(old_file)

            # save bytes to /tmp
            buffw = open(new_file, "wb")
            buffw.write(bytes_data)
            buffw.close()

            return new_file

    return filename


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
            txt = extract_text_from_webdav_pdf(fqp)

            contact_info = extract_contact_info(txt)
            work_experience = extract_work_experience(txt)
            education = extract_education(txt)
            e1 = extract_education_details(txt)
            e2 = extract_education_section(txt)
            skills = extract_skills(txt)
            st.write('Contact Information:')
            st.write(contact_info)
            st.write('Education:')
            st.write(e1)

            st.write('Work Experience:')
            for experience in work_experience:
                st.write(experience)
            st.write('Skills:')
            st.write(skills)

            ###
            st.write('NER - Named Entity Recognition in this document')
            # first do $ python3 -m spacy download en_core_web_trf
            #N/A  - trf = spacy.load('en_core_web_trf')
            ner = spacy.load('en_core_web_sm')
            doc = ner(txt)
            # import pdb
            # pdb.set_trace()
            for ent in doc.ents:
                st.write(ent.text, ent.start_char, ent.end_char, ent.label_)
            ###

            if fqp not in st.session_state.filelist:
                st.session_state.filelist.insert(0, fqp)
            st.session_state.sidebar_combo = fqp
        else:
            st.error(f"Please select a PDF in first dropdown: {fqp}")
            st.stop()

    elif st.sidebar.caption == "Previewer" and fqp:
        #fqp_parts = fqp.split("/")
        ufqp = find_local_file(fqp)
        if ufqp[-4:] == ".pdf":
            sections, paragraphs, txt = parse_pdf(ufqp)
            display_results(sections, paragraphs, txt)
            if fqp not in st.session_state.filelist:
                st.session_state.filelist.insert(0, fqp)
            st.session_state.sidebar_combo = fqp
        else:
            st.error(f"Please select a PDF in first dropdown: {fqp}")
            st.stop()

    elif st.sidebar.caption == "Smart Parser" and fqp:
        # fqp_parts = fqp.split("/")
        ufqp = find_local_file(fqp)
        if ufqp[-4:] == ".pdf":
            sections, paragraphs, txt = parse_pdf(ufqp)
            if (len(sections)>0):
                smart_display_results(sections, paragraphs, txt)
            else:
                st.error("Could not find natural sections in this document.")
            if fqp not in st.session_state.filelist:
                st.session_state.filelist.insert(0, fqp)
            st.session_state.sidebar_combo = fqp
        else:
            st.error(f"Error retrieving PDF -- Please select another: {fqp}")
            st.stop()
    else:
        # preview page
        if fqp[-4:] == ".pdf":
            txt = extract_text_from_webdav_pdf(fqp)
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
