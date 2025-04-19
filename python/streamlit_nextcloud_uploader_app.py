import base64
import os
import pdb
import re as regex0
from urllib.parse import urlparse

# find document sections
import pdfplumber
import torch

# validate english words
import langid
from wordfreq import zipf_frequency

import easywebdav
import fitz  # or the chosen library
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


torch.classes.__path__ = [] # add this line to manually set it to empty.

text_selected = False
file_list = ["No file selected","",""]
folder_path='/Users/personal/Documents'


# Nextcloud WebDAV URL and credentials
#webdav_url = 'https://yournextcloud.domain/remote.php/dav/files/yourusername/'
webdav_url = 'http://localhost:8088/remote.php/dav/files/nextcloud_admin'
webdav_local_path = 'remote.php/dav/files/nextcloud_admin'
username = '****'
password = '****'
remote_path = 'resume_library'

if 'wdurl' not in st.session_state:
    st.session_state['wdurl'] = webdav_url
if 'wdlurl' not in st.session_state:
    st.session_state['wdlpath'] = webdav_local_path
if 'wduser' not in st.session_state:
    st.session_state['wduser'] = username
if 'wdpass' not in st.session_state:
    st.session_state['wdpass'] = password
if 'wdpath' not in st.session_state:
    st.session_state['wdpath'] = remote_path
if 'wdfqpath' not in st.session_state:
    st.session_state['wdfqpath'] = webdav_url + os.sep + remote_path + os.sep


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
    st.sidebar.caption = "Previewer"
    #preview_file = st.selectbox('Select a PDF', st.session_state.filelist)
    # if st.sidebar.caption == "Nothing Selected":
    #     st.error("Nothing Selected")
    #     st.stop()
    # else:
    #     st.sidebar.caption = "Previewer"

# Show title and description of the app.
st.title('Work In Pilot')
st.sidebar.title("Help")
st.sidebar.markdown('Upload, preview or parse resume PDF file \n You can use this page to upload or parse a PDF document text or preview a resume PDF.')
#st.sidebar.caption ="Nothing Selected"

sidebar_selection = st.sidebar.selectbox("Recent Selections", options=st.session_state['filelist']) #options=file_list)

if 'sidebar_combo' not in st.session_state:
    st.session_state['sidebar_combo'] = sidebar_selection

sidebar_selection.join(file_list)

pg = st.navigation([Uploader, Parser, NLP_Parser, Previewer])
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
    client.baseurl = st.session_state["wdurl"]
    fpath = st.session_state["wdpath"]
    files = client.ls()
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
    webdavurl = st.session_state['wdurl']
    rpath = st.session_state['wdpath']
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

    # new
    phone_number_pattern = (
        r"^(\+\d{1,3})?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$"
    )
    match = regex0.findall(phone_number_pattern, text)
    if match:
        contact_info['phone_numbers'] = match.group(0)

    return contact_info


def extract_education_details(text):
    # Use regular expressions to extract education details
    education_details = []
    # patterns = [
    #     r'([A-Za-z\s]+)\s*("b.a.")\s*([A-Za-z\s]+)\s*(from|at)\s*([A-Za-z\s]+)\s*(in|on)\s*(\d{4})',
    #     r'([A-Za-z\s]+)\s*(college|university)\s*([A-Za-z\s]+)\s*(from|at)\s*([A-Za-z\s]+)\s*(in|on)\s*(\d{4})',
    #     r'([A-Za-z\s]+)\s*(earned|received|graduated with)\s*([A-Za-z\s]+)\s*(from|at)\s*([A-Za-z\s]+)\s*(in|on)\s*(\d{4})',
    #     r'([A-Za-z\s]+)\s*(earned|received|graduated with)\s*([A-Za-z\s]+)\s*(from|at)\s*([A-Za-z\s]+)',
    #     r'([A-Za-z\s]+)\s*(graduated|earned|received)\s*(from|at)\s*([A-Za-z\s]+)\s*(in|on)\s*(\d{4})',
    #     r'([A-Za-z\s]+)\s*(graduated|earned|received)\s*(from|at)\s*([A-Za-z\s]+)',
    #     #r'([A-Za-z\s]+)\s*(college|university|bachelors)\s*(from|at)\s*([A-Za-z\s]+)',
    #     #r'([A-Za-z\s]+)\s*(education)\s*([A-Za-z\s]+)',
    #     r'([A-Za-z\s]+)\"M.B.A."\s*([A-Za-z\s]+)',
    #     r'([A-Za-z\s]+)\"B.A."\s*([A-Za-z\s]+)',
    # ]

    #ba_pattern = [r'(["B"]["A"][" "])']
    #ba_pattern = [r'(["B"]["."]*["A"]["."]*[" "|"\n"])']
    masterspattern = regex0.compile(r'Masters? ')
    mbapattern = regex0.compile(r'(M\.?B\.?A\.?)', regex0.IGNORECASE)
    mbamatches = regex0.findall(mbapattern, text) #, regex0.IGNORECASE)
    bapattern = regex0.compile(r"(B\.?A\.?)")
    mastersmatches = regex0.findall(masterspattern, text)
    bamatches = regex0.findall(bapattern, text) #, regex0.IGNORECASE)
    mba_match = ""
    ba_match = ""
    masters_match = ""
    if len(mbamatches) > 0:
        mba_match = mbamatches[len(mbamatches)-1]
    if len(bamatches) > 0:
        ba_match = bamatches[len(bamatches)-1]
    if len(mastersmatches) > 0:
        masters_match = mastersmatches[len(mastersmatches)-1]

    # for pattern in patterns:
        # matches = regex0.findall(pattern, text, regex0.IGNORECASE)
        # import pdb
        # pdb.set_trace()
        # for match in matches:
        #     education_details.append({
        #         'all': match[0],
        #         'one': match[1] if len(match) > 1 else None,
        #         'degree': match[2] if len(match) > 2 else None,
        #         'university': match[3] if len(match) > 3 else None,
        #         'year': match[6] if len(match) > 6 else None,
        #     })
        #

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
            'degree': ba_match
        })
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
                'degree': ba_match
            })


    return education_details


def case_insensitive_split_re(mtext, delimiter):
    return regex0.split(delimiter, mtext, flags=regex0.IGNORECASE)

def extract_work_experience(text):
    work_experience = []
    #import pdb

    # Split text into sections
    if r"experience" in text.lower():
        # pdb.set_trace()
        #sections = text.split(r"EXPERIENCE")
        sections = case_insensitive_split_re(text, 'experience')
        paragraphs = sections[1].split('\n')

        #pdb.set_trace()

        #sections2 = sections[1].split("\n")
        #sections = sections2
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
        section2 = sections[1].split('\n')
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


def extract_text_from_webdav_pdf(fqp):
    parsed_url = urlparse(st.session_state["wdurl"])
    server_address = parsed_url.geturl()
    client = easywebdav.connect(server_address,
                                username=st.session_state["wduser"],
                                password=st.session_state["wdpass"],
                                protocol=parsed_url.scheme
                                )
    client.baseurl = st.session_state["wdurl"]
    byte_data = bytes()
    reconstructed_url = parsed_url.scheme + "://" + parsed_url.netloc + fqp
    try:
        with client.session.get(reconstructed_url, stream=True) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=8192):
                # append each chunk of bytes
                byte_data += chunk
    except Exception as e:
        print(f"An error occurred: {e}")
        st.write(f"An error occurred: {e}")

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

    return high_sections, paragraphs, full_text


def display_results(sections, paragraphs, txt):

    sect_indices = {}

    langid.set_languages(['en'])  # ISO 639-1 codes

    st.write("**Sections Detected**")
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

    # debug
    sect_indices['EOF'] = [len(txt),len(txt)]
    #sect_items = list(sect_indices.items())
    st_list = sorted(list(sect_indices.values()))

    j_list = sorted(list(sect_indices.items()), key=lambda x: x[1][0])
    for idx, key in enumerate(sect_indices):
        if idx > len(sect_indices) - 2:
            break
        sect_indices[j_list[idx][0]].append(st_list[idx + 1][0])

    st.divider()
    if len(sections) > 0 and len(sect_indices)>0:
        st.markdown(":page_facing_up: :blue-background[ Text Block Dump ]")
    else:
        st.markdown(":no_entry_sign: :red[No Sections Detected]")
    ks = list(sect_indices.keys())
    vs = list(sect_indices.values())
    for idx in range (len(sect_indices)-1):
        k = ks[idx]
        v = sect_indices[k]
        #st.page_link()
        st.write("Document Section: ",k)
        section_text = txt[v[0]:v[1]]
        paras = section_text.split('\n') if k == "EDUCATION" else section_text.split('. ')
        # st.write()
        for para in paras:
            if (len(para.strip())>0):
                if '\n' in para:
                    lines = para.replace('\n',' ')
                    st.write(lines)
                else:
                    st.write(f" - {para}")

    st.write(f"*Document EOF at line [{sect_indices['EOF'][1]}]*")
    st.write("***")
    st.write("**Natural Paragraphs Detected**")
    for para in paragraphs:
        st.write(f" - {para}")


def find_local_file(filename):
    if filename:
        if os.path.exists("/Users/personal/Documents/"+filename):
            return os.path.realpath("/Users/personal/Documents/"+filename)
        if os.path.exists("/Users/personal/Downloads/"+filename):
            return os.path.realpath("/Users/personal/Downloads/"+filename)
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
        # import pdb
        # pdb.set_trace()
        fqp_parts = fqp.split("/")
        ufqp = find_local_file(fqp_parts[-1])
        if ufqp[-4:] == ".pdf":
            sections, paragraphs, txt = parse_pdf(ufqp)
            display_results(sections, paragraphs, txt)
            if fqp not in st.session_state.filelist:
                st.session_state.filelist.insert(0, fqp)
            st.session_state.sidebar_combo = fqp
        else:
            st.error(f"Please select a PDF in first dropdown: {fqp}")
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
