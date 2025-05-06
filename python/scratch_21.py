from streamlit_pdf_viewer import pdf_viewer
import streamlit as st
from streamlit_image_select import image_select
import os
import platform
import sys

# Method 1: Using os.name
os_name = os.name
print(f"OS name (os.name): {os_name}")

# Method 2: Using platform.system()
os_system = platform.system()
print(f"OS name (platform.system()): {os_system}")

# Method 3: Using sys.platform
os_platform = sys.platform
print(f"OS name (sys.platform): {os_platform}")

# Method 4: Using platform.uname()
os_uname = platform.uname()
print(f"OS information (platform.uname()): {os_uname}")

#Example of checking for a specific OS
if os_system == "Windows":
    print("Running on Windows")
elif os_system == "Linux":
    print("Running on Linux")
elif os_system == "Darwin":
    print("Running on macOS")
else:
    print("Operating system not recognized")

img = image_select("Choose a Resume Template",
                   ["/Users/personal/Desktop/image1.png",
                    "/Users/personal/Desktop/image2.png",
                    "/Users/personal/Desktop/image3.png"])
st.write(img)

container_pdf, container_chat = st.columns([50, 50])
with container_pdf:
    pdf_file = st.file_uploader("Upload Content (Personal Resume PDF)", type=('pdf'))

    if pdf_file:
        binary_data = pdf_file.getvalue()
        # pdf_viewer(input=binary_data,
        #            width=700)

        lcol, rcol = st.columns([2, 1])
        with lcol.expander("**Preview**", expanded=bool(pdf_file)):
            pdf_viewer(
                input=binary_data,
                height=250
                # key=,
            )
