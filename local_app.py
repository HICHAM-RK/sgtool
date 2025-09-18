import os
import sys
from tempfile import TemporaryDirectory
from typing import Dict, List
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Add the current directory to the path to import local modules
sys.path.append('.')
sys.path.append('./streamlit')

# Import from the local streamlit directory
import sys
import os

# Add paths to find the modules
streamlit_path = os.path.join(os.getcwd(), 'streamlit')
sys.path.insert(0, streamlit_path)

from classes import read_tables_from_list, read_texts_from_list
from utility import dump_excel, get_random_color, merge_text_table

# Import the text recognition module
text_recognition_path = os.path.join(os.getcwd(), 'text_recognition')
sys.path.insert(0, text_recognition_path)
from text_recognition import TextRecognizer

# Import table recognition module  
table_recognition_path = os.path.join(os.getcwd(), 'table_recognition')
sys.path.insert(0, table_recognition_path)
from table_recognition import TableRecognizer

st.set_page_config(layout="wide", page_icon="🖱️", page_title="Interactive Table App")
st.title("👨‍💻 Table Extraction App (Local Version)")

def app():
    st.write("Upload an image containing tables to extract data and convert to Excel format.")
    
    upload_file = st.file_uploader(label="Pick a file", type=["png", "jpg", "jpeg"])
    image = None
    
    if upload_file is not None:
        filename = upload_file.name
        filebyte = bytearray(upload_file.read())

        # Convert byte to image
        image = np.asarray(filebyte, dtype=np.uint8)
        image = cv2.imdecode(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.write(f"Processing image: {filename}")
        
        # Show original image
        st.subheader("Original Image")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        tables = []
        texts = []
        
        try:
            # Table Recognition
            with st.spinner("🤖 Extracting Table Structure!"):
                table_recognizer = TableRecognizer.get_unique_instance()
                tables = table_recognizer.process(image, [])
                st.success(f"Found {len(tables)} table(s)")

            # Text Recognition
            with st.spinner("🤖 Extracting Text Regions and OCR!"):
                text_recognizer = TextRecognizer.get_unique_instance()
                texts = text_recognizer.process(image, [])
                st.success(f"Found {len(texts)} text region(s)")

            # Convert to the expected format for compatibility
            table_dicts = []
            for table in tables:
                table_dict = {
                    "name": "table",
                    "xmin": table.xmin,
                    "ymin": table.ymin, 
                    "xmax": table.xmax,
                    "ymax": table.ymax,
                    "cells": []
                }
                for cell in table.cells:
                    cell_dict = {
                        "xmin": cell.xmin,
                        "ymin": cell.ymin,
                        "xmax": cell.xmax,
                        "ymax": cell.ymax
                    }
                    table_dict["cells"].append(cell_dict)
                table_dicts.append(table_dict)

            text_dicts = []
            for text in texts:
                text_dict = {
                    "name": "text",
                    "xmin": text.xmin,
                    "ymin": text.ymin,
                    "xmax": text.xmax,
                    "ymax": text.ymax,
                    "ocr": text.ocr
                }
                text_dicts.append(text_dict)

            # Convert back to proper objects
            tables = read_tables_from_list(table_dicts)
            texts = read_texts_from_list(text_dicts)
            
            # Merge text with tables
            merge_text_table(tables, texts)
            tables.sort(key=lambda t: t.ymin)

            if len(tables) == 0:
                st.warning("No tables were detected in the image. Please try with an image containing clear table structures.")
                return

            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(
                ["Tabular Data", "Table Visualization", "OCR Visualization"]
            )

            with tab1:
                st.header("Tabular Information")
                
                if len(tables) > 0:
                    for idx, table in enumerate(tables):
                        st.subheader(f"Table {idx + 1}")
                        
                        # Index the table
                        table.indexing()
                        
                        with TemporaryDirectory() as temp_dir_path:
                            xlsx_path = os.path.join(temp_dir_path, f"table_{idx}.xlsx")
                            dump_excel([table], xlsx_path)
                            
                            # Read and display the Excel file
                            try:
                                with open(xlsx_path, "rb") as ref:
                                    df = pd.read_excel(ref)
                                    st.dataframe(df)
                                    
                                    # Download button
                                    ref.seek(0)  # Reset file pointer
                                    file_data = ref.read()
                                    st.download_button(
                                        "Download Excel File",
                                        file_data,
                                        file_name=f"table_{idx}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                            except Exception as e:
                                st.error(f"Error reading Excel file: {str(e)}")
                                
                                # Show raw table data as fallback
                                st.write("Raw table data:")
                                for cell in table.cells:
                                    st.write(f"Cell ({cell.xmin}, {cell.ymin}): {cell.ocr}")
                else:
                    st.info("No tables detected")

            with tab2:
                st.header("Table Detection & Recognition")
                
                if len(tables) > 0:
                    cols = st.columns(len(tables) + 1)
                    
                    # Original image with table boundaries
                    with cols[0]:
                        st.subheader("Detected Tables")
                        vimage = image.copy()
                        for table in tables:
                            cv2.rectangle(
                                vimage,
                                (table.xmin, table.ymin),
                                (table.xmax, table.ymax),
                                (255, 0, 0),  # Red color for table boundaries
                                4,
                            )
                        st.image(vimage, caption="Table Boundaries", use_container_width=True)
                    
                    # Individual table cell visualization
                    for idx, table in enumerate(tables):
                        with cols[idx + 1]:
                            st.subheader(f"Table {idx + 1} Cells")
                            vimage = image.copy()
                            for cell in table.cells:
                                cv2.rectangle(
                                    vimage,
                                    (cell.xmin, cell.ymin),
                                    (cell.xmax, cell.ymax),
                                    get_random_color(),
                                    -1,
                                )
                            blended = vimage // 2 + image // 2
                            st.image(blended, caption=f"Table {idx + 1} Cells", use_container_width=True)
                else:
                    st.info("No tables to visualize")

            with tab3:
                st.header("OCR Visualization")
                
                if len(texts) > 0:
                    vimage = image.copy()
                    
                    # Draw text bounding boxes
                    for table in tables:
                        for cell in table.cells:
                            for text in cell.texts:
                                cv2.rectangle(
                                    vimage,
                                    (text.xmin, text.ymin),
                                    (text.xmax, text.ymax),
                                    (0, 255, 0),  # Green for text boxes
                                    2,
                                )
                    
                    # Add text labels
                    for table in tables:
                        for cell in table.cells:
                            for text in cell.texts:
                                cv2.putText(
                                    vimage,
                                    text.ocr,
                                    (text.xmin, text.ymin - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 0, 0),  # Blue for text
                                    1,
                                )
                    
                    st.image(vimage, caption="OCR Results", use_container_width=True)
                    
                    # Show text details
                    st.subheader("Detected Text Details")
                    for idx, text in enumerate(texts):
                        with st.expander(f"Text {idx + 1}: {text.ocr[:50]}..."):
                            st.write(f"**Text:** {text.ocr}")
                            st.write(f"**Coordinates:** ({text.xmin}, {text.ymin}) to ({text.xmax}, {text.ymax})")
                else:
                    st.info("No text detected")

            st.balloons()

        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            st.error("Please make sure the image contains clear tables and text.")

if __name__ == "__main__":
    app()
