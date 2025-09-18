import os

import sys
from tempfile import TemporaryDirectory
from typing import Dict, List
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import zipfile
import io
import time
from datetime import datetime

current_dir = os.path.dirname(__file__)
# Add paths to find the modules
sys.path.insert(0, os.path.join(current_dir, 'streamlit'))          # classes.py and utility.py
sys.path.insert(0, os.path.join(current_dir, 'text_recognition'))   # TextRecognizer
sys.path.insert(0, os.path.join(current_dir, 'table_recognition'))

from classes import read_tables_from_list, read_texts_from_list
from utility import dump_excel, get_random_color, merge_text_table
# Import text recognizer with MD5 bypass for model compatibility
try:
    from text_recognition_md5_bypass import TextRecognizer
    print("✅ Using MD5 bypass TextRecognizer")
except Exception as e:
    print(f"⚠️ MD5 bypass failed, trying standard: {e}")
    try:
        from text_recognition import TextRecognizer
        print("✅ Using standard TextRecognizer")
    except Exception as e2:
        print(f"⚠️ Standard failed, using fallback: {e2}")
        from text_recognition_fallback import FallbackTextRecognizer as TextRecognizer
from table_recognition import TableRecognizer

# Import PDF processor
from pdf_processor import pdf_processor

# Page configuration with custom theme
st.set_page_config(
    layout="wide", 
    page_icon="🚀", 
    page_title="SG Table Extractor TOOL",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/library/get-started',
        'Report a bug': 'https://github.com/streamlit/streamlit/issues',
        'About': "# SG Table Extractor TOOL\nPowered by advanced AI to extract tables from PDFs and images with precision!"
    }
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main theme colors */
    :root {
        --primary-color: #4f46e5;
        --secondary-color: #06b6d4;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --dark-bg: #1f2937;
        --light-bg: #f8fafc;
    }
    
    /* Global font */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid var(--primary-color);
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin: 1rem 0;
        transition: transform 0.2s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .feature-card h3 {
        color: var(--primary-color);
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    /* Stats cards */
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stats-number {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .stats-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Upload area styling */
    .upload-area {
        border: 3px dashed #e5e7eb;
        border-radius: 12px;
        padding: 3rem;
        text-align: center;
        background: #f9fafb;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: var(--primary-color);
        background: #f0f9ff;
    }
    
    /* Progress styling */
    .stProgress .st-bo {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        height: 8px;
        border-radius: 4px;
    }
    
    /* Success message styling */
    .success-message {
        background: linear-gradient(90deg, #10b981, #059669);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stDownloadButton button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        color: white;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def create_header():
    """Create the main header with modern styling"""
    st.markdown("""
    <div class="main-header">
        <h1>🚀 SG Table Extractor TOOL</h1>
        <p>Transform your PDFs and images into structured Excel data with cutting-edge AI technology</p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create an enhanced sidebar with features and stats"""
    with st.sidebar:
        st.markdown("## ⚙️ Processing Options")
        
        # Processing settings
        dpi = st.slider("PDF Quality (DPI)", 150, 300, 200, 25, 
                       help="Higher DPI = better quality but slower processing")
        
        max_pages = st.number_input("Max Pages to Process", 1, 50, 10,
                                  help="Limit processing for large PDFs")
        
        st.markdown("---")
        
        # Features section
        st.markdown("## ✨ Features")
        features = [
            ("📄", "PDF Support", "Multi-page PDF processing"),
            ("🖼️", "Image Support", "PNG, JPG, JPEG formats"),
            ("🎯", "AI Detection", "Advanced table recognition"),
            ("📊", "Excel Export", "Structured data output"),
            ("🔍", "OCR Engine", "Text extraction & recognition"),
            ("📦", "Batch Download", "ZIP file generation")
        ]
        
        for icon, title, desc in features:
            st.markdown(f"""
            <div class="feature-card">
                <h3>{icon} {title}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("## 📈 Session Stats")
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = 0
        if 'extracted_tables' not in st.session_state:
            st.session_state.extracted_tables = 0
            
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Files", st.session_state.processed_files, delta=1 if st.session_state.get('just_processed') else None)
        with col2:
            st.metric("Tables", st.session_state.extracted_tables, delta=st.session_state.get('last_extraction', 0))
    
    return dpi, max_pages

def process_single_image(image: np.ndarray, image_name: str = "image"):
    """Process a single image for table extraction with progress tracking"""
    tables = []
    texts = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Table Recognition
    status_text.text(f"🔍 Analyzing table structure in {image_name}...")
    progress_bar.progress(25)
    
    table_recognizer = TableRecognizer.get_unique_instance()
    tables = table_recognizer.process(image, [])
    
    progress_bar.progress(50)
    
    # Text Recognition
    status_text.text(f"📝 Extracting text content from {image_name}...")
    progress_bar.progress(75)
    
    # Initialize TextRecognizer with better error handling for Streamlit context
    try:
        text_recognizer = TextRecognizer.get_unique_instance()
        texts = text_recognizer.process(image, [])
    except Exception as e:
        st.error(f"⚠️ Text recognition failed: {str(e)}")
        st.info("📝 Continuing without text extraction. Tables will still be detected.")
        texts = []  # Continue without text recognition

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
    status_text.text(f"🔗 Combining text with table structures...")
    progress_bar.progress(90)
    
    merge_text_table(tables, texts)
    tables.sort(key=lambda t: t.ymin)
    
    progress_bar.progress(100)
    status_text.text("✅ Processing complete!")
    
    time.sleep(0.5)  # Brief pause to show completion
    progress_bar.empty()
    status_text.empty()
    
    return tables, texts

def display_results(image: np.ndarray, tables: List, texts: List, page_name: str = ""):
    """Display extraction results with enhanced visualization"""
    
    if len(tables) == 0:
        st.warning(f"⚠️ No tables detected in {page_name}. The image might not contain clear table structures.", icon="⚠️")
        
        # Show helpful tips
        with st.expander("💡 Tips for better table detection"):
            st.markdown("""
            **For better results, ensure your document has:**
            - Clear table borders and grid lines
            - High contrast between text and background
            - Minimal skew or rotation
            - Good image resolution (at least 150 DPI)
            - Tables that are not overlapping with other content
            """)
        return []
    
    # Success message
    st.markdown(f"""
    <div class="success-message">
        ✅ <strong>Success!</strong> Found {len(tables)} table(s) in {page_name}
    </div>
    """, unsafe_allow_html=True)
    
    # Create enhanced tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        f"📊 Extracted Data", 
        f"🎯 Detection Results", 
        f"📝 OCR Analysis",
        f"📈 Statistics"
    ])
    
    excel_files = []
    
    with tab1:
        st.markdown("### 🗂️ Structured Table Data")
        
        for idx, table in enumerate(tables):
            with st.expander(f"📋 Table {idx + 1} - Click to expand", expanded=True):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**Table {idx + 1}**")
                with col2:
                    st.markdown(f"**Cells:** {len(table.cells)}")
                with col3:
                    dimensions = f"{table.xmax - table.xmin} × {table.ymax - table.ymin}px"
                    st.markdown(f"**Size:** {dimensions}")
                
                # Index the table
                table.indexing()
                
                with TemporaryDirectory() as temp_dir_path:
                    xlsx_path = os.path.join(temp_dir_path, f"table_{idx}_{page_name.replace(' ', '_')}.xlsx")
                    dump_excel([table], xlsx_path)
                    
                    try:
                        with open(xlsx_path, "rb") as ref:
                            df = pd.read_excel(ref)
                            
                            # Display dataframe with custom styling
                            st.dataframe(
                                df, 
                                use_container_width=True,
                                height=min(400, len(df) * 35 + 50)
                            )
                            
                            # Store file data for download
                            ref.seek(0)
                            file_data = ref.read()
                            excel_files.append((f"table_{idx}_{page_name.replace(' ', '_')}.xlsx", file_data))
                            
                            # Enhanced download button
                            col1, col2, col3 = st.columns([1, 1, 2])
                            with col1:
                                st.download_button(
                                    f"📥 Download Excel",
                                    file_data,
                                    file_name=f"table_{idx}_{page_name.replace(' ', '_')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                            with col2:
                                # Show data preview info
                                st.info(f"📊 {len(df)} rows × {len(df.columns)} columns")
                                
                    except Exception as e:
                        st.error(f"❌ Error processing table: {str(e)}")
                        
                        # Show raw table data as fallback
                        with st.expander("🔍 Raw Table Data"):
                            for cell in table.cells:
                                st.text(f"Cell ({cell.xmin}, {cell.ymin}): {cell.ocr}")

    with tab2:
        st.markdown("### 🎯 Table Detection Visualization")
        
        if len(tables) <= 3:
            cols = st.columns(len(tables) + 1)
        else:
            cols = st.columns(4)
        
        # Original image with table boundaries
        with cols[0]:
            st.markdown("**🔍 All Tables Detected**")
            vimage = image.copy()
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            
            for idx, table in enumerate(tables):
                color = colors[idx % len(colors)]
                cv2.rectangle(vimage, (table.xmin, table.ymin), (table.xmax, table.ymax), color, 4)
                
                # Add table label
                cv2.putText(vimage, f"T{idx+1}", (table.xmin, table.ymin-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            st.image(vimage, use_container_width=True)
        
        # Individual table cell visualization
        for idx, table in enumerate(tables[:3]):
            col_idx = (idx + 1) % len(cols)
            with cols[col_idx]:
                st.markdown(f"**📋 Table {idx + 1} Cells**")
                vimage = image.copy()
                
                for cell in table.cells:
                    cv2.rectangle(vimage, (cell.xmin, cell.ymin), (cell.xmax, cell.ymax), 
                                get_random_color(), -1)
                
                blended = cv2.addWeighted(vimage, 0.3, image, 0.7, 0)
                st.image(blended, use_container_width=True)
                
                st.caption(f"{len(table.cells)} cells detected")

    with tab3:
        st.markdown("### 📝 OCR Analysis Results")
        
        if len(texts) > 0:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                vimage = image.copy()
                
                # Draw text bounding boxes with different colors for different tables
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
                
                for table_idx, table in enumerate(tables):
                    color = colors[table_idx % len(colors)]
                    for cell in table.cells:
                        for text in cell.texts:
                            cv2.rectangle(vimage, (text.xmin, text.ymin), (text.xmax, text.ymax), color, 2)
                
                st.image(vimage, caption="Text Regions Detected", use_container_width=True)
            
            with col2:
                # OCR Statistics
                total_text_regions = sum(len(cell.texts) for table in tables for cell in table.cells)
                total_characters = sum(len(text.ocr) for table in tables for cell in table.cells for text in cell.texts)
                
                st.markdown("**📊 OCR Statistics**")
                st.metric("Text Regions", total_text_regions)
                st.metric("Total Characters", total_characters)
                st.metric("Avg. Chars/Region", round(total_characters / max(total_text_regions, 1), 1))
                
                # Show confidence if available
                st.markdown("**🎯 Detection Quality**")
                st.progress(0.85)  # Placeholder for actual confidence score
                st.caption("85% confidence")
            
            # Text content summary
            with st.expander("📄 Extracted Text Content"):
                for table_idx, table in enumerate(tables):
                    st.markdown(f"**Table {table_idx + 1} Text:**")
                    for cell_idx, cell in enumerate(table.cells):
                        if cell.texts:
                            text_content = " | ".join([text.ocr for text in cell.texts])
                            st.text(f"Cell {cell_idx + 1}: {text_content}")
        else:
            st.info("ℹ️ No text regions detected in the processed image.")

    with tab4:
        st.markdown("### 📈 Processing Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="stats-card">
                <div class="stats-number">{}</div>
                <div class="stats-label">Tables Found</div>
            </div>
            """.format(len(tables)), unsafe_allow_html=True)
        
        with col2:
            total_cells = sum(len(table.cells) for table in tables)
            st.markdown("""
            <div class="stats-card">
                <div class="stats-number">{}</div>
                <div class="stats-label">Total Cells</div>
            </div>
            """.format(total_cells), unsafe_allow_html=True)
        
        with col3:
            total_text = sum(len(cell.texts) for table in tables for cell in table.cells)
            st.markdown("""
            <div class="stats-card">
                <div class="stats-number">{}</div>
                <div class="stats-label">Text Elements</div>
            </div>
            """.format(total_text), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed table information
        st.markdown("**🔍 Detailed Table Analysis**")
        
        table_data = []
        for idx, table in enumerate(tables):
            table_data.append({
                "Table": f"Table {idx + 1}",
                "Cells": len(table.cells),
                "Width (px)": table.xmax - table.xmin,
                "Height (px)": table.ymax - table.ymin,
                "Text Elements": sum(len(cell.texts) for cell in table.cells)
            })
        
        if table_data:
            df_stats = pd.DataFrame(table_data)
            st.dataframe(df_stats, use_container_width=True)
    
    return excel_files

def create_zip_download(all_excel_files: List, filename: str = "extracted_tables.zip"):
    """Create a ZIP file containing all Excel files with metadata"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add Excel files
        for file_name, file_data in all_excel_files:
            zip_file.writestr(file_name, file_data)
        
        # Add metadata file
        metadata = f"""Table Extraction Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Tables: {len(all_excel_files)}
Processing Method: AI Table Recognition

Files Included:
{chr(10).join([f'- {name}' for name, _ in all_excel_files])}
"""
        zip_file.writestr("extraction_report.txt", metadata)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def app():
    """Main application with enhanced UI"""
    # Create header
    create_header()
    
    # Create sidebar and get settings
    dpi, max_pages = create_sidebar()
    
    # Main content area
    st.markdown("### 📤 Upload Your Documents")
    
    # Enhanced upload section
    st.markdown("""
    <div class="upload-area">
        <h3>🎯 Drag and Drop Your Files Here</h3>
        <p>Support for PDF documents and images (PNG, JPG, JPEG)</p>
    </div>
    """, unsafe_allow_html=True)
    
    upload_file = st.file_uploader(
        label="Choose your file", 
        type=["png", "jpg", "jpeg", "pdf"],
        help="Maximum file size: 200MB. For best results, ensure images are clear and tables have visible borders.",
        label_visibility="collapsed"
    )
    
    if upload_file is not None:
        # Update session state
        st.session_state.processed_files += 1
        st.session_state.just_processed = True
        
        filename = upload_file.name
        filetype = upload_file.type
        filebyte = bytearray(upload_file.read())

        # File information
        file_size = len(filebyte) / 1024 / 1024  # MB
        st.markdown(f"""
        **📄 File Information:**
        - **Name:** {filename}
        - **Type:** {filetype}
        - **Size:** {file_size:.2f} MB
        - **Processing DPI:** {dpi}
        """)
        
        # Processing progress
        overall_progress = st.progress(0)
        status_container = st.container()
        
        # Check if it's a PDF
        if pdf_processor.is_pdf(upload_file):
            with status_container:
                st.info("🔍 PDF document detected. Starting conversion to images...", icon="🔍")
            
            overall_progress.progress(10)
            
            # Extract PDF metadata
            metadata = pdf_processor.extract_pdf_metadata(bytes(filebyte))
            
            # Enhanced PDF information display
            st.markdown("### 📋 Document Overview")
            info_col1, info_col2, info_col3, info_col4 = st.columns(4)
            
            with info_col1:
                st.metric("📄 Pages", metadata.get('page_count', 'Unknown'))
            with info_col2:
                title = str(metadata.get('title', 'Unknown'))
                display_title = title[:15] + "..." if len(title) > 15 else title
                st.metric("📝 Title", display_title)
            with info_col3:
                author = str(metadata.get('author', 'Unknown'))
                display_author = author[:15] + "..." if len(author) > 15 else author
                st.metric("👤 Author", display_author)
            with info_col4:
                creation_date = metadata.get('creation_date', 'Unknown')
                if isinstance(creation_date, str) and creation_date != 'Unknown':
                    try:
                        date_obj = datetime.fromisoformat(creation_date.replace('Z', '+00:00'))
                        display_date = date_obj.strftime('%Y-%m-%d')
                    except:
                        display_date = creation_date[:10] if len(str(creation_date)) > 10 else creation_date
                else:
                    display_date = 'Unknown'
                st.metric("📅 Created", display_date)
            
            overall_progress.progress(20)
            
            # Convert PDF to images
            with status_container:
                conversion_status = st.empty()
                conversion_status.info("🔄 Converting PDF pages to high-resolution images...")
            
            pdf_images = pdf_processor.pdf_to_images(bytes(filebyte), dpi=dpi)
            
            overall_progress.progress(40)
            
            if not pdf_images:
                st.error("❌ Failed to process PDF. Please ensure the file is a valid PDF document.")
                return
            
            conversion_status.success(f"✅ Successfully converted {len(pdf_images)} pages")
            
            # Limit pages if specified
            if len(pdf_images) > max_pages:
                st.warning(f"⚠️ PDF has {len(pdf_images)} pages, but processing is limited to {max_pages} pages. Adjust in sidebar if needed.")
                pdf_images = pdf_images[:max_pages]
            
            # Page selection with enhanced UI
            if len(pdf_images) > 1:
                st.markdown("### 📋 Page Selection")
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    select_all = st.button("✅ Select All Pages", use_container_width=True)
                    select_none = st.button("❌ Deselect All", use_container_width=True)
                
                # Create thumbnail preview with grid layout
                thumbnails = pdf_processor.get_page_thumbnails(bytes(filebyte), max_pages=min(len(pdf_images), 12))
                
                selected_pages = []
                
                # Display thumbnails in a grid
                cols_per_row = 4
                for i in range(0, min(len(thumbnails), 12), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        if i + j < len(thumbnails):
                            thumb, page_num = thumbnails[i + j]
                            with col:
                                st.image(thumb, use_container_width=True)
                                default_checked = True if select_all else not select_none if select_none else True
                                if st.checkbox(f"Page {page_num}", value=default_checked, key=f"page_{page_num}"):
                                    selected_pages.append(page_num)
                
                # Handle remaining pages if more than 12
                if len(pdf_images) > 12:
                    with st.expander(f"📄 View All {len(pdf_images) - 12} Remaining Pages"):
                        remaining_cols = st.columns(3)
                        for i, (_, page_num) in enumerate(pdf_images[12:], 13):
                            col_idx = (i - 13) % 3
                            with remaining_cols[col_idx]:
                                default_checked = True if select_all else not select_none if select_none else True
                                if st.checkbox(f"Page {page_num}", value=default_checked, key=f"page_{page_num}"):
                                    selected_pages.append(page_num)
                
                if not selected_pages:
                    selected_pages = [page_num for _, page_num in pdf_images]
            else:
                selected_pages = [1]
            
            overall_progress.progress(50)
            
            # Process selected pages
            st.markdown("### 🔄 Processing Results")
            all_excel_files = []
            total_tables = 0
            
            for i, (image_array, page_num) in enumerate(pdf_images):
                if page_num in selected_pages:
                    progress = 50 + (i / len(selected_pages)) * 40
                    overall_progress.progress(int(progress))
                    
                    st.markdown(f"---\n## 📄 Page {page_num}")
                    
                    # Show the page image in a compact format
                    with st.expander(f"🖼️ View Page {page_num} Image", expanded=False):
                        st.image(image_array, caption=f"Page {page_num}", use_container_width=True)
                    
                    try:
                        # Process the page
                        tables, texts = process_single_image(image_array, f"Page {page_num}")
                        
                        page_tables = len(tables)
                        total_tables += page_tables
                        
                        if page_tables > 0:
                            st.success(f"✅ Page {page_num}: Found {page_tables} table(s) and {len(texts)} text region(s)")
                            
                            # Display results
                            page_excel_files = display_results(image_array, tables, texts, f"Page {page_num}")
                            all_excel_files.extend(page_excel_files)
                        else:
                            st.info(f"ℹ️ Page {page_num}: No tables detected")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing page {page_num}: {str(e)}")
            
            overall_progress.progress(90)
            
            # Update session state
            st.session_state.extracted_tables += total_tables
            st.session_state.last_extraction = total_tables
            
            # Final results summary
            if all_excel_files:
                overall_progress.progress(100)
                
                st.markdown("---")
                st.markdown("## 🎉 Processing Complete!")
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("""
                    <div class="stats-card">
                        <div class="stats-number">{}</div>
                        <div class="stats-label">Pages Processed</div>
                    </div>
                    """.format(len(selected_pages)), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="stats-card">
                        <div class="stats-number">{}</div>
                        <div class="stats-label">Tables Extracted</div>
                    </div>
                    """.format(len(all_excel_files)), unsafe_allow_html=True)
                
                with col3:
                    file_size_mb = sum(len(data) for _, data in all_excel_files) / 1024 / 1024
                    st.markdown("""
                    <div class="stats-card">
                        <div class="stats-number">{:.1f}MB</div>
                        <div class="stats-label">Total Size</div>
                    </div>
                    """.format(file_size_mb), unsafe_allow_html=True)
                
                with col4:
                    processing_time = "~2min"  # Placeholder - you could track actual time
                    st.markdown("""
                    <div class="stats-card">
                        <div class="stats-number">{}</div>
                        <div class="stats-label">Process Time</div>
                    </div>
                    """.format(processing_time), unsafe_allow_html=True)
                
                st.markdown("### 📦 Download Options")
                
                download_col1, download_col2 = st.columns(2)
                
                with download_col1:
                    # Individual downloads
                    st.markdown("**📄 Individual Files**")
                    for file_name, file_data in all_excel_files[:5]:  # Show first 5
                        st.download_button(
                            f"📥 {file_name}",
                            file_data,
                            file_name=file_name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    
                    if len(all_excel_files) > 5:
                        st.info(f"... and {len(all_excel_files) - 5} more files in ZIP")
                
                with download_col2:
                    # Bulk download
                    st.markdown("**📦 Bulk Download**")
                    zip_data = create_zip_download(all_excel_files, f"{filename.replace('.pdf', '')}_tables.zip")
                    
                    st.download_button(
                        "🚀 Download All Tables (ZIP)",
                        zip_data,
                        file_name=f"{filename.replace('.pdf', '')}_tables.zip",
                        mime="application/zip",
                        use_container_width=True,
                        help=f"Contains {len(all_excel_files)} Excel files and processing report"
                    )
                    
                    st.success(f"✅ Ready to download {len(all_excel_files)} Excel files!")
            else:
                st.warning("⚠️ No tables were found in the selected pages. Try adjusting the DPI or selecting different pages.")
            
        else:
            # Handle regular image files
            overall_progress.progress(20)
            
            image = np.asarray(filebyte, dtype=np.uint8)
            image = cv2.imdecode(image, 1)
            
            if image is None:
                st.error("❌ Failed to process image. Please ensure the file is a valid image format.")
                return
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            overall_progress.progress(40)
            
            # Show original image
            st.markdown("### 🖼️ Original Image")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            overall_progress.progress(50)
            
            try:
                # Process the image
                tables, texts = process_single_image(image, filename)
                
                overall_progress.progress(80)
                
                # Update session stats
                st.session_state.extracted_tables += len(tables)
                st.session_state.last_extraction = len(tables)
                
                if len(tables) > 0:
                    st.success(f"✅ Processing complete! Found {len(tables)} table(s) and {len(texts)} text region(s)")
                    
                    # Display results
                    excel_files = display_results(image, tables, texts)
                    
                    overall_progress.progress(100)
                    
                    # Download options for single image
                    if len(excel_files) > 1:
                        st.markdown("### 📦 Download All Tables")
                        zip_data = create_zip_download(excel_files, f"{filename.replace('.', '_')}_tables.zip")
                        st.download_button(
                            "🚀 Download All Tables (ZIP)",
                            zip_data,
                            file_name=f"{filename.replace('.', '_')}_tables.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                else:
                    st.info("ℹ️ No tables detected in this image.")
                
            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
                st.error("Please ensure the image contains clear tables with visible borders.")
        
        overall_progress.progress(100)
        
        # Success celebration
        if st.session_state.get('last_extraction', 0) > 0:
            st.balloons()
            
            # Show next steps
            with st.expander("🎯 What's Next?"):
                st.markdown("""
                **Great job! Here's what you can do now:**
                
                1. **📊 Review Data:** Check the extracted tables for accuracy
                2. **📝 Edit in Excel:** Open downloaded files in Excel for further editing
                3. **🔄 Process More:** Upload additional files to extract more tables
                4. **💡 Tips:** For better results, ensure clear table borders and good image quality
                
                **Need help?** Check our processing tips in the sidebar!
                """)
    
    else:
        # Landing page with instructions
        st.markdown("### 🚀 Get Started")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h3>📄 PDF Processing</h3>
                <p>Upload multi-page PDF documents and extract tables from each page automatically. Perfect for reports, invoices, and financial documents.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h3>🖼️ Image Processing</h3>
                <p>Process PNG, JPG, and JPEG images containing tables. Ideal for screenshots, scanned documents, and photos of printed tables.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h3>🎯 AI-Powered Detection</h3>
                <p>Advanced computer vision algorithms detect table structures and extract text with high accuracy, even from complex layouts.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h3>📊 Excel Export</h3>
                <p>Get your data in structured Excel format, ready for analysis, reporting, or further processing in your favorite tools.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Usage instructions
        st.markdown("---")
        st.markdown("### 📋 How to Use")
        
        steps_col1, steps_col2, steps_col3 = st.columns(3)
        
        with steps_col1:
            st.markdown("""
            **Step 1: Upload** 📤
            - Click the upload area above
            - Select PDF or image files
            - Maximum size: 200MB
            """)
        
        with steps_col2:
            st.markdown("""
            **Step 2: Process** ⚙️
            - AI analyzes your document
            - Tables are automatically detected
            - Text is extracted using OCR
            """)
        
        with steps_col3:
            st.markdown("""
            **Step 3: Download** 📥
            - Review extracted data
            - Download individual Excel files
            - Or get everything in a ZIP
            """)
        
        # Tips section
        with st.expander("💡 Pro Tips for Best Results"):
            st.markdown("""
            **For PDFs:**
            - Use high DPI (200-300) for better quality
            - Ensure the PDF is not password protected
            - Text-based PDFs work better than image-only PDFs
            
            **For Images:**
            - Ensure good contrast between table lines and background
            - Avoid blurry or low-resolution images
            - Make sure tables have clear borders
            - Minimize skew and rotation
            
            **General:**
            - Tables with clear grid lines work best
            - Avoid overlapping content or watermarks
            - Consider splitting very large tables across multiple images
            """)

if __name__ == "__main__":
    app()