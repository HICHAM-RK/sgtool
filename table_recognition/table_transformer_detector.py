"""
Table Transformer Detector for Borderless Tables
Implements Microsoft's Table Transformer models for detecting and recognizing table structure
"""

import os
import torch
import numpy as np
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
import cv2
from typing import List, Dict, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TableTransformerDetector:
    """
    Table Transformer detector for borderless tables using Microsoft's pre-trained models
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model names
        self.detection_model_name = "microsoft/table-transformer-detection"
        self.structure_model_name = "microsoft/table-transformer-structure-recognition"
        
        # Initialize models
        self.detection_processor = None
        self.detection_model = None
        self.structure_processor = None
        self.structure_model = None
        
        self._load_models()
    
    def _load_models(self):
        """Load the Table Transformer models"""
        try:
            logger.info("Loading Table Transformer models...")
            
            # Load detection model (finds table regions)
            self.detection_processor = DetrImageProcessor.from_pretrained(
                self.detection_model_name
            )
            self.detection_model = DetrForObjectDetection.from_pretrained(
                self.detection_model_name
            )
            self.detection_model.to(self.device)
            self.detection_model.eval()
            
            # Load structure recognition model (finds rows, columns, cells)
            self.structure_processor = DetrImageProcessor.from_pretrained(
                self.structure_model_name
            )
            self.structure_model = DetrForObjectDetection.from_pretrained(
                self.structure_model_name
            )
            self.structure_model.to(self.device)
            self.structure_model.eval()
            
            logger.info("Table Transformer models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _preprocess_image(self, image: np.ndarray) -> Image.Image:
        """
        Preprocess image for the models
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            PIL Image in RGB format
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Convert to RGB if not already
        pil_image = pil_image.convert('RGB')
        
        return pil_image
    
    def detect_tables(self, image: np.ndarray) -> List[Dict]:
        """
        Detect table regions in the image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected tables with bounding boxes and confidence scores
        """
        # Preprocess image
        pil_image = self._preprocess_image(image)
        
        # Process image
        inputs = self.detection_processor(pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.detection_model(**inputs)
        
        # Post-process results
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)  # (height, width)
        results = self.detection_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
        )[0]
        
        # Convert to our format
        tables = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if label.item() == 0:  # Table class
                xmin, ymin, xmax, ymax = box.tolist()
                tables.append({
                    'confidence': score.item(),
                    'xmin': int(xmin),
                    'ymin': int(ymin),
                    'xmax': int(xmax),
                    'ymax': int(ymax),
                    'label': 'table'
                })
        
        logger.info(f"Detected {len(tables)} tables")
        return tables
    
    def recognize_table_structure(self, image: np.ndarray, table_bbox: Dict) -> Dict:
        """
        Recognize the internal structure of a detected table
        
        Args:
            image: Original image as numpy array
            table_bbox: Bounding box of the table {'xmin', 'ymin', 'xmax', 'ymax'}
            
        Returns:
            Dictionary containing detected rows, columns, and cells
        """
        # Crop table region
        xmin, ymin, xmax, ymax = table_bbox['xmin'], table_bbox['ymin'], table_bbox['xmax'], table_bbox['ymax']
        table_crop = image[ymin:ymax, xmin:xmax]
        
        if table_crop.size == 0:
            return {'rows': [], 'columns': [], 'cells': []}
        
        # Preprocess cropped table
        pil_image = self._preprocess_image(table_crop)
        
        # Process image
        inputs = self.structure_processor(pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.structure_model(**inputs)
        
        # Post-process results
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
        results = self.structure_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
        )[0]
        
        # Classify detected elements
        rows = []
        columns = []
        cells = []
        
        # Structure model class mapping
        # 0: table, 1: table column, 2: table row, 3: table column header, 
        # 4: table projected row header, 5: table spanning cell
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            rel_xmin, rel_ymin, rel_xmax, rel_ymax = box.tolist()
            
            # Convert relative coordinates back to original image coordinates
            abs_xmin = int(rel_xmin + xmin)
            abs_ymin = int(rel_ymin + ymin)
            abs_xmax = int(rel_xmax + xmin)
            abs_ymax = int(rel_ymax + ymin)
            
            element = {
                'confidence': score.item(),
                'xmin': abs_xmin,
                'ymin': abs_ymin,
                'xmax': abs_xmax,
                'ymax': abs_ymax,
                'label_id': label.item()
            }
            
            if label.item() == 1:  # Column
                element['label'] = 'column'
                columns.append(element)
            elif label.item() == 2:  # Row
                element['label'] = 'row'
                rows.append(element)
            elif label.item() in [3, 4, 5]:  # Headers and spanning cells
                element['label'] = 'cell'
                cells.append(element)
        
        logger.info(f"Found {len(rows)} rows, {len(columns)} columns, {len(cells)} cells")
        
        return {
            'rows': rows,
            'columns': columns,
            'cells': cells
        }
    
    def create_grid_cells(self, rows: List[Dict], columns: List[Dict]) -> List[Dict]:
        """
        Create grid cells from detected rows and columns
        
        Args:
            rows: List of detected row elements
            columns: List of detected column elements
            
        Returns:
            List of grid cells
        """
        grid_cells = []
        
        for row in rows:
            for col in columns:
                # Create cell from intersection
                cell_xmin = max(col['xmin'], row['xmin'])
                cell_ymin = max(row['ymin'], col['ymin'])
                cell_xmax = min(col['xmax'], row['xmax'])
                cell_ymax = min(row['ymax'], col['ymax'])
                
                # Check if intersection is valid
                if cell_xmax > cell_xmin and cell_ymax > cell_ymin:
                    grid_cells.append({
                        'xmin': cell_xmin,
                        'ymin': cell_ymin,
                        'xmax': cell_xmax,
                        'ymax': cell_ymax,
                        'label': 'grid_cell',
                        'confidence': min(row['confidence'], col['confidence'])
                    })
        
        return grid_cells
    
    def process_image(self, image: np.ndarray) -> Dict:
        """
        Complete pipeline: detect tables and recognize their structure
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing detected tables and their structure
        """
        results = {
            'tables': [],
            'structure_elements': []
        }
        
        try:
            # Step 1: Detect tables
            logger.info("Detecting tables...")
            tables = self.detect_tables(image)
            results['tables'] = tables
            
            # Step 2: Recognize structure for each table
            for i, table in enumerate(tables):
                logger.info(f"Processing table {i+1}/{len(tables)}")
                structure = self.recognize_table_structure(image, table)
                
                # Add table index to structure elements
                for element_type in ['rows', 'columns', 'cells']:
                    for element in structure[element_type]:
                        element['table_id'] = i
                        results['structure_elements'].append(element)
                
                # Create grid cells if we have both rows and columns
                if structure['rows'] and structure['columns']:
                    grid_cells = self.create_grid_cells(structure['rows'], structure['columns'])
                    for cell in grid_cells:
                        cell['table_id'] = i
                        results['structure_elements'].append(cell)
            
            logger.info(f"Processing complete. Found {len(tables)} tables with {len(results['structure_elements'])} structure elements")
            
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            
        return results
    
    def visualize_results(self, image: np.ndarray, results: Dict, output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize detection results on the image
        
        Args:
            image: Original image
            results: Detection results from process_image
            output_path: Optional path to save the visualization
            
        Returns:
            Image with visualizations drawn
        """
        vis_image = image.copy()
        
        # Draw tables in red
        for table in results['tables']:
            cv2.rectangle(vis_image, 
                         (table['xmin'], table['ymin']), 
                         (table['xmax'], table['ymax']), 
                         (0, 0, 255), 3)
            cv2.putText(vis_image, f"Table ({table['confidence']:.2f})", 
                       (table['xmin'], table['ymin']-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw structure elements in different colors
        colors = {
            'row': (0, 255, 0),      # Green for rows
            'column': (255, 0, 0),    # Blue for columns  
            'cell': (255, 255, 0),    # Cyan for cells
            'grid_cell': (0, 255, 255)  # Yellow for grid cells
        }
        
        for element in results['structure_elements']:
            color = colors.get(element['label'], (128, 128, 128))
            
            cv2.rectangle(vis_image,
                         (element['xmin'], element['ymin']),
                         (element['xmax'], element['ymax']),
                         color, 2)
        
        if output_path:
            cv2.imwrite(output_path, vis_image)
            
        return vis_image


def test_table_transformer():
    """Test function to verify the implementation"""
    # Create a simple test
    detector = TableTransformerDetector()
    
    # Create a test image (white background)
    test_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Add some text to simulate a table
    cv2.putText(test_image, 'ITEM    QTY    DESCRIPTION', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(test_image, '67227   1.00   CARBON MONOXIDE ALARM', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(test_image, '75572   1.00   AUTO BYPASS VALVE', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    print("Testing Table Transformer...")
    results = detector.process_image(test_image)
    print(f"Found {len(results['tables'])} tables")
    print(f"Found {len(results['structure_elements'])} structure elements")
    
    return detector, results


if __name__ == "__main__":
    test_table_transformer()
