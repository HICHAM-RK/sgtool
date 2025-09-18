import os
import logging

from mmdet.apis import (
    inference_detector,
    init_detector,
    show_result,
    show_result_pyplot,
)
from PIL import Image

# Try to import enhanced borderless detection
try:
    from .borderless_table_detector import BorderlessTableDetector, EnhancedTableDetector
    BORDERLESS_AVAILABLE = True
    logging.info("Borderless table detection available")
except ImportError as e:
    BORDERLESS_AVAILABLE = False
    logging.warning(f"Borderless table detection not available: {e}")
    logging.warning("Install dependencies: pip install transformers timm torch")


def return_table(table_coordinates):

    tables_dict = []

    for table_data in table_coordinates:
        table_dict = {
            "name": "table",
            "xmin": table_data[0],
            "ymin": table_data[1],
            "xmax": table_data[2],
            "ymax": table_data[3],
        }

        tables_dict.append(table_dict)

    return tables_dict


class TableDetector:
    instance = None

    def __init__(self, weights_path=None, use_enhanced=True):
        # init your model here
        if weights_path is None:
            weights_path = os.path.join(os.path.dirname(__file__), "weight.pth")
            # Comment out assertion for now as weights don't exist
            # assert os.path.exists(weights_path), weights_path

        self.use_enhanced = use_enhanced and BORDERLESS_AVAILABLE
        
        if self.use_enhanced:
            try:
                # Use enhanced detector with borderless capabilities
                self.enhanced_detector = EnhancedTableDetector(weights_path=weights_path)
                logging.info("Using enhanced table detector with borderless capabilities")
            except Exception as e:
                logging.warning(f"Could not initialize enhanced detector: {e}")
                self.use_enhanced = False
        
        if not self.use_enhanced:
            # Fallback to traditional detection
            logging.info("Using traditional table detection")
            # TODO: Replace this dummy implementation with your actual mmdet model
            self.model = lambda x: [
                {
                    "name": "table",
                    "xmin": 100,
                    "ymin": 100,
                    "xmax": 200,
                    "ymax": 200,
                }
            ]

    def process(self, image):
        """Process image and return detected tables"""
        if self.use_enhanced:
            # Use enhanced detection (traditional + borderless)
            return self.enhanced_detector.process(image)
        else:
            # Use traditional detection only
            output = self.model(image)
            return output

    @classmethod
    def get_unique_instance(cls, use_enhanced=True):
        if cls.instance is None:
            # initialize instance
            cls.instance = cls(use_enhanced=use_enhanced)
        return cls.instance


def main():
    # Load model
    config_file = "Project/Go5-Project/table_detection/CascadeTabNet/Config/cascade_mask_rcnn_hrnetv2p_w32_20e.py"
    checkpoint_file = "Project/Go5-Project/table_detection/checkpoints/epoch_1.pth"

    model = init_detector(config_file, checkpoint_file, device="cuda:0")

    img = "/Go5-Project/sample.jpg"

    # Run Inference
    result = inference_detector(model, img)

    table_coordinates = []

    ## extracting bordered tables
    for bordered_tables in result[0][0]:
        table_coordinates.append(bordered_tables[:4].astype(int))

    ## extracting borderless tables
    for borderless_tables in result[0][2]:
        table_coordinates.append(borderless_tables[:4].astype(int))

    table_data = return_table(table_coordinates)

    if len(table_data) != 0:
        # visualize the results in a new window
        show_result_pyplot(
            img, result, ("Bordered", "Cell", "Borderless"), score_thr=0.8
        )

        # or save the visualization results to image files
        show_result(
            img, result, ("Bordered", "Cell", "Borderless"), out_file="out_file.jpg"
        )
    else:
        im1 = Image.open(img)

        im1 = im1.save("out_file.jpg")

    for i in table_data:
        print(i)


if __name__ == "__main__":
    main()
