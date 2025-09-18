import os
from typing import List

import cv2
from pydantic import BaseModel


def show(img, name="disp", width=1000):
    """
    name: name of window, should be name of img
    img: source of img, should in type ndarray
    """
    cv2.namedWindow(name, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(name, width, 1000)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class Box(BaseModel):
    name: str = "box"
    xmin: int
    xmax: int
    ymin: int
    ymax: int

    @property
    def width(self):
        return max(self.xmax - self.xmin, 0)

    @property
    def height(self):
        return max(self.ymax - self.ymin, 0)


class Text(Box):
    name: str = "text"
    ocr: str = ""


class TextRecognizer:
    instance = None

    def __init__(self):
        import easyocr
        import os
        
        # Try to use project-local models first
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate from text_recognition/ up to tableextract/ then to models/easyocr/
        project_model_dir = os.path.join(current_dir, "..", "models", "easyocr")
        project_model_dir = os.path.abspath(project_model_dir)
        
        # Fallback to user directory
        user_model_dir = os.path.expanduser("~/.EasyOCR/model")
        
        # Check which model directory to use
        if os.path.exists(project_model_dir) and os.path.exists(os.path.join(project_model_dir, "craft_mlt_25k.pth")):
            model_storage_directory = project_model_dir
            print(f"📦 Using project-local models: {model_storage_directory}")
        else:
            model_storage_directory = user_model_dir
            os.makedirs(model_storage_directory, exist_ok=True)
            print(f"🏠 Using user directory models: {model_storage_directory}")
        
        # Check if models exist
        craft_model = os.path.join(model_storage_directory, "craft_mlt_25k.pth")
        english_model = os.path.join(model_storage_directory, "english_g2.pth")
        
        models_exist = os.path.exists(craft_model) and os.path.exists(english_model)
        
        if models_exist:
            print("✅ EasyOCR models found locally, initializing offline mode...")
        
        try:
            # Force offline mode if models exist
            if models_exist:
                # Set environment variable to force offline mode
                os.environ['EASYOCR_MODULE_PATH'] = model_storage_directory
                
            self.reader: easyocr.Reader = easyocr.Reader(
                lang_list=["en"], 
                download_enabled=False,
                model_storage_directory=model_storage_directory,
                gpu=False,  # Force CPU to avoid GPU-related issues
                verbose=False  # Reduce verbose output
            )
            print("✅ EasyOCR initialized successfully in offline mode")
            
        except Exception as e:
            print(f"⚠️ EasyOCR initialization failed with offline mode: {e}")
            
            if not models_exist:
                print("❌ Required models are missing. Please run the model downloader.")
                raise RuntimeError("Failed to initialize EasyOCR. Required models are missing. Please run 'python download_easyocr_models.py' to download them.")
            else:
                print("⚠️ Models exist but initialization failed. Trying alternative approach...")
                try:
                    # Alternative initialization attempt
                    import torch
                    torch.hub.set_dir(model_storage_directory)
                    
                    self.reader: easyocr.Reader = easyocr.Reader(
                        lang_list=["en"],
                        download_enabled=False,
                        model_storage_directory=model_storage_directory,
                        gpu=False
                    )
                    print("✅ EasyOCR initialized with alternative method")
                    
                except Exception as e2:
                    print(f"❌ All EasyOCR initialization attempts failed: {e2}")
                    raise RuntimeError(f"Failed to initialize EasyOCR even with models present. Error: {e2}")

    def process(self, image, text_list: list):
        texts = self.reader.readtext(image, min_size=1, text_threshold=0.3)

        output = []
        for (location, ocr, _) in texts:
            xmin = int(min(p[0] for p in location))
            xmax = int(max(p[0] for p in location))
            ymin = int(min(p[1] for p in location))
            ymax = int(max(p[1] for p in location))

            new_text = Text(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, ocr=ocr)
            output.append(new_text)

        return output

    @classmethod
    def get_unique_instance(cls):
        if cls.instance is None:
            # initialize instance
            cls.instance = cls()
        return cls.instance


def draw(image, text_list: List[Text]):
    for text in text_list:
        cv2.rectangle(
            image, (text.xmin, text.ymin), (text.xmax, text.ymax), (255, 0, 0), 4
        )
    return image


def main():
    # image = cv2.imread("/home/luan/research/Go5-Project/sample.jpg")

    import glob

    from tqdm import tqdm

    for image_path in tqdm(
        glob.glob("/home/luan/research/Go5-Project/data/images/*.jpg")
    ):
        image_name = os.path.basename(image_path)
        file_name = os.path.splitext(image_name)[0]

        image = cv2.imread(image_path)

        model = TextRecognizer.get_unique_instance()
        texts = model.process(image, text_list=[])
        # cv2.imwrite(
        #     f"/home/luan/research/Go5-Project/debug/{image_name}",
        #     draw(image, texts)
        # )
        output = [t.dict() for t in texts]

        import json

        with open(
            f"/home/luan/research/Go5-Project/cache/ocr/{file_name}.json", "w"
        ) as ref:
            json.dump(output, ref)


if __name__ == "__main__":
    main()
