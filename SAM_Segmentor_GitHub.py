import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap
import cv2
import os
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
from pycocotools import mask as mask_utils

class SAMApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Set up the user interface
        self.layout = QVBoxLayout()
        
        # Label for instructions
        self.label = QLabel("Select a folder to process images")
        self.layout.addWidget(self.label)
        
        # Button to select a folder
        self.button = QPushButton("Select Folder", self)
        self.button.clicked.connect(self.showDialog)
        self.layout.addWidget(self.button)

        # Set the initial size and title of the main window
        self.setGeometry(100, 100, 800, 600)  # x, y, width, height
        self.setWindowTitle('SAM Segmentation App')
        self.setLayout(self.layout)
        self.show()

    def showDialog(self):
        # Open a dialog to select the folder containing images
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if folder_path:
            self.processImages(folder_path)

    def processImages(self, folder_path):
        # Load the SAM model
        sam_model_type = "vit_h"  # Choose model type: "vit_h", "vit_l", or "vit_b"
        checkpoint_path = "path_to_checkpoint/sam_vit_h_4b8939.pth"  # Update with your checkpoint path
        sam = sam_model_registry[sam_model_type](checkpoint=checkpoint_path)
        
        # Initialize the predictor and mask generator
        predictor = SamPredictor(sam)
        mask_generator = SamAutomaticMaskGenerator(sam)

        # Iterate through all image files in the selected folder
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    self.detectAndDisplay(image_path, predictor, mask_generator)
                    
        # Update the label to indicate completion
        self.label.setText("Processing complete!")

    def detectAndDisplay(self, image_path, predictor, mask_generator):
        # Read the image
        img = cv2.imread(image_path)

        # Check if the image was loaded successfully
        if img is None:
            print(f"Error: Could not read the image {image_path}.")
            return

        # Generate masks for the image
        masks = mask_generator.generate(img)
        
        # Create an empty mask with the same dimensions as the image
        combined_mask = np.zeros_like(img, dtype=np.uint8)
        
        # Draw masks on the combined mask
        for mask in masks:
            mask_data = mask.get('segmentation')
            if mask_data is not None:
                mask_bool = mask_data.astype(bool)  # Ensure the mask is a boolean array
                combined_mask[mask_bool] = (0, 255, 0)  # Green color for the mask
        
        # Blend the original image with the combined mask
        img = cv2.addWeighted(img, 1, combined_mask, 0.5, 0)
        
        # Display the image with masks
        cv2.imshow("Segmented Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SAMApp()
    sys.exit(app.exec_())
