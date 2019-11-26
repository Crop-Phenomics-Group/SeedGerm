import numpy as np

class Panel:
    
    def __init__(self, label, mask, centroid, bbox):
        self.label = label
        #self.mask = mask
        self.centroid = centroid
        self.bbox = bbox
        
        min_row, min_col, max_row, max_col = self.bbox
        self.mask_crop = mask[min_row:max_row, min_col:max_col]

    def __str__(self):
        return "panel no: " + str(self.label)

    def get_bbox_image(self, img):
        min_row, min_col, max_row, max_col = self.bbox
        cropped_img = img[min_row:max_row, min_col:max_col]
        return cropped_img        
    
    def get_cropped_image(self, img):
        min_row, min_col, max_row, max_col = self.bbox
        mask_crop_stack = np.dstack([self.mask_crop] * 3)
        cropped_img = img[min_row:max_row, min_col:max_col]
        if cropped_img.shape == mask_crop_stack.shape:
            return cropped_img * mask_crop_stack
        else:
            return cropped_img
