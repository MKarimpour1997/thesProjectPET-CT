import os
import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Ensure CUDA is available
assert torch.cuda.is_available()
device = torch.device('cuda')

# Load your model and set to eval mode
model = CustomConvNet(num_classes=3)
model.load_state_dict(torch.load('best_best_model.pth'))
model.to(device).eval()


input_folder = 'testSet/normal'
output_folder = 'heatmap'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
        image_path = os.path.join(input_folder, filename)
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        rgb_image = np.float32(original_image) / 255.0

        input_tensor = torch.from_numpy(rgb_image.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_class = torch.argmax(outputs).item()

        cam = GradCAMPlusPlus(model=model, target_layers=[model.layer5])
        targets = [ClassifierOutputTarget(predicted_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        grayscale_cam_resized = cv2.resize(grayscale_cam, (original_image.shape[1], original_image.shape[0]))

        visualization = show_cam_on_image(
            rgb_image,
            grayscale_cam_resized,
            use_rgb=True,
            image_weight=0.68,
            colormap=cv2.COLORMAP_JET
        )

        output_path = os.path.join(output_folder, f"heatmap_{filename}")
        visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, visualization_bgr)

print("Processing complete. Heatmaps saved in the 'heatmap' folder.")
