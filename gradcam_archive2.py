import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


# =========================
# CONFIGURATION
# =========================

MODEL_PATH = r"models/classification/archive2_best.pt"
IMAGE_FOLDER = r"gradcam_test_images"
OUTPUT_FOLDER = r"gradcam_outputs"
IMG_SIZE = 224

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")


# =========================
# LOAD MODEL
# =========================

print("[INFO] Loading YOLO classification model...")
yolo_model = YOLO(MODEL_PATH)

model = yolo_model.model.to(device)

# Important for Grad-CAM
# We keep model in eval mode for stable prediction,
# but force gradients to be enabled.
model.eval()
for param in model.parameters():
    param.requires_grad_(True)

class_names = yolo_model.names
print("[INFO] Model classes:", class_names)


# =========================
# FIND LAST CONVOLUTION LAYER
# =========================

def find_last_conv_layer(torch_model):
    last_conv_name = None
    last_conv_layer = None

    for name, module in torch_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv_name = name
            last_conv_layer = module

    if last_conv_layer is None:
        raise RuntimeError("No Conv2d layer found in the model.")

    return last_conv_name, last_conv_layer


target_layer_name, target_layer = find_last_conv_layer(model)
print(f"[INFO] Target layer for Grad-CAM: {target_layer_name}")


# =========================
# HOOKS FOR GRAD-CAM
# =========================

activations = None
gradients = None


def forward_hook(module, input, output):
    global activations
    activations = output


def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)


# =========================
# IMAGE PREPROCESSING
# =========================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Important: make input require gradient
    input_tensor.requires_grad_(True)

    return image, input_tensor


# =========================
# OUTPUT EXTRACTION
# =========================

def extract_logits(output):
    """
    Ultralytics classification model may return:
    - Tensor
    - Tuple/list: usually (probabilities, logits)
    This function selects the best tensor for backpropagation.
    """

    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (list, tuple)):
        tensors = [x for x in output if isinstance(x, torch.Tensor)]

        if len(tensors) == 0:
            raise RuntimeError("No tensor found in model output.")

        # Prefer raw logits, usually the last tensor in Ultralytics classification output
        for t in reversed(tensors):
            if t.ndim == 2:
                return t

        return tensors[-1]

    raise RuntimeError(f"Unsupported model output type: {type(output)}")


# =========================
# GRAD-CAM GENERATION
# =========================

def generate_gradcam(image_path, output_path):
    global activations, gradients

    activations = None
    gradients = None

    original_image, input_tensor = preprocess_image(image_path)

    model.zero_grad(set_to_none=True)

    with torch.enable_grad():
        output = model(input_tensor)
        logits = extract_logits(output)

        # Make sure logits can backpropagate
        if not logits.requires_grad:
            # Last fallback: use model in train mode only for Grad-CAM graph creation
            model.train()
            output = model(input_tensor)
            logits = extract_logits(output)
            model.eval()

        if not logits.requires_grad:
            raise RuntimeError("Logits still do not require grad. Grad-CAM cannot be generated.")

        probabilities = torch.softmax(logits, dim=1)
        class_idx = int(torch.argmax(probabilities, dim=1).item())
        confidence = float(probabilities[0, class_idx].item())

        class_label = class_names[class_idx] if isinstance(class_names, dict) else str(class_idx)

        print(f"[INFO] Image: {os.path.basename(image_path)}")
        print(f"       Prediction: {class_label}")
        print(f"       Confidence: {confidence:.4f}")

        score = logits[0, class_idx]
        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

    if activations is None or gradients is None:
        raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

    # Grad-CAM calculation
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)

    cam = torch.nn.functional.interpolate(
        cam,
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False
    )

    cam = cam.squeeze().detach().cpu().numpy()

    # Normalize heatmap
    cam = cam - cam.min()
    if cam.max() != 0:
        cam = cam / cam.max()

    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Prepare original image
    original_resized = original_image.resize((IMG_SIZE, IMG_SIZE))
    original_np = np.array(original_resized)
    original_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)

    # Overlay heatmap
    overlay = cv2.addWeighted(original_bgr, 0.55, heatmap, 0.45, 0)

    # Add prediction text
    text = f"{class_label} ({confidence:.2f})"
    cv2.putText(
        overlay,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imwrite(output_path, overlay)
    print(f"       Saved: {output_path}")


# =========================
# RUN FOR ALL IMAGES
# =========================

valid_extensions = (".jpg", ".jpeg", ".png", ".webp")

image_files = [
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith(valid_extensions)
]

if not image_files:
    print("[ERROR] No images found in gradcam_test_images folder.")
else:
    print(f"[INFO] Found {len(image_files)} image(s). Generating Grad-CAM outputs...")

    for image_file in image_files:
        image_path = os.path.join(IMAGE_FOLDER, image_file)
        name_without_ext = os.path.splitext(image_file)[0]
        output_path = os.path.join(OUTPUT_FOLDER, f"{name_without_ext}_gradcam.png")

        try:
            generate_gradcam(image_path, output_path)
        except Exception as e:
            print(f"[ERROR] Failed for {image_file}: {e}")

    print("[DONE] Grad-CAM generation completed.")