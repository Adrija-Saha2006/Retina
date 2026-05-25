import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, render_template, jsonify, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# --- 1. Load the Hybrid Ensemble ---
MODEL_MAIN_PATH = 'retina_tri_risk_model.h5'
MODEL_HYP_PATH = 'hypertension_specialist.keras'

model_main = load_model(MODEL_MAIN_PATH)
model_hyp = load_model(MODEL_HYP_PATH)

# Warm-up: Initialize symbolic tensors for Keras 3 / Python 3.12
print("🚀 Initializing Dual-Stream Ensemble...")
model_main.predict(np.zeros((1, 224, 224, 3)), verbose=0)
model_hyp.predict(np.zeros((1, 300, 300, 3)), verbose=0)
print("✅ Systems Ready.")

UPLOAD_FOLDER = 'static/heatmaps/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- 2. DYNAMIC CLINICAL INTERPRETATION LOGIC ---
def generate_clinical_insight(stroke_v, ckd_v, hyp_v):
    risks = {"Stroke": stroke_v, "Chronic Kidney Disease": ckd_v, "Hypertension": hyp_v}
    primary_condition = max(risks, key=risks.get)
    primary_val = risks[primary_condition]
    
    if primary_val < 0.15:
        return "Normal physiological appearance. Microvascular biomarkers are within standard reference ranges."

    if primary_condition == "Stroke":
        severity = "Critical" if primary_val > 0.40 else "Moderate"
        return f"{severity} Concern: ResNet-50 backbone has isolated atypical vessel tortuosity. High correlation with cerebrovascular risk markers."
    elif primary_condition == "Chronic Kidney Disease":
        severity = "Significant" if primary_val > 0.40 else "Early-Stage"
        return f"{severity} Risk: Detected micro-aneurysm patterns and vessel leakage markers. Retinal-renal progression markers identified."
    else: 
        severity = "Urgent" if primary_val > 0.40 else "Persistent"
        return f"{severity} Hypertensive Concern: Specialist B3 engine has detected generalized arteriolar narrowing."

# --- 3. THE SURGICAL GRAD-CAM ENGINE (Order of Operations Fixed) ---
def get_heatmap(img_array, model, prob, model_type="resnet"):
    """
    Generates probability-scaled heatmaps. 
    Fix: Normalizes/Sharpen first, Scale by probability last.
    """
    if model_type == "resnet":
        last_conv_layer_name = "conv5_block3_out"
        grad_model = tf.keras.models.Model(model.input, [model.get_layer(last_conv_layer_name).output, model.output])
        with tf.GradientTape() as tape:
            conv_output, preds = grad_model(img_array)
            top_class_channel = preds[:, tf.argmax(preds[0, :2])] 
    else:
        last_conv_layer_name = "block7b_add" 
        backbone = model.get_layer('efficientnetb3')
        backbone_model = tf.keras.models.Model(backbone.input, [backbone.get_layer(last_conv_layer_name).output, backbone.output])
        with tf.GradientTape() as tape:
            conv_output, backbone_final = backbone_model(img_array)
            x = backbone_final
            for layer in model.layers[1:]: x = layer(x)
            top_class_channel = x[:, 0]

    grads = tape.gradient(top_class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) 
    
    heatmap_np = heatmap.numpy()

    # --- THE MATHEMATICAL CORRECTION ---
    
    # 1. PEAK NORMALIZATION: Set highest point to 1.0 to define feature shape
    max_val = np.max(heatmap_np)
    if max_val > 0:
        heatmap_np = heatmap_np / max_val
    
    # 2. FEATURE SHARPENING: Denoise on the 0-1 range
    if model_type == "effnet":
        heatmap_np = np.power(heatmap_np, 3) # Cubic transform
        heatmap_np[heatmap_np < 0.5] = 0     # Hard threshold for vascular clarity
    else:
        heatmap_np[heatmap_np < 0.3] = 0     # Moderate threshold for global ResNet features

    # 3. PROBABILITY SCALING: Map intensity to actual risk (Intensity = $Shape \times Probability$)
    # If risk is 49%, the hottest spot is 0.49 (Visible Orange/Yellow)
    # If risk is 1%, the hottest spot is 0.01 (Faint Blue)
    heatmap_np = heatmap_np * prob

    return heatmap_np

def apply_heatmap(img_path, heatmap):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (512, 512))
    heatmap = cv2.resize(heatmap, (512, 512))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# --- 4. THE ANALYZER ROUTES ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return jsonify({"error": "No file uploaded"})
    file = request.files['file']
    img_path = os.path.join(UPLOAD_FOLDER, "temp.jpg")
    file.save(img_path)

    # Preprocessing
    img_224 = np.expand_dims(image.img_to_array(image.load_img(img_path, target_size=(224, 224)))/255.0, axis=0)
    img_300 = np.expand_dims(image.img_to_array(image.load_img(img_path, target_size=(300, 300)))/255.0, axis=0)

    # Inference
    preds_main = model_main.predict(img_224, verbose=0)[0]
    preds_hyp = model_hyp.predict(img_300, verbose=0)[0]

    stroke_v, ckd_v, hyp_v = float(preds_main[0]), float(preds_main[1]), float(preds_hyp[0])

    # Dynamic Scaling
    prob_main = max(stroke_v, ckd_v)
    
    # Generate Dual Heatmaps
    heatmap_main = get_heatmap(img_224, model_main, prob_main, "resnet")
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, "analysis_stroke_ckd.jpg"), apply_heatmap(img_path, heatmap_main))

    heatmap_hyp = get_heatmap(img_300, model_hyp, hyp_v, "effnet")
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, "analysis_hypertension.jpg"), apply_heatmap(img_path, heatmap_hyp))

    # Get Interpretation
    interpretation = generate_clinical_insight(stroke_v, ckd_v, hyp_v)
    
    return jsonify({
        "stroke": f"{round(stroke_v*100, 1)}%",
        "ckd": f"{round(ckd_v*100, 1)}%",
        "hypertension": f"{round(hyp_v*100, 1)}%",
        "heatmap_main_url": url_for('static', filename='heatmaps/analysis_stroke_ckd.jpg'),
        "heatmap_hyp_url": url_for('static', filename='heatmaps/analysis_hypertension.jpg'),
        "insight": interpretation
    })

@app.route('/')
def index(): return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)