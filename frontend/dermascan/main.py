import io
import torch
import torch.nn as nn
import timm
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torchvision.transforms as transforms
import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import warnings

warnings.filterwarnings('ignore')



# Manually define feature sizes to prevent errors
GHOSTNET_FEATURES = 1280
ALBERT_FEATURES = 768


class DermaScanFlexibleModel(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5, projection_dim=512):
        super().__init__()
        self.image_projection = nn.Linear(GHOSTNET_FEATURES, projection_dim)
        self.text_projection = nn.Linear(ALBERT_FEATURES, projection_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(projection_dim, num_classes)
        )

    def forward(self, image_features, text_features):
        projected_image = self.image_projection(image_features)
        projected_text = self.text_projection(text_features)
        # Weighted fusion 70% image + 30% text
        combined_features = (0.7 * projected_image) + (0.3 * projected_text)
        return self.classifier(combined_features)


# --- PART 2: LOAD MODELS AND SET UP API ---
print("Loading all necessary models... (This may take a moment)")

# Configuration
MODEL_PATH = 'dermascan_finetuned_model.pth'
NUM_CLASSES = 11
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Load backbone models
image_model_backbone = timm.create_model('ghostnet_100', pretrained=True, num_classes=0).to(DEVICE)
text_model_backbone = AutoModel.from_pretrained("albert-base-v2").to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

image_model_backbone.eval()
text_model_backbone.eval()

# Instantiate your fusion model
model = DermaScanFlexibleModel(num_classes=NUM_CLASSES, dropout_rate=0.5).to(DEVICE)

# --- Safe model loading ---
if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(" Models loaded successfully!")
    except Exception as e:
        print(f"Warning: Model found but could not be loaded due to error: {e}")
else:
    print(f"Warning: Model file '{MODEL_PATH}' not found. The API will still run but predictions may fail.")


# --- CLASS LABELS ---
class_names = [
    'Acne/Rosacea', 'Bacterial Infection', 'Contact Dermatitis', 'Eczema', 'Fungal Infection',
    'Hives', 'Infestations/Bites', 'Not a Skin Condition', 'Potentially Malignant Lesions',
    'Psoriasis', 'Viral Infection'
]

# --- IMAGE TRANSFORMS ---
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


feature_database = {
    'Acne/Rosacea': {
        'care': [
            "Wash face with a gentle cleanser twice a day.",
            "Avoid oily or greasy cosmetics.",
            "Do not pick or squeeze pimples."
        ],
        'link': "https://www.aad.org/public/diseases/acne"
    },
    'Eczema': {
        'care': [
            "Moisturize skin frequently with a fragrance-free cream.",
            "Avoid long, hot showers or baths.",
            "Identify and avoid personal triggers (e.g., certain soaps, fabrics)."
        ],
        'link': "https://nationaleczema.org/eczema/"
    },
    'Psoriasis': {
        'care': [
            "Use medicated creams or ointments as prescribed.",
            "Keep skin moisturized to reduce scaling and itching.",
            "Manage stress, as it can be a trigger."
        ],
        'link': "https://www.psoriasis.org/about-psoriasis/"
    },
    'Potentially Malignant Lesions': {
        'care': [
            "See a dermatologist immediately for a professional evaluation.",
            "Perform regular self-examinations of your skin.",
            "Always use broad-spectrum sunscreen with SPF 30 or higher."
        ],
        'link': "https://www.skincancer.org/"
    },
    'Fungal Infection': {
        'care': [
            "Keep the affected area clean and dry.",
            "Use over-the-counter antifungal creams or powders.",
            "Wear clean, dry clothing and avoid sharing towels."
        ],
        'link': "https://www.cdc.gov/fungal/diseases/ringworm/index.html"
    },
    'Not a Skin Condition': {
        'care': [
            "The uploaded image does not appear to be a skin condition. Please upload a clear photo of the affected skin area for analysis."
        ],
        'link': ""
    },
    'Default': {
        'care': [
            "Keep the area clean and dry.",
            "Avoid scratching the affected area.",
            "Consult a healthcare professional for an accurate diagnosis."
        ],
        'link': "https://www.aad.org/public/diseases"
    }
}


# --- PREDICTION FUNCTION ---
def predict(image_bytes, symptoms):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = inference_transforms(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        image_features = image_model_backbone(image_tensor)
        text_inputs = tokenizer(
            symptoms,
            padding='max_length',
            max_length=128,
            truncation=True,
            return_tensors="pt"
        )
        text_inputs = {key: val.to(DEVICE) for key, val in text_inputs.items()}
        text_output = text_model_backbone(**text_inputs)
        text_features = text_output.last_hidden_state[:, 0, :]

        logits = model(image_features, text_features)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_class = class_names[predicted_idx.item()]
    confidence_score = confidence.item()
    return predicted_class, confidence_score


# --- PART 3: CREATE THE FASTAPI APP ---
app = FastAPI(title="DermaScan AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def handle_prediction(image: UploadFile = File(...), symptoms: str = Form(...)):
    image_bytes = await image.read()
    predicted_class, confidence = predict(image_bytes, symptoms)

    features = feature_database.get(predicted_class, feature_database['Default'])

    return {
        "prediction": predicted_class,
        "confidence": f"{confidence:.2%}",
        "care_suggestions": features['care'],
        "learn_more_link": features['link']
    }


@app.get("/")
def home():
    return {"status": "DermaScan AI server is running successfully "}
