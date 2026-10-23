
import io
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from transformers import pipeline

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

load_dotenv(BASE_DIR / ".env")

MODEL_NAME = "openai/clip-vit-base-patch32"

app = FastAPI(
    title="WasteWorth AI",
    description="Local pretrained image classification for waste analysis",
    version="2.0.0",
)

# --------------------------------------------------
# Indicative material rates in INR per kg.
# These are illustrative values, not live market prices.
# --------------------------------------------------

MATERIALS = {
    "plastic": {
        "rate": 15.0,
        "disposal": "Sort by plastic type and check local recycling rules.",
    },
    "paper": {
        "rate": 12.0,
        "disposal": "Keep clean and dry for paper recycling.",
    },
    "cardboard": {
        "rate": 10.0,
        "disposal": "Flatten boxes and keep them dry.",
    },
    "glass": {
        "rate": 3.0,
        "disposal": "Separate carefully and follow local glass collection rules.",
    },
    "metal": {
        "rate": 35.0,
        "disposal": "Separate metal from other waste for scrap collection.",
    },
    "aluminium": {
        "rate": 100.0,
        "disposal": "Collect clean aluminium separately for recycling.",
    },
    "steel": {
        "rate": 25.0,
        "disposal": "Separate ferrous metal for scrap recycling.",
    },
    "organic": {
        "rate": 0.0,
        "disposal": "Consider composting suitable food and garden waste.",
    },
    "food": {
        "rate": 0.0,
        "disposal": "Separate food scraps for composting where available.",
    },
    "electronic": {
        "rate": 20.0,
        "disposal": "Use an authorised e-waste collection centre.",
    },
    "textile": {
        "rate": 5.0,
        "disposal": "Reuse, donate, or use a textile recycling service.",
    },
    "wood": {
        "rate": 5.0,
        "disposal": "Reuse untreated wood or check local collection rules.",
    },
    "rubber": {
        "rate": 5.0,
        "disposal": "Use a suitable tyre or rubber recycling service.",
    },
    "battery": {
        "rate": 0.0,
        "disposal": "Take batteries to an authorised collection point.",
    },
    "mixed": {
        "rate": 0.0,
        "disposal": "Separate materials before deciding how to dispose of them.",
    },
}

# Each description is a candidate label for the pretrained model.
CANDIDATE_LABELS = [
    ("plastic", "a photograph of plastic waste, plastic bottles or plastic packaging"),
    ("paper", "a photograph of waste paper, newspapers or sheets of paper"),
    ("cardboard", "a photograph of discarded cardboard boxes"),
    ("glass", "a photograph of glass bottles or broken glass waste"),
    ("metal", "a photograph of scrap metal waste"),
    ("aluminium", "a photograph of aluminium cans or aluminium scrap"),
    ("steel", "a photograph of steel scrap or steel objects"),
    ("organic", "a photograph of garden waste, leaves or organic waste"),
    ("food", "a photograph of leftover food waste"),
    ("electronic", "a photograph of discarded electronic waste or circuit boards"),
    ("textile", "a photograph of discarded clothes or textile waste"),
    ("wood", "a photograph of discarded wood waste"),
    ("rubber", "a photograph of rubber waste or used tyres"),
    ("battery", "a photograph of discarded batteries"),
    ("mixed", "a photograph of mixed household garbage containing different materials"),
]

# The model is loaded once and reused for subsequent requests.
classifier = None


def get_classifier():
    global classifier

    if classifier is None:
        classifier = pipeline(
            task="zero-shot-image-classification",
            model=MODEL_NAME,
        )

    return classifier


def parse_custom_rates(raw_rates: str) -> dict:
    if not raw_rates:
        return {}

    try:
        data = json.loads(raw_rates)
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(
            status_code=400,
            detail="custom_rates must contain valid JSON.",
        )

    if not isinstance(data, dict):
        raise HTTPException(
            status_code=400,
            detail="custom_rates must be a JSON object.",
        )

    cleaned = {}

    for name, value in data.items():
        key = str(name).strip().lower()

        if key not in MATERIALS:
            continue

        try:
            rate = float(value)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid rate for {key}.",
            )

        if rate < 0:
            raise HTTPException(
                status_code=400,
                detail=f"Rate for {key} cannot be negative.",
            )

        cleaned[key] = rate

    return cleaned


# --------------------------------------------------
# Pages and health check
# --------------------------------------------------

@app.get("/")
def home():
    index_file = WEB_DIR / "index.html"

    if not index_file.exists():
        raise HTTPException(
            status_code=404,
            detail="web/index.html was not found.",
        )

    return FileResponse(index_file)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "application": "WasteWorth AI",
        "inference": "local_pretrained_model",
        "model": MODEL_NAME,
        "api_required": False,
        "model_loaded": classifier is not None,
    }


@app.get("/api/materials")
def get_materials():
    return {
        name: {
            "rate": details["rate"],
            "disposal": details["disposal"],
        }
        for name, details in MATERIALS.items()
    }


# --------------------------------------------------
# Image analysis
# --------------------------------------------------

@app.post("/api/analyze")
async def analyze_waste(
    image: UploadFile = File(...),
    model: str = Form(default=""),
    handling_cost: float = Form(default=0.0),
    custom_rates: str = Form(default=""),
):
    # The model is fixed in this application. A form field named
    # "model" is accepted for compatibility with the existing frontend.
    if handling_cost < 0:
        raise HTTPException(
            status_code=400,
            detail="Handling cost cannot be negative.",
        )

    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image is too large. Maximum size is 10 MB.",
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as original:
            picture = original.convert("RGB")
            picture.thumbnail((1024, 1024))
            # Copy the image so it remains usable after the file closes.
            picture = picture.copy()

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be read as an image.",
        )

    rates_override = parse_custom_rates(custom_rates)

    try:
        model_pipeline = get_classifier()

        predictions = model_pipeline(
            picture,
            candidate_labels=[
                description for _, description in CANDIDATE_LABELS
            ],
        )

    except Exception as exc:
        print(f"Local model error: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail=(
                "The local pretrained model could not run. "
                "Check your internet connection for the initial model download "
                "and verify that torch and transformers are installed."
            ),
        )

    # Translate the model's descriptive labels into material categories.
    label_to_material = {
        description: material
        for material, description in CANDIDATE_LABELS
    }

    ranked = []

    for prediction in predictions:
        description = prediction["label"]
        material = label_to_material.get(description, "mixed")

        ranked.append({
            "material": material,
            "label": description,
            "confidence": round(float(prediction["score"]), 4),
        })

    if not ranked:
        raise HTTPException(
            status_code=500,
            detail="The model did not return any predictions.",
        )

    best = ranked[0]
    category = best["material"]
    details = MATERIALS[category]
    rate = rates_override.get(category, details["rate"])

    # No weight is inferred from a photograph. Report the per-kg value
    # and calculate a total only when the user provides a weight.
    return {
        "success": True,
        "analysis_type": "local_pretrained_image_classification",
        "model": MODEL_NAME,
        "api_required": False,
        "filename": image.filename,
        "material": category,
        "category": category.title(),
        "confidence": best["confidence"],
        "confidence_percent": round(best["confidence"] * 100, 2),
        "rate_per_kg": round(rate, 2),
        "currency": "INR",
        "estimated_value_per_kg": round(rate, 2),
        "handling_cost": round(handling_cost, 2),
        "estimated_net_per_kg": round(rate - handling_cost, 2),
        "weight_kg": None,
        "estimated_total_value": None,
        "disposal_recommendation": details["disposal"],
        "top_predictions": ranked[:5],
        "all_material_rates": {
            name: rates_override.get(name, info["rate"])
            for name, info in MATERIALS.items()
        },
        "notice": (
            "Predictions are generated by a general-purpose pretrained model. "
            "They may be inaccurate for waste-specific images. Prices are "
            "illustrative, not live market rates. Weight cannot be estimated "
            "reliably from an image alone."
        ),
    }


# Serve static files only if the directory exists.
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")
