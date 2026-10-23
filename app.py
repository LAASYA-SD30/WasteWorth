
import streamlit as st
from PIL import Image, UnidentifiedImageError
from transformers import pipeline
from pathlib import Path

st.set_page_config(
    page_title="WasteWorth | AI Waste Value",
    page_icon="♻️",
    layout="wide",
)

MATERIALS = {
    "plastic": (15.0, "Sort by plastic type and check local recycling rules."),
    "paper": (12.0, "Keep clean and dry for paper recycling."),
    "cardboard": (10.0, "Flatten boxes and keep them dry."),
    "glass": (3.0, "Separate carefully and follow local collection rules."),
    "metal": (35.0, "Separate metal for scrap collection."),
    "aluminium": (100.0, "Collect clean aluminium separately."),
    "steel": (25.0, "Separate steel for scrap recycling."),
    "organic": (0.0, "Consider composting suitable organic waste."),
    "food": (0.0, "Separate food scraps for composting."),
    "electronic": (20.0, "Use an authorised e-waste collection centre."),
    "textile": (5.0, "Reuse, donate, or recycle textiles."),
    "wood": (5.0, "Reuse untreated wood where possible."),
    "rubber": (5.0, "Use a suitable rubber recycling service."),
    "battery": (0.0, "Take batteries to an authorised collection point."),
    "mixed": (0.0, "Separate materials before disposal."),
}

LABELS = {
    "plastic": "plastic waste, bottles or packaging",
    "paper": "waste paper, newspapers or sheets of paper",
    "cardboard": "discarded cardboard boxes",
    "glass": "glass bottles or broken glass",
    "metal": "scrap metal waste",
    "aluminium": "aluminium cans or aluminium scrap",
    "steel": "steel scrap or steel objects",
    "organic": "garden leaves or organic waste",
    "food": "leftover food waste",
    "electronic": "electronic waste or circuit boards",
    "textile": "discarded clothes or textile waste",
    "wood": "discarded wood waste",
    "rubber": "rubber waste or used tyres",
    "battery": "discarded batteries",
    "mixed": "mixed household garbage",
}

@st.cache_resource(show_spinner="Loading AI model for the first time...")
def load_classifier():
    return pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32",
        device=-1,
    )

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(140deg, #fff9f2, #f0f7ef);
    }
    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
    }
    .hero {
        padding: 2rem;
        border-radius: 22px;
        background: linear-gradient(120deg, #173f35, #36785a);
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        color: white;
        font-size: 3rem;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #e4f2e9;
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>♻️ WasteWorth</h1>
        <p>See the value in waste.</p>
        Turn a waste image into an AI-powered material estimate.
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Upload a clear image of one waste item. The AI prediction and "
    "material prices are estimates, not guaranteed valuations."
)

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("📸 Upload your waste")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    handling_cost = st.number_input(
        "Handling cost (₹ per kg)",
        min_value=0.0,
        value=0.0,
        step=1.0,
    )

    st.caption("Optional: adjust the indicative price for each material.")

    with st.expander("Customize material prices"):
        custom_rates = {}
        for material, (default_rate, _) in MATERIALS.items():
            custom_rates[material] = st.number_input(
                f"{material.title()} (₹/kg)",
                min_value=0.0,
                value=float(default_rate),
                step=1.0,
                key=f"rate_{material}",
            )

    analyse = st.button(
        "✨ Analyse Waste",
        type="primary",
        use_container_width=True,
    )

with right:
    st.subheader("🔍 Image preview")
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            image.thumbnail((1024, 1024))
            st.image(image, use_container_width=True)
        except (UnidentifiedImageError, OSError):
            st.error("This image could not be opened.")
            image = None
    else:
        image = None
        st.markdown(
            """
            Upload an image to preview it here.

            **Tips for better results**
            - Use good lighting.
            - Keep the waste item clearly visible.
            - Avoid cluttered backgrounds.
            """
        )

if analyse:
    if image is None:
        st.warning("Please upload a valid image first.")
        st.stop()

    try:
        with st.spinner("AI is analysing your image..."):
            classifier = load_classifier()
            descriptions = [
                f"a photograph of {description}"
                for description in LABELS.values()
            ]
            predictions = classifier(
                image,
                candidate_labels=descriptions,
            )

        description_to_material = {
            f"a photograph of {description}": material
            for material, description in LABELS.items()
        }

        ranked = [
            {
                "material": description_to_material.get(
                    prediction["label"], "mixed"
                ),
                "confidence": float(prediction["score"]),
            }
            for prediction in predictions
        ]

        if not ranked:
            st.error("The model returned no predictions.")
            st.stop()

        best = ranked[0]
        material = best["material"]
        rate = custom_rates[material]
        net = rate - handling_cost

        st.divider()
        st.header("📊 Analysis results")

        a, b, c = st.columns(3)
        a.metric("Detected material", material.title())
        b.metric("Model confidence", f"{best['confidence'] * 100:.1f}%")
        c.metric("Estimated value", f"₹{rate:.2f}/kg")

        st.subheader("💰 Value breakdown")
        st.write(f"Indicative material price: ₹{rate:.2f} per kg")
        st.write(f"Handling cost: ₹{handling_cost:.2f} per kg")
        st.metric("Estimated net value", f"₹{net:.2f}/kg")

        st.subheader("♻️ Recommended disposal")
        st.success(MATERIALS[material][1])

        st.subheader("Other possible materials")
        for item in ranked[:5]:
            st.write(
                f"**{item['material'].title()}** — "
                f"{item['confidence'] * 100:.1f}%"
            )
            st.progress(min(max(item["confidence"], 0.0), 1.0))

        st.download_button(
            "Download analysis",
            data=__import__("json").dumps(
                {
                    "material": material,
                    "confidence_percent": round(
                        best["confidence"] * 100, 2
                    ),
                    "rate_per_kg_inr": rate,
                    "handling_cost_per_kg_inr": handling_cost,
                    "net_value_per_kg_inr": net,
                    "disposal_recommendation": MATERIALS[material][1],
                },
                indent=2,
            ),
            file_name="wasteworth_analysis.json",
            mime="application/json",
        )

        st.caption(
            "Prices are illustrative, not live market rates. "
            "Confidence is a model score, not a guarantee of accuracy. "
            "The image alone cannot determine the item's weight."
        )

    except Exception as exc:
        st.error(
            "Analysis failed. Check the app logs for the technical details."
        )
        st.code(f"{type(exc).__name__}: {exc}")
