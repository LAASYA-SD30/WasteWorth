
# Trashure

**See the value in waste.**

Trashure is an AI-powered waste classification and valuation
website. It uses a pretrained CLIP model to predict a likely
material category from an image and display an illustrative
estimated value per kilogram.

## Features

- Image upload and preview
- Pretrained AI image classification
- Material confidence estimates
- Estimated value per kilogram
- Optional handling cost
- Disposal and recycling suggestions
- Downloadable JSON analysis
- FastAPI backend and responsive frontend
- No paid AI API key required

## Important limitations

The model is a general-purpose image model, not a certified
waste-sorting system. Predictions may be incorrect.

Material rates are illustrative estimates, not live market prices.
Update the rates in main.py using verified local information.

The system cannot determine an item's weight from an image.

The model downloads during first use, so internet access is needed
for the initial download.

## Run locally on Windows

Python 3.11 is recommended.

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000.

## Deploy on Render

1. Push this project to GitHub.
2. Create a Render Web Service connected to the repository.
3. Build command: pip install -r requirements.txt
4. Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
5. Deploy and test /api/health.

The first startup may take longer because the model is downloaded.
A free service may not have enough memory for PyTorch and CLIP.

## GitHub

```bat
git init
git add .
git commit -m "Build Trashure AI waste classification app"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/trashure.git
git push -u origin main
```
