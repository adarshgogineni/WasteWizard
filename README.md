# WasteWizard

## Introduction
WasteWizard is an innovative application designed to enhance recycling efforts by utilizing machine learning. This app analyzes photos of recycling symbols on items to determine their recyclability, promoting better waste management and environmental consciousness.

## Features
- **Image Recognition:** Users can upload pictures of recycling symbols, and the app will use machine learning to analyze and recognize the symbol.
- **Recyclability Status:** After analysis, the app informs the user whether the item is recyclable.
- **Interactive UI:** A modern, user-friendly interface that makes recycling easier and more accessible.
- **Incentives:** Adds incentives for users to keep scanning more images, hoping to lead to more items recycled. 

## Technologies Used
- Machine Learning: TensorFlow, Keras
- Frontend Framework: Flask
- Backend: Python

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
What things you need to install the software and how to install them:

# Clone the repository
```bash
git clone https://github.com/adarshgogineni/WasteWizard.git
```
# Go into the repository
```bash
cd WasteWizard
```
# Create a virtual environment
```bash
python3 -m venv venv
```
# Activate the virtual environment
```bash
source venv/bin/activate
```
# Install dependencies
```bash
pip install -r requirements.txt
```

# Running the Model:
Run the jupyter notebook file in the model folder first. This will generate the machine learning model required for the classification. Place the model in the WasteWizard directory.

# On Windows
```bash
set FLASK_APP=app.py
flask run
```
# On Unix/Linux
```bash
export FLASK_APP=app.py
flask run
```
# Alternatively, you can use
```bash
python app.py
```
