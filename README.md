# Food Classifier with CNN & Nutrition Analysis

A deep learning project that classifies 7 food categories and provides nutritional information using PyTorch CNN trained on RTX 4050 GPU.

## Project Structure
```
cal_dnn/
├── train_food_classifier_pytorch.py    # Training script
├── web_app.py                          # Flask web application
├── test_prediction.py                  # CLI prediction tool
├── requirements_pytorch.txt            # Dependencies
├── templates/                          # HTML templates
│   └── index.html                      # Web UI
├── models/                             # Trained models
│   └── food_classifier_pytorch.pth     # PyTorch model weights
├── outputs/                            # Training results
│   ├── training_history.png            # Accuracy/Loss plots
│   ├── confusion_matrix.png            # Confusion matrix
│   └── prediction_*.png                # Sample predictions
├── docs/                               # Documentation
│   ├── README.md                       # Project overview
│   ├── TRAINING_SUMMARY.md             # Training details
│   └── NEXT_STEPS.md                   # Future improvements
└── gpu_env/                            # Virtual environment
```

## Features
- **7 Food Classes**: Chicken Curry, Chicken Wings, French Fries, Grilled Cheese Sandwich, Omelette, Pizza, Samosa
- **CNN Architecture**: 3 convolutional layers, 16.8M parameters
- **GPU Training**: CUDA-accelerated training on RTX 4050
- **Web Interface**: Beautiful Flask app with drag-and-drop upload
- **Nutrition Info**: Calories, protein, fat, fiber, carbs for each food

## Quick Start

### 1. Setup Environment
```bash
# Activate virtual environment
gpu_env\Scripts\activate

# Install dependencies (if needed)
pip install -r requirements_pytorch.txt
```

### 2. Run Web App
```bash
python web_app.py
```

### 3. Test from CLI
```bash
python test_prediction.py path/to/food_image.jpg
```

## Model Performance
- **Training Accuracy**: 94.09%
- **Validation Accuracy**: 44.86%
- **Training Time**: ~5-10 minutes on RTX 4050
- **Dataset Size**: 7,000 images (80-20 train-val split)

## Technical Details
- **Framework**: PyTorch 2.5.1 with CUDA 12.1
- **Architecture**: Custom CNN (Conv2D → MaxPool → Dense)
- **Input Size**: 128×128 RGB images
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy
- **Data Augmentation**: Rotation, flip, color jitter

## Usage Examples

### Web Interface
1. Upload food image
2. Click "Classify Food"
3. View prediction, confidence, and nutrition data

### Command Line
```bash
python test_prediction.py ../FOOD_DATA/pizza/image.jpg
```

This project demonstrates:
- Deep Learning fundamentals (CNN architecture)
- GPU-accelerated training
- Full-stack ML application (training + deployment)
- Real-world problem solving (food classification + nutrition)
- Understanding of overfitting and model evaluation

## Future Improvements
- Reduce overfitting with better regularization
- Add more food categories
- Deploy to cloud (Heroku/AWS)
- Mobile app integration
- Real-time calorie tracking

---
**Note**: Model trained from scratch (not pre-trained) as educational project.
