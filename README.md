Week 4 Assignment — SVMs for Image Classification

Dataset Source:
The dataset used is the Chest X-Ray Images (Pneumonia) dataset, available publicly on Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
A balanced subset of 200 images was selected (100 normal, 100 pneumonia) to ensure reproducibility and manage compute time.

Environment Setup:
1. Clone this repository: git clone (https://github.com/aadenaCSP/cst600-week04-svm-images-aadenaCSP)
2. Create and activate a virtual environment:
   python -m venv .venv
   source .venv/bin/activate  (Mac/Linux)
   .venv\Scripts\activate   (Windows)
3. Install dependencies:
   pip install -r requirements.txt

Run Instructions:
To train and evaluate the model on the subset:
python -m src.main --data_dir data/raw --img_size 128 128 --grayscale --probabilities --test_size 0.2 --random_state 42

Optional speed-up for quick tests:
python -m src.main --data_dir data/raw --img_size 128 128 --grayscale --probabilities --test_size 0.2 --random_state 42 --cv_splits 3 --max_per_class 100

Outputs:
- outputs/cv_results.csv : GridSearchCV results
- outputs/test_report.txt : Accuracy, precision, recall, F1 metrics
- figures/confusion_matrix.png : Confusion matrix of predictions
- figures/roc_curve.png : ROC curve with AUC score

Summary of Decisions and Results:
- Feature extraction used grayscale conversion, resizing to 128x128, and HOG descriptors.
- Models tested: SVM with linear, polynomial, and RBF kernels.
- Best model: Linear kernel with C=0.1.
- Test results on held-out data:
  Accuracy = 95%
  Precision = 0.95
  Recall = 0.95
  F1-score = 0.95
  ROC-AUC = 0.995
- The linear kernel generalized best, outperforming more complex kernels.
- Limitations include small dataset size (200 images) and reliance on HOG features instead of deep learning.

These results demonstrate the value of careful preprocessing and simple, interpretable models in healthcare proxy tasks.
