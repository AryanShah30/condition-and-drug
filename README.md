# Patient Condition Classification and Drug Recommendation

[Condition and Drug](https://condition-and-drug.streamlit.app/)

## Overview

This project is a high-accuracy (98.6%) patient condition classification and drug recommendation system built on a dataset of over 161,000 entries. The system predicts medical conditions such as Birth Control, Depression, High Blood Pressure, and Diabetes Type 2 based on symptom descriptions and offers relevant drug recommendations.

## Features

- **Condition Prediction**: Accurately classifies patient conditions based on symptom descriptions
- **Drug Recommendations**: Suggests top-rated and most useful drugs for the predicted condition
- **Natural Language Processing**: Utilizes NLP techniques for symptom analysis
- **Named Entity Recognition**: Employs NER for extracting relevant medical information

## Technologies Used

- Python
- Streamlit
- Pandas
- NLTK (Natural Language Toolkit)
- BeautifulSoup
- Scikit-learn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/patient-condition-classification.git
   cd patient-condition-classification
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure necessary datasets and models are in place:
   - `drugsComTrain.csv` in the data directory
   - `patient_classification.sav` (prediction model) in the models directory
   - `tfidf_vectorizer_3.sav` (vectorizer) in the models directory

## Usage

Run the application:
```bash
streamlit run app.py
```

Navigate to the provided URL in your web browser, enter symptom descriptions, and click "Check" to receive condition predictions and drug recommendations.

## Model Performance

The system achieves a 98.6% accuracy in classifying patient conditions, demonstrating high reliability in medical condition prediction.

## Supported Conditions

- Birth Control
- Depression
- High Blood Pressure
- Diabetes Type 2

## Contributing
Contributions to enhance the analysis or extend the project are welcome. Please feel free to fork the repository and submit pull requests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For inquiries, suggestions, or feedback, please reach out to: [AryanShah30](https://github.com/AryanShah30)
