# face-analysis-and-recognition
## Project Structure

The project directory structure is organized as follows:

```
ml_models_project/
├── emotions_model/
│   ├── data/
│   │   └── raw/
│   ├── src/
│   │   ├── data_preprocessing.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── models/
│   │   └── saved_model.h5
│   └── notebooks/
├── age_model/
│   ├── data/
│   │   └── raw/
│   ├── src/
│   │   ├── data_preprocessing.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── models/
│   │   └── saved_model.h5
│   └── notebooks/
├── face_recognition_model/
│   ├── data/
│   │   └── raw/
│   ├── src/
│   │   ├── data_preprocessing.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── models/
│   │   └── saved_model.h5
│   └── notebooks/
├── gender_model/
│   ├── data/
│   │   └── raw/
│   ├── src/
│   │   ├── data_preprocessing.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── models/
│   │   └── saved_model.h5
│   └── notebooks/
├── main.py
└── README.md
```

This structure provides a clear separation of models, data, source code, and notebooks for each model category. The `main.py` file serves as the entry point for the project, and the `README.md` file contains additional information about the project.
