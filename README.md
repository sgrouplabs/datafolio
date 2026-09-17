# Data Science & Machine Learning Portfolio

I'm a graduate student finishing my master's indata science, and these 
are the projects I've built so far: a churn model with a live
dashboard, an image classifier, a text classifier, and a BI data model. I
chose each one because it mirrors a real business problem I either saw
first-hand or utilized in my course study.


## Projects

| # | Project | Tech | Headline metric |
|---|---------|------|-----------------|
| 1 | [Customer Retention & Churn Risk Intelligence](01_customer_retention_intelligence/) | Python, scikit-learn, Streamlit, Plotly | ROC-AUC 0.84 hold-out, recall 0.80 |
| 2 | [Plant Seedling Vision — CNN Species Classifier](02_plant_seedling_vision/) | TensorFlow/Keras, MobileNetV2 | 80.7% test accuracy, 12 classes |
| 3 | [NLP Sentiment Analysis — Bidirectional LSTM](03_nlp_sentiment_analysis/) | TensorFlow/Keras, word embeddings | 87.9% test accuracy on IMDb |
| 4 | [Aviation Logistics BI — Star Schema & DAX](04_aviation_logistics_bi/) | Python, pandas, Power BI DAX | 4-table star schema from 10k BTS flights |

## Live Demo

You can try my churn dashboard right in your browser — no installation
needed. I compiled it to WebAssembly with [stlite](https://github.com/whitphx/stlite)
so it runs entirely client-side:

**[📉 Customer Retention & Churn Risk Intelligence — Live Demo](https://sgrouplabs.github.io/datafolio/)**
