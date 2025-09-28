ProjectQuasar
ProjectQuasar is a machine learning pipeline for **stock prediction and portfolio forecasting**. It integrates **PyTorch Lightning**, **Temporal Fusion Transformers**, and **custom data pipelines** to analyze both historical stock data and real-time news sentiment.

Project Structure
```
ProjectQuasar/
│── data/                  # Raw & processed datasets
│── scrapers/              # Scripts for news & stock data scraping
│── phase2_predict.py      # Prediction scripts
│── phase2_train_model.py  # Training pipeline
│── run_pipeline.py        # Main entry point for running the pipeline
│── run_portfolio_forecast.py # Portfolio-level forecasting
│── update_tickers.py      # Update tickers list
│── rss_scraper.py         # RSS news scraper
│── intelligence_layer.py  # LLM-powered news analysis
│── lightning_logs/        # Training logs
│── checkpoints/           # Saved models

````
Features
- Stock price forecasting using **Temporal Fusion Transformers (TFT)**
- Real-time news scraping & **LLM-based sentiment analysis**
- Modular pipeline design with reusable data-prep functions
- Portfolio-level forecasts, not just single stock predictions
- Organized logging and checkpoint saving with PyTorch Lightning

Getting Started
1. Clone the repo
```
git clone https://github.com/your-username/ProjectQuasar.git
cd ProjectQuasar
````

2. Create a virtual environment
```
python3 -m venv quasar-env
source quasar-env/bin/activate   # On Windows: quasar-env\Scripts\activate
```

3. Install dependencies
```
pip install -r requirements.txt
```

4. Run the pipeline
```
python run_pipeline.py
```

Example Workflow
1. **Scrape data**
   * Stock data (historical OHLCV)
   * News headlines via RSS

2. **Prepare datasets**
   * Feature engineering with time indices, groups, and sentiment features

3. **Train model**
   ```
   python phase2_train_model.py
   ```
4. **Predict & visualize**
   ```
   python phase2_predict.py
   ```

Tech Stack
* **Python 3.10+**
* **PyTorch Lightning**
* **PyTorch Forecasting**
* **Pandas / NumPy**
* **tqdm**
* **News + RSS scraping**
* **LLM-based sentiment analysis**

## Next Steps
* Add Docker support for easy deployment
* Build interactive dashboard for forecasts
* Expand to crypto and commodities

## Author
Developed by **Soham Shetty**
*(MBA student + builder of data-driven tools)*

## License
This project is open-sourced – feel free to use and modify!
