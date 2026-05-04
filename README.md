# bank_statements_agent

This project is an AI assistant for processing bank statements and accounting entries in a fund accounting workflow.  
It helps operations and finance teams normalize raw statement data, classify transactions, and prepare posting-ready outputs faster.  
The assistant combines rule-based processing with lightweight AI retrieval to support day-to-day accounting review.

## Key Features

- RAG-lite similarity search for transaction matching and classification support
- FX recalculation for multi-currency accounting scenarios
- Approval workflow for reviewed and validated posting decisions
- Excel export for downstream accounting and reporting processes

## Tech Stack

- Python
- Streamlit
- SQLite
- pandas
- scikit-learn
- Docker
- OpenAI API

## Run Locally (Simple)

1. Clone the repository and open the project folder.
2. Create and activate a virtual environment.
3. Install dependencies:
   - `pip install -r requirements.txt`
4. Set your OpenAI API key:
   - PowerShell: `$env:OPENAI_API_KEY="your_api_key"`
5. Start the app:
   - `streamlit run app.py`

Then open the local URL shown by Streamlit in your browser.

## Run with Docker (Simple)

1. Build the image:
   - `docker build -t bank-statements-agent .`
2. Run the container:
   - `docker run --rm -p 8501:8501 -e OPENAI_API_KEY=your_api_key bank-statements-agent`
3. Open:
   - [http://localhost:8501](http://localhost:8501)

### Docker with `.env` (recommended)

1. Create a `.env` file in the project root:
   - `OPENAI_API_KEY=your_api_key`
2. Run the container with env file:
   - `docker run --rm -p 8501:8501 --env-file .env bank-statements-agent`