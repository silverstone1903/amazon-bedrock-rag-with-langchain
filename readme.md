# Amazon Bedrock RAG with LangChain

A Python-based document processing system that leverages Retrieval Augmented Generation (RAG) using AWS Bedrock and Chroma Vector DB for efficient document search and retrieval.

<p style="text-align:center">
<img src="https://img.youtube.com/vi/_vdK5PgcNvc/maxresdefault.jpg" width="550">
</p>
<br>


Detailed blog post: [LangChain ile Amazon Bedrock RAG Kullanımı](https://silverstone1903.github.io/posts/2024/11/langchain-ile-amazon-bedrock/) (Turkish)

## 🚀 Features

- PDF document processing and chunking
- Vector embeddings using AWS Bedrock Titan
- Document storage in Chroma vector database
- Incremental document updates
- Unique document chunk identification
- Efficient text search and retrieval

## 🛠️ Technologies

- Python 3.9+
- LangChain
- AWS Bedrock
- ChromaDB
- Streamlit (for UI)

## 📋 Prerequisites

- AWS Account with Bedrock access
- AWS CLI configured with appropriate credentials
- Python 3.9 or higher

## 🔧 Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Install dependencies
poetry install

# To run the app using Poetry
poetry run streamlit run app.py
```

## ⚙️ Configuration

1. Set up AWS credentials in your profile
2. Update the following constants in `create_db.py` and `app.py`:
```python
aws_profile_name = "aws-profile-name"
region_name = "region-name"
emb_model_name = "amazon.titan-embed-text-v2:0"
txt_model_name = "amazon.titan-text-express-v1"

```

## 🚀 Usage

1. Place your PDF documents in the configured `DATA_PATH`
2. Run the document processing script:
```bash
python create_db.py
```
3. Start the Streamlit app:
```python
streamlit run app.py
```

### 💻 Streamlit App

<p style="text-align:center">
<img src="https://silverstone1903.github.io/images/ml_assistant_long.gif" width="900">
</p>
<br>

## 📁 Project Structure

```
.
├── create_db.py          # Main document processing script
├── app.py               # Streamlit web interface
├── requirements.txt     # Project dependencies
├── db/                  # ChromaDB storage
└── data/               # PDF documents directory
```

