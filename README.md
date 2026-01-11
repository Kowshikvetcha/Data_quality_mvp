# AI-Assisted Data Cleaning MVP

This project is a demo-ready AI-assisted data cleaning tool.

Users can:
- Upload a CSV file
- View data quality issues
- Use a chat interface (like ChatGPT) to clean data
- Review cleaned data and updated quality metrics
- Apply or cancel each cleaning action safely

The AI **does not modify data automatically** — all actions require user confirmation.

---

## ✅ Prerequisites

Before running the project, make sure you have:

- **Python 3.9 or higher**
- **An OpenAI API key**
- Internet connection (for AI calls)

---

## 🚀 Step-by-Step Setup Instructions

### 1️⃣ Download the project

Clone the repository:

```bash
git clone https://github.com/Kowshikvetcha/Data_quality_mvp.git
cd Data_quality_mvp

```
### 2️⃣ Install Dependencies

Create and activate a virtual environment, then install the required packages:

```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
pip install -r requirements.txt
```

### 3️⃣ Configure Environment

Create a `.env` file in the root directory and add your OpenAI API key:

```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 4️⃣ Run the Application

Start the Pro version of the application:

```bash
streamlit run app_enhanced.py
```

`app_enhanced.py` is the comprehensive Pro version containing the AI chat, manual transformations, advanced visualizations, and data type overrides.






