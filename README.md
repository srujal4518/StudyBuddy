# StudyBuddy AI

### Your Personal AI Learning Companion

---

## About the Project

StudyBuddy is an AI-powered study assistant built to help students learn smarter, not harder.
Instead of acting like a generic chatbot, StudyBuddy changes its behavior based on the study mode selected — so the AI responds differently when tutoring, quizzing, summarizing, or solving problems.

Students can upload notes, PDFs, images, audio files, and spreadsheets, and the system understands and works with all of them.

This project was developed as part of the **Edunet Foundation – SkillsBuild AI/ML Virtual Internship**, with a focus on real-world use of large language models, multimodal AI, and full-stack development.

---

## What Problem Does It Solve?

Most AI chatbots give the same type of response no matter what you’re studying.
StudyBuddy fixes that by adapting how the AI thinks based on how the student wants to study.

Whether you want:

* a teacher-style explanation
* exam-oriented summaries
* step-by-step math solutions
* or quiz-based practice

StudyBuddy adjusts automatically.

---

## File Upload Support

StudyBuddy can process:

* PDFs
* Images
* Audio recordings
* Excel files
* Text notes

Each file type is handled differently so the AI understands the content properly.

---

## How to Run the Project

### Requirements

* Python 3.10 or higher
* A free Groq API key (no credit card required)

---

### Clone the Repository

```bash
git clone https://github.com/srujal4518/StudyBuddy.git
cd StudyBuddy
```

---

### Install Dependencies

```bash
pip install -r Requirements.txt
```

**Note for Windows users:**
If `python-magic` causes issues, run:

```bash
pip install python-magic-bin
```

---

### Environment Setup

Create a `.env` file in the project root and add:

```
GROQ_API_KEY=your_groq_api_key_here
SECRET_KEY=any_random_string
```

---

### Run the Application

```bash
python App.py
```

Then open the following URL in your browser:

```
http://localhost:5000
```

---

## Alternative: Run Without Cloning (ZIP Method)

If you don’t want to clone the repository:

1. Click **Code → Download ZIP** on GitHub
2. Extract the ZIP file to any folder
3. Open a terminal in the extracted folder

Install dependencies:

```bash
pip install -r Requirements.txt
```

Set up the `.env` file as explained above, then run:

```bash
python App.py
```

Open:

```
http://localhost:5000
```

---

## Tech Stack

### Backend

* Python (Flask)
* Groq API for LLM inference
* Flask-Login and Flask-Bcrypt for authentication
* Whisper for audio transcription
* Pandas for spreadsheet handling

### Frontend

* HTML, CSS, and JavaScript (no framework)
* Marked.js for rendering markdown responses
* Font Awesome icons
* Google Fonts
* localStorage for session persistence

---

## Project Structure

```
StudyBuddy/
│
├── App.py
├── chatbot.html
├── Requirements.txt
├── .env.example
├── README.md
├── .gitignore
│
└── uploads/
```

---

## Security Measures

* API keys stored using environment variables
* No sensitive data committed to GitHub
* Uploaded files processed locally

---

## Author

Developed during the **Edunet Foundation – SkillsBuild AI/ML Virtual Internship**
as a practical project to explore AI-powered learning systems.

---

⭐ If you find this project useful, feel free to star the repository
