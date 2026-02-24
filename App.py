from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import base64
from datetime import datetime
import json
import pandas as pd
from groq import Groq

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

app = Flask(__name__)
CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
}})

# ── Config ──────────────────────────────────────────────────────────
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB
app.config['ALLOWED_EXTENSIONS'] = {
    'png', 'jpg', 'jpeg', 'gif', 'webp',
    'pdf', 'txt', 'doc', 'docx',
    'mp3', 'wav', 'mp4', 'avi',
    'xlsx', 'xls', 'csv'
}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', 'your-groq-api-key-here')
client = Groq(api_key=GROQ_API_KEY)

# ── State ────────────────────────────────────────────────────────────
conversation_history: dict = {}
current_dataframe: pd.DataFrame | None = None
current_file_path: str | None = None
current_file_name: str | None = None


# ── Study Mode Personas ──────────────────────────────────────────────
STUDY_MODES = {
    "tutor": {
        "name": "Tutor Mode", "icon": "🎓",
        "system": """You are StudyBuddy, an expert AI tutor. Help students truly *understand* — not just get answers.

Rules:
- Break complex topics into digestible steps with analogies and mnemonics
- Use the Socratic method — ask guiding questions to check understanding
- Gently correct mistakes, explain *why* the correct answer is right
- Adapt vocabulary to the student's apparent level
- After explaining, always offer: "Does that make sense? Want me to quiz you on this?"
- Use emojis occasionally to keep energy up 📚✨
- Format responses with markdown headers and bullets for clarity"""
    },
    "quiz": {
        "name": "Quiz Mode", "icon": "🧪",
        "system": """You are a Quiz Master. Generate quizzes and test students interactively.

Rules:
- Generate 5–10 questions per topic by default
- Mix types: MCQ (4 options A–D), True/False, Fill-in-the-blank, Short answer
- After each student answer, give detailed feedback — right or wrong and WHY
- Track and announce score at the end of a quiz session
- Gradually increase difficulty
- Always encourage the student regardless of score

MCQ format:
**Q: [Question]**
A) Option 1  B) Option 2  C) Option 3  D) Option 4"""
    },
    "summarizer": {
        "name": "Summarizer", "icon": "📝",
        "system": """You are a Study Summarizer. Condense and organise study material perfectly.

Rules:
- Bullet the most important points; **bold** key terms
- Produce mind-map style outlines when asked
- Generate flashcard-ready Q&A pairs from notes
- List key vocabulary with definitions
- Flag what to prioritise for exams
- Always use markdown headers, bullets, and tables for scannability"""
    },
    "essay": {
        "name": "Essay Coach", "icon": "✍️",
        "system": """You are an Essay Writing Coach for students.

Rules:
- Help brainstorm and outline essays before writing
- Give feedback on structure, argument clarity, transitions, and conclusions
- Suggest improvements — don't just rewrite for them; explain *why* each change helps
- Ask for subject, grade level, and word count before diving in
- Provide example sentences to illustrate techniques
- Be honest but kind — always end feedback with one genuine strength"""
    },
    "math": {
        "name": "Math Solver", "icon": "🔢",
        "system": """You are MathBuddy, a patient step-by-step math tutor.

Rules:
- ALWAYS show every step — never just give the final answer
- Explain the reasoning at each step in plain English
- Point out common mistakes students make on similar problems
- Wrap equations in backticks for clarity: `3x² - 5x + 2 = 0`
- If a student is stuck, give a hint first rather than the full solution
- Relate concepts to real-life examples when possible
- After solving, ask: "Want a similar practice problem to try yourself?" """
    },
    "language": {
        "name": "Language Tutor", "icon": "🌍",
        "system": """You are a friendly Language Learning tutor supporting all languages.

Rules:
- Gently correct mistakes, showing the correct form and the grammar rule behind it
- Provide 2–3 example sentences for every new vocabulary word
- Conduct conversation practice in the target language when asked
- Teach idioms and native-speaker phrases, not just textbook language
- Use the student's native language for explanations when they're confused
- Celebrate progress — even small wins deserve recognition!"""
    },
    "data": {
        "name": "Data Analyst", "icon": "📊",
        "system": """You are a data analysis expert. You receive the COMPLETE dataset in each message.

Critical rules:
1. Always use the ACTUAL data provided — never guess or approximate
2. Show ALL rows when asked for complete lists
3. Present tabular results as markdown tables
4. After answering the question, suggest 1–2 additional insights the student might find interesting
5. Explain what the numbers *mean*, not just what they are

When data is provided, structure your analysis as:
- **Answer** — direct response to the question
- **Key Finding** — the most interesting thing you noticed
- **Suggestion** — one follow-up question worth asking"""
    }
}


# ── File Helpers ─────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def get_file_type(file_path: str) -> str:
    if MAGIC_AVAILABLE:
        try:
            return magic.Magic(mime=True).from_file(file_path)
        except Exception:
            pass
    ext = file_path.rsplit('.', 1)[-1].lower()
    return {
        'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
        'gif': 'image/gif', 'webp': 'image/webp', 'pdf': 'application/pdf',
        'txt': 'text/plain', 'mp3': 'audio/mpeg', 'wav': 'audio/wav',
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'xls': 'application/vnd.ms-excel', 'csv': 'text/csv',
    }.get(ext, 'application/octet-stream')


def encode_image(image_path: str) -> str:
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


# ── File Processors ──────────────────────────────────────────────────
def process_image(file_path: str) -> str:
    try:
        b64 = encode_image(file_path)
        mime = get_file_type(file_path)
        resp = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": (
                    "Describe this image in detail. "
                    "If it contains text, diagrams, equations, or study material, "
                    "extract and explain all content clearly and thoroughly."
                )},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            ]}],
            temperature=0.5, max_tokens=1200
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ Image processing error: {e}"


def process_audio(file_path: str) -> str:
    try:
        with open(file_path, 'rb') as f:
            t = client.audio.transcriptions.create(
                file=(os.path.basename(file_path), f.read()),
                model="whisper-large-v3",
                response_format="json",
                language="en",
                temperature=0.0
            )
        return t.text
    except Exception as e:
        return f"⚠️ Audio transcription error: {e}"


def process_text_file(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        if len(content) > 12000:
            return content[:12000] + "\n\n[... file truncated at 12 000 characters ...]"
        return content
    except Exception as e:
        return f"⚠️ Text read error: {e}"


def process_pdf(file_path: str) -> str:
    try:
        import PyPDF2
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages[:20]:
                text += page.extract_text() or ""
        if len(text) > 12000:
            text = text[:12000] + "\n\n[... PDF truncated at 12 000 characters ...]"
        return text or "No extractable text found in PDF."
    except ImportError:
        return "⚠️ PyPDF2 not installed. Run: pip install PyPDF2"
    except Exception as e:
        return f"⚠️ PDF error: {e}"


def process_excel(file_path: str) -> str:
    """Load a spreadsheet into the global dataframe and return a summary."""
    global current_dataframe, current_file_path, current_file_name
    try:
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path, sheet_name=0)
        current_dataframe = df
        current_file_path = file_path
        current_file_name = os.path.basename(file_path)

        rows, cols = df.shape
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(exclude='number').columns.tolist()

        # Build a richer summary
        stat_lines = []
        for col in numeric_cols[:5]:          # cap at 5 columns
            stat_lines.append(
                f"  - **{col}**: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}"
            )

        summary = f"""✅ **Spreadsheet loaded: `{os.path.basename(file_path)}`**

📋 **Shape:** {rows} rows × {cols} columns
📌 **Columns:** {', '.join(f'`{c}`' for c in df.columns)}
🔢 **Numeric columns:** {', '.join(numeric_cols) or 'none'}
🔤 **Text columns:** {', '.join(cat_cols) or 'none'}

{"📊 **Quick stats:**" + chr(10) + chr(10).join(stat_lines) if stat_lines else ""}

📝 **First 5 rows:**
{df.head(5).to_markdown(index=False)}

💡 Switch to **Data Analyst** mode and ask anything about this data!"""
        return summary
    except ImportError:
        return "⚠️ Install pandas & openpyxl: `pip install pandas openpyxl`"
    except Exception as e:
        return f"⚠️ Spreadsheet error: {e}"


# ── Dataframe Operations ─────────────────────────────────────────────
def build_data_context(df: pd.DataFrame) -> str:
    """Inject the full dataset into a message so the LLM can answer accurately."""
    try:
        rows = len(df)
        # For very large frames, send stats + sample instead of full dump
        if rows > 300:
            context = (
                f"Dataset: {rows} rows × {len(df.columns)} columns\n"
                f"Columns: {df.columns.tolist()}\n\n"
                f"Statistical summary:\n{df.describe(include='all').to_string()}\n\n"
                f"First 50 rows:\n{df.head(50).to_string()}\n\n"
                f"Last 20 rows:\n{df.tail(20).to_string()}"
            )
        else:
            context = (
                f"Dataset: {rows} rows × {len(df.columns)} columns\n"
                f"Columns: {df.columns.tolist()}\n\n"
                f"Complete data:\n{df.to_string()}\n\n"
                f"Statistical summary:\n{df.describe(include='all').to_string()}"
            )
        return context
    except Exception as e:
        return f"Error building data context: {e}"


def modify_dataframe_with_ai(message: str, df: pd.DataFrame):
    """
    Ask the LLM to generate pandas code, then execute it safely.
    Returns (modified_df, success: bool, error_or_None).
    """
    code_prompt = f"""
DataFrame columns: {df.columns.tolist()}
DataFrame dtypes: {df.dtypes.to_dict()}
Sample (3 rows): {df.head(3).to_dict()}

User request: "{message}"

Write ONLY executable Python. The variable is called `df`.
Do NOT wrap in a function. Do NOT include imports.
Assign the result back to `df` when needed.

Examples:
- add column Grade → df['Grade'] = ''
- add column Pass/Fail based on Score >= 50 → df['Pass/Fail'] = df['Score'].apply(lambda x: 'Pass' if x >= 50 else 'Fail')
- delete column Age → df.drop(columns=['Age'], inplace=True)
- rename Name to Student Name → df.rename(columns={{'Name': 'Student Name'}}, inplace=True)
- sort by Score descending → df.sort_values('Score', ascending=False, inplace=True)

Code (Python only):"""

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Output ONLY raw Python code. No markdown fences, no comments, no explanations."},
                {"role": "user", "content": code_prompt}
            ],
            temperature=0.05, max_tokens=600
        )
        code = resp.choices[0].message.content.strip()
        code = code.replace('```python', '').replace('```', '').strip()

        # Execute in a namespace that includes df and pd
        namespace = {'df': df.copy(), 'pd': pd}
        exec(code, namespace)
        # exec might have reassigned df inside namespace
        result_df = namespace.get('df', df)

        # Validate: must still be a DataFrame
        if not isinstance(result_df, pd.DataFrame):
            raise ValueError("Execution did not return a DataFrame.")

        return result_df, True, None

    except Exception as e:
        return df, False, str(e)


# ── Conversation History ─────────────────────────────────────────────
def get_context(session_id: str, max_msgs: int = 18) -> list:
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    return list(conversation_history[session_id][-max_msgs:])


def add_msg(session_id: str, role: str, content: str):
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    conversation_history[session_id].append({"role": role, "content": content})


# ── Study Tools ──────────────────────────────────────────────────────
def generate_flashcards(topic: str, count: int = 10) -> dict:
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"""Generate {count} flashcards for: "{topic}"

Return ONLY valid JSON — no markdown, no preamble:
{{
  "topic": "{topic}",
  "flashcards": [
    {{"front": "Question or term", "back": "Answer or definition"}},
    ...
  ]
}}"""}],
        temperature=0.7, max_tokens=2500
    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.replace('```json', '').replace('```', '').strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON block inside the response
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {"error": "JSON parse failed", "raw": raw}


def generate_study_plan(subject: str, exam_date: str, hours_per_day: int, level: str = "intermediate") -> str:
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"""Create a detailed study plan:
Subject: {subject}
Exam/deadline: {exam_date}
Hours available per day: {hours_per_day}
Student level: {level}

Include:
- Day-by-day schedule with specific topics
- Recommended resources (textbooks, videos, practice sites)
- Active recall and spaced repetition checkpoints
- Buffer days for review
- Subject-specific study tips

Format with markdown headers and bullets."""}],
        temperature=0.7, max_tokens=2500
    )
    return resp.choices[0].message.content


def explain_concept(concept: str, level: str = "high school") -> str:
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"""Explain "{concept}" for a {level} student.

Structure your explanation as:
1. **Simple definition** (1–2 sentences, no jargon)
2. **How it works** (detailed but accessible explanation)
3. **Real-world example** (concrete, relatable)
4. **Common misconceptions** (what students often get wrong)
5. **Helpful analogy** (something familiar to a {level} student)
6. **Key points to remember** (bullet list, exam-ready)"""}],
        temperature=0.6, max_tokens=1800
    )
    return resp.choices[0].message.content


# ═══════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    for fname in ['index.html', 'chatbot.html']:
        if os.path.exists(fname):
            with open(fname, 'r', encoding='utf-8') as f:
                return render_template_string(f.read())
    return "<h1>StudyBuddy — place index.html in the same folder as app.py</h1>", 404


# ── Chat ─────────────────────────────────────────────────────────────
@app.route('/chat', methods=['POST'])
def chat():
    try:
        body = request.get_json(force=True)
        message = (body.get('message') or '').strip()
        session_id = body.get('session_id', 'default')
        mode = body.get('mode', 'tutor')

        if not message:
            return jsonify({'error': 'No message provided'}), 400

        mode_cfg = STUDY_MODES.get(mode, STUDY_MODES['tutor'])
        system_prompt = mode_cfg['system']

        messages = get_context(session_id)

        # Always keep system prompt fresh / correct for the current mode
        if messages and messages[0].get('role') == 'system':
            messages[0]['content'] = system_prompt
        else:
            messages.insert(0, {"role": "system", "content": system_prompt})

        # Data mode: enrich the message with the actual dataframe
        enhanced = message
        global current_dataframe
        if current_dataframe is not None:
            data_kw = [
                'show', 'list', 'find', 'calculate', 'how many', 'what', 'which',
                'who', 'average', 'mean', 'sum', 'count', 'total', 'minimum', 'maximum',
                'failed', 'passed', 'top', 'bottom', 'highest', 'lowest',
                'students', 'all', 'percentage', 'percent', 'rows', 'columns', 'data'
            ]
            if mode == 'data' or any(kw in message.lower() for kw in data_kw):
                ctx = build_data_context(current_dataframe)
                enhanced = f"User question: {message}\n\n--- DATASET ---\n{ctx}\n---\n\nAnswer using ONLY the data above."

        add_msg(session_id, "user", message)
        messages.append({"role": "user", "content": enhanced})

        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7, max_tokens=2048, top_p=0.9
        )
        answer = resp.choices[0].message.content

        # Check for dataframe modification intent
        mod_kw = [
            'add column', 'add a column', 'create column', 'insert column',
            'delete column', 'remove column', 'rename column', 'drop column',
            'sort by', 'filter rows', 'update column'
        ]
        file_modified = False
        if current_dataframe is not None and any(kw in message.lower() for kw in mod_kw):
            new_df, ok, err = modify_dataframe_with_ai(message, current_dataframe)
            if ok:
                current_dataframe = new_df
                file_modified = True
                preview = current_dataframe.head(10).to_markdown(index=False)
                answer = (
                    f"✅ **Modification applied!**\n\n{answer}\n\n"
                    f"📊 **Preview (first 10 rows):**\n\n{preview}\n\n"
                    f"💾 Click **Export** in the top bar to download the updated file."
                )
            else:
                answer += f"\n\n⚠️ Couldn't auto-apply the modification: `{err}`\nTry rephrasing or be more specific."

        add_msg(session_id, "assistant", answer)

        return jsonify({
            'response': answer,
            'mode': mode,
            'mode_name': mode_cfg['name'],
            'file_modified': file_modified,
            'has_dataframe': current_dataframe is not None,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"[/chat] {e}")
        return jsonify({'error': str(e)}), 500


# ── Upload ────────────────────────────────────────────────────────────
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file attached'}), 400

        file = request.files['file']
        message = request.form.get('message', '')
        session_id = request.form.get('session_id', 'default')
        mode = request.form.get('mode', 'tutor')

        if not file.filename or not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed. Supported: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'}), 400

        fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secure_filename(file.filename)}"
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(fpath)

        ftype = get_file_type(fpath)
        is_spreadsheet = (
            'spreadsheet' in ftype or
            'excel' in ftype or
            ftype == 'text/csv' or
            file.filename.lower().endswith(('.xlsx', '.xls', '.csv'))
        )

        if ftype.startswith('image/'):
            content = process_image(fpath)
            desc = f"[📷 Image uploaded: {file.filename}]\n\n{content}"
        elif ftype.startswith('audio/'):
            content = process_audio(fpath)
            desc = f"[🎵 Audio transcribed: {file.filename}]\n\nTranscription:\n{content}"
        elif ftype == 'text/plain':
            content = process_text_file(fpath)
            desc = f"[📄 Text file: {file.filename}]\n\n{content}"
        elif ftype == 'application/pdf':
            content = process_pdf(fpath)
            desc = f"[📑 PDF: {file.filename}]\n\n{content}"
        elif is_spreadsheet:
            content = process_excel(fpath)
            desc = f"[📊 Spreadsheet: {file.filename}]\n\n{content}"
            # Auto-hint: switch to data mode for spreadsheets
            if mode not in ('data',):
                desc += "\n\n💡 **Tip:** Switch to **Data Analyst** mode for the best analysis experience."
        else:
            desc = f"[📎 File uploaded: {file.filename} — type: {ftype}]"

        ctx_msg = f"{message}\n\n{desc}" if message else desc

        mode_cfg = STUDY_MODES.get(mode, STUDY_MODES['tutor'])
        messages = get_context(session_id)
        if messages and messages[0].get('role') == 'system':
            messages[0]['content'] = mode_cfg['system']
        else:
            messages.insert(0, {"role": "system", "content": mode_cfg['system']})

        add_msg(session_id, "user", ctx_msg)
        messages.append({"role": "user", "content": ctx_msg})

        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7, max_tokens=2048
        )
        answer = resp.choices[0].message.content
        add_msg(session_id, "assistant", answer)

        return jsonify({
            'response': answer,
            'file_processed': True,
            'file_type': ftype,
            'is_spreadsheet': is_spreadsheet,
            'has_dataframe': current_dataframe is not None,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"[/upload] {e}")
        return jsonify({'error': str(e)}), 500


# ── Flashcards ────────────────────────────────────────────────────────
@app.route('/flashcards', methods=['POST'])
def flashcards():
    try:
        body = request.get_json(force=True)
        topic = (body.get('topic') or '').strip()
        count = int(body.get('count', 10))
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        return jsonify(generate_flashcards(topic, count))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Study Plan ────────────────────────────────────────────────────────
@app.route('/study-plan', methods=['POST'])
def study_plan():
    try:
        body = request.get_json(force=True)
        subject = (body.get('subject') or '').strip()
        if not subject:
            return jsonify({'error': 'Subject is required'}), 400
        plan = generate_study_plan(
            subject=subject,
            exam_date=body.get('exam_date', 'in 2 weeks'),
            hours_per_day=int(body.get('hours_per_day', 2)),
            level=body.get('level', 'intermediate')
        )
        return jsonify({'plan': plan})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Concept Explainer ─────────────────────────────────────────────────
@app.route('/explain', methods=['POST'])
def explain():
    try:
        body = request.get_json(force=True)
        concept = (body.get('concept') or '').strip()
        level = body.get('level', 'high school')
        if not concept:
            return jsonify({'error': 'Concept is required'}), 400
        return jsonify({'explanation': explain_concept(concept, level)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Transcribe ────────────────────────────────────────────────────────
@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file'}), 400
        audio = request.files['audio']
        fname = f"voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        audio.save(fpath)
        text = process_audio(fpath)
        try:
            os.remove(fpath)
        except Exception:
            pass
        return jsonify({'transcription': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Export ────────────────────────────────────────────────────────────
@app.route('/export-excel', methods=['POST'])
def export_excel():
    try:
        global current_dataframe
        if current_dataframe is None:
            return jsonify({'error': 'No spreadsheet loaded. Upload one first.'}), 400

        body = request.get_json(force=True) if request.content_length else {}
        fmt = (body or {}).get('format', 'xlsx')
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        if fmt == 'csv':
            fname = f'export_{ts}.csv'
            fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            current_dataframe.to_csv(fpath, index=False)
        else:
            fname = f'export_{ts}.xlsx'
            fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            current_dataframe.to_excel(fpath, index=False, engine='openpyxl')

        return send_from_directory(app.config['UPLOAD_FOLDER'], fname, as_attachment=True)
    except Exception as e:
        print(f"[/export] {e}")
        return jsonify({'error': str(e)}), 500


# ── Dataframe Info ────────────────────────────────────────────────────
@app.route('/dataframe-info', methods=['GET'])
def dataframe_info():
    """Return metadata about the currently loaded dataframe."""
    if current_dataframe is None:
        return jsonify({'loaded': False})
    df = current_dataframe
    return jsonify({
        'loaded': True,
        'filename': current_file_name,
        'rows': len(df),
        'columns': df.columns.tolist(),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'preview': df.head(5).to_dict(orient='records')
    })


# ── Clear History ─────────────────────────────────────────────────────
@app.route('/clear-history', methods=['POST'])
def clear_history():
    try:
        body = request.get_json(force=True)
        sid = body.get('session_id', 'default')
        conversation_history[sid] = []
        return jsonify({'message': 'History cleared', 'session_id': sid})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Modes ─────────────────────────────────────────────────────────────
@app.route('/modes', methods=['GET'])
def get_modes():
    return jsonify({'modes': {k: {'name': v['name'], 'icon': v['icon']} for k, v in STUDY_MODES.items()}})


# ── Health / Ping ─────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
@app.route('/ping', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'groq_configured': GROQ_API_KEY not in ('', 'your-groq-api-key-here'),
        'dataframe_loaded': current_dataframe is not None,
        'dataframe_file': current_file_name,
        'sessions_active': len(conversation_history),
        'timestamp': datetime.now().isoformat()
    })


# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    pad = "=" * 62
    print(pad)
    print("  📚  StudyBuddy AI — Server Starting")
    print(pad)
    api_ok = GROQ_API_KEY not in ('', 'your-groq-api-key-here')
    print(f"  🔑  Groq API key : {'✅ configured' if api_ok else '❌ NOT SET — set GROQ_API_KEY env var'}")
    print(f"  📁  Upload folder: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
    print(f"  🔬  python-magic  : {'✅ available' if MAGIC_AVAILABLE else '⚠️  not found (extension-based detection)'}")
    print(pad)
    print("  Endpoints")
    for ep in [
        "GET  /           → serve frontend",
        "POST /chat       → chat with AI",
        "POST /upload     → process a file",
        "POST /flashcards → generate flashcards",
        "POST /study-plan → personalised study plan",
        "POST /explain    → deep concept explainer",
        "POST /transcribe → voice → text",
        "POST /export-excel → download spreadsheet",
        "GET  /dataframe-info → metadata of loaded sheet",
        "GET  /ping       → server health check",
    ]:
        print(f"  {ep}")
    print(pad)
    print("  🚀  http://localhost:5000")
    print(pad)
    app.run(debug=True, host='0.0.0.0', port=5000)