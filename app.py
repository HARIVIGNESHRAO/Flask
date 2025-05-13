from flask import Flask, request, jsonify
from deep_translator import GoogleTranslator
from transformers import pipeline
import os
from werkzeug.utils import secure_filename
from groq import Groq
import json
from flask_cors import CORS
import logging
from threading import Lock
import uuid

# Custom LoggerAdapter for request tracking
class RequestLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        request_id = self.extra.get('request_id', 'N/A')
        return f'[{request_id}] {msg}', kwargs

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = RequestLoggerAdapter(logging.getLogger(), {'request_id': 'N/A'})

app = Flask(__name__)
CORS(app)

# Lock for thread-safe model access
model_lock = Lock()

# Load Distil-Whisper model
try:
    model = pipeline("automatic-speech-recognition", model="distil-whisper/distil-small.en")
    logger.info("Distil-Whisper model 'distil-small.en' loaded successfully")
except Exception as e:
    logger.error("Failed to load Distil-Whisper model: %s", str(e))
    raise

translator = GoogleTranslator(source='auto', target='en')
UPLOAD_FOLDER = "Uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "webm"}

try:
    client = Groq(api_key="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")
except Exception as e:
    logger.error("Failed to initialize Groq client: %s", str(e))
    raise

SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "te": "Telugu",
    "hi": "Hindi"
}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_response(response):
    if response.startswith('```json'):
        return response[7:-3].strip()
    elif response.startswith('```'):
        return response[3:-3].strip()
    return response

def analyze_transcription(transcribed_text, question, language="en"):
    language_name = SUPPORTED_LANGUAGES.get(language, "English")
    prompt = f"""
    Return only a valid JSON object with the following structure, without any explanations or extra text:
    {{
      "Tones": ["List of detected tones"],
      "Emotions": ["List of inferred emotions"],
      "Reasons": "Explanation of how tones and words suggest these emotions",
      "Suggestions": ["List of practical advice based on emotions"],
      "Language": "{language_name}",
      "Question": "{question}"
    }}
    Analyze the given transcribed speech text in {language_name} for the question: "{question}".
    - Detect tones based on punctuation, phrasing, and structure (e.g., happy, sad, angry, calm, excited, anxious, sarcastic, neutral).
    - Identify key emotional words or phrases that carry emotional weight.
    - Infer emotions by combining tones and emotional words.
    Text: "{transcribed_text}"
    """
    try:
        logger.info("Sending analysis prompt to Groq API for question: %s", question)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an emotion analysis assistant."},
                {"role": "user", "content": prompt},
            ],
            model="llama-3.3-70b-versatile",
        )
        response = clean_response(chat_completion.choices[0].message.content)
        logger.debug("Raw Groq API response: %s", response)
        try:
            json.loads(response)
            return response
        except json.JSONDecodeError as jde:
            logger.error("JSON decode error: %s, response: %s", str(jde), response)
            return json.dumps({
                "Tones": [],
                "Emotions": [],
                "Reasons": "Invalid JSON from API",
                "Suggestions": [],
                "Language": language_name,
                "Question": question,
                "error": f"JSON decode error: {str(jde)}"
            })
    except Exception as e:
        logger.error("Groq API error: %s", str(e))
        return json.dumps({
            "error": f"Groq API error: {str(e)}",
            "Language": language_name,
            "Question": question
        })

@app.route("/transcribe_audio", methods=["POST"])
def transcribe_audio():
    request_id = str(uuid.uuid4())
    app.logger = RequestLoggerAdapter(logging.getLogger(), {"request_id": request_id})
    app.logger.info("Received request to /transcribe_audio")

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    language = request.form.get("language", "en")

    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid or no selected file"}), 400

    if language != "en":  # Distil-Whisper is English-only
        return jsonify({"error": "Distil-Whisper only supports English transcription"}), 400

    filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    file.save(filepath)
    app.logger.info("Processing file: %s, size: %s bytes", filepath, os.path.getsize(filepath))

    try:
        app.logger.info("Transcribing audio...")
        with model_lock:
            transcription = model(filepath)
        transcribed_text = transcription["text"]
        app.logger.info("Transcription: %s", transcribed_text)

        result = {
            "text": transcribed_text,
            "language": language
        }
        app.logger.info("Returning result: %s", result)
        return jsonify(result)
    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        app.logger.error(error_msg)
        return jsonify({"error": error_msg}), 500
    finally:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                app.logger.info("Deleted file: %s", filepath)
            except Exception as e:
                app.logger.error("Failed to delete %s: %s", filepath, str(e))

@app.route("/analyze_audio", methods=["POST"])
def analyze_audio():
    request_id = str(uuid.uuid4())
    app.logger = RequestLoggerAdapter(logging.getLogger(), {"request_id": request_id})
    app.logger.info("Received request to /analyze_audio")

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    language = request.form.get("language", "en")
    question = request.form.get("question", "General conversation")

    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid or no selected file"}), 400

    if language != "en":
        return jsonify({"error": "Distil-Whisper only supports English transcription"}), 400

    filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    file.save(filepath)
    app.logger.info("Processing file: %s, size: %s bytes", filepath, os.path.getsize(filepath))

    try:
        app.logger.info("Transcribing audio...")
        with model_lock:
            transcription = model(filepath)
        transcribed_text = transcription["text"]
        app.logger.info("Transcription: %s", transcribed_text)

        app.logger.info("Analyzing transcription...")
        api_output = analyze_transcription(transcribed_text, question, language)
        app.logger.info("API Output: %s", api_output)

        try:
            analysis = json.loads(api_output)
            result = {
                "transcription": transcribed_text,
                "analysis": analysis,
                "language": language,
                "question": question
            }
        except json.JSONDecodeError as e:
            result = {
                "transcription": transcribed_text,
                "analysis_raw": api_output,
                "error": f"Failed to parse API output as JSON: {str(e)}",
                "language": language,
                "question": question
            }

        app.logger.info("Returning result: %s", result)
        return jsonify(result)
    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        app.logger.error(error_msg)
        return jsonify({"error": error_msg}), 500
    finally:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                app.logger.info("Deleted file: %s", filepath)
            except Exception as e:
                app.logger.error("Failed to delete %s: %s", filepath, str(e))

@app.route("/translate", methods=["POST"])
def translate_text():
    request_id = str(uuid.uuid4())
    app.logger = RequestLoggerAdapter(logging.getLogger(), {"request_id": request_id})
    try:
        data = request.get_json()
        text = data.get('text')
        target_language = data.get('targetLanguage', data.get('target_language', 'en'))

        if not text:
            app.logger.error("No text provided")
            return jsonify({'error': 'No text provided'}), 400

        if target_language not in SUPPORTED_LANGUAGES:
            app.logger.error("Unsupported language code: %s", target_language)
            return jsonify({'error': f"Unsupported language code: {target_language}"}), 400

        app.logger.info("Translating: %s to %s", text, target_language)
        translator = GoogleTranslator(source='auto', target=target_language)
        translated = translator.translate(text)
        app.logger.info("Translated text: %s", translated)
        return jsonify({'translatedText': translated}), 200
    except Exception as e:
        error_msg = f"Translation error: {str(e)}"
        app.logger.error(error_msg)
        return jsonify({'error': f"Translation failed: {str(e)}"}), 500

if __name__ == "__main__":
    try:
        logger.info("Starting Flask application")
        app.run(debug=False, host="0.0.0.0", port=5000)
    except Exception as e:
        error_msg = f"Failed to start Flask app: {str(e)}"
        logger.error(error_msg)
        raise
