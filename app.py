from flask import Flask, request, jsonify, send_file
from deep_translator import GoogleTranslator
import os
from werkzeug.utils import secure_filename
from groq import Groq
import json
from flask_cors import CORS
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import librosa
import numpy as np
import traceback
import logging
from threading import Lock
import uuid
import boto3
import time
import requests

# Custom LoggerAdapter
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

# AWS clients
s3_client = boto3.client('s3')
transcribe_client = boto3.client('transcribe')
bucket_name = 'my-whisper-app-bucket'  # Replace with your S3 bucket name
UPLOAD_FOLDER = '/tmp'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'webm'}

# Initialize Groq client
try:
    client = Groq(api_key="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")
except Exception as e:
    logger.error("Failed to initialize Groq client: %s", str(e))
    raise

SUPPORTED_LANGUAGES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
    'zh': 'Chinese', 'te': 'Telugu', 'hi': 'Hindi'
}

translator = GoogleTranslator(source='auto', target='en')

# Locks for thread safety
file_locks = {}

def get_file_lock(filepath):
    if filepath not in file_locks:
        file_locks[filepath] = Lock()
    return file_locks[filepath]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_response(response):
    if response.startswith('```json'):
        return response[7:-3].strip()
    elif response.startswith('```'):
        return response[3:-3].strip()
    return response

def extract_acoustic_features(filepath):
    try:
        y, sr = librosa.load(filepath, sr=None)
        logger.info("Loaded audio file %s with sample rate %s", filepath, sr)

        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        rms = librosa.feature.rms(y=y)[0]
        energy_mean = np.mean(rms)
        pitch_values = pitches[pitches > 0]
        jitter = np.mean(np.abs(np.diff(pitch_values))) if len(pitch_values) > 1 else 0
        shimmer = np.mean(np.abs(np.diff(rms))) if len(rms) > 1 else 0
        silence_threshold = 0.01 * np.max(np.abs(y))
        pauses = librosa.effects.split(y, top_db=20)
        pause_count = len(pauses) - 1
        pause_duration = np.sum(np.diff(pauses, axis=1)) / sr if pause_count > 0 else 0

        features = {
            'pitch_mean': float(pitch_mean),
            'energy_mean': float(energy_mean),
            'jitter': float(jitter),
            'shimmer': float(shimmer),
            'silence_threshold': float(silence_threshold),
            'pause_count': int(pause_count),
            'pause_duration': float(pause_duration)
        }
        logger.info("Extracted acoustic features: %s", features)
        return features
    except Exception as e:
        logger.error("Failed to extract acoustic features for %s: %s", filepath, str(e))
        raise Exception(f"Failed to extract acoustic features: {str(e)}")

def analyze_transcription(transcribed_text, acoustic_features, question, language='en'):
    language_name = SUPPORTED_LANGUAGES.get(language, 'English')
    prompt = f"""
    Return only a valid JSON object with the following structure, without any explanations or extra text:
    {{
      "Tones": ["List of detected tones"],
      "Emotions": ["List of inferred emotions, including 'sarcasm' if text and acoustic features conflict"],
      "Reasons": "Explanation of how tones, words, and acoustic features suggest these emotions, noting any conflict between text and acoustic features",
      "Suggestions": ["List of practical advice based on emotions"],
      "Language": "{language_name}",
      "Question": "{question}",
      "AcousticAnalysis": {{
        "Pitch": "Interpretation of pitch_mean",
        "Energy": "Interpretation of energy_mean",
        "Jitter": "Interpretation of jitter",
        "Shimmer": "Interpretation of shimmer",
        "Pauses": "Interpretation of pause_count and pause_duration"
      }}
    }}
    Analyze the given transcribed speech text and acoustic features in {language_name} for the question: "{question}".
    - Detect tones based on punctuation, phrasing, structure, and acoustic features (e.g., happy, sad, angry, calm, excited, anxious, sarcastic, neutral).
    - Identify key emotional words or phrases that carry emotional weight.
    - Infer emotions by combining tones, emotional words, and acoustic features (pitch, energy, jitter, shimmer, pauses).
    - If text and acoustic features conflict, prioritize acoustic features, note the conflict in 'Reasons', and include 'sarcasm' in 'Emotions' if the conflict suggests it.
    Text: "{transcribed_text}"
    Acoustic Features: {json.dumps(acoustic_features)}
    """
    try:
        logger.info("Sending analysis prompt to Groq API for question: %s", question)
        chat_completion = client.chat.completions.create(
            messages=[
                {'role': 'system', 'content': 'You are an emotion analysis assistant.'},
                {'role': 'user', 'content': prompt},
            ],
            model='llama-3.3-70b-versatile',
        )
        response = clean_response(chat_completion.choices[0].message.content)
        logger.debug("Raw Groq API response: %s", response)
        try:
            json.loads(response)
            return response
        except json.JSONDecodeError as jde:
            logger.error("JSON decode error: %s, response: %s", str(jde), response)
            return json.dumps({
                'Tones': [],
                'Emotions': [],
                'Reasons': 'Invalid JSON from API',
                'Suggestions': [],
                'Language': language_name,
                'Question': question,
                'AcousticAnalysis': {},
                'error': f'JSON decode error: {str(jde)}'
            })
    except Exception as e:
        logger.error("Groq API error: %s", str(e))
        return json.dumps({
            'error': f'Groq API error: {str(e)}',
            'Language': language_name,
            'Question': question
        })

def combine_analyses(analyses, transcriptions):
    try:
        emotions = set()
        tones = set()
        suggestions = set()
        reasons = []

        for i, analysis in enumerate(analyses):
            emotions.update(analysis.get('Emotions', []))
            tones.update(analysis.get('Tones', []))
            suggestions.update(analysis.get('Suggestions', []))
            reasons.append(
                f"Response to '{transcriptions[i]['question']}': {analysis.get('Reasons', 'No reasons provided.')}"
            )

        language = transcriptions[0]['language'] if transcriptions else 'en'
        language_name = SUPPORTED_LANGUAGES.get(language, 'English')

        prompt = f"""
        Summarize the emotional analysis of {len(transcriptions)} responses in {language_name}.
        Combine the following emotions, tones, reasons, and suggestions into a concise paragraph.
        Emotions: {', '.join(emotions)}
        Tones: {', '.join(tones)}
        Reasons: {'; '.join(reasons)}
        Suggestions: {', '.join(suggestions)}
        Provide a holistic view of the user's emotional state across all responses, highlighting key patterns and offering overarching recommendations.
        """
        logger.info("Sending combined analysis prompt to Groq API")
        chat_completion = client.chat.completions.create(
            messages=[
                {'role': 'system', 'content': 'You are an emotion analysis assistant.'},
                {'role': 'user', 'content': prompt},
            ],
            model='llama-3.3-70b-versatile',
        )
        summary = chat_completion.choices[0].message.content.strip()
        logger.info("Combined analysis summary: %s", summary)
        return summary
    except Exception as e:
        logger.error("Error combining analyses: %s\n%s", str(e), traceback.format_exc())
        return f"Error combining analyses: {str(e)}"

def draw_border(canvas, doc):
    width, height = A4
    margin = 20
    canvas.setLineWidth(2)
    canvas.setStrokeColor(colors.darkblue)
    canvas.rect(margin, margin, width - 2 * margin, height - 2 * margin)

def generate_pdf(api_response, filename=None):
    data = json.loads(api_response)
    individual_analyses = data.get('individual_analyses', [])
    combined_analysis = data.get('combined_analysis', 'No combined analysis provided.')
    username = data.get('username', 'Unknown')

    if filename is None:
        filename = f"mental_health_report_{username}.pdf"

    doc = SimpleDocTemplate(filename, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Title'],
        alignment=TA_CENTER,
        fontSize=22,
        spaceAfter=12
    )
    normal_style = ParagraphStyle(
        'NormalStyle',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6
    )
    justified_style = ParagraphStyle(
        'JustifiedStyle',
        parent=styles['Normal'],
        alignment=TA_JUSTIFY,
        fontSize=10,
        spaceAfter=12
    )

    elements.append(Paragraph(f"Mental Health Analysis Report for {username}", title_style))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    elements.append(Spacer(1, 20))

    for idx, analysis in enumerate(individual_analyses, 1):
        elements.append(Paragraph(f"Analysis {idx}", styles['Heading2']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Inferred Emotions:", styles['Heading3']))
        emotions = analysis.get('Emotions', [])
        if emotions:
            for emotion in emotions:
                elements.append(Paragraph(f"• {emotion}", normal_style))
        else:
            elements.append(Paragraph("No emotions detected.", normal_style))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Detected Tones:", styles['Heading3']))
        tones = analysis.get('Tones', [])
        if tones:
            for tone in tones:
                elements.append(Paragraph(f"• {tone}", normal_style))
        else:
            elements.append(Paragraph("No tones detected.", normal_style))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Reasons Behind These Emotions:", styles['Heading3']))
        reasons = analysis.get('Reasons', 'No reasons provided.')
        elements.append(Paragraph(reasons, justified_style))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Emotional Support Suggestions:", styles['Heading3']))
        suggestions = analysis.get('Suggestions', [])
        if suggestions:
            for suggestion in suggestions:
                elements.append(Paragraph(f"✔ {suggestion}", normal_style))
        else:
            elements.append(Paragraph("No suggestions provided.", normal_style))
        elements.append(Spacer(1, 20))

    elements.append(Paragraph("Combined Analysis Summary:", styles['Heading2']))
    elements.append(Paragraph(combined_analysis, justified_style))
    elements.append(Spacer(1, 20))

    doc.build(elements, onFirstPage=draw_border, onLaterPages=draw_border)
    return filename

@app.route('/transcribe_audio', methods=['POST'])
def transcribe_audio():
    request_id = str(uuid.uuid4())
    app.logger = RequestLoggerAdapter(logging.getLogger(), {'request_id': request_id})
    app.logger.info('Received request to /transcribe_audio')
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    language = request.form.get('language', 'en')
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid or no selected file'}), 400
    if language not in SUPPORTED_LANGUAGES:
        return jsonify({'error': f'Unsupported language: {language}'}), 400
    
    filename = secure_filename(f'{uuid.uuid4()}_{file.filename}')
    s3_key = f'uploads/{filename}'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    with get_file_lock(filepath):
        try:
            # Save file locally and upload to S3
            file.save(filepath)
            s3_client.upload_file(filepath, bucket_name, s3_key)
            
            # Transcribe
            job_name = f'transcribe_{uuid.uuid4()}'
            s3_uri = f's3://{bucket_name}/{s3_key}'
            language_code = {
                'en': 'en-US', 'es': 'es-ES', 'fr': 'fr-FR', 'de': 'de-DE',
                'zh': 'zh-CN', 'te': 'te-IN', 'hi': 'hi-IN'
            }.get(language, 'en-US')
            
            transcribe_client.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': s3_uri},
                LanguageCode=language_code
            )
            while True:
                result = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
                if result['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                    break
                time.sleep(5)
            if result['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
                transcript_uri = result['TranscriptionJob']['Transcript']['TranscriptFileUri']
                transcript = requests.get(transcript_uri).json()['results']['transcripts'][0]['transcript']
                result = {'text': transcript, 'language': language}
                app.logger.info('Transcription result: %s', result)
                return jsonify(result)
            raise Exception('Transcription failed')
        except Exception as e:
            app.logger.error('Processing error: %s\n%s', str(e), traceback.format_exc())
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    app.logger.info('Deleted local file: %s', filepath)
                except Exception as e:
                    app.logger.error('Failed to delete local file %s: %s', filepath, str(e))
            try:
                s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
                app.logger.info('Deleted S3 object: %s', s3_key)
            except Exception as e:
                app.logger.error('Failed to delete S3 object %s: %s', s3_key, str(e))

@app.route('/analyze_audio', methods=['POST'])
def analyze_audio():
    request_id = str(uuid.uuid4())
    app.logger = RequestLoggerAdapter(logging.getLogger(), {'request_id': request_id})
    app.logger.info('Received request to /analyze_audio')

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    language = request.form.get('language', 'en')
    question = request.form.get('question', 'General conversation')

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid or no selected file'}), 400
    if language not in SUPPORTED_LANGUAGES:
        return jsonify({'error': f'Unsupported language: {language}'}), 400

    filename = secure_filename(f'{uuid.uuid4()}_{file.filename}')
    s3_key = f'uploads/{filename}'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    with get_file_lock(filepath):
        try:
            # Save file locally and upload to S3
            file.save(filepath)
            s3_client.upload_file(filepath, bucket_name, s3_key)

            # Transcribe
            job_name = f'transcribe_{uuid.uuid4()}'
            s3_uri = f's3://{bucket_name}/{s3_key}'
            language_code = {
                'en': 'en-US', 'es': 'es-ES', 'fr': 'fr-FR', 'de': 'de-DE',
                'zh': 'zh-CN', 'te': 'te-IN', 'hi': 'hi-IN'
            }.get(language, 'en-US')
            
            transcribe_client.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': s3_uri},
                LanguageCode=language_code
            )
            while True:
                result = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
                if result['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                    break
                time.sleep(5)
            if result['TranscriptionJob']['TranscriptionJobStatus'] != 'COMPLETED':
                raise Exception('Transcription failed')
            
            transcript_uri = result['TranscriptionJob']['Transcript']['TranscriptFileUri']
            transcribed_text = requests.get(transcript_uri).json()['results']['transcripts'][0]['transcript']
            app.logger.info('Transcription: %s', transcribed_text)

            # Extract acoustic features
            app.logger.info('Extracting acoustic features...')
            acoustic_features = extract_acoustic_features(filepath)
            app.logger.info('Acoustic Features: %s', acoustic_features)

            # Analyze transcription
            app.logger.info('Analyzing transcription...')
            api_output = analyze_transcription(transcribed_text, acoustic_features, question, language)
            app.logger.info('API Output: %s', api_output)

            try:
                analysis = json.loads(api_output)
                result = {
                    'transcription': transcribed_text,
                    'analysis': analysis,
                    'acoustic_features': acoustic_features,
                    'language': language,
                    'question': question
                }
            except json.JSONDecodeError as e:
                result = {
                    'transcription': transcribed_text,
                    'analysis_raw': api_output,
                    'error': f'Failed to parse API output as JSON: {str(e)}',
                    'language': language,
                    'question': question
                }

            app.logger.info('Returning result: %s', result)
            return jsonify(result)
        except Exception as e:
            error_msg = f'Processing error: {str(e)}\n{traceback.format_exc()}'
            app.logger.error(error_msg)
            return jsonify({'error': error_msg}), 500
        finally:
            # Clean up
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    app.logger.info('Deleted local file: %s', filepath)
                except Exception as e:
                    app.logger.error('Failed to delete local file %s: %s', filepath, str(e))
            try:
                s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
                app.logger.info('Deleted S3 object: %s', s3_key)
            except Exception as e:
                app.logger.error('Failed to delete S3 object %s: %s', s3_key, str(e))

@app.route('/analyze_multiple_audio', methods=['POST'])
def analyze_multiple_audio():
    request_id = str(uuid.uuid4())
    app.logger = RequestLoggerAdapter(logging.getLogger(), {'request_id': request_id})
    app.logger.info('Received request to /analyze_multiple_audio')
    app.logger.info('Incoming files: %s', list(request.files.keys()))

    try:
        file_keys = [key for key in request.files.keys() if key.startswith('file_')]
        if not file_keys:
            error_msg = 'At least one audio file is required'
            app.logger.error(error_msg)
            return jsonify({'error': error_msg}), 400

        transcriptions = []
        individual_analyses = []
        files_to_delete = []

        for i, file_key in enumerate(file_keys):
            question_key = f'question_{i}'
            language_key = f'language_{i}'

            file = request.files[file_key]
            question = request.form.get(question_key, 'Unknown question')
            language = request.form.get(language_key, 'en')

            if file.filename == '' or not allowed_file(file.filename):
                error_msg = f'Invalid file for response {i}'
                app.logger.error(error_msg)
                return jsonify({'error': error_msg}), 400
            if language not in SUPPORTED_LANGUAGES:
                error_msg = f'Unsupported language: {language} for response {i}'
                app.logger.error(error_msg)
                return jsonify({'error': error_msg}), 400

            filename = secure_filename(f'{uuid.uuid4()}_{file.filename}')
            s3_key = f'uploads/{filename}'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            files_to_delete.append((filepath, s3_key))

            with get_file_lock(filepath):
                app.logger.info('Saving file %s as %s', file.filename, filepath)
                file.save(filepath)
                s3_client.upload_file(filepath, bucket_name, s3_key)
                app.logger.info('Saved file to S3: %s', s3_key)

                try:
                    app.logger.info('Transcribing file %s', filepath)
                    job_name = f'transcribe_{uuid.uuid4()}'
                    s3_uri = f's3://{bucket_name}/{s3_key}'
                    language_code = {
                        'en': 'en-US', 'es': 'es-ES', 'fr': 'fr-FR', 'de': 'de-DE',
                        'zh': 'zh-CN', 'te': 'te-IN', 'hi': 'hi-IN'
                    }.get(language, 'en-US')
                    
                    transcribe_client.start_transcription_job(
                        TranscriptionJobName=job_name,
                        Media={'MediaFileUri': s3_uri},
                        LanguageCode=language_code
                    )
                    while True:
                        result = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
                        if result['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                            break
                        time.sleep(5)
                    if result['TranscriptionJob']['TranscriptionJobStatus'] != 'COMPLETED':
                        raise Exception('Transcription failed')
                    
                    transcript_uri = result['TranscriptionJob']['Transcript']['TranscriptFileUri']
                    transcribed_text = requests.get(transcript_uri).json()['results']['transcripts'][0]['transcript']
                    app.logger.info('Transcription for %s: %s', filepath, transcribed_text)

                    app.logger.info('Extracting acoustic features for %s', filepath)
                    acoustic_features = extract_acoustic_features(filepath)

                    app.logger.info('Analyzing transcription for response %s', i)
                    api_output = analyze_transcription(transcribed_text, acoustic_features, question, language)
                    analysis = json.loads(api_output)
                    app.logger.info('Analysis for response %s: %s', i, analysis)

                    transcriptions.append({
                        'question': question,
                        'text': transcribed_text,
                        'language': language
                    })
                    individual_analyses.append(analysis)

                except Exception as e:
                    error_msg = f'Processing error for response {i}: {str(e)}\n{traceback.format_exc()}'
                    app.logger.error(error_msg)
                    return jsonify({'error': error_msg}), 500

        app.logger.info('Generating combined analysis')
        combined_analysis = combine_analyses(individual_analyses, transcriptions)

        result = {
            'transcriptions': transcriptions,
            'individual_analyses': individual_analyses,
            'combined_analysis': combined_analysis
        }
        app.logger.info('Returning result: %s', result)
        return jsonify(result)
    except Exception as e:
        error_msg = f'Processing error: {str(e)}\n{traceback.format_exc()}'
        app.logger.error(error_msg)
        with open('error.log', 'a') as f:
            f.write(error_msg + '\n')
        return jsonify({'error': error_msg}), 500
    finally:
        for filepath, s3_key in files_to_delete:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    app.logger.info('Deleted local file: %s', filepath)
                except Exception as e:
                    app.logger.error('Failed to delete local file %s: %s', filepath, str(e))
            try:
                s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
                app.logger.info('Deleted S3 object: %s', s3_key)
            except Exception as e:
                app.logger.error('Failed to delete S3 object %s: %s', s3_key, str(e))

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf_route():
    request_id = str(uuid.uuid4())
    app.logger = RequestLoggerAdapter(logging.getLogger(), {'request_id': request_id})
    try:
        data = request.get_json()
        app.logger.info('Received payload: %s', json.dumps(data, indent=2))

        if not data or 'individual_analyses' not in data or 'combined_analysis' not in data:
            error_msg = 'Individual analyses and combined analysis are required'
            app.logger.error(error_msg)
            return jsonify({'error': error_msg, 'error_id': request_id}), 400

        individual_analyses = data.get('individual_analyses')
        combined_analysis = data.get('combined_analysis')
        username = data.get('username', 'Unknown')

        if not isinstance(individual_analyses, list) or not combined_analysis:
            error_msg = 'Invalid data format: individual_analyses must be a list and combined_analysis must be non-empty'
            app.logger.error(error_msg)
            return jsonify({'error': error_msg, 'error_id': request_id}), 400

        pdf_filename = generate_pdf(json.dumps(data))
        app.logger.info('Generated PDF: %s', pdf_filename)
        
        # Upload PDF to S3
        s3_key = f'pdfs/{pdf_filename}'
        s3_client.upload_file(pdf_filename, bucket_name, s3_key)
        app.logger.info('Uploaded PDF to S3: %s', s3_key)
        
        # Generate pre-signed URL for download
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': s3_key},
            ExpiresIn=3600  # 1 hour
        )
        
        # Clean up local file
        if os.path.exists(pdf_filename):
            os.remove(pdf_filename)
            app.logger.info('Deleted local PDF: %s', pdf_filename)
        
        return jsonify({'pdf_url': presigned_url})
    except Exception as e:
        error_msg = f'PDF generation error: {str(e)}\n{traceback.format_exc()}'
        app.logger.error(error_msg)
        return jsonify({'error': error_msg, 'error_id': request_id}), 500

@app.route('/translate', methods=['POST'])
def translate_text():
    request_id = str(uuid.uuid4())
    app.logger = RequestLoggerAdapter(logging.getLogger(), {'request_id': request_id})
    try:
        data = request.get_json()
        text = data.get('text')
        target_language = data.get('targetLanguage', data.get('target_language', 'en'))

        if not text:
            app.logger.error('No text provided')
            return jsonify({'error': 'No text provided'}), 400

        if target_language not in SUPPORTED_LANGUAGES:
            app.logger.error('Unsupported language code: %s', target_language)
            return jsonify({'error': f'Unsupported language code: {target_language}'}), 400

        app.logger.info('Translating: %s to %s', text, target_language)
        translator = GoogleTranslator(source='auto', target=target_language)
        translated = translator.translate(text)
        app.logger.info('Translated text: %s', translated)
        return jsonify({'translatedText': translated}), 200
    except Exception as e:
        error_msg = f'Translation error: {str(e)}\n{traceback.format_exc()}'
        app.logger.error(error_msg)
        return jsonify({'error': f'Translation failed: {str(e)}'}), 500

if __name__ == '__main__':
    try:
        logger.info('Starting Flask application')
        app.run(debug=False, host='0.0.0.0', port=5000 
