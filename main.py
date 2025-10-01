import os
import logging
import requests
import telebot
import json
from flask import Flask, request, abort, render_template_string, jsonify
from datetime import datetime
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, KeyboardButton
import threading
import time
import io
from pymongo import MongoClient
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
import subprocess
import tempfile
import glob
import math
import speech_recognition as sr
from concurrent.futures import ThreadPoolExecutor
import re
from collections import Counter
import wave

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants & Environment Variables (No Change needed here, good for configuration) ---

CHUNK_DURATION_SEC = int(os.environ.get("CHUNK_DURATION_SEC", "55"))
CHUNK_BATCH_SIZE = int(os.environ.get("CHUNK_BATCH_SIZE", "30"))
CHUNK_BATCH_PAUSE_SEC = int(os.environ.get("CHUNK_BATCH_PAUSE_SEC", "5"))
RECOGNITION_MAX_RETRIES = int(os.environ.get("RECOGNITION_MAX_RETRIES", "3"))
RECOGNITION_RETRY_WAIT = int(os.environ.get("RECOGNITION_RETRY_WAIT", "3"))
AUDIO_SAMPLE_RATE = int(os.environ.get("AUDIO_SAMPLE_RATE", "16000"))
AUDIO_CHANNELS = int(os.environ.get("AUDIO_CHANNELS", "1"))
TELEGRAM_MAX_BYTES = int(os.environ.get("TELEGRAM_MAX_BYTES", str(20 * 1024 * 1024)))
MAX_WEB_UPLOAD_MB = int(os.environ.get("MAX_WEB_UPLOAD_MB", "250"))
REQUEST_TIMEOUT_TELEGRAM = int(os.environ.get("REQUEST_TIMEOUT_TELEGRAM", "300"))
REQUEST_TIMEOUT_LLM = int(os.environ.get("REQUEST_TIMEOUT_LLM", "300"))
TRANSCRIBE_MAX_WORKERS = int(os.environ.get("TRANSCRIBE_MAX_WORKERS", "4"))
PREPEND_SILENCE_SEC = int(os.environ.get("PREPEND_SILENCE_SEC", "5"))
AMBIENT_CALIB_SEC = float(os.environ.get("AMBIENT_CALIB_SEC", "3"))
REQUEST_TIMEOUT_GEMINI = int(os.environ.get("REQUEST_TIMEOUT_GEMINI", "300"))
REQUEST_TIMEOUT_ASSEMBLY = int(os.environ.get("REQUEST_TIMEOUT_ASSEMBLY", "180"))
ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "3da64715f8304ca3a7c78638c4bfd90c")
BOT_TOKENS = [
    "7770743573:AAEhVz2jKwdJjhrm3XH0bAhhLucg2H4AvMY",
    "7790991731:AAF4NHGm0BJCf08JTdBaUWKzwfs82_Y9Ecw",
]
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBRWP4BaPLPpYdB5_E3C3TVDGqiHrjv4vQ")
WEBHOOK_BASE = os.environ.get("WEBHOOK_BASE", "https://mediatotextbot.onrender.com")
SECRET_KEY = os.environ.get("SECRET_KEY", "super-secret-please-change")
MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://hoskasii:GHyCdwpI0PvNuLTg@cluster0.dy7oe7t.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
DB_NAME = os.environ.get("DB_NAME", "telegram_bot_db")
REQUIRED_CHANNEL = os.environ.get("REQUIRED_CHANNEL", "@guruubka_wasmada")

# --- Initialization ---

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
users_collection = db["users"]
groups_collection = db["groups"]

app = Flask(__name__)

bots = [telebot.TeleBot(token, threaded=True, parse_mode='HTML') for token in BOT_TOKENS]
serializer = URLSafeTimedSerializer(SECRET_KEY)

# --- Language Configuration (Same as before) ---

LANG_OPTIONS = [
    ("🇬🇧 English", "en"), ("🇸🇦 العربية", "ar"), ("🇪🇸 Español", "es"), ("🇫🇷 Français", "fr"),
    ("🇷🇺 Русский", "ru"), ("🇩🇪 Deutsch", "de"), ("🇮🇳 हिन्दी", "hi"), ("🇮🇷 فارسی", "fa"),
    ("🇮🇩 Indonesia", "id"), ("🇺🇦 Українська", "uk"), ("🇦🇿 Azərbaycan", "az"), ("🇮🇹 Italiano", "it"),
    ("🇹🇷 Türkçe", "tr"), ("🇧🇬 Български", "bg"), ("🇷🇸 Srpski", "sr"), ("🇵🇰 اردو", "ur"),
    ("🇹🇭 ไทย", "th"), ("🇻🇳 Tiếng Việt", "vi"), ("🇯🇵 日本語", "ja"), ("🇰🇷 한국어", "ko"),
    ("🇨🇳 中文", "zh"), ("🇳🇱 Nederlands", "nl"), ("🇸🇪 Svenska", "sv"), ("🇳🇴 Norsk", "no"),
    ("🇮🇱 עברית", "he"), ("🇩🇰 Dansk", "da"), ("🇪🇹 አማርኛ", "am"), ("🇫🇮 Suomi", "fi"),
    ("🇧🇩 বাংলা", "bn"), ("🇰🇪 Kiswahili", "sw"), ("🇪🇹 Oromoo", "om"), ("🇳🇵 नेपाली", "ne"),
    ("🇵🇱 Polski", "pl"), ("🇬🇷 Ελληνικά", "el"), ("🇨🇿 Čeština", "cs"), ("🇮🇸 Íslenska", "is"),
    ("🇱🇹 Lietuvių", "lt"), ("🇱🇻 Latviešu", "lv"), ("🇭🇷 Hrvatski", "hr"), ("🇷🇸 Bosanski", "bs"),
    ("🇭🇺 Magyar", "hu"), ("🇷🇴 Română", "ro"), ("🇸🇴 Somali", "so"), ("🇲🇾 Melayu", "ms"),
    ("🇺🇿 O'zbekcha", "uz"), ("🇵🇭 Tagalog", "tl"), ("🇵🇹 Português", "pt"),
]

CODE_TO_LABEL = {code: label for (label, code) in LANG_OPTIONS}
LABEL_TO_CODE = {label: code for (label, code) in LANG_OPTIONS}
STT_LANGUAGES = {}
for label, code in LANG_OPTIONS:
    STT_LANGUAGES[label.split(" ", 1)[-1]] = {
        "code": code, "emoji": label.split(" ", 1)[0], "native": label.split(" ", 1)[-1]
    }

user_transcriptions = {}
memory_lock = threading.Lock()
in_memory_data = {"pending_media": {}}
action_usage = {}
ALLOWED_EXTENSIONS = {
    "mp3", "wav", "m4a", "ogg", "webm", "flac", "mp4", "mkv", "avi", "mov", "hevc", "aac", "aiff", "amr", "wma", "opus", "m4v", "ts", "flv", "3gp"
}
ASSEMBLY_LANG_SET = {"en", "ar", "es", "fr", "ru", "de", "hi", "fa", "zh", "ko", "ja", "it", "uk"}

# --- FFMPEG Binary Check (Same as before) ---

FFMPEG_ENV = os.environ.get("FFMPEG_BINARY", "")
POSSIBLE_FFMPEG_PATHS = [FFMPEG_ENV, "./ffmpeg", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "ffmpeg"]
FFMPEG_BINARY = None
for p in POSSIBLE_FFMPEG_PATHS:
    if not p:
        continue
    try:
        subprocess.run([p, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
        FFMPEG_BINARY = p
        break
    except Exception:
        continue
if FFMPEG_BINARY is None:
    logging.warning("ffmpeg binary not found. Set FFMPEG_BINARY env var or place ffmpeg in ./ffmpeg or /usr/bin/ffmpeg")

# --- Database & User Helper Functions (Same as before) ---

def update_user_activity(user_id: int):
    user_id_str = str(user_id)
    now = datetime.now()
    users_collection.update_one(
        {"_id": user_id_str},
        {"$set": {"last_active": now}, "$setOnInsert": {"first_seen": now, "stt_conversion_count": 0}},
        upsert=True
    )

def increment_processing_count(user_id: str, service_type: str):
    field_to_inc = f"{service_type}_conversion_count"
    users_collection.update_one(
        {"_id": str(user_id)},
        {"$inc": {field_to_inc: 1}}
    )

def get_stt_user_lang(user_id: str) -> str:
    user_data = users_collection.find_one({"_id": user_id})
    if user_data and "stt_language" in user_data:
        return user_data["stt_language"]
    return "en"

def set_stt_user_lang(user_id: str, lang_code: str):
    users_collection.update_one(
        {"_id": user_id},
        {"$set": {"stt_language": lang_code}},
        upsert=True
    )

def get_user_send_mode(user_id: str) -> str:
    user_data = users_collection.find_one({"_id": user_id})
    if user_data and "stt_send_mode" in user_data:
        return user_data["stt_send_mode"]
    return "file"

def set_user_send_mode(user_id: str, mode: str):
    if mode not in ("file", "split"):
        mode = "file"
    users_collection.update_one(
        {"_id": user_id},
        {"$set": {"stt_send_mode": mode}},
        upsert=True
    )

def user_has_stt_setting(user_id: str) -> bool:
    user_data = users_collection.find_one({"_id": user_id})
    return user_data is not None and "stt_language" in user_data

def save_pending_media(user_id: str, media_type: str, data: dict):
    with memory_lock:
        in_memory_data["pending_media"][user_id] = {
            "media_type": media_type, "data": data, "saved_at": datetime.now()
        }

def pop_pending_media(user_id: str):
    with memory_lock:
        return in_memory_data["pending_media"].pop(user_id, None)

def delete_transcription_later(user_id: str, message_id: int):
    time.sleep(86400)
    with memory_lock:
        if user_id in user_transcriptions and message_id in user_transcriptions[user_id]:
            del user_transcriptions[user_id][message_id]

# --- Core Utility Functions (Same as before) ---

def select_speech_model_for_lang(language_code: str):
    return "universal"

def is_transcoding_like_error(msg: str) -> bool:
    if not msg:
        return False
    m = msg.lower()
    checks = [
        "transcoding failed", "file does not appear to contain audio",
        "text/html", "html document", "unsupported media type", "could not decode",
    ]
    return any(ch in m for ch in checks)

def signed_upload_token(chat_id: int, lang_code: str, bot_index: int = 0):
    payload = {"chat_id": chat_id, "lang": lang_code, "bot_index": int(bot_index)}
    return serializer.dumps(payload)

def unsign_upload_token(token: str, max_age_seconds: int = 3600):
    data = serializer.loads(token, max_age=max_age_seconds)
    return data

def animate_processing_message(bot_obj, chat_id, message_id, stop_event):
    frames = ["🔄 Processing", "🔄 Processing.", "🔄 Processing..", "🔄 Processing..."]
    idx = 0
    while not stop_event():
        try:
            bot_obj.edit_message_text(frames[idx % len(frames)], chat_id=chat_id, message_id=message_id)
        except Exception:
            pass
        idx = (idx + 1) % len(frames)
        time.sleep(0.6)

def normalize_text_offline(text: str) -> str:
    if not text:
        return text
    t = re.sub(r'\s+', ' ', text).strip()
    return t

def extract_key_points_offline(text: str, max_points: int = 6) -> str:
    if not text:
        return ""
    sentences = re.split(r'(?<=[\.\!\?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return ""
    words = re.findall(r'\w+', text.lower())
    words = [w for w in words if len(w) > 3]
    if not words:
        selected = sentences[:max_points]
        return "\n".join(f"- {s}" for s in selected)
    freq = Counter(words)
    sentence_scores = []
    for s in sentences:
        s_words = re.findall(r'\w+', s.lower())
        score = sum(freq.get(w, 0) for w in s_words)
        sentence_scores.append((score, s))
    sentence_scores.sort(key=lambda x: x[0], reverse=True)
    top = sentence_scores[:max_points]
    top_sentences = sorted(top, key=lambda x: sentences.index(x[1]))
    result_lines = [f"- {s}" for _, s in top_sentences]
    return "\n".join(result_lines)

def safe_extension_from_filename(filename: str):
    if not filename or "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()

def telegram_file_info_and_url(bot_token: str, file_id):
    import urllib.request
    url = f"https://api.telegram.org/bot{bot_token}/getFile?file_id={file_id}"
    resp = requests.get(url, timeout=REQUEST_TIMEOUT_TELEGRAM)
    resp.raise_for_status()
    j = resp.json()
    file_path = j.get("result", {}).get("file_path")
    file_url = f"https://api.telegram.org/file/bot{bot_token}/{file_path}"
    class Dummy:
        pass
    d = Dummy()
    d.file_path = file_path
    return d, file_url

def convert_to_wav(input_path: str, output_wav_path: str):
    if FFMPEG_BINARY is None:
        raise RuntimeError("ffmpeg binary not found")
    cmd = [
        FFMPEG_BINARY, "-y", "-i", input_path, "-ar", str(AUDIO_SAMPLE_RATE),
        "-ac", str(AUDIO_CHANNELS), output_wav_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def split_wav_to_chunks(wav_path: str, out_dir: str, chunk_duration_sec: int):
    if FFMPEG_BINARY is None:
        raise RuntimeError("ffmpeg binary not found")
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "chunk%03d.wav")
    cmd = [
        FFMPEG_BINARY, "-y", "-i", wav_path, "-ar", str(AUDIO_SAMPLE_RATE),
        "-ac", str(AUDIO_CHANNELS), "-f", "segment", "-segment_time",
        str(chunk_duration_sec), "-reset_timestamps", "1", pattern
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    files = sorted(glob.glob(os.path.join(out_dir, "chunk*.wav")))
    return files

def create_prepended_chunk(chunk_path: str, silence_sec: int):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    out_path = tmp.name
    try:
        prepend_silence_to_wav(chunk_path, out_path, silence_sec)
        return out_path
    except Exception:
        try:
            os.remove(out_path)
        except Exception:
            pass
        raise

def prepend_silence_to_wav(original_wav: str, output_wav: str, silence_sec: int):
    if FFMPEG_BINARY is None:
        raise RuntimeError("ffmpeg binary not found")
    tmp_dir = os.path.dirname(output_wav) or tempfile.gettempdir()
    silence_file = os.path.join(tmp_dir, f"silence_{int(time.time()*1000)}.wav")
    cmd_create_silence = [
        FFMPEG_BINARY, "-y", "-f", "lavfi",
        "-i", f"anullsrc=channel_layout=mono:sample_rate={AUDIO_SAMPLE_RATE}",
        "-t", str(silence_sec), "-ar", str(AUDIO_SAMPLE_RATE),
        "-ac", str(AUDIO_CHANNELS), silence_file
    ]
    subprocess.run(cmd_create_silence, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    cmd_concat = [
        FFMPEG_BINARY, "-y", "-i", silence_file, "-i", original_wav,
        "-filter_complex", "[0:0][1:0]concat=n=2:v=0:a=1[out]",
        "-map", "[out]", output_wav
    ]
    subprocess.run(cmd_concat, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    try:
        os.remove(silence_file)
    except Exception:
        pass

def recognize_chunk_file(recognizer, file_path: str, language: str):
    last_exc = None
    prepended_path = None
    for attempt in range(1, RECOGNITION_MAX_RETRIES + 1):
        try:
            try:
                prepended_path = create_prepended_chunk(file_path, PREPEND_SILENCE_SEC)
            except Exception:
                prepended_path = None
            use_path = prepended_path if prepended_path else file_path
            with sr.AudioFile(use_path) as source:
                try:
                    recognizer.adjust_for_ambient_noise(source, duration=AMBIENT_CALIB_SEC)
                except Exception:
                    pass
                audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language=language) if language else recognizer.recognize_google(audio)
            if prepended_path:
                try: os.remove(prepended_path)
                except Exception: pass
            return text
        except sr.UnknownValueError:
            if prepended_path:
                try: os.remove(prepended_path)
                except Exception: pass
            return ""
        except (sr.RequestError, ConnectionResetError) as e:
            last_exc = e
            if prepended_path:
                try: os.remove(prepended_path)
                except Exception: pass
            time.sleep(RECOGNITION_RETRY_WAIT * attempt)
            continue
        except OSError as e:
            last_exc = e
            if prepended_path:
                try: os.remove(prepended_path)
                except Exception: pass
            break
    if last_exc is not None:
        raise last_exc
    return ""

def transcribe_file_with_speech_recognition(input_file_path: str, language_code: str):
    tmpdir = tempfile.mkdtemp(prefix="stt_")
    try:
        base_wav = os.path.join(tmpdir, "converted.wav")
        try:
            convert_to_wav(input_file_path, base_wav)
        except Exception as e:
            raise RuntimeError("Conversion to WAV failed: " + str(e))
        chunk_files = split_wav_to_chunks(base_wav, tmpdir, CHUNK_DURATION_SEC)
        if not chunk_files:
            raise RuntimeError("No audio chunks created")
        def transcribe_chunk(chunk_path):
            recognizer = sr.Recognizer()
            return recognize_chunk_file(recognizer, chunk_path, language_code)
        with ThreadPoolExecutor(max_workers=TRANSCRIBE_MAX_WORKERS) as executor:
            results = list(executor.map(transcribe_chunk, chunk_files))
        final_text = "\n".join([r for r in results if r])
        return final_text
    finally:
        try:
            for f in glob.glob(os.path.join(tmpdir, "*")):
                try: os.remove(f)
                except Exception: pass
            try: os.rmdir(tmpdir)
            except Exception: pass
        except Exception: pass

def transcribe_with_assemblyai(file_path: str, language_code: str, timeout_seconds: int = REQUEST_TIMEOUT_ASSEMBLY):
    headers = {"authorization": ASSEMBLYAI_API_KEY}
    upload_url = None
    with open(file_path, "rb") as f:
        try:
            resp = requests.post("https://api.assemblyai.com/v2/upload", headers=headers, data=f, timeout=timeout_seconds)
            resp.raise_for_status()
            j = resp.json()
            upload_url = j.get("upload_url") or j.get("url") or j.get("data")
            if not upload_url and isinstance(j, dict) and len(j) == 1:
                val = next(iter(j.values()))
                if isinstance(val, str) and val.startswith("http"):
                    upload_url = val
            if not upload_url:
                raise RuntimeError("Upload failed: no upload_url returned")
        except Exception as e:
            raise RuntimeError("AssemblyAI upload failed: " + str(e))
    try:
        payload = {"audio_url": upload_url}
        if language_code:
            payload["language_code"] = language_code
        resp = requests.post("https://api.assemblyai.com/v2/transcript", headers={**headers, "content-type": "application/json"}, json=payload, timeout=timeout_seconds)
        resp.raise_for_status()
        job = resp.json()
        job_id = job.get("id")
        if not job_id:
            raise RuntimeError("AssemblyAI transcript creation failed")
        poll_url = f"https://api.assemblyai.com/v2/transcript/{job_id}"
        start = time.time()
        while True:
            r = requests.get(poll_url, headers=headers, timeout=30)
            r.raise_for_status()
            status_json = r.json()
            status = status_json.get("status")
            if status == "completed":
                return status_json.get("text", "")
            if status == "error":
                raise RuntimeError("AssemblyAI transcription error: " + str(status_json.get("error", "")))
            if time.time() - start > timeout_seconds:
                raise RuntimeError("AssemblyAI transcription timed out")
            time.sleep(3)
    except Exception as e:
        raise RuntimeError("AssemblyAI transcription failed: " + str(e))

def transcribe_via_selected_service(input_path: str, lang_code: str):
    use_assembly = lang_code in ASSEMBLY_LANG_SET
    try:
        if use_assembly:
            text = transcribe_with_assemblyai(input_path, lang_code)
            if text is None:
                raise RuntimeError("AssemblyAI returned no text")
            return text, "assemblyai"
        else:
            text = transcribe_file_with_speech_recognition(input_path, lang_code)
            return text, "speech_recognition"
    except Exception as primary_e:
        logging.exception("Primary STT failed, attempting fallback")
        try:
            if use_assembly:
                text = transcribe_file_with_speech_recognition(input_path, lang_code)
                return text, "speech_recognition"
            else:
                text = transcribe_with_assemblyai(input_path, lang_code)
                return text, "assemblyai"
        except Exception as fallback_e:
            raise RuntimeError(f"Both STT services failed: Primary ({primary_e}), Fallback ({fallback_e})")

def split_text_into_chunks(text: str, limit: int = 4096):
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + limit, n)
        if end < n:
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space
        chunk = text[start:end].strip()
        if not chunk:
            end = start + limit
            chunk = text[start:end].strip()
        chunks.append(chunk)
        start = end
    return chunks

def check_subscription(user_id: int, bot_obj) -> bool:
    if not REQUIRED_CHANNEL or not REQUIRED_CHANNEL.strip():
        return True
    try:
        member = bot_obj.get_chat_member(REQUIRED_CHANNEL, user_id)
        return member.status in ['member', 'administrator', 'creator']
    except Exception:
        return False

def send_subscription_message(chat_id: int, bot_obj):
    if not REQUIRED_CHANNEL or not REQUIRED_CHANNEL.strip():
        return
    try:
        chat = bot_obj.get_chat(chat_id)
        if chat.type != 'private':
            return
    except Exception:
        pass
    try:
        markup = InlineKeyboardMarkup()
        markup.add(
            InlineKeyboardButton(
                "Click here to join the Channel",
                url=f"https://t.me/{REQUIRED_CHANNEL.lstrip('@')}"
            )
        )
        bot_obj.send_message(
            chat_id,
            "🔒 Access Locked You cannot use this bot until you join the Channel.",
            reply_markup=markup
        )
    except Exception:
        pass

# --- Keyboard/UI Helper Functions (Refactored to reduce repetition) ---

def build_lang_keyboard(callback_prefix: str, row_width: int = 3, message_id: int = None):
    markup = InlineKeyboardMarkup(row_width=row_width)
    buttons = []
    for label, code in LANG_OPTIONS:
        cb = f"{callback_prefix}|{code}|{message_id}" if message_id is not None else f"{callback_prefix}|{code}"
        buttons.append(InlineKeyboardButton(label, callback_data=cb))
    for i in range(0, len(buttons), row_width):
        markup.add(*buttons[i:i+row_width])
    return markup

def build_result_mode_keyboard(prefix: str = "result_mode"):
    markup = InlineKeyboardMarkup(row_width=2)
    markup.add(InlineKeyboardButton("📄 .txt file", callback_data=f"{prefix}|file"))
    markup.add(InlineKeyboardButton("💬 Split messages", callback_data=f"{prefix}|split"))
    return markup

def build_action_keyboard(chat_id: int, msg_id: int, text_length: int) -> InlineKeyboardMarkup:
    """Wuxuu dhisaa keyboard-ka Clean transcript iyo Get Summarize."""
    markup = InlineKeyboardMarkup()
    buttons = []
    buttons.append(InlineKeyboardButton("⭐️Clean transcript", callback_data=f"clean_up|{chat_id}|{msg_id}"))
    if text_length > 1000:
        buttons.append(InlineKeyboardButton("Get Summarize", callback_data=f"get_key_points|{chat_id}|{msg_id}"))
    markup.add(*buttons)
    return markup

def send_transcription_result(bot_obj, chat_id, message_id, corrected_text, user_mode):
    """Wuxuu maamulaa u dirista natiijada (file ama split messages) iyo keyboaardka."""
    uid_key = str(chat_id)
    sent_msg = None
    if len(corrected_text) > 4000:
        if user_mode == "file":
            f = io.BytesIO(corrected_text.encode("utf-8"))
            f.name = "transcription.txt"
            sent_msg = bot_obj.send_document(chat_id, f, reply_to_message_id=message_id)
        else:
            chunks = split_text_into_chunks(corrected_text, limit=4096)
            last_sent = None
            for idx, chunk in enumerate(chunks):
                if idx == 0:
                    last_sent = bot_obj.send_message(chat_id, chunk, reply_to_message_id=message_id)
                else:
                    last_sent = bot_obj.send_message(chat_id, chunk)
            sent_msg = last_sent
    else:
        sent_msg = bot_obj.send_message(chat_id, corrected_text or "No transcription text was returned.", reply_to_message_id=message_id)

    if sent_msg:
        try:
            markup = build_action_keyboard(chat_id, sent_msg.message_id, len(corrected_text))
            bot_obj.edit_message_reply_markup(chat_id, sent_msg.message_id, reply_markup=markup)
        except Exception:
            pass

        try:
            user_transcriptions.setdefault(uid_key, {})[sent_msg.message_id] = corrected_text
            threading.Thread(target=delete_transcription_later, args=(uid_key, sent_msg.message_id), daemon=True).start()
        except Exception:
            pass

        try:
            action_usage[f"{chat_id}|{sent_msg.message_id}|clean_up"] = 0
            action_usage[f"{chat_id}|{sent_msg.message_id}|get_key_points"] = 0
        except Exception:
            pass
    return sent_msg

# --- Gemini Helper Function (Same as before) ---

def ask_gemini(text: str, instruction: str, timeout=REQUEST_TIMEOUT_GEMINI) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": instruction},
                    {"text": text}
                ]
            }
        ]
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    result = resp.json()
    if "candidates" in result and isinstance(result["candidates"], list) and len(result["candidates"]) > 0:
        cand = result['candidates'][0]
        try:
            return cand['content']['parts'][0]['text']
        except Exception:
            return json.dumps(cand)
    return json.dumps(result)


# --- Handle Media Logic (Refactored using send_transcription_result) ---

def handle_media_common(message, bot_obj, bot_token, bot_index=0):
    user_id_str = str(message.from_user.id)
    update_user_activity(message.from_user.id)
    file_id = None
    file_size = None
    filename = None
    media = message.voice or message.audio or message.video or message.document
    if media:
        if message.voice:
            file_id, file_size, filename = message.voice.file_id, message.voice.file_size, "voice.ogg"
        elif message.audio:
            file_id, file_size, filename = message.audio.file_id, message.audio.file_size, getattr(message.audio, "file_name", "audio")
        elif message.video:
            file_id, file_size, filename = message.video.file_id, message.video.file_size, getattr(message.video, "file_name", "video.mp4")
        elif message.document:
            mime = getattr(message.document, "mime_type", None)
            filename = getattr(message.document, "file_name", None) or "file"
            ext = safe_extension_from_filename(filename)
            if (mime and ("audio" in mime or "video" in mime)) or ext in ALLOWED_EXTENSIONS:
                file_id, file_size = message.document.file_id, message.document.file_size
            else:
                bot_obj.send_message(message.chat.id, "Sorry, I can only transcribe audio or video files.")
                return
    else:
        return

    lang = get_stt_user_lang(user_id_str)
    if file_size and file_size > TELEGRAM_MAX_BYTES:
        token = signed_upload_token(message.chat.id, lang, bot_index)
        upload_link = f"{WEBHOOK_BASE.rstrip('/')}/upload/{token}"
        max_display_mb = TELEGRAM_MAX_BYTES // (1024 * 1024)
        text = f'😓Telegram API doesn’t allow me to download your file if it’s larger than {max_display_mb}MB:👉🏻 <a href="{upload_link}">Click here to Upload  your file</a>'
        bot_obj.send_message(message.chat.id, text, disable_web_page_preview=True, parse_mode='HTML', reply_to_message_id=message.message_id)
        return

    processing_msg = bot_obj.send_message(message.chat.id, "🔄 Processing...", reply_to_message_id=message.message_id)
    processing_msg_id = processing_msg.message_id
    stop_animation = {"stop": False}
    def stop_event():
        return stop_animation["stop"]
    animation_thread = threading.Thread(target=animate_processing_message, args=(bot_obj, message.chat.id, processing_msg_id, stop_event))
    animation_thread.start()

    tmpf = None
    try:
        tf, file_url = telegram_file_info_and_url(bot_token, file_id)
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix="." + (safe_extension_from_filename(filename) or "tmp"))
        with requests.get(file_url, stream=True, timeout=REQUEST_TIMEOUT_TELEGRAM) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=256*1024):
                if chunk:
                    tmpf.write(chunk)
        tmpf.flush()
        tmpf.close()

        text, used_service = transcribe_via_selected_service(tmpf.name, lang)
        corrected_text = normalize_text_offline(text)
        user_mode = get_user_send_mode(user_id_str)
        send_transcription_result(bot_obj, message.chat.id, message.message_id, corrected_text, user_mode)
        increment_processing_count(user_id_str, "stt")
    except Exception as e:
        error_msg = str(e)
        logging.exception("Error in transcription process")
        if "ffmpeg" in error_msg.lower():
            bot_obj.send_message(message.chat.id, "⚠️ Server error: ffmpeg not found or conversion failed. Contact admin @boyso.", reply_to_message_id=message.message_id)
        elif is_transcoding_like_error(error_msg):
            bot_obj.send_message(message.chat.id, "⚠️ Transcription error: file is not audible. Please send a different file.", reply_to_message_id=message.message_id)
        else:
            bot_obj.send_message(message.chat.id, f"Error during transcription: {error_msg}", reply_to_message_id=message.message_id)
    finally:
        stop_animation["stop"] = True
        animation_thread.join()
        try:
            bot_obj.delete_message(message.chat.id, processing_msg_id)
        except Exception:
            pass
        if tmpf and os.path.exists(tmpf.name):
            try:
                os.remove(tmpf.name)
            except Exception:
                pass


# --- Web Upload Endpoint (Refactored using send_transcription_result) ---

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
... (Same HTML Template content as before) ...
</html>
"""

@app.route("/upload/<token>", methods=['GET', 'POST'])
def upload_large_file(token):
    try:
        data = unsign_upload_token(token, max_age_seconds=3600)
    except SignatureExpired:
        return "<h3>Link expired</h3>", 400
    except BadSignature:
        return "<h3>Invalid link</h3>", 400
    chat_id = data.get("chat_id")
    lang = data.get("lang", "en")
    bot_index = int(data.get("bot_index", 0))
    if bot_index < 0 or bot_index >= len(bots):
        bot_index = 0

    if request.method == 'GET':
        return render_template_string(HTML_TEMPLATE, lang_options=LANG_OPTIONS, selected_lang=lang, max_mb=MAX_WEB_UPLOAD_MB)

    file = request.files.get('file')
    if not file:
        return "No file uploaded", 400
    file_bytes = file.read()
    if len(file_bytes) > MAX_WEB_UPLOAD_MB * 1024 * 1024:
        return f"File too large. Max allowed is {MAX_WEB_UPLOAD_MB}MB.", 400

    def bytes_to_tempfile(b):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".upload")
        tmp.write(b)
        tmp.flush()
        tmp.close()
        return tmp.name

    def process_uploaded_file(chat_id_inner, lang_inner, path, bot_index_inner):
        bot_to_use = bots[bot_index_inner] if 0 <= bot_index_inner < len(bots) else bots[0]
        try:
            try:
                text, used = transcribe_via_selected_service(path, lang_inner)
            except Exception:
                bot_to_use.send_message(chat_id_inner, "Error occurred while transcribing the uploaded file.")
                return

            corrected_text = normalize_text_offline(text)
            user_mode = get_user_send_mode(str(chat_id_inner))
            # Ku xidh message_id oo ah 0 maadaama aysan ahayn reply
            sent_msg = send_transcription_result(bot_to_use, chat_id_inner, None, corrected_text, user_mode)

            if not sent_msg:
                bot_to_use.send_message(chat_id_inner, "Error sending transcription message. The transcription completed but could not be delivered as a message.")
        except Exception:
            logging.exception("Error in processing uploaded file thread")
            bot_to_use.send_message(chat_id_inner, "An unexpected error occurred during file processing.")
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

    tmp_path = bytes_to_tempfile(file_bytes)
    threading.Thread(target=process_uploaded_file, args=(chat_id, lang, tmp_path, bot_index), daemon=True).start()
    return jsonify({"status": "accepted", "message": "Upload accepted. Processing started. Your transcription will be sent to your Telegram chat when ready."})


# --- LLM/Clean Up/Summarize Common Handler (Refactored to reduce repetition) ---

def handle_ai_action(call, bot_obj, action_type: str, instruction_template: str, offline_func):
    """Maamulaha guud ee shaqooyinka 'clean_up' iyo 'get_key_points'."""
    try:
        if call.message.chat.type == 'private' and not check_subscription(call.from_user.id, bot_obj):
            send_subscription_message(call.message.chat.id, bot_obj)
            try: bot_obj.answer_callback_query(call.id)
            except Exception: pass
            return

        parts = call.data.split("|")
        chat_id_val = int(parts[1]) if len(parts) >= 3 else call.message.chat.id
        msg_id = int(parts[-1])
        uid_key = str(chat_id_val)

        usage_key = f"{chat_id_val}|{msg_id}|{action_type}"
        usage = action_usage.get(usage_key, 0)
        if usage >= 2:
            bot_obj.answer_callback_query(call.id, f"⚠️ {action_type.replace('_', ' ').title()} unavailable (maybe expired/used twice)", show_alert=True)
            return

        action_usage[usage_key] = usage + 1
        stored = user_transcriptions.get(uid_key, {}).get(msg_id)
        if not stored:
            bot_obj.answer_callback_query(call.id, f"⚠️ {action_type.replace('_', ' ').title()} unavailable (maybe expired)", show_alert=True)
            return

        bot_obj.answer_callback_query(call.id, "Generating..." if action_type == "get_key_points" else "Cleaning up...")
        status_msg = bot_obj.send_message(call.message.chat.id, "🔄 Processing...", reply_to_message_id=call.message.message_id)
        stop_animation = {"stop": False}
        def stop_event(): return stop_animation["stop"]
        animation_thread = threading.Thread(target=animate_processing_message, args=(bot_obj, call.message.chat.id, status_msg.message_id, stop_event))
        animation_thread.start()

        result_text = ""
        try:
            lang = get_stt_user_lang(uid_key) or "en"
            instruction = instruction_template.format(lang=lang)
            try:
                result_text = ask_gemini(stored, instruction)
            except Exception:
                result_text = offline_func(stored)
        except Exception:
            pass

        stop_animation["stop"] = True
        animation_thread.join()

        if not result_text:
            bot_obj.edit_message_text(f"No {action_type.replace('_', ' ')} returned.", chat_id=call.message.chat.id, message_id=status_msg.message_id)
            return

        # Maamulista dirista natiijada: file ama split messages
        if action_type == "clean_up":
            try: bot_obj.delete_message(call.message.chat.id, status_msg.message_id)
            except Exception: pass
            user_mode = get_user_send_mode(uid_key)
            send_transcription_result(bot_obj, call.message.chat.id, call.message.message_id, result_text, user_mode)
        else: # get_key_points (Summarize)
            try:
                bot_obj.edit_message_text(f"{result_text}", chat_id=call.message.chat.id, message_id=status_msg.message_id)
            except Exception:
                pass

    except Exception:
        logging.exception(f"Error in {action_type}_callback")


# --- Handler Registration (Reduced duplication) ---

def register_handlers(bot_obj, bot_token, bot_index):
    @bot_obj.message_handler(commands=['start'])
    def start_handler(message):
        try:
            update_user_activity(message.from_user.id)
            if message.chat.type == 'private' and not check_subscription(message.from_user.id, bot_obj):
                send_subscription_message(message.chat.id, bot_obj)
                return
            bot_obj.send_message(
                message.chat.id,
                "Choose your file language for transcription using the below buttons:",
                reply_markup=build_lang_keyboard("start_select_lang")
            )
        except Exception:
            logging.exception("Error in start_handler")

    @bot_obj.callback_query_handler(func=lambda c: c.data and c.data.startswith("start_select_lang|"))
    def start_select_lang_callback(call):
        try:
            if call.message.chat.type == 'private' and not check_subscription(call.from_user.id, bot_obj):
                send_subscription_message(call.message.chat.id, bot_obj)
                try: bot_obj.answer_callback_query(call.id)
                except Exception: pass
                return
            uid = str(call.from_user.id)
            _, lang_code = call.data.split("|", 1)
            lang_label = CODE_TO_LABEL.get(lang_code, lang_code)
            set_stt_user_lang(uid, lang_code)
            try: bot_obj.delete_message(call.message.chat.id, call.message.message_id)
            except Exception: pass
            welcome_text = (
                f"👋 Salaam!    \n"
                "• Send me\n"
                "• voice message\n"
                "• audio file\n"
                "• video\n"
                "• to transcribe for free"
            )
            bot_obj.send_message(call.message.chat.id, welcome_text)
            bot_obj.answer_callback_query(call.id, f"✅ Language set to {lang_label}")
        except Exception:
            logging.exception("Error in start_select_lang_callback")
            try: bot_obj.answer_callback_query(call.id, "❌ Error setting language", show_alert=True)
            except Exception: pass

    @bot_obj.message_handler(commands=['help'])
    def handle_help(message):
        try:
            update_user_activity(message.from_user.id)
            if message.chat.type == 'private' and not check_subscription(message.from_user.id, bot_obj):
                send_subscription_message(message.chat.id, bot_obj)
                return
            text = (
              "Commands supported:\n"
                "/start - Show welcome message\n"
                "/lang  - Change language\n"
                "/mode  - Change result delivery mode\n"
                "/help  - This help message\n\n"
                "Send a voice/audio/video (up to 20MB for Telegram) and I will transcribe it.\n"
                "If it's larger than Telegram limits, you'll be provided a secure web upload link (supports up to 250MB) Need more help? Contact: @boyso20"
            )
            bot_obj.send_message(message.chat.id, text)
        except Exception:
            logging.exception("Error in handle_help")

    @bot_obj.message_handler(commands=['lang'])
    def handle_lang(message):
        try:
            if message.chat.type == 'private' and not check_subscription(message.from_user.id, bot_obj):
                send_subscription_message(message.chat.id, bot_obj)
                return
            kb = build_lang_keyboard("stt_lang")
            bot_obj.send_message(message.chat.id, "Choose your file language for transcription using the below buttons:", reply_markup=kb)
        except Exception:
            logging.exception("Error in handle_lang")

    @bot_obj.message_handler(commands=['mode'])
    def handle_mode(message):
        try:
            if message.chat.type == 'private' and not check_subscription(message.from_user.id, bot_obj):
                send_subscription_message(message.chat.id, bot_obj)
                return
            current_mode = get_user_send_mode(str(message.from_user.id))
            mode_text = "📄 .txt file" if current_mode == "file" else "💬 Split messages"
            bot_obj.send_message(message.chat.id, f"Result delivery mode: {mode_text}. Change it below:", reply_markup=build_result_mode_keyboard())
        except Exception:
            logging.exception("Error in handle_mode")

    @bot_obj.callback_query_handler(lambda c: c.data and c.data.startswith("stt_lang|"))
    def on_stt_language_select(call):
        try:
            if call.message.chat.type == 'private' and not check_subscription(call.from_user.id, bot_obj):
                send_subscription_message(call.message.chat.id, bot_obj)
                try: bot_obj.answer_callback_query(call.id)
                except Exception: pass
                return
            uid = str(call.from_user.id)
            _, lang_code = call.data.split("|", 1)
            lang_label = CODE_TO_LABEL.get(lang_code, lang_code)
            set_stt_user_lang(uid, lang_code)
            bot_obj.answer_callback_query(call.id, f"✅ Language set: {lang_label}")
            try: bot_obj.delete_message(call.message.chat.id, call.message.message_id)
            except Exception: pass
        except Exception:
            logging.exception("Error in on_stt_language_select")
            try: bot_obj.answer_callback_query(call.id, "❌ Error setting language", show_alert=True)
            except Exception: pass

    @bot_obj.callback_query_handler(lambda c: c.data and c.data.startswith("result_mode|"))
    def on_result_mode_select(call):
        try:
            if call.message.chat.type == 'private' and not check_subscription(call.from_user.id, bot_obj):
                send_subscription_message(call.message.chat.id, bot_obj)
                try: bot_obj.answer_callback_query(call.id)
                except Exception: pass
                return
            uid = str(call.from_user.id)
            _, mode = call.data.split("|", 1)
            set_user_send_mode(uid, mode)
            mode_text = "📄 .txt file" if mode == "file" else "💬 Split messages"
            try: bot_obj.delete_message(call.message.chat.id, call.message.message_id)
            except Exception: pass
            bot_obj.answer_callback_query(call.id, f"✅ Result mode set: {mode_text}")
        except Exception:
            logging.exception("Error in on_result_mode_select")
            try: bot_obj.answer_callback_query(call.id, "❌ Error setting result mode", show_alert=True)
            except Exception: pass

    @bot_obj.message_handler(content_types=['new_chat_members'])
    def handle_new_chat_members(message):
        try:
            if message.new_chat_members[0].id == bot_obj.get_me().id:
                group_data = {
                    '_id': str(message.chat.id), 'title': message.chat.title,
                    'type': message.chat.type, 'added_date': datetime.now()
                }
                groups_collection.update_one({'_id': group_data['_id']}, {'$set': group_data}, upsert=True)
                bot_obj.send_message(message.chat.id, "Thanks for adding me! I'm ready to transcribe your media files.")
        except Exception:
            logging.exception("Error in handle_new_chat_members")

    @bot_obj.message_handler(content_types=['left_chat_member'])
    def handle_left_chat_member(message):
        try:
            if message.left_chat_member.id == bot_obj.get_me().id:
                groups_collection.delete_one({'_id': str(message.chat.id)})
        except Exception:
            logging.exception("Error in handle_left_chat_member")

    @bot_obj.message_handler(content_types=['voice', 'audio', 'video', 'document'])
    def handle_media_types(message):
        try:
            if message.chat.type == 'private' and not check_subscription(message.from_user.id, bot_obj):
                send_subscription_message(message.chat.id, bot_obj)
                return
            handle_media_common(message, bot_obj, bot_token, bot_index)
        except Exception:
            logging.exception("Error in handle_media_types")

    @bot_obj.message_handler(content_types=['text'])
    def handle_text_messages(message):
        try:
            if message.chat.type == 'private' and not check_subscription(message.from_user.id, bot_obj):
                send_subscription_message(message.chat.id, bot_obj)
                return
            bot_obj.send_message(message.chat.id, "For Text to Audio Use @TextToSpeechBBot")
        except Exception:
            logging.exception("Error in handle_text_messages")

    @bot_obj.callback_query_handler(lambda c: c.data and c.data.startswith("get_key_points|"))
    def get_key_points_callback(call):
        instruction = "Summarize this text (lang={lang}) without adding any introductions, notes, or extra phrases."
        handle_ai_action(call, bot_obj, "get_key_points", instruction, lambda text: extract_key_points_offline(text, max_points=6))

    @bot_obj.callback_query_handler(lambda c: c.data and c.data.startswith("clean_up|"))
    def clean_up_callback(call):
        instruction = "Clean and normalize this transcription (lang={lang}). Remove ASR artifacts like [inaudible], repeated words, filler noises, timestamps, and incorrect punctuation. Produce a clean, well-punctuated, readable text in the same language. Do not add introductions or explanations."
        handle_ai_action(call, bot_obj, "clean_up", instruction, normalize_text_offline)


for idx, bot_obj in enumerate(bots):
    register_handlers(bot_obj, BOT_TOKENS[idx], idx)

# --- Webhook & Startup (Same as before) ---

@app.route("/", methods=["GET", "POST", "HEAD"])
def webhook_root():
    if request.method in ("GET", "HEAD"):
        bot_index = request.args.get("bot_index")
        try:
            bot_index_val = int(bot_index) if bot_index is not None else 0
        except Exception:
            bot_index_val = 0
        now_iso = datetime.utcnow().isoformat() + "Z"
        return jsonify({"status": "ok", "time": now_iso, "bot_index": bot_index_val}), 200
    if request.method == "POST":
        content_type = request.headers.get("Content-Type", "")
        if content_type and content_type.startswith("application/json"):
            raw = request.get_data().decode("utf-8")
            try:
                payload = json.loads(raw)
            except Exception:
                payload = None
            bot_index = request.args.get("bot_index")
            if not bot_index and isinstance(payload, dict):
                bot_index = payload.get("bot_index")
            header_idx = request.headers.get("X-Bot-Index")
            if header_idx:
                bot_index = header_idx
            try:
                bot_index_val = int(bot_index) if bot_index is not None else 0
            except Exception:
                bot_index_val = 0
            if bot_index_val < 0 or bot_index_val >= len(bots):
                return abort(404)
            try:
                update = telebot.types.Update.de_json(raw)
                bots[bot_index_val].process_new_updates([update])
            except Exception:
                logging.exception("Error processing incoming webhook update")
            return "", 200
    return abort(403)

@app.route("/set_webhook", methods=["GET", "POST"])
def set_webhook_route():
    results = []
    for idx, bot_obj in enumerate(bots):
        try:
            url = WEBHOOK_BASE.rstrip("/") + f"/?bot_index={idx}"
            bot_obj.delete_webhook()
            time.sleep(0.2)
            bot_obj.set_webhook(url=url)
            results.append({"index": idx, "url": url, "status": "ok"})
        except Exception as e:
            logging.error(f"Failed to set webhook for bot {idx}: {e}")
            results.append({"index": idx, "error": str(e)})
    return jsonify({"results": results}), 200

@app.route("/delete_webhook", methods=["GET", "POST"])
def delete_webhook_route():
    results = []
    for idx, bot_obj in enumerate(bots):
        try:
            bot_obj.delete_webhook()
            results.append({"index": idx, "status": "deleted"})
        except Exception as e:
            logging.error(f"Failed to delete webhook for bot {idx}: {e}")
            results.append({"index": idx, "error": str(e)})
    return jsonify({"results": results}), 200

def set_webhook_on_startup():
    for idx, bot_obj in enumerate(bots):
        try:
            bot_obj.delete_webhook()
            time.sleep(0.2)
            url = WEBHOOK_BASE.rstrip("/") + f"/?bot_index={idx}"
            bot_obj.set_webhook(url=url)
            logging.info(f"Main bot webhook set successfully to {url}")
        except Exception as e:
            logging.error(f"Failed to set main bot webhook on startup: {e}")

def set_bot_info_and_startup():
    set_webhook_on_startup()

if __name__ == "__main__":
    try:
        set_bot_info_and_startup()
        try:
            client.admin.command('ping')
            logging.info("Successfully connected to MongoDB!")
        except Exception as e:
            logging.error("Could not connect to MongoDB: %s", e)
    except Exception:
        logging.exception("Failed during startup")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
