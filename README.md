Telegram MediaToTextBot

A compact Telegram bot + Flask backend that accepts voice/audio/video files, transcribes them (using local speech_recognition or AssemblyAI), and returns cleaned transcripts to users. Designed to run as a webhook-enabled Flask app or locally for testing.

Key features
	•	Transcribe voice/audio/video from Telegram (or via secure web upload for large files).
	•	Choose transcription language & result delivery mode (file or split messages).
	•	Supports AssemblyAI (optional) and local speech_recognition fallback.
	•	Admin panel for broadcasts and basic stats.
	•	Per-user language and simple action buttons: clean / summarize.

Quick start
	1.	Install dependencies (example):

python -m venv venv
source venv/bin/activate
pip install flask pyTelegramBotAPI pymongo itsdangerous SpeechRecognition requests python-dotenv
# ffmpeg must be installed on the system (apt / brew / or provide FFMPEG_BINARY env)

	2.	Create .env (example):

BOT_TOKENS=@bot_token_1,@bot_token_2
ADMIN_USER_IDS=12345678
MONGO_URI=mongodb://localhost:27017
DB_NAME=telegram_bot_db
SECRET_KEY=your_secret_here
ADMIN_PANEL_SECRET=your_admin_secret
WEBHOOK_BASE=https://your.domain.com
ASSEMBLYAI_API_KEY=your_assemblyai_key    # optional
GEMINI_API_KEY=your_gemini_key            # optional (for cleaning/summarize)
FFMPEG_BINARY=/usr/bin/ffmpeg             # optional if ffmpeg on PATH
MAX_WEB_UPLOAD_MB=150
REQUIRED_CHANNEL=@yourchannel             # optional: restrict usage to channel members
PORT=8080

	3.	Run:

export $(cat .env | xargs)    # or set env vars by your method
python main.py                # or python <your_file>.py

	4.	(Optional) Set webhooks:

	•	Visit GET /set_webhook or call the route to set bot webhook URLs to WEBHOOK_BASE/?bot_index=<n>.

Important endpoints
	•	GET/POST / — webhook root (incoming Telegram updates are POSTed here).
	•	GET/POST /upload/<token> — secure web upload (for files larger than Telegram limit).
	•	GET /admin?secret=<ADMIN_PANEL_SECRET> — admin UI.
	•	POST /assemblyai — helper endpoint to transcribe via AssemblyAI.

Environment notes & requirements
	•	ffmpeg required for audio conversions/splitting. If not in PATH, set FFMPEG_BINARY.
	•	MongoDB is used for basic persistence (users/groups/settings).
	•	Telegram file download limits apply; large files are uploaded via the web UI.
	•	Provide valid Telegram bot tokens in BOT_TOKENS (comma-separated).

Troubleshooting
	•	If transcripts are empty or errors mention ffmpeg, ensure ffmpeg is installed and reachable.
	•	Check logs for AssemblyAI/Gemini API errors and network timeouts.
	•	For webhook issues, ensure WEBHOOK_BASE is a reachable HTTPS URL.



