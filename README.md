Telegram Media transcriber Bot
This is a Telegram Bot designed to convert voice messages, audio files, and video files into text transcriptions. It utilizes various Speech-to-Text (STT) services such as Google Speech Recognition and AssemblyAI, and leverages the Gemini LLM for text cleanup and summarization.
Prerequisites
To run this bot, the following must be installed:
 * Python 3.x
 * FFmpeg: An essential tool for audio and video conversion. It must be installed and accessible in the system's execution path.
 * Python Libraries:
   pip install pyTelegramBotAPI Flask requests pymongo itsdangerous speechrecognition pydub pydub_ng google-genai

   Note: Some modules like subprocess, glob, tempfile, and wave are included in standard Python.
Configuration / Environment Variables
The bot relies on environment variables for sensitive keys and settings. The following must be set up:
| Variable | Description | Required? |
|---|---|---|
| BOT_TOKENS | Your Telegram Bot tokens, separated by a comma (e.g., token1,token2). | Yes |
| WEBHOOK_BASE | The base URL for your bot (e.g., https://yourdomain.com). | Yes |
| SECRET_KEY | A secret key used for signing tokens and sessions. | Yes |
| ADMIN_USER_IDS | Comma-separated user IDs for bot administrators. | No |
| MONGO_URI | MongoDB connection URI. | No (Defaults to local MongoDB) |
| DB_NAME | The name of the MongoDB database to use. | No (Default: telegram_bot_db) |
| REQUIRED_CHANNEL | The channel username or ID users must join to use the bot. | No |
| ASSEMBLYAI_API_KEY | AssemblyAI API key for STT. | No (Falls back to Google STT) |
| GEMINI_API_KEY | Gemini API key for transcription cleanup and summarization. | Yes (if cleanup/summarization is desired) |
| FFMPEG_BINARY | The path to the ffmpeg executable. | No (Attempts to find it in common locations) |
| TELEGRAM_MAX_BYTES | The maximum file size (in bytes) downloadable via the Telegram API (Default: 20MB). | No |
| MAX_WEB_UPLOAD_MB | The maximum file size (in MB) allowed for web uploads (Default: 250MB). | No |
| STT Settings | CHUNK_DURATION_SEC, RECOGNITION_MAX_RETRIES, etc. (See code for advanced STT operation settings). | No (Have default values) |
Bot Architecture and Features
The bot operates using a Webhook model with Flask handling incoming requests and the large file web upload page.
1. Speech-to-Text (STT) Capabilities
 * Media Support: Transcribes voice messages, audio files, and video files.
 * Conversion & Chunking: Uses FFmpeg to convert media to WAV format and splits it into smaller chunks for processing.
 * STT Services:
   * Google Speech Recognition: Used as a primary or fallback service.
   * AssemblyAI: Preferred if the API key is provided and the language is supported by AssemblyAI.
 * Multithreading: Transcription is handled in separate threads to avoid blocking the bot's Telegram message processing.
 * Language Support: Supports a wide array of languages (over 40), selectable by the user.
2. Gemini LLM Features
 * Transcription Cleanup (⭐️Clean transcript): Uses Gemini to remove ASR artifacts, filler words, noise, and normalize punctuation to produce a clean, readable text.
 * Summarization (Get Summarize): Uses Gemini to summarize the transcription and extract key points.
3. Large File Handling
 * Telegram Limit Bypass: If a file exceeds the Telegram download limit (e.g., 20MB), the user is provided a secure web upload link.
 * Secure Web Upload: A Flask-based web page allows file uploads up to MAX_WEB_UPLOAD_MB (Default: 250MB), which is then processed by the bot.
4. User and Group Management
 * MongoDB: Used to persistently store user activity, language settings, and group information.
 * Subscription Check: Enforces a mandatory channel join (REQUIRED_CHANNEL) before private chat usage.
5. Admin Functionality
 * Admin Panel: A simple web administration panel is available at /admin (requires ADMIN_PANEL_SECRET).
 * Broadcast: Allows administrators to send mass messages or media to all users or groups for announcements or advertisements.
How to Use
 * Start: Send /start to begin interaction and select your preferred transcription language.
 * Transcribe: Send a voice message, audio file, or video to the bot.
 * Change Settings:
   * Send /lang to change the STT language.
   * Send /mode to select the result delivery format (as a .txt file or as split messages).
 * Admin: Administrators can send /admin to get statistics and the link to the Admin Web Panel.
