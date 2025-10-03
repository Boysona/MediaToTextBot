MediaToTextBot is a powerful Telegram bot that provides free transcription and summarization services for voice messages, audio files, and videos. The bot supports multiple languages and offers both online and offline processing capabilities.

Features

· Multi-format Support: Handles voice messages, audio files (MP3, WAV, M4A, OGG, etc.), and video files (MP4, MKV, AVI, etc.)
· Multi-language Transcription: Supports 50+ languages including English, Arabic, Spanish, French, Russian, and many more
· Dual Transcription Engine: Uses both AssemblyAI API and offline speech recognition
· Large File Support: Web upload interface for files up to 250MB
· Text Cleaning & Summarization: AI-powered text normalization and key point extraction
· Multiple Delivery Modes: Send results as text files or split messages
· Multi-bot Support: Scalable architecture supporting multiple bot instances
· Admin Panel: Comprehensive administration and broadcasting capabilities
· MongoDB Integration: User and group management with activity tracking

Technical Architecture

Core Components

1. Flask Web Framework: Handles webhooks and file uploads
2. PyTelegramBotAPI: Telegram bot integration
3. SpeechRecognition Library: Offline transcription fallback
4. AssemblyAI API: Premium transcription service
5. FFmpeg: Audio/video processing and conversion
6. MongoDB: Data persistence for users, groups, and settings
7. ThreadPoolExecutor: Parallel processing for large files

Key Technical Features

· Chunked Processing: Splits large audio files into manageable chunks
· Retry Mechanism: Automatic retry with exponential backoff
· Memory Management: Efficient handling of large files with temp files
· Rate Limiting: Protection against API abuse
· Error Handling: Comprehensive exception handling with fallbacks

Configuration

Environment Variables

```bash
# Bot Configuration
BOT_TOKENS=token1,token2,token3
SECRET_KEY=your-secret-key-here
WEBHOOK_BASE=https://your-domain.com

# API Keys
ASSEMBLYAI_API_KEY=your-assemblyai-key
GEMINI_API_KEY=your-gemini-key

# Database
MONGO_URI=mongodb://localhost:27017
DB_NAME=telegram_bot_db

# Features
REQUIRED_CHANNEL=@yourchannel
ADMIN_USER_IDS=123456789,987654321

# Performance Tuning
CHUNK_DURATION_SEC=55
CHUNK_BATCH_SIZE=30
TRANSCRIBE_MAX_WORKERS=4
MAX_WEB_UPLOAD_MB=250
```

Supported Languages

The bot supports 50+ languages including:

· European: English, Spanish, French, German, Italian, Russian, etc.
· Asian: Chinese, Japanese, Korean, Hindi, Thai, Vietnamese, etc.
· Middle Eastern: Arabic, Persian, Hebrew, Turkish, Urdu, etc.
· African: Amharic, Swahili, Somali, Oromo, etc.

API Endpoints

Webhook Endpoints

· POST / - Main webhook for Telegram updates
· GET/POST /set_webhook - Configure webhooks
· GET/POST /delete_webhook - Remove webhooks

File Processing

· GET/POST /upload/<token> - Large file upload interface
· POST /assemblyai - Direct AssemblyAI transcription API

Administration

· GET /admin - Admin panel interface
· POST /admin/send_ads - Broadcast messages
· POST /admin/save_settings - Update admin settings

Database Schema

Collections

users

```javascript
{
  user_id: String,
  first_seen: DateTime,
  last_active: DateTime,
  stt_language: String,
  stt_send_mode: String,
  stt_conversion_count: Number
}
```

groups

```javascript
{
  _id: String (chat_id),
  title: String,
  type: String,
  added_date: DateTime
}
```

settings

```javascript
{
  _id: String,
  value: Mixed
}
```

Deployment

Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

Required Dependencies

· Flask
· pyTelegramBotAPI
· pymongo
· speechrecognition
· pydub
· requests
· itsdangerous

Usage Patterns

Basic Transcription Flow

1. User sends media file to bot
2. Bot validates file type and size
3. File is downloaded and processed
4. Transcription via AssemblyAI or offline recognition
5. Results delivered based on user preferences

Large File Handling

1. For files >20MB, provide secure upload link
2. User uploads via web interface
3. Background processing with progress updates
4. Results sent to Telegram chat

Development Notes

Adding New Languages

1. Update LANG_OPTIONS list with emoji, label, and code
2. Add language to appropriate speech recognition sets
3. Test with sample audio in the new language

Customizing Transcription

· Modify transcribe_via_selected_service() for engine selection
· Adjust CHUNK_DURATION_SEC for performance tuning
· Update ASSEMBLY_LANG_SET for API language support

Extending Features

· Add new media types in ALLOWED_EXTENSIONS
· Implement additional AI services in transcription pipeline
· Create new callback handlers for interactive features

Monitoring & Analytics

The bot includes:

· User activity tracking
· Conversion count statistics
· Group management analytics
· Admin panel with broadcast capabilities

Support & Contribution

For issues and contributions:

· Telegram: https://t.me/boyso20
· Repository: [GitHub Link]
· Documentation: This README

License

This project is open source and available under the MIT License.
