# 🧠 Talk to YouTube

A comprehensive YouTube video analysis tool powered by AI that can summarize videos, generate quizzes, create memes, and answer questions about video content in multiple languages.

## ✨ Features

### 🎯 Core Functionality
- **Multi-language transcript extraction** with automatic fallback (YouTube API → Auto-generated → Whisper)
- **AI-powered video summarization** using Mistral 7B model
- **Interactive Q&A** with context-aware responses
- **Semantic search** through video transcripts
- **Chat memory** for ongoing conversations

### 🎨 Creative Features
- **Quiz generation** from video content
- **Meme creation** using ImgFlip API and AI image generation
- **Text-to-speech** for summaries and answers
- **Multilingual support** with auto-translation

### 💾 Data Management
- **Vector embeddings** stored in Pinecone for fast similarity search
- **Automatic caching** to avoid reprocessing videos
- **Export options** for summaries, transcripts, and quizzes

## 🚀 Setup & Installation

### Prerequisites
- Python 3.8+
- FFmpeg (for audio processing)
- Valid API keys for required services

### 1. Clone and Install
```bash
git clone <repository-url>
cd youtube-llm-assistant
pip install -r requirements.txt
```

### 2. Environment Setup
Copy the `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

### 3. Required API Keys

#### 🤗 HuggingFace API Key (Required)
1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Add to `.env` as `HUGGINGFACE_API_KEY`

#### 📌 Pinecone API Key (Required)
1. Sign up at [Pinecone](https://app.pinecone.io/)
2. Create a new project
3. Get your API key from the dashboard
4. Add to `.env` as `PINECONE_API_KEY`

#### 🎭 ImgFlip Credentials (Optional)
1. Register at [ImgFlip](https://imgflip.com/signup)
2. Get API access from [ImgFlip API](https://imgflip.com/api)
3. Add username and password to `.env`

### 4. FFmpeg Installation

#### Windows
```bash
# Using chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

#### macOS
```bash
brew install ffmpeg
```

#### Linux
```bash
sudo apt update
sudo apt install ffmpeg
```

## 🏃‍♂️ Usage

### Start the Application
```bash
streamlit run app.py
```

### Basic Workflow
1. **Enter YouTube URL** in the input field
2. **Processing** happens automatically:
   - Transcript extraction
   - AI summarization
   - Content indexing
3. **Interact** with the generated content:
   - Read summaries with TTS
   - Take generated quizzes
   - View AI-generated memes
   - Search transcript content
4. **Ask questions** about the video content
5. **Download** results as text files

## 🔧 Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HUGGINGFACE_API_KEY` | ✅ | - | HuggingFace API key |
| `PINECONE_API_KEY` | ✅ | - | Pinecone vector database key |
| `MISTRAL_MODEL` | ❌ | `mistralai/Mistral-7B-Instruct-v0.1` | LLM model to use |
| `PINECONE_ENV` | ❌ | `us-east-1` | Pinecone environment |
| `INDEX_NAME` | ❌ | `youtube-assistant` | Pinecone index name |
| `IMGFLIP_USERNAME` | ❌ | - | ImgFlip username for memes |
| `IMGFLIP_PASSWORD` | ❌ | - | ImgFlip password for memes |

### Model Configuration
The app uses several AI models:
- **Mistral 7B**: For text generation and Q&A
- **Whisper Base**: For audio transcription
- **SentenceTransformers**: For text embeddings
- **Google Translator**: For multilingual support

## 📊 Technical Architecture

### Data Flow
1. **Input**: YouTube URL
2. **Transcript Extraction**: 
   - YouTube API (preferred)
   - Auto-generated captions
   - Whisper transcription (fallback)
3. **Text Processing**:
   - Chunking with LangChain
   - Embedding generation
   - Vector storage in Pinecone
4. **AI Processing**:
   - Summarization with Mistral
   - Q&A with RAG (Retrieval-Augmented Generation)
   - Quiz and meme generation

### Components
- **Frontend**: Streamlit web interface
- **Vector DB**: Pinecone for semantic search
- **LLM**: Mistral 7B via HuggingFace
- **Speech**: Whisper for transcription, gTTS for synthesis
- **Translation**: Google Translator API

## 🎨 Features Deep Dive

### 📝 Summarization
- Processes up to 2000 characters of transcript
- Generates concise 2-3 sentence summaries
- Available in original language and English

### 🧠 Quiz Generation
- Creates multiple-choice questions
- Based on video content understanding
- Downloadable as text file

### 🎭 Meme Creation
- **Traditional memes**: Using ImgFlip templates
- **AI-generated memes**: Using DALL-E style models
- Contextual captions based on video content

### 💬 Interactive Chat
- Context-aware responses
- Multilingual question support
- Chat history maintenance
- Semantic search for relevant content

## 🔍 Troubleshooting

### Common Issues

**"No transcript found"**
- Video might not have captions
- Try videos with English audio
- Check if video is publicly accessible

**"API rate limit exceeded"**
- Wait a few minutes before retrying
- Consider upgrading API plans
- Check API key validity

**"Audio processing failed"**
- Ensure FFmpeg is installed
- Check internet connection
- Verify video URL format

**"Vector storage failed"**
- Verify Pinecone API key
- Check Pinecone service status
- Ensure index exists

### Performance Tips
- Use shorter videos for faster processing
- Enable caching to avoid reprocessing
- Consider using more powerful models for better results

## 📈 Future Enhancements

### Planned Features
- 📺 **Video timeline navigation**
- 🎬 **Batch video processing**
- 📊 **Analytics dashboard**
- 🔄 **Real-time streaming support**
- 🌐 **More language models**
- 📱 **Mobile app version**

### Contributing
Contributions are welcome! Please feel free to submit issues and enhancement requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI** for Whisper
- **HuggingFace** for model hosting
- **Pinecone** for vector database
- **Streamlit** for the web framework
- **YouTube API** for transcript access

---

**Made with ❤️ for the AI community**