import asyncio
from flask import Flask, render_template, request, jsonify, send_file
from voicegroq import ConversationManager, TextToSpeech
import os

app = Flask(__name__, static_folder="static")
conversation_manager = ConversationManager()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(conversation_manager.transcribe())
    return jsonify(result)

@app.route('/respond', methods=['POST'])
def respond():
    user_input = request.json['user_input']
    response_text = conversation_manager.llm.process(user_input)
    
    # Generate the audio file and get its URL
    audio_filename = "response_audio.mp3"
    tts = TextToSpeech()
    file_path = tts.speak(response_text, filename=audio_filename)
    
    if file_path:
        audio_url = f"/static/{audio_filename}"
        # Return both the text response and audio URL as JSON
        return jsonify({"text": response_text, "audio_url": audio_url})
    else:
        # Return only the text response if audio generation fails
        return jsonify({"text": response_text})
    
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_file(os.path.join("static", filename))

if __name__ == "__main__":
    app.run(debug=True)