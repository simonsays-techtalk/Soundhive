# Soundhive
Soundhive is a Homassistant service and client that turns a raspberry pi and audio device into an mqtt, mediaplayer and tts streaming device. Current version is still very rudimentary. What works:

Soundhive Mediaplayer client
TTS speak and streaming audio/radio: 

service: media_player.play_media
data:
  entity_id: media_player.soundhive_media_player
  media_content_id: "http://icecast.omroep.nl/radio2-bb-mp3"
  media_content_type: "music"

When running the Soundhive mediaplayer for the first time, you must enter mqtt credentials.
