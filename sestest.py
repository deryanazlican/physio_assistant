import pyttsx3

engine = pyttsx3.init()
engine.setProperty("rate", 160)
engine.setProperty("volume", 1.0)
engine.say("Test sesi bir iki üç")
engine.runAndWait()
print("bitti")