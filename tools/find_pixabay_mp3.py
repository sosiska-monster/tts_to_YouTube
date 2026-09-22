import urllib.request
import re

url = "https://pixabay.com/sound-effects/laughing-man-117725/"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
html = urllib.request.urlopen(req).read().decode("utf-8", errors="ignore")
mp3s = re.findall(r"https://[^\s\"']+\.mp3[^\s\"']*", html)
print("Found MP3s:", set(mp3s))
