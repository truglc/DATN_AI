import yt_dlp

url = "https://www.youtube.com/watch?v=2wDRmJWRjTc"

ydl_opts = {
    'format': 'mp4',
    'outtmpl': 'downloaded_video.%(ext)s'
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

print("Download complete!")