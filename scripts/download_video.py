# Install yt-dlp first if not installed
# pip install yt-dlp

import yt_dlp


def download_youtube_video(youtube_url, output_path='./'):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',  # best video and audio
        'outtmpl': f'{output_path}%(title)s.%(ext)s',  # save as video title
        'merge_output_format': 'mp4',  # merge into mp4
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])


if __name__ == "__main__":
    # ⭐⭐ Put your YouTube link here
    video_url = "https://www.youtube.com/watch?v=9txrdKwBXJQ"

    # ⭐⭐ Optional: Choose where you want to save (default = current folder)
    save_folder = "data/downloaded_videos/"

    download_youtube_video(video_url, save_folder)
