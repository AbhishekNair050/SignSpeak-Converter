import requests
from bs4 import BeautifulSoup
import os


def download_video(query, output_dir=""):
    url = f"https://www.signasl.org/sign/{query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    meta_tag = soup.find("meta", itemprop="contentURL")
    if meta_tag is None:
        print("Meta tag with itemprop='contentURL' not found for query:", query)
        return False

    video_url = meta_tag.get("content")
    if not video_url:
        print("Video URL not found for query:", query)
        return False

    video_filename = query + ".mp4"
    with open(os.path.join(output_dir, video_filename), "wb") as f:
        video_response = requests.get(video_url)
        f.write(video_response.content)

    print("Video downloaded successfully!")
    return True
