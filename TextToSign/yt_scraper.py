import requests
import urllib.parse
import json
import pytube
import os


def search_youtube(query, channel_id, max_results=None):
    encoded_query = urllib.parse.quote_plus(query)
    BASE_URL = "https://youtube.com"
    url = f"{BASE_URL}/results?search_query={encoded_query}&sp=EgIQAg%253D%253D"
    response = requests.get(url).text
    while "ytInitialData" not in response:
        response = requests.get(url).text
    results = parse_html(response, channel_id)
    if max_results is not None and len(results) > max_results:
        return results[:max_results]
    return results


def parse_html(response, channel_id):
    results = []
    start = response.index("ytInitialData") + len("ytInitialData") + 3
    end = response.index("};", start) + 1
    json_str = response[start:end]
    data = json.loads(json_str)

    try:
        contents = data["contents"]["twoColumnSearchResultsRenderer"][
            "primaryContents"
        ]["sectionListRenderer"]["contents"]
        for content in contents:
            if "itemSectionRenderer" in content:
                items = content["itemSectionRenderer"]["contents"]
                for item in items:
                    if "videoRenderer" in item:
                        video_data = item["videoRenderer"]
                        if (
                            "ownerText" in video_data
                            and "runs" in video_data["ownerText"]
                            and video_data["ownerText"]["runs"][0]
                            .get("navigationEndpoint", {})
                            .get("browseEndpoint", {})
                            .get("browseId")
                            == channel_id
                        ):
                            res = {
                                "id": video_data.get("videoId", None),
                                "title": video_data.get("title", {})
                                .get("runs", [[{}]])[0]
                                .get("text", None),
                                "url_suffix": video_data.get("navigationEndpoint", {})
                                .get("commandMetadata", {})
                                .get("webCommandMetadata", {})
                                .get("url", None),
                            }
                            results.append(res)
    except KeyError as e:
        print(f"Error parsing HTML response: {e}")

    return results


def download_top_result_video(query, channel_id="UCZy9xs6Tn9vWqN_5l0EEIZA"):
    hardcoded_query = query + "asl"
    videos = search_youtube(hardcoded_query, channel_id, max_results=10)

    if videos:
        video_id = videos[0]["id"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        yt = pytube.YouTube(video_url)
        stream = yt.streams.get_highest_resolution()
        stream.download()
        os.rename(f"{yt.title}.mp4", f"{query}.mp4")
        print(
            f"Successfully downloaded the top result video for '{query}' from the specified channel."
        )
        return True
    else:
        print(f"No search results found for '{query}' from the specified channel.")

    return False
