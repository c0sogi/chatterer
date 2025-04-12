import json
import unicodedata
import urllib.parse
from dataclasses import dataclass
from typing import Any, Optional, Self, cast

import requests


def get_youtube_video_details(
    query: str,
) -> list[dict[str, Optional[str]]]:
    """Search for video metadata on YouTube using the given query. Returns a list of dictionaries containing `video_id`, `title`, `channel`, `duration`, `views`, `publish_time`, and `long_desc`."""
    return [
        {
            "video_id": video_id,
            "title": video.title,
            "channel": video.channel,
            "duration": video.duration,
            "views": video.views,
            "publish_time": video.publish_time,
            "long_desc": video.long_desc,
        }
        for video in YoutubeSearchResult.from_query(base_url="https://youtube.com", query=query, max_results=10)
        if (video_id := _get_video_id(video.url_suffix))
    ]


def get_youtube_video_subtitle(video_id: str) -> str:
    """Get the transcript of a YouTube video using the given video ID."""

    from youtube_transcript_api import YouTubeTranscriptApi  # pyright: ignore[reportPrivateImportUsage]

    get_transcript = YouTubeTranscriptApi.get_transcript  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    list_transcripts = YouTubeTranscriptApi.list_transcripts  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    result: str = ""
    buffer_timestamp: str = "0s"
    buffer_texts: list[str] = []
    for entry in get_transcript(video_id, languages=(next(iter(list_transcripts(video_id))).language_code,)):  # pyright: ignore[reportUnknownVariableType]
        entry = cast(dict[object, object], entry)
        text: str = str(entry.get("text", "")).strip().replace("\n", " ")
        if not text:
            continue
        if len(buffer_texts) >= 10 or _is_special_char(text) or (buffer_texts and _is_special_char(buffer_texts[-1])):
            result += f"[{buffer_timestamp}] {'. '.join(buffer_texts)}\n"
            start = entry.get("start", 0)
            if start:
                buffer_timestamp = f"{start:.0f}s"
            buffer_texts = [text]
        else:
            buffer_texts.append(text)

    if buffer_texts:
        result += f"[{buffer_timestamp}] {' '.join(buffer_texts)}"
    return result


def _get_video_id(suffix: str) -> str:
    urllib_parse_result = urllib.parse.urlparse(suffix)
    if urllib_parse_result.path.startswith("/shorts/"):
        # Fore shorts (/shorts/...) the video ID is in the path
        parts = urllib_parse_result.path.split("/")
        if len(parts) < 3:
            print(f"Failed to get video ID from {suffix}")
            return ""
        return parts[2]

    query: str = urllib.parse.urlparse(suffix).query
    query_strings = urllib.parse.parse_qs(query)
    if "v" not in query_strings:
        print(f"Failed to get video ID from {suffix}")
        return ""
    return next(iter(query_strings["v"]), "")


def _is_special_char(text: str) -> bool:
    if not text:
        return False
    return not unicodedata.category(text[0]).startswith("L")


@dataclass
class YoutubeSearchResult:
    url_suffix: str
    id: Optional[str]
    thumbnails: list[str]
    title: Optional[str]
    long_desc: Optional[str]
    channel: Optional[str]
    duration: Optional[str]
    views: Optional[str]
    publish_time: Optional[str]

    @classmethod
    def from_query(cls, base_url: str, query: str, max_results: int) -> list[Self]:
        url: str = f"{base_url}/results?search_query={urllib.parse.quote_plus(query)}"
        response: str = requests.get(url).text
        while "ytInitialData" not in response:
            response = requests.get(url).text
        results: list[Self] = cls.parse_html(response)
        return results[:max_results]

    @classmethod
    def parse_html(cls, html: str) -> list[Self]:
        results: list[Self] = []
        start: int = html.index("ytInitialData") + len("ytInitialData") + 3
        end: int = html.index("};", start) + 1
        data: Any = json.loads(html[start:end])
        for contents in data["contents"]["twoColumnSearchResultsRenderer"]["primaryContents"]["sectionListRenderer"][
            "contents"
        ]:
            for video in contents["itemSectionRenderer"]["contents"]:
                if "videoRenderer" in video.keys():
                    video_data = video.get("videoRenderer", {})
                    suffix = (
                        video_data.get("navigationEndpoint", {})
                        .get("commandMetadata", {})
                        .get("webCommandMetadata", {})
                        .get("url", None)
                    )
                    if not suffix:
                        continue
                    res = cls(
                        id=video_data.get("videoId", None),
                        thumbnails=[
                            thumb.get("url", None) for thumb in video_data.get("thumbnail", {}).get("thumbnails", [{}])
                        ],
                        title=video_data.get("title", {}).get("runs", [[{}]])[0].get("text", None),
                        long_desc=video_data.get("descriptionSnippet", {}).get("runs", [{}])[0].get("text", None),
                        channel=video_data.get("longBylineText", {}).get("runs", [[{}]])[0].get("text", None),
                        duration=video_data.get("lengthText", {}).get("simpleText", 0),
                        views=video_data.get("viewCountText", {}).get("simpleText", 0),
                        publish_time=video_data.get("publishedTimeText", {}).get("simpleText", 0),
                        url_suffix=suffix,
                    )
                    results.append(res)

            if results:
                break
        return results


if __name__ == "__main__":
    print(get_youtube_video_details("BTS"))
    # print(get_youtube_transcript("y7jrpS8GHxs"))
    