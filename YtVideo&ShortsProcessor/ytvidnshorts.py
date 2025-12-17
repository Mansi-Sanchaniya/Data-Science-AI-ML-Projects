import streamlit as st
import torch
import yt_dlp
from moviepy.video.compositing.concatenate import concatenate_videoclips
from pytube import Playlist
from sklearn.metrics.pairwise import cosine_similarity
from streamlit import session_state
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.feature_extraction.text import TfidfVectorizer
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
import io
import os
import re
import cv2
import nltk
import tempfile
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from yt_dlp import YoutubeDL
from transformers import T5ForConditionalGeneration, T5Tokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# Initialize Sentence Transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# Load a summarization model (optional for further shortening)
summarizer = pipeline("summarization")

model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Function to convert cookies in JSON or TXT format to Netscape format
def convert_cookies_to_netscape(cookies_file):
    try:
        if cookies_file.name.endswith('.json'):
            cookies = json.load(cookies_file)
        elif cookies_file.name.endswith('.txt'):
            cookies = []
            for line in cookies_file.readlines():
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split('\t')
                    cookies.append({
                        'domain': parts[0],
                        'name': parts[5],
                        'value': parts[6],
                        'path': parts[2],
                        'expirationDate': parts[4] if len(parts) > 4 else None
                    })
        else:
            raise ValueError("Invalid cookie file format. Only JSON or TXT are supported.")

        # Convert cookies to Netscape format
        netscape_cookies = []
        for cookie in cookies:
            expiry = cookie.get('expiry', '')
            if expiry:
                expiry = str(int(expiry))
            netscape_cookies.append(
                f"{cookie['domain']}\tTRUE\t{cookie['path']}\t{expiry}\t{cookie['name']}\t{cookie['value']}"
            )

        return netscape_cookies

    except Exception as e:
        st.error(f"Error converting cookies: {str(e)}")
        return None


# Function to download a YouTube video using yt-dlp and a cookie file
def download_video(url, cookie_file=None):
    download_status = ""  # Initialize download_status to avoid referencing undefined variable
    downloaded_video_path = None

    # Define the directory to store the video
    temp_dir = "temp_video"  # Name of the temporary directory

    # Create the directory if it doesn't exist
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)  # Create the directory

    # Set up yt-dlp options, saving video to the temp directory
    ydl_opts = {
        'outtmpl': os.path.join(temp_dir, '%(title)s_%(id)s.%(ext)s'),  # Save video to the temp directory
    }

    # Add cookie file to yt-dlp options if provided
    if cookie_file:
        ydl_opts['cookiefile'] = cookie_file

    # Use yt-dlp to download the video
    with YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            download_status = "Video downloaded successfully!"  # Set the success status
            # Prepare the full path of the downloaded video (fixing path issue)
            downloaded_video_path = os.path.join(f"{ydl.prepare_filename(ydl.extract_info(url, download=False))}")
            st.success(download_status)
        except Exception as e:
            download_status = f"Error downloading video: {str(e)}"  # Set the error message
            st.error(download_status)  # Display error message

    return download_status, downloaded_video_path  # Return the status for further checking if needed


# Function to get video URLs from multiple playlists or individual video links
def get_video_urls_multiple(input_urls):
    video_urls = []
    urls = input_urls.split(",")  # Split input by comma
    for url in urls:
        url = url.strip()  # Remove any leading/trailing spaces
        if "playlist" in url:
            playlist = Playlist(url)
            video_urls.extend(playlist.video_urls)  # Add all video URLs in the playlist
        else:
            video_urls.append(url)  # Treat as a single video URL
    return video_urls


# Function to get transcript for a video using its YouTube ID
def get_transcript(video_url):
    video_id = video_url.split("v=")[-1]
    try:
        # Fetch the transcript (if available)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        return None


# Function to format the transcript into a readable form
def format_transcript(transcript):
    formatted_transcript = []
    for entry in transcript:
        start_time = entry['start']  # Timestamp
        duration = entry['duration']
        text = entry['text']  # Transcript text
        formatted_transcript.append(f"[{start_time}s - {start_time + duration}s] {text}")
    return formatted_transcript


# Function to process input (multiple playlists or individual videos) and fetch transcripts for all videos
def process_input(input_urls):
    video_urls = get_video_urls_multiple(input_urls)
    if not video_urls:
        return []

    all_transcripts = []  # List to hold all transcripts

    video_chunks = {}  # Dictionary to store video-specific transcripts

    # Use another ThreadPoolExecutor to fetch transcripts concurrently
    with ThreadPoolExecutor() as transcript_executor:
        future_to_video = {transcript_executor.submit(get_transcript, video_url): video_url for video_url in video_urls}
        for idx, future in enumerate(as_completed(future_to_video)):
            video_url = future_to_video[future]
            try:
                transcript = future.result()
                if transcript:
                    formatted_transcript = format_transcript(transcript)
                    video_chunks[video_url] = formatted_transcript  # Store by video URL
                else:
                    video_chunks[video_url] = ["Transcript not available"]
            except Exception as e:
                video_chunks[video_url] = ["Transcript extraction failed"]
                print(f"Error getting transcript for {video_url}: {e}")

    # Reassemble the output in the original order of video URLs
    for video_url in video_urls:
        all_transcripts.append(
            {"video_url": video_url, "transcript": video_chunks.get(video_url, ["No transcript found"])})
    return all_transcripts

exclude_words = {"promotion", "offer", "limited-time" "buy", "buy now", "ad", "discount", "sale"}

def remove_stop_words(text):
    """Remove stop words from the text."""
    words = text.split()
    return " ".join([word for word in words if word.lower() not in stop_words])

def remove_stop_words_and_excluded_words(texts, exclude_words):
    """Remove stop words and excluded words from a batch of texts."""
    filtered_texts = []
    for text in texts:
        text = remove_stop_words(text)  # Remove stop words
        words = text.lower().split()
        filtered_words = [word for word in words if word not in exclude_words]
        filtered_texts.append(" ".join(filtered_words))
    return filtered_texts


def is_relevant_to_query(snippet, query, threshold=0.35):
    """Check if the snippet is relevant to the query using keyword matching."""

    # Ensure query is a string
    if isinstance(query, list):
        query = " ".join(query)  # Join list into a single string

    query_words = set(query.lower().split())
    snippet_words = set(snippet.lower().split())

    common_words = query_words.intersection(snippet_words)
    return len(common_words) / len(query_words) > threshold


def is_relevant_to_query_embedding(snippet, query, model, threshold=0.35):
    """Check if the snippet is relevant to the query using semantic similarity."""
    query_embedding = model.encode(query)
    snippet_embedding = model.encode(snippet)

    similarity = cosine_similarity([query_embedding], [snippet_embedding])
    return similarity[0][0] > threshold  # Check if similarity score exceeds the threshold

def process_query(query, stored_transcripts, threshold=0.3):
    if not query:
        print("Please enter a query to search in the transcripts.")
        return []

    if not stored_transcripts:
        print("No transcripts available. Please process a playlist or video first.")
        return []

    # Encode the query once
    inputs = tokenizer.encode(query, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    query_embedding = model.encoder(inputs).last_hidden_state.mean(dim=1).detach().cpu().numpy()

    all_transcripts_text = []
    transcript_lines = []
    timestamps = []
    video_urls = []

    for video in stored_transcripts:
        video_url = video['video_url']
        if isinstance(video['transcript'], list):
            for line in video['transcript']:

                if isinstance(line, str):
                    # Example line format: "[2.03s - 4.54s] some text"
                    parts = line.split(']')  # Split at the closing bracket
                    # Ensure the line has the expected number of parts
                    if len(parts) > 1:
                        time_text = parts[0].strip('[')  # Extract the time part by removing '['
                        text = parts[1].strip()
                        # Extract start_time and end_time from time_text
                        time_parts = time_text.split('-')  # Split at the dash
                    try:
                        start_time = float(time_parts[0].replace('s', '').strip())  # Convert to float and remove 's'
                        end_time = float(time_parts[1].replace('s', '').strip())  # Convert to float and remove 's'
                    except ValueError:
                        # Default to 0.0 if conversion fails
                        start_time = 0.0
                        end_time = 0.0

                    # Create a dictionary for the line
                    line = {'text': text, 'start_time': start_time, 'end_time': end_time}
                # Check if the transcript line is a dictionary with 'text', 'start_time', and 'end_time'

                if isinstance(line, dict) and 'text' in line and 'start_time' in line and 'end_time' in line:
                    text = line['text']
                    start_time = line['start_time']  # Ensure it's a float if it's not
                    end_time = line['end_time']

                else:
                    # If the line does not match the expected format, assign default values
                    start_time = 0.0
                    end_time = 0.0
                    text = line  # The entire line is treated as text
                all_transcripts_text.append(f"{video_url} [{start_time} - {end_time}] {text}")
                transcript_lines.append(text)
                timestamps.append((start_time, end_time))
                video_urls.append(video_url)

    # Remove stop words and excluded words from the entire batch of transcript lines
    transcript_lines = remove_stop_words_and_excluded_words(transcript_lines, exclude_words)

    # Encode all transcript lines in one batch
    transcript_inputs = tokenizer(transcript_lines, padding=True, truncation=True, return_tensors="pt",max_length=512)
    transcript_embeddings = model.encoder(transcript_inputs['input_ids']).last_hidden_state.mean(dim=1).detach().cpu().numpy()

    cosine_similarities = cosine_similarity(query_embedding, transcript_embeddings)

    # Collect relevant snippets based on cosine similarity
    relevant_snippets = []
    for idx, score in enumerate(cosine_similarities[0]):
        if score > threshold:
            relevant_snippets.append((video_urls[idx], transcript_lines[idx], timestamps[idx]))

    # Sort the relevant snippets by timestamps for coherent output
    relevant_snippets.sort(key=lambda x: x[2][0])

    final_snippets = []
    current_snippet = ""
    current_start_time = None
    current_end_time = None
    current_video_url = None

    for (video_url, snippet, (start_time, end_time)) in relevant_snippets:
        if is_relevant_to_query(snippet, query) or is_relevant_to_query_embedding(snippet, query,embedding_model):
            if current_end_time is None or start_time - current_end_time < 1.5:  # Merge if close enough (2s threshold)
                if current_video_url is None:
                    current_video_url = video_url  # Initialize the video URL for the first snippet
                current_snippet += " " + snippet  # Append the summarized snippet
                current_end_time = end_time  # Extend the end time
            else:
                final_snippets.append((current_video_url, current_snippet.strip(), current_start_time, current_end_time))

                # Start a new snippet
                current_snippet = snippet
                current_start_time = start_time
                current_end_time = end_time
                current_video_url = video_url  # Track video_url with new snippet

    # Summarize and append the last remaining snippet
    if current_snippet:
        final_snippets.append((current_video_url, current_snippet.strip(), current_start_time, current_end_time))



    # Ensure that start_time and end_time are valid floats before formatting
    formatted_relevant_snippets = []
    for (video_url, snippet, start_time, end_time) in final_snippets:
        # If start_time or end_time are None, set them to 0.0 (or another default value)
        start_time = start_time if start_time is not None else 0.0
        end_time = end_time if end_time is not None else 0.0
        formatted_relevant_snippets.append(
            f"Video: {video_url}\n[{start_time:.2f}s - {end_time:.2f}s] {snippet}")

    return formatted_relevant_snippets


def extract_timestamps_from_section(section):
    try:
        # Strip any leading/trailing whitespaces
        section = section.strip()

        # Check if the section contains timestamp information in the correct format
        if '[' not in section or ']' not in section:
            return None  # Skip sections that do not contain timestamps in '[start_time - end_time]' format

        # Extract the timestamp part of the section (the part inside the brackets)
        timestamp_part = section[section.find('[') + 1:section.find(']')].strip()  # Extract content inside brackets
        times = timestamp_part.split(" - ")

        if len(times) != 2:
            return None  # Return None to skip this section

        # Clean timestamps and remove any unnecessary decimal precision
        start_time = float(times[0].strip().replace("s", ""))
        end_time = float(times[1].strip().replace("s", ""))

        start_time = round(start_time, 2)
        end_time = round(end_time, 2)

        return start_time, end_time
    except Exception as e:
        print(f"Error extracting timestamps from section '{section}'. Exception: {e}")
        return None  # Return None in case of an error


def extract_video_segments(input_string):

    # This pattern looks for YouTube URLs with timestamps in the format of [start_time - end_time]
    pattern = r"(https://www\.youtube\.com/watch\?v=[\w-]+(?:&t=\d+s)?)\s*\[([\d\.]+s)\s*-\s*([\d\.]+s)\]"
    # This pattern matches the format of YouTube URLs and timestamps

    # Find all matching segments
    matches = re.findall(pattern, input_string)

    video_segments = []
    last_end_time = 0

    # For each match, process the video URL and timestamps
    for match in matches:
        url, start, end = match
        start_time = float(start[:-1])  # Remove the 's' and convert to float
        end_time = float(end[:-1])  # Remove the 's' and convert to float
        # Ensure that the current segment does not overlap with the previous one
        if start_time < last_end_time:
            start_time = last_end_time  # Update the start time to avoid overlap

        # Update last_end_time to the end of the current segment
        last_end_time = end_time
        video_segments.append((url, start_time, end_time))

    return video_segments


# Helper: Extract short clip
def create_short_clip(video_path, transcript, keyword, min_duration=15, max_duration=60):
    # Load the video file using moviepy
    video = VideoFileClip(video_path)
    clips = []
    occurrences = []

    total_duration = 0  # Track total duration of selected clips
    max_padding = 7  # Initial padding value for the end

    for segment in transcript:
        # Parse the segment string to extract start, end, and text
        match = re.match(r'\[(\d+\.\d+)s\s*-\s*(\d+\.\d+)s\]\s*(.+)', segment)
        if match:
            start = float(match.group(1))  # Extract start time
            end = float(match.group(2))  # Extract end time
            text = match.group(3).strip()  # Extract text content

            # Normalize text and keyword for comparison
            normalized_text = " ".join(text.split())
            normalized_keyword = " ".join(keyword.split())

            # Check if the normalized keyword exists in normalized text
            if normalized_keyword.lower() in normalized_text.lower():
                # Add padding to start and end times
                start = max(0.0, start - 3)  # 5 seconds padding before the clip
                end = min(start + max_duration, end + max_padding)  # Add padding to the end

                duration = end - start
                # Adjust duration if needed
                if duration > max_duration:
                    end = start + max_duration  # Trim the end if it's too long
                elif duration < min_duration:
                    start = end - min_duration  # Trim the start if it's too short


                final_duration = end - start

                # Ensure the duration is within the allowed range
                if min_duration <= final_duration <= max_duration:
                    # Check if the clip already exists in the list
                    if not any(occ['start'] == start and occ['end'] == end for occ in occurrences):
                        # Check for partial overlap with previous clips
                        for prev_clip in clips:
                            prev_start, prev_end = prev_clip[0], prev_clip[1]

                            # Check for overlap (partial or complete)
                            if not (end <= prev_start or start >= prev_end):
                                print(
                                    f"Partial overlap detected: Clip {start}-{end} overlaps with {prev_start}-{prev_end}")
                                # Adjust the start or end time to avoid overlap
                                if start < prev_end:
                                    start = prev_end  # Shift start to avoid overlap
                                if end > prev_start:
                                    end = prev_start  # Shift end to avoid overlap
                                break  # Only adjust once, don't check more clips

                        # Adjust clip duration
                        final_duration = end - start
                        if final_duration > max_duration:
                            end = start + max_duration  # Trim if too long
                        elif final_duration < min_duration:
                            start = end - min_duration  # Trim if too short

                        # Ensure the duration is valid
                        if min_duration <= final_duration <= max_duration:
                            # Extract the short clip and resize it
                            short_clip = video.subclip(start, end).resize(height=1080, width=1920)
                            clips.append((start, end, short_clip))  # Append the clip to the list
                            occurrences.append({
                                'start': start,
                                'end': end,
                                'text': text,
                                'duration': final_duration
                            })
                            print(f"Adding clip: Start: {start}, End: {end}, Text: {text}")
                        else:
                            print(f"Clip too short or too long after adjustment: Start: {start}, End: {end}")
                    else:
                        print(f"Duplicate clip detected: Start: {start}, End: {end}")

    # Combine the clips
    if clips:
        # Remove duplicate clips by ensuring unique start and end times
        unique_clips = []
        seen = set()

        for occ in occurrences:
            key = (occ['start'], occ['end'])
            if key not in seen:
                seen.add(key)
                unique_clips.append(occ)

        # Create unique clip list
        clips = [video.subclip(occ['start'], occ['end']).resize(height=1080, width=1920) for occ in unique_clips]

        # Combine all clips
        combined_clip = concatenate_videoclips(clips, method="compose")  # Use compose for compatibility
        combined_duration = combined_clip.duration

        print(f"Total Combined Duration: {combined_duration} seconds")

        # Trim the combined video to 60 seconds if it's too long
        if combined_duration > max_duration:
            combined_clip = combined_clip.subclip(0, max_duration)
            print(f"Trimmed Combined Duration: {combined_clip.duration} seconds")

        return combined_clip
    else:
        print("No valid clips found for the given keyword.")
        return None

import moviepy.editor as mp
from moviepy.editor import VideoFileClip

def clip_and_merge_videos(segments, downloaded_video_paths, output_filename):
    temp_dir = "temp_vid"
    total_duration = 0

    # Create the directory if it doesn't exist
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)  # Create the directory

    # Full output path for the final video
    output_path = os.path.join(temp_dir, output_filename)
    temp_clips = []

    # Extract video IDs from the downloaded video paths
    video_id_to_path = {}
    for path in downloaded_video_paths:
        # Extract video ID from the filename assuming format "temp_video\\<title>_<video_id>.mp4"
        title_with_id = os.path.basename(path).replace(".mp4", "")
        video_id_match = re.search(r"[A-Za-z0-9_-]{11}$", title_with_id)  # Match video ID at the end
        if video_id_match:
            video_id = video_id_match.group()
            video_id_to_path[video_id] = path

    # Process each segment
    for segment in segments:
        url, start_time, end_time = segment

        # Extract video ID from the URL
        video_id_match = re.search(r"v=([A-Za-z0-9_-]{11})", url)
        if not video_id_match:
            raise ValueError(f"Invalid YouTube URL: {url}")

        video_id = video_id_match.group(1)

        # Check if the video ID exists in the downloaded paths
        if video_id not in video_id_to_path:
            raise FileNotFoundError(f"No downloaded video matches the URL: {url}")

        downloaded_video_path = video_id_to_path[video_id]

        # Ensure the file exists before proceeding
        if not os.path.exists(downloaded_video_path):
            raise FileNotFoundError(f"Video file not found at: {downloaded_video_path}")


        # Using moviepy to clip both audio and video
        video_clip = mp.VideoFileClip(downloaded_video_path)
        # Ensure that the end_time does not exceed the video's duration
        # Ensure that start_time and end_time are within the video duration
        video_duration = video_clip.duration
        start_time = min(start_time, video_duration)
        end_time = min(end_time, video_duration)

        if start_time >= end_time:
            raise ValueError(f"Invalid time range: start_time ({start_time}) should be less than end_time ({end_time})")

        video_clip = video_clip.subclip(start_time, end_time)
        min_clip_duration = 1.0
        # Check the duration of the clip before adding it to the list
        clip_duration = video_clip.duration
        if clip_duration < min_clip_duration:
            st.warning(
                f"Skipping clip with duration {clip_duration} seconds, as it's below the minimum threshold of {min_clip_duration} seconds.")
            video_clip.close()
            continue  # Skip this clip if its duration is too small

        temp_clips.append(video_clip)
        clip_duration = video_clip.duration
        total_duration += clip_duration

    # Convert total duration to minutes
    total_duration_minutes = total_duration / 60

    # Combine all clips into a final video (both video and audio)
    if temp_clips:
        final_clip = mp.concatenate_videoclips(temp_clips)

        # Write the final video with audio
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=24)

        # Clean up temporary clips
        for clip in temp_clips:
            clip.close()
        # Clean up the downloaded video after merging
        if os.path.exists(downloaded_video_path):
            os.remove(downloaded_video_path)

        return output_path  # Return the path to the merged video
    else:
        st.text("No clips to merge")
        return "No clips to merge"

def main():
    st.set_page_config(page_title="Video & Playlist Processor", page_icon="🎬", layout="wide")

    st.title("🎥 Videos & Shorts Creator")

    st.markdown("""
    <style>
        .css-1d391kg {padding: 30px;}
        .stTextArea>div>div>textarea {
            font-size: 14px;
            line-height: 1.8;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
        }
        .stButton>button {
            background-color: #ff5c5c;
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #ff7d7d;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state if not already present
    if 'selected_option' not in st.session_state:
        st.session_state.selected_option = 'Choose an option'  # Default option

    # Dropdown menu for user choice
    options = ["Choose an option", "Create Shorts", "Combine Videos"]
    choice = st.selectbox("What would you like to do?", options)

    # Update session state when the user selects an option
    if choice != st.session_state.selected_option:
        st.session_state.selected_option = choice

    if st.session_state.selected_option == "Choose an option":
        st.text('Please choose your desired option to proceed')


    # Step 3: Process and create shorts
    if st.session_state.selected_option == "Create Shorts":
        # Streamlit App
        st.header("Shorts Creator 🎬")

        # Step 1: Enter YouTube URL
        youtube_url = st.text_input("Enter the YouTube video URL:")
        # Step 2: Enter keywords/phrases to create shorts
        st.text("Enter keywords/phrases to create shorts:")

        # Handle multiple keyword inputs dynamically
        if "keywords" not in st.session_state:
            st.session_state.keywords = [""]  # Start with one input field for keywords

        def add_keyword():
            st.session_state.keywords.append("")

        if st.button("Add another keyword"):
            add_keyword()

        for i, keyword in enumerate(st.session_state.keywords):
            st.session_state.keywords[i] = st.text_input(f"Keyword {i + 1}:", value=keyword, key=f"keyword_{i}")

        min_duration = 15
        max_duration = 60

        # Step 3: Process and create shorts
        if st.button("Create Shorts"):
            with st.spinner("Processing..."):
                try:
                    download_status, video_path = download_video(youtube_url)

                    # Fetch transcript
                    fetch_transcripts = get_transcript(youtube_url)
                    transcript = format_transcript(fetch_transcripts)

                    # Create a short for each keyword
                    created_shorts = []
                    for i, keyword in enumerate(st.session_state.keywords, start=1):
                        if keyword.strip():  # Ensure keyword is not empty
                            short_clip = create_short_clip(video_path, transcript, keyword, min_duration, max_duration)
                            if short_clip:
                                output_file = f"short_{i}.mp4"
                                short_clip.write_videofile(output_file, codec="libx264")
                                created_shorts.append(output_file)
                                st.success(f"Short created for keyword: {keyword}")
                                st.video(output_file)
                            else:
                                st.warning(f"No matching segment found for: {keyword}")

                    if created_shorts:
                        st.success("All shorts have been processed!")
                    else:
                        st.error("No shorts could be created. Check the keywords or video content.")

                except Exception as e:
                    st.error(f"An error occurred: {e}")

    elif st.session_state.selected_option == "Combine Videos":
        st.header("Contextual Video Clipper ✂️")

        input_urls = st.text_input(
            "Enter YouTube Playlist(s) or Video URL(s) or both (comma-separated): \n\n Example of link: https://www.youtube.com/watch?v=abc123xyz or https://www.youtube.com/playlist?list=xyz456abc")

        if 'stored_transcripts' not in st.session_state:
            st.session_state.stored_transcripts = []
        if 'transcript_text' not in st.session_state:
            st.session_state.transcript_text = ""

        # Initialize session state for queries if not already initialized
        if "queries" not in st.session_state:
            st.session_state["queries"] = [""]  # Start with one query box
        if "query_output" not in st.session_state:
            st.session_state["query_output"] = {}  # Store results for each query

        # Dynamic query handling
        st.text("Enter your query to extract relevant information:")

        # Add Another Query button to dynamically add more query input boxes
        if st.button("Add Another Query"):
            st.session_state["queries"].append("")  # Add an empty query placeholder

        queries = [st.text_input(f"Query {i + 1}", key=f"query_{i}") for i in range(len(st.session_state["queries"]))]


        if st.button('Combine and Play'):
            with st.spinner('Processing'):

                if input_urls:
                    st.session_state.stored_transcripts = process_input(input_urls)
                    st.success("Transcripts extracted successfully.")
                    if st.session_state.stored_transcripts:
                        transcript_text = ""
                        for video in st.session_state.stored_transcripts:
                            transcript_text += f"\nTranscript for video {video['video_url']}:\n"
                            if isinstance(video['transcript'], list):
                                for line in video['transcript']:
                                    transcript_text += line + "\n"
                            else:
                                transcript_text += video['transcript'] + "\n"
                            transcript_text += "-" * 50 + "\n"
                        st.session_state.transcript_text = transcript_text



                    st.session_state.query_list = [q for q in queries if q.strip()]

                    if not st.session_state.query_list:
                        st.error("Please provide at least one valid query.")

                    # Process each query independently
                    for query in st.session_state.query_list:
                        if query not in st.session_state["query_output"]:  # Skip already processed queries
                            # Simulate transcript fetching and processing for the query
                            results = process_query(query, st.session_state.stored_transcripts)

                            if results:
                                st.session_state["query_output"][query] = results
                                st.success("Query Processed Sucessfully")
                            else:
                                st.session_state["query_output"][query] = "No relevant content found for the query."

                downloaded_video_paths = []

                # Ensure that `input_urls` is set and split correctly
                if input_urls:
                    for url in input_urls.split(","):
                        url = url.strip()
                        # Call the download_video function to download the video and get the path
                        download_status, downloaded_video_path = download_video(url)
                        if downloaded_video_path:
                            downloaded_video_paths.append(downloaded_video_path)

                if 'query_output' in st.session_state and st.session_state.query_output:
                    for query, query_output in st.session_state.query_output.items():

                        # Ensure output is a string before passing it to extract_video_segments
                        if isinstance(query_output, dict) or isinstance(query_output, list):
                            output = '\n'.join(query_output)  # Convert dict or list to a string

                            # Extract video segments from query output
                            try:
                                video_segments = extract_video_segments(output)
                            except Exception as e:
                                st.error(f"Error extracting video segments for query '{query}': {str(e)}")
                                continue  # Skip to the next query if extraction fails

                        # Combine and create the final video for the current query
                        output_filename = f"{query.replace(' ', '_')}_final_video.mp4"
                        final_path = clip_and_merge_videos(video_segments, downloaded_video_paths, output_filename)

                        # Check if the final video file exists
                        if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                            st.success(f"Final video for '{query}' created successfully!")
                            st.video(final_path)  # Display the final video
                        else:
                            st.error(f"Failed to create the final video for query: {query}.")
                else:
                    st.error("No segments to combine. Process a query first.")


if __name__ == "__main__":
    main()