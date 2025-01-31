import os
import time
import argparse
import subprocess
from queue import Queue
from threading import Thread
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Define the video file extensions to monitor
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv'}

# Queue to hold file paths for processing
file_queue = Queue()

def process_video(file_path):
    """
    Function to run the SubFlow.py script with the given file path and display its output in real-time.
    :param file_path: Path to the video file to process.
    """
    print(f"Processing video: {file_path}")
    try:
        # Start the SubFlow.py script as a subprocess
        process = subprocess.Popen(
            ["python.exe", "SubFlow.py", file_path, "LOG"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Decode output as text (Python 3.7+)
        )

        # Read and print output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"SubFlow output: {output.strip()}")

        # Capture any remaining errors
        stderr = process.stderr.read()
        if stderr:
            print(f"SubFlow error: {stderr.strip()}")

        # Check the return code
        if process.returncode != 0:
            print(f"SubFlow exited with error code {process.returncode}")

    except Exception as e:
        print(f"Error processing video {file_path}: {e}")

def worker():
    """
    Worker function to process files from the queue.
    """
    while True:
        file_path = file_queue.get()  # Get the next file path from the queue
        if file_path is None:  # Sentinel value to stop the worker
            break
        process_video(file_path)
        file_queue.task_done()  # Mark the task as done

class VideoFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        """
        Triggered when a new file is created in the monitored directory.
        :param event: The file system event.
        """
        if not event.is_directory:  # Ensure it's a file, not a directory
            file_path = event.src_path
            _, ext = os.path.splitext(file_path)
            if ext.lower() in VIDEO_EXTENSIONS:  # Check if the file is a video
                print(f"New video file detected: {file_path}")
                file_queue.put(file_path)  # Add the file to the queue

def start_monitoring(folder_path):
    """
    Start monitoring the specified folder for new video files.
    :param folder_path: Path to the folder to monitor.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"The specified path is not a valid directory: {folder_path}")

    # Start the worker thread
    worker_thread = Thread(target=worker, daemon=True)
    worker_thread.start()

    # Create an event handler and observer
    event_handler = VideoFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=folder_path, recursive=True)  # Recursive for subfolders

    print(f"Monitoring folder: {folder_path}")
    observer.start()

    try:
        while True:
            time.sleep(1)  # Keep the daemon running
    except KeyboardInterrupt:
        print("Stopping the monitor...")
        observer.stop()
    observer.join()

    # Stop the worker thread gracefully
    file_queue.put(None)  # Sentinel value to stop the worker
    worker_thread.join()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Monitor a folder for new video files and process them.")
    parser.add_argument("folder", type=str, help="Path to the folder to monitor")
    args = parser.parse_args()

    # Start the monitoring daemon
    start_monitoring(args.folder)