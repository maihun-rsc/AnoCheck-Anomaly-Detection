import os
import cv2

RAW_VIDEOS_DIR = r"/Users/Video anomaly/data/raw_videos"

def check_videos():
    print("🔍 Deep Video File Checker 🔍")
    print(f"Base directory: {RAW_VIDEOS_DIR}\n")
    
    # First verify the root raw_videos exists
    if not os.path.exists(RAW_VIDEOS_DIR):
        print(f"❌ CRITICAL: Main directory doesn't exist at {RAW_VIDEOS_DIR}")
        print("Create this structure:")
        print("data/raw_videos/")
        print("├── anomaly/")
        print("└── normal/")
        return
    
    for category in ["anomaly", "normal"]:
        category_dir = os.path.join(RAW_VIDEOS_DIR, category)
        print(f"\n=== Checking {category.upper()} ===")
        print(f"Full path: {category_dir}")
        
        # Check if category folder exists
        if not os.path.exists(category_dir):
            print(f"❌ Missing {category} folder!")
            print(f"Create this folder: {category_dir}")
            continue
            
        # List all files
        all_files = os.listdir(category_dir)
        print(f"\nAll items in folder ({len(all_files)} total):")
        for f in all_files[:5]:  # Show first 5 items
            print(f"  - {f} {'(DIR)' if os.path.isdir(os.path.join(category_dir, f)) else ''}")
        if len(all_files) > 5:
            print(f"  (...and {len(all_files)-5} more)")
        
        # Filter video files
        video_exts = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI")  # Added common variants
        videos = [f for f in all_files if f.lower().endswith(video_exts)]
        
        print(f"\nFound {len(videos)} video files:")
        if not videos:
            print("❌ No videos found! Supported formats: .mp4, .avi, .mov")
            continue
            
        # Test reading the first video
        test_video = os.path.join(category_dir, videos[0])
        print(f"\nTesting first video: {videos[0]}")
        
        try:
            cap = cv2.VideoCapture(test_video)
            if not cap.isOpened():
                print("❌ FAILED: Couldn't open video file (may be corrupted)")
                print("Try opening this file manually with a video player.")
            else:
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"✅ Video opened successfully!")
                print(f"  Frames: {frame_count}")
                print(f"  Resolution: {width}x{height}")
                cap.release()
        except Exception as e:
            print(f"❌ ERROR reading video: {str(e)}")

if __name__ == "__main__":
    check_videos()