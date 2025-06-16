import os

RAW_VIDEOS_DIR = r"/Users/Video anomaly/data/raw_videos"

def check_videos():
    print("🔍 Checking video files...")
    
    for category in ["anomaly", "normal"]:
        category_dir = os.path.join(RAW_VIDEOS_DIR, category)
        print(f"\nChecking {category_dir}:")
        
        if not os.path.exists(category_dir):
            print(f"❌ Folder doesn't exist! Create it first.")
            continue
            
        videos = [f for f in os.listdir(category_dir) 
                 if f.lower().endswith((".mp4", ".avi", ".mov"))]
        
        if not videos:
            print("❌ No videos found! Add MP4/AVI/MOV files here.")
        else:
            print(f"✅ Found {len(videos)} videos:")
            for v in videos[:3]:  # Print first 3 as sample
                print(f"  - {v}")
            if len(videos) > 3:
                print(f"  (...and {len(videos)-3} more)")

if __name__ == "__main__":
    check_videos()