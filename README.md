# 🎥 Video Anomaly Detection  

Video Anomaly Detection is all about spotting weird or unexpected stuff in video footage—like a security camera catching someone sneaking around or a glitch in a live stream. This project uses Random Forest, a machine learning algorithm, to analyze video frames and flag anything that doesn’t fit the norm, with Pandas handling the data crunching. Think of it as giving your video feed a smart pair of eyes! Built with Python.  

---

## 🚀 **Features**  
- Analyze video streams to spot anomalies in real-time.  
- Leverage Random Forest for accurate pattern detection.  
- Process and visualize data with Pandas.  
- Lightweight and easy to integrate into security systems.  

---

## 🛠️ **Setup Instructions**  

1. **Clone the repository:**  
   ```bash
   git clone https://github.com/maihun-rsc/AnoCheck-Anomaly-Detection.git
   ```

2. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Get the dataset:**  
   - Download the dataset from [Dataset Link](https://www.dropbox.com/scl/fo/2aczdnx37hxvcfdo4rq4q/AOjRokSTaiKxXmgUyqdcI6k?rlkey=5bg7mxxbq46t7aujfch46dlvz&e=2&dl=0) and place it in the project directory (e.g., a `data/` folder).  
   - Alternatively, update the script with the path to your own video dataset.  

4. **Prepare your video data:**  
   - Add your video files (e.g., `.mp4`, `.avi`) to the project directory or specify the path in the script.  

---

## 💻 **How to Run**  
1. Start the anomaly detection script:  
   ```bash
   python anomaly_detection.py
   ```  
2. Check the output for detected anomalies (e.g., logs, visualizations, or alerts).  

---

## ✨ **How to Use**  

### **Detecting Anomalies:**  
- Input a video file or stream URL to the script.  
- Run the detection—Random Forest analyzes frames for unusual patterns.  
- Check the console or output files for anomaly reports (e.g., timestamps, frame numbers).  

---

## 🌐 **Demo**  

Check out the working demo below:  
![Demo Screenshot](https://github.com/LexViper/Video-Anomaly-Detection/blob/main/Output/output.png)  

---

## 📂 **Project Structure**  
- `anomaly_detection.py`: Main script for running the detection system.  
- `requirements.txt`: Python dependencies (e.g., pandas, scikit-learn).  
- `README.md`: Project documentation.  
- `images/`: Folder for demo screenshots or visuals (optional).  
- `data/`: Suggested folder for storing the dataset (optional).  

---


## 📧 **Created by**

1. Abhay Chaudhary - 23BAI10760
   (uploaded on his GitHub as well. Link: https://github.com/LexViper/video-anomaly-detection)
2. Rananjay Singh Chauhan - 23BAI10080 (me)
3. Anugya Chaubey - 23BAI10550
4. Harsh Damodar Pandit - 23BAI10772
5. Tushar Kumar Tiwari - 23BAI10551
6. Nidhi Joshi - 23BAI10551
